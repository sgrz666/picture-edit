#!/usr/bin/env python3
"""CPU-only, per-tensor provenance of the preserved V6/Qwen initial model.

This describes initialization, not a claim that every branch was trained in the
single-person experiments. Checkpoint contents are checked before any omission
is accepted. No model weights or training state are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.pose_control.qwen21_full.checkpoint import FACE_ONLY_PREFIXES
from src.pose_control.qwen21_full.execution import BODY_CHECKPOINT, FACE_CHECKPOINT, BODY_PREFIXES, RUN_ROOT
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def short_descriptor(name, tensor, *, source):
    return dict(key=name, source=source, shape=list(tensor.shape), dtype=str(tensor.dtype), numel=tensor.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-checkpoint', type=Path, default=BODY_CHECKPOINT)
    parser.add_argument('--face-checkpoint', type=Path, default=FACE_CHECKPOINT)
    parser.add_argument('--deepgen-config', type=Path, default=ROOT / 'models/DeepGen-1.0-diffusers/transformer/config.json')
    parser.add_argument('--output', type=Path, default=RUN_ROOT / 'acceptance/weight_inventory.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    body_saved = torch.load(args.body_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    face_saved = torch.load(args.face_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    body, face = body_saved['adapter_state_dict'], face_saved['adapter_state_dict']
    adapter = Qwen21FullAdapter.from_checkpoints(args.deepgen_config, args.body_checkpoint, args.face_checkpoint)
    initial = adapter.state_dict()
    legacy = adapter.legacy.state_dict()
    parameters = set(dict(adapter.named_parameters()))
    ignored = {}
    for name, value in face.items():
        if name.startswith(FACE_ONLY_PREFIXES):
            continue
        # The selective loader legitimately ignores known non-Face legacy state,
        # not arbitrary unexpected tensors. Unknown/malformed keys are failures.
        if name not in legacy:
            raise RuntimeError(f'Unexpected non-Face checkpoint key cannot be silently ignored: {name}')
        if not torch.is_tensor(value) or value.shape != legacy[name].shape:
            raise RuntimeError(f'Ignored non-Face tensor has incompatible shape: {name}')
        if not torch.isfinite(value).all():
            raise RuntimeError(f'Ignored non-Face tensor is nonfinite: {name}')
        ignored[name] = {
            **short_descriptor(name, value, source='Face prototype checkpoint; not loaded'),
            'reason': 'Known non-Face state is excluded so the Face prototype cannot overwrite the Body20k state.',
            'initialization_used_instead': 'Body20k' if name in body else 'new preserved legacy module; inactive in this acceptance run',
        }
    entries = []
    for name, tensor in initial.items():
        entry = short_descriptor(name, tensor, source='')
        entry['kind'] = 'parameter' if name in parameters else 'buffer'
        entry['checkpoint_key'] = None
        entry['checkpoint_dtype'] = None
        entry['checkpoint_value_verified_exact'] = None
        entry['trained_in_current_bounded_run'] = []
        if name.startswith('legacy.'):
            key = name[len('legacy.'):]
            if key in body:
                source_tensor = body[key]
                entry.update(source='inherited_DeepGen_Body20k', checkpoint_key=key,
                             checkpoint_dtype=str(source_tensor.dtype), checkpoint_value_verified_exact=torch.equal(tensor, source_tensor.to(tensor.dtype)))
                if key.startswith(BODY_PREFIXES):
                    entry['previous_training_status'] = 'included_in_single_person_Body_optimizer'
                    if name in parameters:
                        entry['trained_in_current_bounded_run'] = ['body']
                else:
                    entry['previous_training_status'] = 'inherited_but_not_updated_by_single_person_Body_training'
            elif key.startswith(FACE_ONLY_PREFIXES) and key in face:
                source_tensor = face[key]
                entry.update(source='inherited_DeepGen_Face_single_pair_1500', checkpoint_key=key,
                             checkpoint_dtype=str(source_tensor.dtype), checkpoint_value_verified_exact=torch.equal(tensor, source_tensor.to(tensor.dtype)),
                             previous_training_status='Face_prototype_only; not a full-data generalization claim')
                if name in parameters:
                    entry['trained_in_current_bounded_run'] = ['face']
            else:
                entry.update(source='new_initialization_of_preserved_legacy_module',
                             previous_training_status='not_inherited; untrained_in_this_run',
                             known_but_intentionally_ignored_in_face_checkpoint=key in ignored)
        else:
            entry.update(source='new_Qwen_boundary_parameter', previous_training_status='new_interface')
            if name.startswith(('latent_bridge.', 'text_bridge.', 'pooled_bridge.', 'output_bridges.geometry.')):
                entry['trained_in_current_bounded_run'] = ['body']
            elif name.startswith('output_bridges.face.'):
                entry['trained_in_current_bounded_run'] = ['face']
            if name.startswith('output_bridges.'):
                branch = name.split('.')[1]
                entry['initialization'] = adapter.loading_report['output_initialization'][branch]
            elif name.startswith(('latent_bridge.', 'text_bridge.')):
                entry['initialization'] = 'identity weight and zero bias'
            elif name.startswith('pooled_bridge.'):
                entry['initialization'] = 'torch Linear default random initialization'
        if entry['kind'] == 'buffer':
            entry['trained_in_current_bounded_run'] = []
            entry['previous_training_status'] = 'fixed_buffer; inherited source is recorded separately'
        if entry['checkpoint_value_verified_exact'] is False:
            raise AssertionError(f'Loaded value does not equal selected checkpoint: {name}')
        entries.append(entry)
    payload = dict(
        purpose='Complete initialization provenance, including inherited but inactive and newly untrained modules',
        checkpoint_sources={
            'body': dict(path=str(args.body_checkpoint), sha256=sha256_file(args.body_checkpoint), tensors=len(body)),
            'face': dict(path=str(args.face_checkpoint), sha256=sha256_file(args.face_checkpoint), tensors=len(face)),
        },
        definitions={
            'trained_in_current_bounded_run': 'Optimizer membership for the planned Body/Face runs; actual completed steps are recorded in training logs, not inferred here.',
            'previous_training_status': 'Original entrypoint/module eligibility; a parameter tensor can contain inactive rows or zero gradients on individual samples.',
            'new_initialization_of_preserved_legacy_module': 'Architecture retained; its trained state is not substituted from the Face prototype. These detail/hand modules stay inactive in this single-person acceptance.',
        },
        summary=dict(total_state_tensors=len(entries), total_parameter_elements=sum(p.numel() for p in adapter.parameters()),
                     by_source=dict(Counter(row['source'] for row in entries)),
                     by_training_membership=dict(Counter('+'.join(row['trained_in_current_bounded_run']) or 'inactive_or_buffer' for row in entries)),
                     face_nonface_tensors_ignored=len(ignored), unexpected_keys=[], shape_mismatches=[],
                     all_selected_checkpoint_values_verified_exact=True),
        loading_report=adapter.loading_report, tensors=entries,
        ignored_face_checkpoint_tensors=list(ignored.values()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(args.output)
    print(json.dumps(payload['summary'], ensure_ascii=False), flush=True)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
