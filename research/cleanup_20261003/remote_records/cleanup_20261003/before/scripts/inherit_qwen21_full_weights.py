#!/usr/bin/env python3
"""Create a full-controller initialization checkpoint from an audited whitelist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.audit_qwen21_weight_compatibility import (
    RUN, SOURCES, apply_mapping_in_memory, cases, file_sha, local_checks,
    module_source_evidence, read_checkpoint, state_hashes,
)
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter
from src.pose_control.qwen21_full.execution import ARCHITECTURE, load_adapter_checkpoint, new_adapter


def verify_state(actual, receiver, source, entries):
    """Exact verification, including every unselected embedding row."""
    if set(actual) != set(receiver):
        raise ValueError('Full architecture parameter keys changed')
    selected = {entry['target']: entry for entry in entries}
    preserved, partial = 0, []
    for name, tensor in actual.items():
        before = receiver[name]
        if tensor.shape != before.shape or tensor.dtype != before.dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid output tensor: {name}')
        entry = selected.get(name)
        if entry is None:
            if not torch.equal(tensor, before):
                raise ValueError(f'Changed outside whitelist: {name}')
            preserved += 1
            continue
        expected = source[entry['source']].to(tensor)
        rows = entry.get('rows')
        if rows is None:
            if not torch.equal(tensor, expected):
                raise ValueError(f'Inherited value differs: {name}')
        else:
            rest = [index for index in range(tensor.shape[0]) if index not in rows]
            if not torch.equal(tensor[rows], expected[rows]) or not torch.equal(tensor[rest], before[rest]):
                raise ValueError(f'Embedding selection violated: {name}')
            partial.append({'target': name, 'copied_rows': rows, 'preserved_rows': rest})
    return {'passed': True, 'mapped_tensors': len(entries), 'fully_preserved_tensors': preserved,
            'partial_embeddings': partial, 'all_keys_shapes_dtypes_preserved': True}


def validate_policy(policy, source_id):
    if source_id not in ('body512', 'body1024'):
        raise ValueError('Only the two audited Body sources are allowed')
    entries = policy['entries']
    if len(entries) != 208:
        raise ValueError('The approved policy must contain exactly 208 tensor mappings')
    partial = [entry for entry in entries if 'rows' in entry]
    expected = {'condition_injector.spatial_encoder.role_film.role_embedding.weight',
                'condition_injector.task_encoder.embedding.weight', 'reasoner.role_embedding.weight'}
    if {entry['source'] for entry in partial} != expected or any(entry['rows'] != [0] for entry in partial):
        raise ValueError('The three approved partial embeddings must copy only row zero')
    for entry in entries:
        if not entry['source'].startswith(('condition_injector.', 'reasoner.', 'condition_bridge.')):
            raise ValueError('Mapping reaches an unapproved module')
        if entry['target'] != 'legacy.' + entry['source']:
            raise ValueError('Unexpected name conversion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id', required=True, choices=('body512', 'body1024'))
    parser.add_argument('--receiver', type=Path, default=RUN / 'face_512_100/checkpoint_last.pt')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--whitelist', type=Path, default=RUN / 'weight_audit/load_whitelists.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.source = args.source or SOURCES[args.source_id]
    args.output = args.output or RUN / 'weight_inheritance' / f'inherit_{args.source_id}.pt'
    report_path = args.output.with_suffix('.json')
    if args.output.exists() or report_path.exists():
        raise FileExistsError('Use a new output path; existing checkpoints are never overwritten')
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    policy = json.loads(args.whitelist.read_text())[args.source_id]
    validate_policy(policy, args.source_id)
    fingerprint = {'receiver': file_sha(args.receiver), 'source': file_sha(args.source),
                   'whitelist': file_sha(args.whitelist)}
    if fingerprint['receiver'] != policy['receiver_sha256'] or fingerprint['source'] != policy['source_sha256']:
        raise ValueError('Checkpoint differs from the source or receiver approved in the audit')
    audit = json.loads((args.whitelist.parent / 'summary.json').read_text())
    if not audit['passed'] or file_sha(RUN / 'val_pairs.jsonl') != audit['validation_pairs_sha256']:
        raise ValueError('Audit was not passed or the fixed validation pairs changed')
    adapter = new_adapter('cpu').eval().requires_grad_(False)
    receiver = load_adapter_checkpoint(adapter, args.receiver)
    original_hashes = state_hashes(adapter)
    saved_source = read_checkpoint(args.source)
    source = saved_source['adapter_state_dict']
    for name, tensor in source.items():
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid source tensor: {name}')
    original = Qwen21BodyAdapter().eval().requires_grad_(False)
    original.load_state_dict(source, strict=True)
    evidence = module_source_evidence(original, adapter)
    apply_mapping_in_memory(adapter, source, policy['entries'], policy['declared_ignored_source_keys'])
    protection = verify_state(adapter.state_dict(), receiver['adapter_state_dict'], source, policy['entries'])
    mapped_hashes = state_hashes(adapter)
    local = local_checks(original, adapter, [case for case in cases() if case[0].startswith('validation_')])
    if state_hashes(adapter) != mapped_hashes:
        raise AssertionError('Local validation changed adapter parameters')
    source_resolution = 512 if args.source_id == 'body512' else 1024
    provenance = {
        'receiver': {'path': str(args.receiver), 'sha256': fingerprint['receiver'],
                     'step': receiver['step'], 'branch': receiver['branch'], 'resolution': receiver['resolution'],
                     'initialization': receiver.get('initialization')},
        'inherited_source': {'id': args.source_id, 'path': str(args.source), 'sha256': fingerprint['source'],
                             'step': saved_source['step'], 'resolution': source_resolution},
        'whitelist_path': str(args.whitelist), 'whitelist_sha256': fingerprint['whitelist'],
        'entries': policy['entries'], 'declared_ignored_source_keys': policy['declared_ignored_source_keys'],
        'source_step_is_not_new_training_progress': True,
    }
    payload = {'architecture': ARCHITECTURE, 'model_revision': receiver['model_revision'],
               'checkpoint_purpose': 'weight_initialization', 'branch': 'weight_initialization', 'step': 0,
               'new_optimizer_steps': 0, 'resolution': receiver['resolution'],
               'adapter_state_dict': adapter.state_dict(), 'initialization': provenance,
               'full_controller_training_resolution': receiver['resolution'],
               'inherited_module_training_resolution': source_resolution,
               'usage': 'Inference or --init-checkpoint only; not a resumable training checkpoint.',
               'no_new_training': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(payload, temporary)
        restored = read_checkpoint(temporary)
        reloaded = verify_state(restored['adapter_state_dict'], receiver['adapter_state_dict'], source, policy['entries'])
        if 'optimizer_state_dict' in restored or restored['step'] != 0:
            raise AssertionError('Initialization checkpoint carried training state')
        adapter.load_state_dict(restored['adapter_state_dict'], strict=True)
        changed = [name for name, digest in state_hashes(adapter).items() if digest != original_hashes[name]]
        for name, path in [('receiver', args.receiver), ('source', args.source), ('whitelist', args.whitelist)]:
            if file_sha(path) != fingerprint[name]:
                raise AssertionError(f'Input file changed: {path}')
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {'passed': True, 'source_id': args.source_id, 'output': str(args.output),
              'output_sha256': file_sha(args.output), 'provenance': provenance,
              'protection': protection, 'reload_verification': reloaded,
              'local_checks': local, 'module_evidence': evidence,
              'actually_changed_tensor_count': len(changed), 'actually_changed_tensors': changed,
              'source_files_unchanged': True, 'training_performed': False, 'optimizer_state_inherited': False,
              'seconds': time.perf_counter() - started}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'source_id': args.source_id, 'passed': True, 'output': str(args.output),
                      'changed_tensors': len(changed), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
