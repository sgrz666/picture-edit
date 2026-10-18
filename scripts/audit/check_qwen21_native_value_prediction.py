#!/usr/bin/env python3
"""Direct real-Qwen prediction parity for the fixed native-value audit.

This does not train, sample complete images, or encode B RGB as a latent.
The target latent is fixed synthetic noise. A reference latent, text/image
prefix, sigma and all Body conditions remain unchanged across each comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path

import torch

from scripts.eval.eval_qwen21_native_value import (
    ROOT, RUN, BODY_PATH, FACE_PATH, AuditDataset, AuditFaceStore, SamplingWitness,
    json_write, read_rows, read_prompts, save_inputs, sha256,
    verify_frozen_body_checkpoint_pair,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=RUN / 'data/pairs.jsonl')
    parser.add_argument('--prompts', type=Path, default=RUN / 'prompts/locked_prompts.json')
    parser.add_argument('--pair-ids', nargs='*', default=[],
                        help='Default: first iPER and first TikTok locked case, at most two')
    parser.add_argument('--output', type=Path, default=RUN / 'acceptance/direct_prediction')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--body-checkpoint', type=Path, default=BODY_PATH)
    parser.add_argument('--face-checkpoint', type=Path, default=FACE_PATH)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--validate-only', action='store_true', help='Check locked inputs and Face source contracts on CPU')
    return parser.parse_args()


def tensor_tree(value, prefix=''):
    if torch.is_tensor(value):
        return {prefix: value}
    if is_dataclass(value):
        output = {}
        for field in fields(value):
            output.update(tensor_tree(getattr(value, field.name), f'{prefix}.{field.name}'))
        return output
    if isinstance(value, (tuple, list)):
        output = {}
        for index, item in enumerate(value):
            output.update(tensor_tree(item, f'{prefix}.{index}'))
        return output
    if isinstance(value, dict):
        output = {}
        for name, item in value.items():
            output.update(tensor_tree(item, f'{prefix}.{name}'))
        return output
    return {}


def compare_body_conditions(a, b):
    """A Face preparation must not silently change any Body/shared condition."""
    a = {name: value for name, value in tensor_tree(a).items() if not name.startswith('.face.')}
    b = {name: value for name, value in tensor_tree(b).items() if not name.startswith('.face.')}
    if set(a) != set(b):
        raise AssertionError('Prepared Body condition schemas differ')
    changed = [name for name in a if a[name].shape != b[name].shape or not torch.equal(a[name], b[name])]
    if changed:
        raise AssertionError(f'Face preparation changed Body/shared conditions: {changed}')
    return {'tensors_compared': len(a), 'exactly_equal': True}


def invalidate_face_only(prepared):
    if prepared.face is None:
        raise ValueError('Cannot test invalid Face without a valid prepared Face branch')
    # PreparedFaceConditioning's existing schema requires every tensor of an
    # invalid person to be zero. Clear Face records only; preserve every Body
    # and shared tensor by reference, including the source identity binding.
    empty = type(prepared.face)(**{field.name: torch.zeros_like(getattr(prepared.face, field.name))
                                  for field in fields(prepared.face)}).validate()
    invalid = replace(prepared, face=empty).validate()
    compare_body_conditions(prepared, invalid)
    return invalid


def parity(a, b, atol, *, require_exact=False):
    maximum = float((a.float() - b.float()).abs().max())
    equal = torch.equal(a, b)
    return {'max_abs': maximum, 'exactly_equal': equal, 'atol': atol,
            'require_exact': require_exact,
            'passed': equal if require_exact else maximum <= atol}


def predict(pipe, adapter, prepared, inputs, *, resolution, disabled=False, face_strength=0):
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
    kwargs = dict(face_strength=face_strength, hand_strength=0, hand_mode='off',
                  progress_mode='sigma', num_inference_steps=1)
    if disabled:
        kwargs.update(geometry_strength=0, interaction_strength=0, detail_strength=0, face_strength=0)
    context = None if adapter is None else Qwen21FullControlContext(pipe.transformer, adapter, prepared, **kwargs)
    witness = SamplingWitness(pipe.transformer, context, 1, resolution // 16, prepared)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), witness:
        result = pipe.transformer(**inputs)[0].detach().cpu().clone()
    if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
        raise AssertionError('Control hook leaked after parity forward')
    return result, witness.record()


def main():
    args = arguments()
    torch.set_num_threads(4)
    all_rows = read_rows(args.pairs)
    by_id = {row['pair_id']: row for row in all_rows}
    if args.pair_ids:
        if len(set(args.pair_ids)) != len(args.pair_ids) or len(args.pair_ids) > 2:
            raise ValueError('Use one or two unique already-locked cases')
        rows = [by_id[name] for name in args.pair_ids]
    else:
        rows = [next(row for row in all_rows if row['dataset'] == name)
                for name in ('iper', 'tiktok') if any(row['dataset'] == name for row in all_rows)]
    prompts = read_prompts(args.prompts, rows, ['full_body', 'full_body_face'])
    if any(prompts[row['pair_id']]['full_body'] != prompts[row['pair_id']]['full_body_face'] for row in rows):
        raise AssertionError('Parity and Face increment require the same detailed text prefix')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'pairs': str(args.pairs), 'pairs_sha256': sha256(args.pairs),
                'prompts': str(args.prompts), 'prompts_sha256': sha256(args.prompts),
                'pair_ids': [row['pair_id'] for row in rows], 'resolution': args.resolution,
                'sigma': 0.5, 'seed': args.seed, 'atol': args.atol,
                'body_checkpoint': str(args.body_checkpoint), 'face_checkpoint': str(args.face_checkpoint),
                'body_checkpoint_sha256': sha256(args.body_checkpoint),
                'face_checkpoint_sha256': sha256(args.face_checkpoint),
                'latent_source': 'fixed synthetic noise; no B RGB/Target clean latent encoded',
                'inference_steps': 0, 'optimizer_steps': 0,
                'weight_check': verify_frozen_body_checkpoint_pair(args.body_checkpoint, args.face_checkpoint)}
    json_write(args.output / 'manifest.json', manifest)
    dataset, store = AuditDataset(rows, args.resolution, args.output), AuditFaceStore(rows)
    inputs_data = [save_inputs(row, args.resolution, args.output) for row in rows]
    if args.validate_only:
        checks = []
        for index, row in enumerate(rows):
            images, paths, alignment, crop = inputs_data[index]
            sample = dataset.sample(index, images, alignment)
            condition, references = store.for_pairs([row], device='cpu', dtype=torch.float32, alignments=[alignment])
            if not condition.face_valid.any():
                raise AssertionError('Direct parity sample must have a reliable A reference and valid B face geometry')
            checks.append({'pair_id': row['pair_id'], 'face_valid': condition.face_valid.tolist(),
                           'source_arcface_nonzero': bool(references.arcface.count_nonzero()),
                           'normal_shape': list(sample['normal'].shape), 'alignment': alignment})
        json_write(args.output / 'cpu_inputs.json', {'passed': True, 'checks': checks, 'gpu_loaded': False})
        print(json.dumps({'status': 'CPU inputs passed', 'cases': len(checks)}), flush=True)
        return
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.execution import new_adapter, load_adapter_checkpoint, prepare, seed_all
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache, transformer_inputs
    seed_all(args.seed)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe.transformer, 'peft_config', {}):
        raise AssertionError('Acceptance must not load LoRA')
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    results = []
    for index, row in enumerate(rows):
        images, paths, alignment, crop = inputs_data[index]
        sample = dataset.sample(index, images, alignment)
        batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                 for name, value in sample.items()}
        prompt = prompts[row['pair_id']]['full_body']
        cache = Qwen21IPERCache(args.output / 'native_cache' / row['pair_id'], prompt=prompt,
                               resolution=args.resolution, preprocessing='letterbox_v1')
        condition = cache.source(pipe, images['A'])
        grid, tokens = args.resolution // 16, (args.resolution // 16) ** 2
        pair = {'source_latent': condition['reference_latent'].to('cuda'),
                'prompt_embeds': condition['prompt_embeds'].to('cuda'),
                'prompt_mask': condition['prompt_mask'].to('cuda'),
                'img_mask': torch.cat((condition['image_mask'], torch.ones(1, tokens // 4, dtype=torch.bool)), dim=1).to('cuda'),
                'resolution': args.resolution}
        noise = torch.randn((1, tokens, 64), generator=torch.Generator(device='cuda').manual_seed(args.seed),
                            device='cuda', dtype=pair['source_latent'].dtype)
        sigma = torch.tensor([.5], device='cuda', dtype=pair['source_latent'].dtype)
        inputs = transformer_inputs(pair, noise, sigma)
        input_fingerprint = hashlib.sha256(noise.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        load_adapter_checkpoint(adapter, args.body_checkpoint)
        with torch.no_grad():
            prepared_body = prepare(adapter, batch, pair, [row], store, with_face=False)
        native, native_proof = predict(pipe, None, None, inputs, resolution=args.resolution)
        disabled, disabled_proof = predict(pipe, adapter, prepared_body, inputs,
                                           resolution=args.resolution, disabled=True)
        body, body_proof = predict(pipe, adapter, prepared_body, inputs, resolution=args.resolution)
        load_adapter_checkpoint(adapter, args.face_checkpoint)
        with torch.no_grad():
            prepared_face = prepare(adapter, batch, pair, [row], store, with_face=True)
        if prepared_face.face is None or not prepared_face.face.face_valid.any():
            raise AssertionError('The unchanged real Face input must be valid before fallback testing')
        body_condition_check = compare_body_conditions(prepared_body, prepared_face)
        invalid = invalidate_face_only(prepared_face)
        face_off, face_off_proof = predict(pipe, adapter, prepared_face, inputs,
                                           resolution=args.resolution, face_strength=0)
        face_invalid, face_invalid_proof = predict(pipe, adapter, invalid, inputs,
                                                   resolution=args.resolution, face_strength=1)
        comparisons = {'all_control_disabled_vs_native': parity(native, disabled, args.atol),
                       'face_checkpoint_Face_off_vs_body_checkpoint': parity(body, face_off, args.atol, require_exact=True),
                       'face_checkpoint_invalid_Face_vs_body_checkpoint': parity(body, face_invalid, args.atol, require_exact=True)}
        result = {'pair_id': row['pair_id'], 'dataset': row['dataset'], 'resolution': args.resolution,
                  'sigma': .5, 'noise_sha256': input_fingerprint,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'latent_input_shape': list(noise.shape), 'prediction_shape': list(native.shape),
                  'comparisons': comparisons, 'Body_condition_parity': body_condition_check,
                  'observations': {'native': native_proof, 'all_control_disabled': disabled_proof,
                                   'body': body_proof, 'Face_off': face_off_proof, 'invalid_Face': face_invalid_proof},
                  'all_models_frozen': all(not p.requires_grad for module in (pipe.transformer, pipe.vae, pipe.text_encoder, adapter)
                                           for p in module.parameters()),
                  'parameter_gradients_absent': all(p.grad is None for module in (pipe.transformer, adapter) for p in module.parameters()),
                  'passed': all(item['passed'] for item in comparisons.values())}
        results.append(result)
        json_write(args.output / 'results.json', {'passed': all(item['passed'] for item in results), 'cases': results})
        print(json.dumps({'pair_id': row['pair_id'], 'comparisons': comparisons, 'passed': result['passed']}), flush=True)
        if not result['passed'] or not result['all_models_frozen'] or not result['parameter_gradients_absent']:
            raise AssertionError('Real native/Body/Face prediction parity failed; preserve results for investigation')
    print('PASS: real-model prediction parity; no complete image generation and no optimizer updates', flush=True)


if __name__ == '__main__':
    main()
