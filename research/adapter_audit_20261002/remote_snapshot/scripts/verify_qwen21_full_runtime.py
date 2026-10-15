#!/usr/bin/env python3
"""Bounded GPU acceptance for the full Qwen controller; never updates weights.

Run only after the Body/Face checkpoints exist and the GPU is free. This checks
velocity predictions and one backward per branch per sample, not image-quality scores
or complete sampling. Face backward intentionally uses only flow loss: it does
not claim to verify differentiable VAE/identity/perceptual losses at 1024.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline

from src.pose_control.qwen21_full.checkpoint import FACE_ONLY_PREFIXES
from src.pose_control.qwen21_full.data_runtime import FaceConditionStore, QwenFullDataset, pair_metadata
from src.pose_control.qwen21_full.execution import (
    ARCHITECTURE, MODEL, RUN_ROOT, batch_one, face_mask, json_write,
    load_adapter_checkpoint, new_adapter, parameter_groups, prepare, seed_all,
    set_training_phase,
)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
from src.pose_control.qwen21_full.qwen_runtime import (
    MODEL_REVISION, PROMPT, Qwen21IPERCache, dataset_image_to_pil, transformer_inputs,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Completed Face checkpoint containing frozen Body and adapted Face.')
    parser.add_argument('--body-checkpoint', type=Path,
                        default=RUN_ROOT / 'body_512_100/checkpoint_last.pt')
    parser.add_argument('--resolution', type=int, choices=(512, 1024), required=True)
    parser.add_argument('--max-samples', type=int, default=2)
    parser.add_argument('--output', type=Path,
                        help='JSON output file; defaults to acceptance/runtime_RESOLUTION.json.')
    parser.add_argument('--pairs', type=Path, default=RUN_ROOT / 'val_pairs.jsonl')
    parser.add_argument('--features', type=Path, default=RUN_ROOT / 'face_cache')
    parser.add_argument('--cache', type=Path, default=RUN_ROOT / 'native_cache')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--atol', type=float, default=0.001,
                        help='Absolute tolerance for native BF16 velocity parity; also records exact equality.')
    result = parser.parse_args()
    if result.max_samples < 1 or result.atol < 0:
        parser.error('max-samples must be positive and atol nonnegative')
    result.output = result.output or RUN_ROOT / 'acceptance' / f'runtime_{result.resolution}.json'
    return result


def cpu_checkpoint_comparison(body_path, face_path):
    """A Face-stage checkpoint must preserve every non-Face parameter exactly."""
    body = torch.load(body_path, map_location='cpu', weights_only=False, mmap=True)
    face = torch.load(face_path, map_location='cpu', weights_only=False, mmap=True)
    if body.get('architecture') != ARCHITECTURE or face.get('architecture') != ARCHITECTURE:
        raise ValueError('Both checkpoints must belong to qwen21_full_v1')
    a, b = body['adapter_state_dict'], face['adapter_state_dict']
    if set(a) != set(b):
        raise ValueError('Body and Face checkpoint model keys differ')
    changed, compared, omitted = [], 0, 0
    for name, value in a.items():
        if name.startswith('output_bridges.face.') or (
            name.startswith('legacy.') and name[len('legacy.'):].startswith(FACE_ONLY_PREFIXES)
        ):
            omitted += 1
            continue
        compared += 1
        if value.dtype != b[name].dtype or value.shape != b[name].shape or not torch.equal(value, b[name]):
            changed.append(name)
    result = dict(body_step=body.get('step'), face_step=face.get('step'),
                  body_branch=body.get('branch'), face_branch=face.get('branch'),
                  checked_tensors=compared, face_tensors_excluded=omitted,
                  changed_nonface_tensors=changed, passed=not changed)
    if changed:
        raise AssertionError(f'Face checkpoint changed non-Face tensors: {changed}')
    if face.get('branch') != 'face':
        raise ValueError('--checkpoint must be a Face-stage checkpoint')
    return result


def tensor_leaves(value, prefix=''):
    if torch.is_tensor(value):
        yield prefix, value
    elif is_dataclass(value):
        for item in fields(value):
            yield from tensor_leaves(getattr(value, item.name), f'{prefix}.{item.name}')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, f'{prefix}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f'{prefix}.{index}')


def compare_prepared(first, second):
    a, b = dict(tensor_leaves(first)), dict(tensor_leaves(second))
    if set(a) != set(b):
        raise AssertionError('Prepared tensor schemas differ')
    changed = [name for name, value in a.items() if not torch.equal(value, b[name])]
    return dict(checked_tensors=len(a), changed_tensors=changed, exactly_equal=not changed)


def empty_face(face):
    return type(face)(**{item.name: torch.zeros_like(getattr(face, item.name)) for item in fields(face)}).validate()


def prediction(pipe, adapter, prepared, inputs, *, face_strength):
    with Qwen21FullControlContext(pipe.transformer, adapter, prepared, face_strength=face_strength,
                                 hand_strength=0, hand_mode='off') as context:
        output = pipe.transformer(**inputs)[0]
        tokens = math.prod(inputs['img_shapes'][0][-1])
        result = output[:, -tokens:].clone()
    return result, context.output.diagnostics


def parity(a, b, atol):
    maximum = float((a.float() - b.float()).abs().max())
    return dict(max_abs=maximum, exactly_equal=torch.equal(a, b), atol=atol, passed=maximum <= atol)


def model_digest(adapter):
    digest = hashlib.sha256()
    for name, value in adapter.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_initial_native_parity(pipe, adapter, cache, dataset, store, index, args):
    """Exercise real inherited nonzero heads through zero new output bridges."""
    if any(bool(torch.count_nonzero(adapter.output_bridges[name].weight)) for name in ('geometry', 'face')):
        raise AssertionError('Native parity requires the true newly initialized geometry/Face output bridges')
    device = torch.device('cuda')
    batch = batch_one(dataset, index)
    pair = cache.pair(pipe, batch, device)
    target = pair['target_latent']
    generator = torch.Generator(device=device).manual_seed(args.seed + 3000 + index)
    sigma = target.new_tensor([.2])
    noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    inputs = transformer_inputs(pair, noisy, sigma)
    tokens = (args.resolution // 16) ** 2
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prepared = prepare(adapter, batch, pair, [dataset.samples[index]], store, with_face=True)
        native = pipe.transformer(**inputs)[0][:, -tokens:]
        with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                geometry_strength=0, interaction_strength=0, detail_strength=0,
                face_strength=0, hand_strength=0, hand_mode='off'):
            disabled = pipe.transformer(**inputs)[0][:, -tokens:]
        initialized, diagnostics = prediction(pipe, adapter, prepared, inputs, face_strength=1)
    zero_parity = parity(native, initialized, args.atol)
    disabled_parity = parity(native, disabled, args.atol)
    raw = diagnostics['raw_branch_rms']
    raw_heads_exercised = (any(value > 0 for value in raw['geometry'])
                          and raw['face'] is not None and any(value > 0 for value in raw['face']))
    result = dict(pair_index=index, sigma=float(sigma),
                  initialized_vs_native=zero_parity, disabled_vs_native=disabled_parity,
                  raw_branch_rms=raw, inherited_nonzero_heads_exercised=raw_heads_exercised,
                  initialized_residual_rms=diagnostics['residual_rms'],
                  passed=zero_parity['passed'] and disabled_parity['passed'] and raw_heads_exercised)
    if not result['passed']:
        raise AssertionError(f'Initial native parity failed: {result}')
    return result


def verify_pair(pipe, adapter, cache, dataset, store, index, args):
    device = torch.device('cuda')
    batch = batch_one(dataset, index)
    metadata = [dataset.samples[index]]
    pair = cache.pair(pipe, batch, device)
    grid = args.resolution // 16
    generator = torch.Generator(device=device).manual_seed(args.seed + index)
    noise = torch.randn(pair['target_latent'].shape, generator=generator,
                        device=device, dtype=pair['target_latent'].dtype)
    sigma = noise.new_tensor([0.2])
    noisy = (1 - sigma[:, None, None]) * pair['target_latent'] + sigma[:, None, None] * noise
    inputs = transformer_inputs(pair, noisy, sigma)
    with torch.no_grad():
        prepared_body = prepare(adapter, batch, pair, metadata, store, with_face=False)
        prepared_face = prepare(adapter, batch, pair, metadata, store, with_face=True)
        if not bool(prepared_face.face.face_valid.any()):
            raise AssertionError(f'Validation pair {index} has no valid face; acceptance requires a valid face case')
        # B geometry is intentionally held fixed. Replace only Target RGB and
        # clean supervised latent; neither may enter static generation conditions.
        changed_batch = dict(batch, tgt_image=-batch['tgt_image'])
        changed_pair = dict(pair, target_latent=torch.full_like(pair['target_latent'], float('nan')))
        changed = prepare(adapter, changed_batch, changed_pair, metadata, store, with_face=True)
        leakage = compare_prepared(prepared_face, changed)
        if not leakage['exactly_equal']:
            raise AssertionError(f'Target RGB/clean latent entered conditions: {leakage}')
        del changed, changed_pair, changed_batch
        disabled_face = replace(prepared_face, face=empty_face(prepared_face.face)).validate()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            body, _ = prediction(pipe, adapter, prepared_body, inputs, face_strength=0)
            face_off, _ = prediction(pipe, adapter, prepared_face, inputs, face_strength=0)
            invalid, _ = prediction(pipe, adapter, disabled_face, inputs, face_strength=1)
            comparisons = dict(face_off_vs_body=parity(body, face_off, args.atol),
                               invalid_face_vs_body=parity(body, invalid, args.atol))
            del body, face_off, invalid
            full = adapter.build_residuals(noisy_target=noisy, prepared=prepared_face,
                    prompt_embeds=pair['prompt_embeds'], prompt_mask=pair['prompt_mask'],
                    sigma=sigma, target_token_hw=(grid, grid), face_strength=1,
                    hand_strength=0, hand_mode='off')
            raw_face = full.raw_residuals.face
            if raw_face is None:
                raise AssertionError('Face residual branch did not execute')
            support = face_mask(prepared_face, grid).squeeze(-1) > 0
            projected = [adapter.output_bridges['face'](value) for value in raw_face]
            max_outside = max(float(value[~support].abs().max()) if (~support).any() else 0. for value in projected)
            face_norm = [float(value.float().square().mean().sqrt()) for value in projected]
            roi = dict(outside_max_abs=max_outside, outside_exactly_zero=max_outside == 0,
                       projected_face_rms=face_norm, support_tokens=int(support.sum()),
                       total_tokens=grid * grid,
                       output_projection_nonzero=bool(torch.count_nonzero(adapter.output_bridges['face'].weight)),
                       raw_branch_rms=full.diagnostics['raw_branch_rms'])
            if not roi['outside_exactly_zero'] or not roi['output_projection_nonzero'] or not any(value > 0 for value in face_norm):
                raise AssertionError(f'Nonzero Face/ROI residual acceptance failed: {roi}')
            if not all(row['passed'] for row in comparisons.values()):
                raise AssertionError(f'Face fallback parity failed: {comparisons}')
    return dict(pair_index=index, pair=pair_metadata(metadata[0]), target_rgb_isolation=leakage,
                note='Target geometry remains a condition; only Target RGB and clean supervised latent were replaced.',
                comparisons=comparisons, face_roi=roi, passed=True)


def backward_once(pipe, adapter, cache, dataset, store, index, branch, args):
    adapter.zero_grad(set_to_none=True)
    pipe.transformer.zero_grad(set_to_none=True)
    set_training_phase(adapter, branch, warmup=False)
    adapter.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    device = torch.device('cuda')
    batch = batch_one(dataset, index)
    pair = cache.pair(pipe, batch, device)
    target = pair['target_latent']
    sigma = target.new_tensor([.2 if branch == 'face' else .5])
    generator = torch.Generator(device=device).manual_seed(args.seed + 9000 + index)
    noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    velocity = noise - target
    grid = args.resolution // 16
    inputs = transformer_inputs(pair, noisy, sigma)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prepared = prepare(adapter, batch, pair, [dataset.samples[index]], store, with_face=branch == 'face')
        with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                face_strength=1 if branch == 'face' else 0, hand_strength=0, hand_mode='off') as context:
            estimate = pipe.transformer(**inputs)[0][:, -(grid * grid):]
            squared = (estimate.float() - velocity.float()).square()
            if branch == 'body':
                human = F.interpolate(batch['human_mask'].to(device=device, dtype=torch.float32),
                                      size=(grid, grid), mode='bilinear', align_corners=False)
                weight = 1 + 2 * human.flatten(2).transpose(1, 2)
                loss = (squared * weight).sum() / (weight.sum() * squared.shape[-1])
            else:
                loss = squared.mean()
            loss.backward()
    torch.cuda.synchronize()
    _, groups = parameter_groups(adapter, branch)
    named = dict(adapter.named_parameters())
    group_norms, nonzero_counts, nonfinite = {}, {}, []
    for name, names in groups.items():
        total = torch.zeros((), device=device)
        count = 0
        for key in names:
            gradient = named[key].grad
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                nonfinite.append(key)
            total += gradient.detach().float().square().sum()
            count += int(bool(torch.count_nonzero(gradient)))
        group_norms[name] = float(total.sqrt())
        nonzero_counts[name] = count
    forbidden_gradients = [name for name, value in named.items() if not value.requires_grad and value.grad is not None]
    base_grads = [name for name, value in pipe.transformer.named_parameters() if value.grad is not None]
    projection = adapter.output_bridges['geometry' if branch == 'body' else 'face'].weight.grad
    raw = context.output.diagnostics['raw_branch_rms']
    raw_active = raw['geometry' if branch == 'body' else 'face']
    result = dict(branch=branch, pair_index=index, sigma=float(sigma), loss=float(loss.detach()),
                  loss_definition='human-weighted flow only' if branch == 'body' else 'flow only',
                  auxiliary_vae_backward_tested=False, optimizer_steps=0,
                  gradient_checkpointing=bool(pipe.transformer.gradient_checkpointing),
                  group_grad_norms=group_norms, nonzero_grad_tensor_counts=nonzero_counts,
                  nonfinite_gradients=nonfinite, forbidden_adapter_gradients=forbidden_gradients,
                  base_parameter_gradients=base_grads, raw_branch_rms=raw,
                  output_projection_gradient_nonzero=projection is not None and bool(torch.count_nonzero(projection)),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                  elapsed_seconds=time.perf_counter() - started)
    result['passed'] = (math.isfinite(result['loss']) and not nonfinite and not forbidden_gradients and not base_grads
                        and result['output_projection_gradient_nonzero'] and raw_active is not None
                        and any(value > 0 for value in raw_active)
                        and all(value > 0 and math.isfinite(value) for value in group_norms.values()))
    adapter.zero_grad(set_to_none=True)
    adapter.eval().requires_grad_(False)
    if not result['passed']:
        raise AssertionError(f'{branch} backward acceptance failed: {result}')
    return result


def main():
    args = arguments()
    seed_all(args.seed)
    torch.set_num_threads(4)
    report: dict[str, Any] = dict(
        architecture=ARCHITECTURE, model_revision=MODEL_REVISION,
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        resolution=args.resolution, optimizer_steps=0,
        scope='validation engineering acceptance; no final test set; no training updates; no complete sampling',
        face_1024_auxiliary_vae_backward_tested=False,
        native_parity=[], samples=[], backward=[], passed=False, status='started',
    )
    json_write(args.output, report)
    try:
        report['body_preserved_by_face_checkpoint'] = cpu_checkpoint_comparison(args.body_checkpoint, args.checkpoint)
        if not torch.cuda.is_available():
            raise RuntimeError('This runtime acceptance requires a free CUDA GPU')
        dataset = QwenFullDataset(args.pairs, resolution=args.resolution)
        count = min(args.max_samples, len(dataset))
        if not count:
            raise ValueError('Validation manifest has no samples')
        report['selected_indices'] = list(range(count))
        report['validation_manifest_sha256'] = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
        store = FaceConditionStore(args.features)
        cache = Qwen21IPERCache(args.cache, resolution=args.resolution, prompt=args.prompt)
        pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
        for component in (pipe.transformer, pipe.text_encoder, pipe.vae):
            component.eval().requires_grad_(False)
        pipe.text_encoder.to('cuda')
        pipe.vae.to('cuda')
        for index in range(count):
            item = dataset[index]
            cache.source(pipe, dataset_image_to_pil(item['src_image']))
            cache.target(pipe, dataset_image_to_pil(item['tgt_image']))
        pipe.text_encoder.to('cpu')
        pipe.vae.to('cpu')
        torch.cuda.empty_cache()
        pipe.transformer.to('cuda')
        pipe.transformer.enable_gradient_checkpointing()
        adapter = new_adapter('cpu')
        adapter.to('cuda').eval().requires_grad_(False)
        for index in range(count):
            row = verify_initial_native_parity(pipe, adapter, cache, dataset, store, index, args)
            report['native_parity'].append(row)
            json_write(args.output, report)
            print(json.dumps(dict(stage='native_parity', index=index, passed=row['passed'])), flush=True)
            torch.cuda.empty_cache()
        saved = load_adapter_checkpoint(adapter, args.checkpoint)
        report['loaded_checkpoint'] = dict(step=saved.get('step'), branch=saved.get('branch'),
                                            training_resolution=saved.get('resolution'))
        del saved
        before = model_digest(adapter)
        adapter.to('cuda').eval().requires_grad_(False)
        report['adapter_weights_sha256_before'] = before
        json_write(args.output, report)
        for index in range(count):
            row = verify_pair(pipe, adapter, cache, dataset, store, index, args)
            report['samples'].append(row)
            json_write(args.output, report)
            print(json.dumps(dict(stage='sample', index=index, passed=row['passed'])), flush=True)
            torch.cuda.empty_cache()
        for index in range(count):
            for branch in ('body', 'face'):
                row = backward_once(pipe, adapter, cache, dataset, store, index, branch, args)
                report['backward'].append(row)
                json_write(args.output, report)
                print(json.dumps(dict(stage='backward', index=index, branch=branch, passed=row['passed'],
                                      peak_allocated_gib=row['peak_allocated_gib'])), flush=True)
                torch.cuda.empty_cache()
        after = model_digest(adapter)
        report['adapter_weights_sha256_after'] = after
        report['weights_unchanged'] = before == after
        report['base_frozen'] = all(not p.requires_grad and p.grad is None for p in pipe.transformer.parameters())
        report['passed'] = report['weights_unchanged'] and report['base_frozen']
        report['status'] = 'complete' if report['passed'] else 'failed'
        if not report['passed']:
            raise AssertionError('Runtime acceptance altered weights or unfroze base parameters')
        json_write(args.output, report)
        print(f'Acceptance passed: {args.output}', flush=True)
    except Exception as error:
        report.update(status='failed', passed=False, error=repr(error), traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            report['failure_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        json_write(args.output, report)
        raise


if __name__ == '__main__':
    main()
