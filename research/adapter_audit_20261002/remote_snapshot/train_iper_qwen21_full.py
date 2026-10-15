#!/usr/bin/env python3
"""Bounded Body/Face training of the complete V6 architecture on Qwen21."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline

from src.pose_control.qwen21_full.execution import (
    ARCHITECTURE, MODEL, ROOT, RUN_ROOT, FaceConditionStore, QwenFullDataset,
    Qwen21IPERCache, batch_one, frozen_digest, json_write, load_adapter_checkpoint,
    new_adapter, parameter_groups, precompute, prepare, probe, run_loss,
    restore_rng, save_checkpoint, seed_all, set_training_phase, transformer_inputs,
)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION, PROMPT
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', choices=('body', 'face'), required=True)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--train-pairs', type=Path, default=RUN_ROOT / 'train_pairs.jsonl')
    p.add_argument('--val-pairs', type=Path, default=RUN_ROOT / 'val_pairs.jsonl')
    p.add_argument('--features', type=Path, default=RUN_ROOT / 'face_cache')
    p.add_argument('--face-supervision', type=Path, default=RUN_ROOT / 'face_supervision.json')
    p.add_argument('--cache', type=Path, default=RUN_ROOT / 'native_cache')
    p.add_argument('--output', type=Path)
    p.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--warmup-steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--init-checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--save-every', type=int, default=20)
    p.add_argument('--save-steps', nargs='*', type=int, default=[])
    p.add_argument('--diagnostic-probes', action='store_true')
    p.add_argument('--require-active-gradients', action='store_true')
    p.add_argument('--face-evaluator-range', choices=('legacy','normalized'), default='legacy')
    p.add_argument('--face-flow-loss', choices=('global','roi'), default='global',
                   help='Historical whole-image flow or separately normalized face-region flow')
    p.add_argument('--face-aux-every', type=int, default=4,
                   help='Identity/perceptual interval; zero disables these terms for a diagnosis')
    p.add_argument('--face-aux-alignment', choices=('box','five_point'), default='box',
                   help='Historical square crop or fixed Target five-point alignment for offline losses')
    p.add_argument('--verify-resume', action='store_true')
    p.add_argument('--gradient-only', action='store_true', help='One forward/backward, no parameter update')
    p.add_argument('--precompute-only', action='store_true')
    return p.parse_args()


def parameter_digest(adapter):
    h = hashlib.sha256()
    for name, p in adapter.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_digest(value):
    """Exact digest of tensors plus optimizer groups and scalar metadata."""
    h = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def main():
    args = arguments()
    # The fixed-noise diagnostic run requires an actual update at every counted
    # training step; historical runs without this option keep their old behavior.
    args.require_active_gradients = args.require_active_gradients or args.diagnostic_probes
    if args.diagnostic_probes:
        args.face_evaluator_range = 'normalized'
    args.output = args.output or RUN_ROOT / f'{args.branch}_{args.resolution}_100'
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / 'train_metrics.jsonl'
    if metrics_path.exists() and not (args.resume or args.gradient_only or args.precompute_only):
        raise FileExistsError('Existing training run: use --resume or a new output directory')
    if not 0 <= args.warmup_steps <= args.max_steps or args.max_steps <= 0:
        raise ValueError('Invalid step count')
    if any(step < 0 or step > args.max_steps for step in args.save_steps):
        raise ValueError('save-steps must lie between zero and max-steps')
    if args.face_aux_every < 0:
        raise ValueError('face-aux-every must be nonnegative')
    if args.branch == 'face' and not (args.init_checkpoint or args.resume):
        raise ValueError('Face training must start from the completed qwen21_full Body checkpoint')
    seed_all(args.seed)
    torch.set_num_threads(4)
    train = QwenFullDataset(args.train_pairs, resolution=args.resolution)
    val = QwenFullDataset(args.val_pairs, resolution=args.resolution)
    from src.pose_control.qwen21_full.data_runtime import pair_metadata
    for row in train.samples:
        meta = pair_metadata(row)
        if meta['source_appearance'] != meta['target_appearance']:
            raise ValueError('Different-person A/B examples are inference checks, never reconstruction training targets')
    store = FaceConditionStore(args.features)
    cache = Qwen21IPERCache(args.cache, resolution=args.resolution, prompt=args.prompt)
    run_manifest = {
        'architecture': ARCHITECTURE, 'args': vars(args), 'model_revision': MODEL_REVISION,
        'train_pairs_sha256': hashlib.sha256(args.train_pairs.read_bytes()).hexdigest(),
        'val_pairs_sha256': hashlib.sha256(args.val_pairs.read_bytes()).hexdigest(),
        'train_pairs': len(train), 'val_pairs': len(val), 'micro_batch_size': 1, 'accumulation': 1,
        'versions': {key: version(key) for key in ('torch', 'diffusers', 'transformers')},
        'scope': 'bounded migration acceptance; final 50-pair test set excluded',
    }
    json_write(args.output / ('resume_manifest.json' if args.resume else 'run_manifest.json'), run_manifest)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.text_encoder.to('cuda')
    pipe.vae.to('cuda')
    precompute(pipe, cache, (train, val))
    if args.precompute_only:
        return
    pipe.text_encoder.to('cpu')
    if args.branch == 'body' or args.gradient_only:
        pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    pipe.transformer.to('cuda')
    pipe.transformer.enable_gradient_checkpointing()
    adapter = new_adapter('cuda').train()
    json_write(args.output / 'weight_loading.json', adapter.loading_report)
    start_step = 0
    sample_order = []
    restored = None
    if args.init_checkpoint:
        load_adapter_checkpoint(adapter, args.init_checkpoint)
    if args.resume:
        restored = load_adapter_checkpoint(adapter, args.resume)
        if restored['resolution'] != args.resolution or restored['branch'] != args.branch:
            raise ValueError('Resume requires the same resolution and branch')
        for key in ('seed', 'prompt', 'model', 'warmup_steps', 'train_pairs', 'val_pairs'):
            if str(restored['args'][key]) != str(getattr(args, key)):
                raise ValueError(f'Resume configuration differs: {key}')
        for key, historical_default in (('face_flow_loss', 'global'), ('face_aux_every', 4),
                                        ('face_aux_alignment', 'box'),
                                        ('face_evaluator_range', 'legacy')):
            if restored['args'].get(key, historical_default) != getattr(args, key):
                raise ValueError(f'Resume loss configuration differs: {key}; use init-checkpoint for a new experiment')
        previous_manifest_path = args.resume.parent / 'run_manifest.json'
        previous_manifest = json.loads(previous_manifest_path.read_text()) if previous_manifest_path.exists() else {}
        for key in ('train_pairs', 'val_pairs'):
            actual = hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            expected = restored.get(f'{key}_sha256', previous_manifest.get(f'{key}_sha256'))
            if expected is None or expected != actual:
                raise ValueError(f'Resume dataset fingerprint differs or is missing: {key}')
        start_step = restored['step']
        sample_order = restored['sample_order']
    groups, names = parameter_groups(adapter, args.branch)
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    if restored:
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        restore_rng(restored)
    json_write(args.output / 'optimizer_parameters.json', names)
    evaluators = None
    supervision = None
    if args.branch == 'face' and not args.gradient_only and args.face_aux_every:
        supervision = FaceSupervisionStore(args.face_supervision)
        evaluators = tuple(torch.jit.load(str(ROOT / 'models/face_eval' / filename), map_location='cuda').eval()
                           for filename in ('identity_encoder.ts', 'perceptual_encoder.ts'))
        for evaluator in evaluators:
            for parameter in evaluator.parameters():
                parameter.requires_grad_(False)
    base_versions = [p._version for p in pipe.transformer.parameters()]
    frozen_before = frozen_digest(adapter, args.branch)
    set_training_phase(adapter, args.branch, warmup=start_step < args.warmup_steps and not args.gradient_only)
    optimizer.zero_grad(set_to_none=True)
    before_path = args.output / 'probe_before.json'
    if args.resume and before_path.exists():
        before = json.loads(before_path.read_text())
    else:
        before = probe(pipe, adapter, cache, train, store, args.branch,
                       face_flow_loss=args.face_flow_loss)
        json_write(before_path, before)
    def save_probes(step):
        if not args.diagnostic_probes:
            return
        sigmas = (.05, .10, .15, .20, .30, .40, .50, .60) if args.branch == 'face' else (.05, .15, .25, .35, .50, .65, .80, .95)
        rows = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for index, sigma in enumerate(sigmas):
                rows.append(run_loss(pipe, adapter, cache, train, store, 0, args.branch,
                                     seed=41042 + index, backward=False, fixed_sigma=sigma,
                                     face_flow_loss=args.face_flow_loss))
        json_write(args.output / f'diagnostic_probe_{step:06d}.json',
                   {'step': step, 'fixed_noise_seeds': list(range(41042, 41050)),
                    'rows': rows, 'mean_flow': float(np.mean([r['flow'] for r in rows]))})
    if start_step == 0 and 0 in args.save_steps:
        save_checkpoint(args.output / 'checkpoint_step_000000.pt', adapter, optimizer,
                        step=0, branch=args.branch, args=args, sample_order=sample_order)
        save_probes(0)
    if args.branch == 'body' and not (args.init_checkpoint or args.resume):
        batch = batch_one(train, 0)
        pair = cache.pair(pipe, batch, torch.device('cuda'))
        sigma = torch.tensor([.5], device='cuda', dtype=pair['target_latent'].dtype)
        inputs = transformer_inputs(pair, pair['target_latent'] * .5, sigma)
        tokens = (args.resolution // 16) ** 2
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            native = pipe.transformer(**inputs)[0][:, -tokens:]
            prepared = prepare(adapter, batch, pair, [train.samples[0]], store, with_face=True)
            with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                         geometry_strength=0, interaction_strength=0, face_strength=0,
                                         hand_strength=0, hand_mode='off'):
                off = pipe.transformer(**inputs)[0][:, -tokens:]
            with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                         face_strength=1, hand_strength=0, hand_mode='off'):
                zero = pipe.transformer(**inputs)[0][:, -tokens:]
        acceptance = {'disabled_equal': torch.equal(native, off),
                      'zero_max_abs': float((native.float() - zero.float()).abs().max())}
        json_write(args.output / 'native_parity.json', acceptance)
        if not acceptance['disabled_equal'] or acceptance['zero_max_abs'] > .001:
            raise AssertionError(acceptance)
        del native, off, zero, prepared, inputs, pair, batch
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    elapsed_offset = 0.
    if args.resume and metrics_path.exists():
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        elapsed_offset = float(history[-1]['elapsed_seconds']) if history else 0.

    def train_step(step, *, update=True):
        epoch, offset = divmod(step, len(train))
        index = int(torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch))[offset])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            row = run_loss(pipe, adapter, cache, train, store, index, args.branch,
                           seed=args.seed + step + (100000 if args.branch == 'face' else 0),
                           auxiliary=args.branch == 'face' and args.face_aux_every > 0
                                     and (step + 1) % args.face_aux_every == 0 and not args.gradient_only,
                           evaluators=evaluators, supervision=supervision,
                           require_active_face=args.require_active_gradients,
                           normalized_face_evaluators=args.face_evaluator_range == 'normalized',
                           face_flow_loss=args.face_flow_loss,
                           face_aux_alignment=args.face_aux_alignment,
                           check_auxiliary_gradients=args.diagnostic_probes and args.branch == 'face'
                                                     and args.face_aux_every > 0 and step + 1 == args.face_aux_every,
                           fixed_sigma=.2 if args.gradient_only and args.branch == 'face' else None)
        grads = {key: float(sum((p.grad.detach().float().square().sum() for name, p in adapter.named_parameters()
                               if name in group and p.grad is not None), torch.tensor(0., device='cuda')).sqrt())
                 for key, group in names.items()}
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0))
        if not np.isfinite(norm) or not np.isfinite(row['loss']):
            raise FloatingPointError(f'Non-finite loss/gradient at {step + 1}: {row}')
        if args.require_active_gradients and norm == 0:
            raise AssertionError(f'Gradient interruption at step {step + 1}')
        if 'auxiliary_gradient_norms' in row and not all(v > 0 for v in row['auxiliary_gradient_norms'].values()):
            raise AssertionError('An auxiliary Face loss has no gradient to its output interface')
        if any(p.grad is not None for p in pipe.transformer.parameters()):
            raise AssertionError('Base model received parameter gradients')
        if update:
            optimizer.step()
        row.update(step=step + 1, warmup=step < args.warmup_steps, grad_norm=norm,
                   group_grad_norms=grads, elapsed_seconds=elapsed_offset + time.perf_counter() - started,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3)
        return row

    if args.gradient_only:
        row = train_step(21, update=False)
        json_write(args.output / 'gradient_only.json', row)
        print(json.dumps(row, default=str), flush=True)
        return
    for step in range(start_step, args.max_steps):
        set_training_phase(adapter, args.branch, warmup=step < args.warmup_steps)
        row = train_step(step)
        sample_order.append(row['pair_index'])
        with (args.output / 'train_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, default=str) + '\n')
        print(json.dumps(row, default=str), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 in (21, args.max_steps):
            save_checkpoint(args.output / 'checkpoint_last.pt', adapter, optimizer,
                            step=step + 1, branch=args.branch, args=args, sample_order=sample_order)
        if step + 1 in args.save_steps:
            save_checkpoint(args.output / f'checkpoint_step_{step + 1:06d}.pt', adapter, optimizer,
                            step=step + 1, branch=args.branch, args=args, sample_order=sample_order)
            save_probes(step + 1)
        if args.verify_resume and step + 1 == 21:
            checkpoint = args.output / 'checkpoint_last.pt'
            parameters_before = state_digest(adapter.state_dict())
            optimizer_before = state_digest(optimizer.state_dict())
            restore = load_adapter_checkpoint(adapter, checkpoint)
            optimizer.load_state_dict(restore['optimizer_state_dict'])
            restore_rng(restore)
            comparison = {
                'checkpoint_step': step + 1,
                'parameters_equal': parameters_before == state_digest(adapter.state_dict()),
                'optimizer_equal': optimizer_before == state_digest(optimizer.state_dict()),
                'rng_restored': True,
                'default_cuda_bitwise_update': False,
                'note': 'Exact checkpoint loading; CUDA backward is nondeterministic. '
                        'Separate deterministic-reference report verifies next-step continuation.',
            }
            json_write(args.output / 'resume_verification.json', comparison)
            if not comparison['parameters_equal'] or not comparison['optimizer_equal']:
                raise AssertionError(f'Resume differs: {comparison}')
            optimizer.zero_grad(set_to_none=True)
    after = probe(pipe, adapter, cache, train, store, args.branch,
                  face_flow_loss=args.face_flow_loss)
    frozen_after = frozen_digest(adapter, args.branch)
    report = {'branch': args.branch, 'optimizer_steps': args.max_steps,
              'unique_training_pairs': len(set(sample_order)), 'training_pair_uses': len(sample_order),
              'probe_before': before, 'probe_after': after,
              'frozen_parameters_unchanged': frozen_before == frozen_after,
              'base_versions_unchanged': base_versions == [p._version for p in pipe.transformer.parameters()],
              'fixed_probe_improved': after['loss'] < before['loss'],
              'seconds': elapsed_offset + time.perf_counter() - started, 'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3}
    history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    report['nonzero_gradient_steps'] = sum(row['grad_norm'] > 0 for row in history)
    report['face_active_steps'] = sum(row.get('control_diagnostics', {}).get('face_active_rows', 0) > 0 for row in history)
    json_write(args.output / 'summary.json', report)
    if not report['frozen_parameters_unchanged'] or not report['base_versions_unchanged']:
        raise AssertionError('Frozen parameters changed')
    print(json.dumps(report, default=str), flush=True)


if __name__ == '__main__':
    main()
