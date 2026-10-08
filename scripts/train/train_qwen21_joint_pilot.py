#!/usr/bin/env python3
"""Bounded joint Body+Face pilot with fixed optional BFS and strict resume."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from src.pose_control.qwen21_full.execution import (
    ARCHITECTURE, MODEL, MODEL_REVISION, PROMPT, RUN_ROOT, FaceConditionStore,
    QwenFullDataset, Qwen21IPERCache, assert_fp32_training_state, assert_frozen_gradients,
    capture_rng, file_sha256, json_write, load_adapter_checkpoint, new_adapter, restore_rng, seed_all,
    reconcile_resume_metrics,
)
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore
from src.pose_control.qwen21_full.frozen_bfs_lora import load_frozen_bfs
from src.pose_control.qwen21_full.joint_training import (
    assert_exact_state, enable_joint_training, frozen_joint_digest, freeze_evaluator, joint_group_gradients,
    joint_parameter_groups, run_joint_loss, validate_joint_signature, validate_locked_split,
    validate_group_gradients, validate_preflight,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', type=Path, required=True)
    parser.add_argument('--val-pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--face-supervision', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--lora', type=Path)
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-offset', type=int, default=0)
    parser.add_argument('--body-lr-scale', type=float, default=.25)
    parser.add_argument('--face-lr-scale', type=float, default=.5)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--face-weight', type=float, default=.5)
    parser.add_argument('--aux-every', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=128)
    parser.add_argument('--gradient-only', action='store_true')
    parser.add_argument('--gradient-report', type=Path)
    parser.add_argument('--probe-indices', type=int, nargs='+', default=[0, 64, 128, 192])
    args = parser.parse_args()
    if bool(args.init_checkpoint) == bool(args.resume):
        parser.error('Exactly one of --init-checkpoint and --resume is required')
    if not 1 <= args.steps <= 256 or args.save_every <= 0 or args.aux_every < 0:
        parser.error('Pilot is bounded to 1..256 updates; invalid save/aux interval')
    if args.noise_offset < 0 or args.warmup_steps < 0 or args.face_weight < 0:
        parser.error('Noise offset, warmup and Face weight must be nonnegative')
    if not args.gradient_only and not args.gradient_report:
        parser.error('Formal training requires --gradient-report from a matching passed preflight')
    return args


def save(path, adapter, optimizer, args, *, step, signature, sample_order):
    payload = {
        'architecture': ARCHITECTURE, 'model_revision': MODEL_REVISION,
        'branch': 'joint', 'resolution': 512, 'step': step,
        'adapter_state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'joint_signature': signature, 'sample_order': sample_order,
        'joint_initialization_sha256': adapter.joint_initialization_sha256,
        'initialization': adapter.loading_report,
        'initialization_provenance': adapter.initialization_provenance, **capture_rng(),
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    # Select deterministic CUDA bilinear reductions. Grid-sample and default
    # Flash SDP still have small nondeterministic reductions, recorded below.
    torch.use_deterministic_algorithms(True, warn_only=True)
    split = validate_locked_split(args.train_pairs, args.val_pairs)
    train = QwenFullDataset(args.train_pairs, resolution=512, dataset_type='tiktok_v3', align_mode='letterbox')
    validation = QwenFullDataset(args.val_pairs, resolution=512, dataset_type='tiktok_v3', align_mode='letterbox')
    if len(train) != 256 or len(validation) != 24:
        raise ValueError('Expected exactly 256 train / 24 held-out pairs')
    if args.steps > len(train) or any(not 0 <= index < len(train) for index in args.probe_indices):
        raise ValueError('Out-of-range training budget/probe')
    cache = Qwen21IPERCache(args.cache, resolution=512, prompt=PROMPT, preprocessing='letterbox_v1')
    store = FaceConditionStore(args.features)
    supervision = FaceSupervisionStore(args.face_supervision)
    signature = {
        'schema': 'joint_body_face_pilot_v1', 'model_revision': MODEL_REVISION,
        'model': str(args.model.resolve()), 'lora_sha256': file_sha256(args.lora) if args.lora else None,
        'strength': args.strength, 'train_sha256': file_sha256(args.train_pairs),
        'val_sha256': file_sha256(args.val_pairs), 'features': str(args.features.resolve()),
        'supervision_sha256': file_sha256(args.face_supervision), 'cache': str(args.cache.resolve()),
        'seed': args.seed, 'noise_offset': args.noise_offset, 'body_lr_scale': args.body_lr_scale,
        'face_lr_scale': args.face_lr_scale, 'face_weight': args.face_weight,
        'aux_every': args.aux_every, 'warmup_steps': args.warmup_steps,
        'joint_code_sha256': file_sha256(ROOT / 'src/pose_control/qwen21_full/joint_training.py'),
        'trainer_code_sha256': file_sha256(Path(__file__)), 'split': split,
        'deterministic_algorithms': 'warn_only', 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
    }
    metrics_path = args.output / 'metrics.jsonl'
    if not args.resume and (metrics_path.exists() or (args.output / 'checkpoint_last.pt').exists()):
        raise ValueError('Existing training output requires an explicit resume or a new directory')
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    # Native source/target caches already exist from the locked old Face run.
    # No text-encoder forward or B RGB reference is required during training.
    pipe.text_encoder.to('cpu')
    pipe.vae.to('cuda')
    pipe.transformer.to('cuda')
    lora_report = load_frozen_bfs(pipe, args.lora, strength=args.strength)
    json_write(args.output / 'lora_loading.json', lora_report)
    pipe.transformer.enable_gradient_checkpointing()
    adapter = new_adapter('cuda').train()
    origin = args.resume or args.init_checkpoint
    saved = load_adapter_checkpoint(adapter, origin)
    if args.resume:
        assert_exact_state(saved['adapter_state_dict'], adapter.state_dict(), 'adapter')
    initialization_sha = saved['joint_initialization_sha256'] if args.resume else file_sha256(origin)
    adapter.joint_initialization_sha256 = initialization_sha
    if not args.gradient_only:
        validate_preflight(json.loads(args.gradient_report.read_text()), signature, initialization_sha)
    adapter.initialization_provenance = {
        'mode': 'resume' if args.resume else 'init_checkpoint',
        'checkpoint_path': str(origin), 'checkpoint_sha256': file_sha256(origin),
        'checkpoint_step': saved.get('step'), 'checkpoint_branch': saved.get('branch'),
        'optimizer_state_restored': bool(args.resume),
    }
    groups, names = joint_parameter_groups(adapter, body_scale=args.body_lr_scale, face_scale=args.face_lr_scale)
    enable_joint_training(adapter, names)
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    start_step, sample_order = 0, []
    if args.resume:
        validate_joint_signature(saved['joint_signature'], signature)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        assert_exact_state(saved['optimizer_state_dict'], optimizer.state_dict(), 'optimizer')
        start_step, sample_order = int(saved['step']), list(saved['sample_order'])
        reconcile_resume_metrics(metrics_path, saved, checkpoint_path=origin, checkpoint_sha256=file_sha256(origin))
        logged = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(logged) != start_step or sample_order != list(range(start_step)):
            raise ValueError('Resume metrics/checkpoint trajectory differs; inspect before continuing')
        if [row['pair_index'] for row in logged] != sample_order or [row['step'] for row in logged] != list(range(1, start_step + 1)):
            raise ValueError('Resume metric sequence differs from checkpoint')
    assert_fp32_training_state(adapter, optimizer)
    evaluators = tuple(freeze_evaluator(torch.jit.load(str(ROOT / 'models/face_eval' / name), map_location='cuda'))
                       for name in ('identity_encoder.ts', 'perceptual_encoder.ts'))
    if args.resume:
        restore_rng(saved)
        assert_exact_state({key: saved[key] for key in capture_rng()}, capture_rng(), 'rng')
        json_write(args.output / f'resume_state_step_{start_step:06d}.json', {
            'adapter_restored_exact': True, 'optimizer_restored_exact': True, 'rng_restored_exact': True, 'checkpoint_step': start_step,
            'checkpoint_sha256': file_sha256(origin), 'joint_signature': signature})
    del saved
    versions = [parameter._version for parameter in pipe.transformer.parameters()]
    frozen_before = frozen_joint_digest(adapter, names)
    manifest = {
        'architecture': ARCHITECTURE, 'args': vars(args), 'joint_signature': signature,
        'initialization_provenance': adapter.initialization_provenance,
        'optimizer_parameters': names, 'lora': lora_report,
        'controls': {'body': 1, 'face': 1, 'interaction': 0, 'hand': 0, 'shared_detail_payload': False},
        'loss': {'body_flow': 1, 'face_roi': args.face_weight, 'identity': .10, 'perceptual': .05, 'outside_teacher': 0},
        'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16',
        'face_evaluator_range': 'normalized', 'sigma': 'uniform_0_1',
        'numerical_reproducibility': 'partial determinism: deterministic bilinear CUDA reductions; grid_sample/Flash SDP may retain small variation',
    }
    json_write(args.output / ('resume_manifest.json' if args.resume else 'run_manifest.json'), manifest)
    if args.gradient_only:
        rows = []
        for index in args.probe_indices:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                row = run_joint_loss(pipe, adapter, cache, train, store, index, seed=24042 + index,
                                     auxiliary=True, evaluators=evaluators, supervision=supervision,
                                     fixed_sigma=.2, face_weight=args.face_weight, check_auxiliary_gradients=True)
            row['group_grad_norms'] = joint_group_gradients(adapter, names)
            validate_group_gradients(row['group_grad_norms'])
            if row['auxiliary_eligible_faces'] and not all(value > 0 for value in row['auxiliary_gradient_norms'].values()):
                raise AssertionError('Auxiliary gradient path interrupted')
            assert_frozen_gradients(adapter, transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder)
            rows.append(row)
            print(json.dumps({'preflight_index': index, **row}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            high = run_joint_loss(pipe, adapter, cache, train, store, args.probe_indices[0], seed=24042,
                                  fixed_sigma=.8, face_weight=args.face_weight)
        high['group_grad_norms'] = joint_group_gradients(adapter, names)
        validate_group_gradients(high['group_grad_norms'], branches=('body',))
        if high['face_active']:
            raise AssertionError('High-noise Body path or Face time gate incorrect')
        assert_frozen_gradients(adapter, transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder)
        if versions != [parameter._version for parameter in pipe.transformer.parameters()]:
            raise AssertionError('Frozen transformer mutated')
        if frozen_before != frozen_joint_digest(adapter, names):
            raise AssertionError('Frozen Adapter mutated during gradient preflight')
        json_write(args.output / 'gradient_only.json', {'passed': True, 'joint_signature': signature,
                   'initialization_sha256': initialization_sha, 'low_noise': rows, 'high_noise': high,
                   'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True})
        return
    if not args.resume:
        # PEFT initializes different numbers of temporary tensors for Body/Head.
        # Reset global RNG after loading so Adapter stochastic ops also match.
        seed_all(args.seed + args.noise_offset)
    if start_step == 0:
        save(args.output / 'checkpoint_step_000000.pt', adapter, optimizer, args, step=0, signature=signature, sample_order=[])
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, args.steps):
        scale = min((step + 1) / args.warmup_steps, 1.) if args.warmup_steps else 1.
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * scale
        with torch.autocast('cuda', dtype=torch.bfloat16):
            row = run_joint_loss(pipe, adapter, cache, train, store, step,
                                 seed=args.seed + args.noise_offset + step,
                                 auxiliary=args.aux_every > 0 and step % args.aux_every == 0,
                                 evaluators=evaluators, supervision=supervision, face_weight=args.face_weight)
        grads = joint_group_gradients(adapter, names)
        validate_group_gradients(grads, branches=('body', 'face') if row['face_active'] else ('body',))
        norm = float(torch.nn.utils.clip_grad_norm_([parameter for parameter in adapter.parameters() if parameter.requires_grad], 1.))
        if not math.isfinite(norm) or norm <= 0 or not math.isfinite(row['loss']):
            raise FloatingPointError(f'Invalid loss/gradient: {row}, norm={norm}')
        if row['face_active'] and sum(value for key, value in grads.items() if key.startswith('face')) <= 0:
            raise AssertionError('Active Face received no gradient')
        assert_frozen_gradients(adapter, transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder,
                               **{f'evaluator_{index}': model for index, model in enumerate(evaluators)})
        optimizer.step()
        assert_fp32_training_state(adapter, optimizer)
        optimizer.zero_grad(set_to_none=True)
        sample_order.append(step)
        row.update(step=step + 1, group_grad_norms=grads, grad_norm=norm,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3)
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        print(json.dumps(row), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(args.output / f'checkpoint_step_{step + 1:06d}.pt', adapter, optimizer, args,
                 step=step + 1, signature=signature, sample_order=sample_order)
            save(args.output / 'checkpoint_last.pt', adapter, optimizer, args,
                 step=step + 1, signature=signature, sample_order=sample_order)
    frozen_after = frozen_joint_digest(adapter, names)
    unchanged = versions == [parameter._version for parameter in pipe.transformer.parameters()]
    if not unchanged or frozen_before != frozen_after:
        raise AssertionError('Frozen model/Adapter parameters changed')
    all_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    json_write(args.output / 'summary.json', {
        'completed': True, 'steps': args.steps, 'unique_pairs': len(set(sample_order)),
        'face_active_steps': sum(row['face_active'] for row in all_rows),
        'auxiliary_steps': sum(row['auxiliary_eligible_faces'] > 0 for row in all_rows),
        'frozen_transformer_unchanged': unchanged, 'frozen_adapter_unchanged': frozen_before == frozen_after,
        'last_loss': all_rows[-1]['loss'], 'active_seconds': time.perf_counter() - started,
    })


if __name__ == '__main__':
    main()
