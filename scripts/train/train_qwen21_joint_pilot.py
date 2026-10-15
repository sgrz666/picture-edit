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
from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
from src.pose_control.qwen21_full.data_runtime import HandConditionStore
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore
from src.pose_control.qwen21_full.frozen_bfs_lora import load_frozen_bfs
from src.pose_control.qwen21_full.joint_training import (
    assert_exact_state, enable_joint_training, frozen_joint_digest, freeze_evaluator, joint_group_gradients,
    joint_parameter_groups, run_joint_loss, validate_joint_signature, validate_locked_split,
    validate_group_gradients, validate_preflight, init_hand_weights_if_zero,
)


from src.pose_control.qwen21_full.prompt_study.protocol import (
    SCHEMA, ARMS, P0_PROMPT, TRAIN_SHA256, VAL_SHA256, text_sha256,
)
from src.pose_control.qwen21_full.prompt_study.sampling import (
    make_order, noise_seed, warmup_scale, validate_trajectory, validate_update_bounds,
)
from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
from src.pose_control.qwen21_full.data_runtime import (
    pair_metadata, study_prompt_signature, study_asset_signature, study_code_signature, validate_study_cache,
    validate_study_supervision,
)


def validate_study_training_inputs(args):
    from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock, validate_study_binding
    lock = load_study_lock(args.study_lock)
    for split, actual in (('train', args.train_pairs), ('val', args.val_pairs)):
        if actual.resolve() != (args.study_lock.parent / lock['splits'][split]['path']).resolve():
            raise ValueError(f'Study {split} manifest differs from the lock')
    validate_study_binding(lock, args.arm, file_sha256(args.lora) if args.lora else None, args.model)
    if args.init_checkpoint and file_sha256(args.init_checkpoint) != lock['initial_checkpoint_sha256']:
        raise ValueError('Study initialization differs from locked J256 checkpoint')
    return lock


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
    parser.add_argument('--with-hand', action='store_true', help='Enable Hand branch joint optimization')
    parser.add_argument('--hand-lr-scale', type=float, default=1.0)
    parser.add_argument('--hand-weight', type=float, default=.5)
    parser.add_argument('--tiktok-raw-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    parser.add_argument('--tiktok-assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--aux-every', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=128)
    parser.add_argument('--gradient-only', action='store_true')
    parser.add_argument('--gradient-report', type=Path)
    parser.add_argument('--probe-indices', type=int, nargs='+', default=[0, 64, 128, 192])
    parser.add_argument('--resolution', default='512', help='512, 1024, or native')
    parser.add_argument('--align-mode', choices=('letterbox', 'resize', 'native'), default='letterbox')
    parser.add_argument('--study-lock', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--prompt-manifest', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    args = parser.parse_args()
    if bool(args.init_checkpoint) == bool(args.resume):
        parser.error('Exactly one of --init-checkpoint and --resume is required')
    if args.study_lock is not None:
        if args.steps != 1024:
            parser.error('Study mode requires --steps 1024')
        if not args.arm:
            parser.error('Study mode requires --arm')
        if not args.prompt_manifest:
            parser.error('Study mode requires --prompt-manifest')
        if args.stop_after_step is None:
            args.stop_after_step = args.steps
        if not 0 <= args.stop_after_step <= args.steps:
            parser.error(f'--stop-after-step must be between 0 and {args.steps}')
        if args.resolution != 'native' or args.align_mode != 'native':
            parser.error("Study mode requires resolution=native and align_mode=native")
        if args.with_hand:
            parser.error("Study mode requires hand adapter to be disabled")
        if args.seed != 42 or args.noise_offset != 256:
            parser.error("Study mode requires seed=42 and noise_offset=256")
        fixed = {'body_lr_scale': .25, 'face_lr_scale': .5, 'face_weight': .5,
                 'warmup_steps': 20, 'aux_every': 4, 'save_every': 128, 'strength': 1.,
                 'hand_lr_scale': 1., 'hand_weight': .5}
        if any(getattr(args, key) != value for key, value in fixed.items()):
            parser.error(f'Study mode requires fixed training parameters: {fixed}')
        if (args.arm.startswith('head_')) != bool(args.lora):
            parser.error('Study Head arms require Head LoRA; Qwen arms require no LoRA')
    else:
        if not 1 <= args.steps <= 256 or args.save_every <= 0 or args.aux_every < 0:
            parser.error('Pilot is bounded to 1..256 updates; invalid save/aux interval')
        args.stop_after_step = args.steps
    if args.noise_offset < 0 or args.warmup_steps < 0 or args.face_weight < 0:
        parser.error('Noise offset, warmup and Face weight must be nonnegative')
    if args.hand_lr_scale <= 0 or args.hand_weight < 0:
        parser.error('Hand LR scale must be positive and hand weight must be nonnegative')
    if not args.gradient_only and not args.gradient_report:
        parser.error('Formal training requires --gradient-report from a matching passed preflight')
    return args


def save(path, adapter, optimizer, args, *, step, signature, sample_order):
    payload = {
        'architecture': ARCHITECTURE, 'model_revision': MODEL_REVISION,
        'branch': 'joint',
        'resolution': args.resolution if args.resolution == 'native' else int(args.resolution),
        'align_mode': getattr(args, 'align_mode', 'letterbox'),
        'step': step,
        'adapter_state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'joint_signature': signature, 'sample_order': sample_order,
        'joint_initialization_sha256': adapter.joint_initialization_sha256,
        'initialization': adapter.loading_report,
        'initialization_provenance': adapter.initialization_provenance, **capture_rng(),
    }
    if getattr(args, 'study_lock', None) is not None:
        payload.update(
            study_schema=SCHEMA, arm=args.arm,
            protocol=args.arm.split("_", 1)[1],
            max_updates=1024, initial_history_steps=256,
            study_lock_sha256=signature.get("study_lock_sha256"),
            prompt_manifest_sha256=signature.get("prompt_manifest_sha256"),
            next_update_index=step,
        )
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
    study_mode = args.study_lock is not None
    res_arg = args.resolution if args.resolution == 'native' else int(args.resolution)
    if study_mode:
        lock = validate_study_training_inputs(args)
        train_sha = file_sha256(args.train_pairs)
        val_sha = file_sha256(args.val_pairs)
        if train_sha != TRAIN_SHA256:
            raise ValueError(f"train512 sha mismatch: {train_sha} != {TRAIN_SHA256}")
        if val_sha != VAL_SHA256:
            raise ValueError(f"val24 sha mismatch: {val_sha} != {VAL_SHA256}")
        order = lock['epoch_order']
        train = QwenFullDataset(args.train_pairs, resolution=res_arg, dataset_type='tiktok_v3', align_mode=args.align_mode,
                                tiktok_raw_root=args.tiktok_raw_root, tiktok_assets_root=args.tiktok_assets_root)
        validation = QwenFullDataset(args.val_pairs, resolution=res_arg, dataset_type='tiktok_v3', align_mode=args.align_mode,
                                     tiktok_raw_root=args.tiktok_raw_root, tiktok_assets_root=args.tiktok_assets_root)
        if len(train) != 512 or len(validation) != 24:
            raise ValueError('Study mode requires exactly 512 train and 24 val pairs')
        for s in train.samples:
            if pair_metadata(s)["cross_person"]:
                raise ValueError("Train pairs must not contain cross_person")
        protocol = args.arm.split("_", 1)[1]
        expected_ids = [pair_metadata(s)["pair_id"] for s in train.samples]
        bank = PromptBank(args.prompt_manifest, protocol, expected_ids)
        if args.prompt_manifest.resolve() != (args.study_lock.parent.parent / 'prompts' / protocol / 'train.jsonl').resolve():
            raise ValueError('Study training prompt manifest must be the complete locked train split')
        if args.cache.resolve() != (args.study_lock.parent.parent / 'cache' / protocol).resolve():
            raise ValueError('Study training cache does not match the protocol')
        cache_prompt = P0_PROMPT if protocol == 'p0' else PROMPT
        split = "study"
    else:
        split = validate_locked_split(args.train_pairs, args.val_pairs)
        order = list(range(args.steps))
        train = QwenFullDataset(args.train_pairs, resolution=res_arg, dataset_type='tiktok_v3', align_mode=args.align_mode)
        validation = QwenFullDataset(args.val_pairs, resolution=res_arg, dataset_type='tiktok_v3', align_mode=args.align_mode)
        if len(train) != 256 or len(validation) != 24:
            raise ValueError('Expected exactly 256 train / 24 held-out pairs')
        if args.steps > len(train) or any(not 0 <= index < len(train) for index in args.probe_indices):
            raise ValueError('Out-of-range training budget/probe')
        bank = None
        protocol = None
        cache_prompt = PROMPT
    cache = Qwen21IPERCache(
        args.cache,
        resolution=res_arg,
        prompt=cache_prompt,
        preprocessing='native_v1' if args.align_mode == 'native' else ('letterbox_v1' if args.align_mode == 'letterbox' else 'resize_v1'),
    )
    store = FaceConditionStore(args.features)
    hand_store = HandConditionStore(
        raw_root=args.tiktok_raw_root,
        assets_root=args.tiktok_assets_root,
        align_mode=args.align_mode,
    ) if args.with_hand else None
    supervision = FaceSupervisionStore(args.face_supervision)
    signature = {
        'schema': SCHEMA if study_mode else ('joint_body_face_hand_pilot_v1' if args.with_hand else 'joint_body_face_pilot_v1'),
        'model_revision': MODEL_REVISION,
        'model': str(args.model.resolve()), 'lora_sha256': file_sha256(args.lora) if args.lora else None,
        'strength': args.strength, 'train_sha256': file_sha256(args.train_pairs),
        'val_sha256': file_sha256(args.val_pairs), 'features': str(args.features.resolve()),
        'supervision_sha256': file_sha256(args.face_supervision), 'cache': str(args.cache.resolve()),
        'resolution': str(args.resolution), 'align_mode': args.align_mode,
        'seed': args.seed, 'noise_offset': args.noise_offset, 'body_lr_scale': args.body_lr_scale,
        'face_lr_scale': args.face_lr_scale, 'face_weight': args.face_weight,
        'with_hand': bool(args.with_hand), 'hand_lr_scale': args.hand_lr_scale, 'hand_weight': args.hand_weight,
        'aux_every': args.aux_every, 'warmup_steps': args.warmup_steps,
        'joint_code_sha256': file_sha256(ROOT / 'src/pose_control/qwen21_full/joint_training.py'),
        'trainer_code_sha256': file_sha256(Path(__file__)), 'split': split,
        'deterministic_algorithms': 'warn_only', 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
    }
    if study_mode:
        signature.update(
            study_schema=SCHEMA,
            study_lock_sha256=file_sha256(args.study_lock),
            arm=args.arm,
            protocol=protocol,
            prompt_manifest_sha256=file_sha256(args.prompt_manifest),
            sample_order_sha256=text_sha256(json.dumps(order, separators=(',', ':'))),
            max_updates=1024,
            initial_history_steps=256,
            preprocessing='native_v1',
            model_manifest_sha256=lock['artifacts']['model']['manifest_sha256'],
        )
        signature.update(study_prompt_signature(bank))
        signature['instruction_sha256'] = file_sha256(args.study_lock.parent.parent / 'prompts/instruction_v1.json')
        signature.update(study_asset_signature(args.features, args.study_lock.parent))
        signature.update(study_code_signature())
        from scripts.precompute_qwen21_prompt_study_cache import validate_cache_manifest
        validate_cache_manifest(args.study_lock.parent.parent, protocol, ['train'])
        validate_study_cache(cache, train, bank, indices=range(len(train)), require_target=True)
        validate_study_supervision(supervision, [pair_metadata(row) for row in train.samples])
        if any(not 0 <= index < len(train) for index in args.probe_indices):
            raise ValueError('Out-of-range study gradient probe')
    metrics_path = args.output / 'metrics.jsonl'
    if not args.resume and (metrics_path.exists() or (args.output / 'checkpoint_last.pt').exists()):
        raise ValueError('Existing training output requires an explicit resume or a new directory')
    if study_mode and args.resume:
        header = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        validate_joint_signature(header['joint_signature'], signature)
        if header.get('joint_initialization_sha256') != lock['initial_checkpoint_sha256']:
            raise ValueError('Study resume initialization differs from locked J256 checkpoint')
        validate_update_bounds(header['step'], args.stop_after_step)
        if (header.get('study_schema'), header.get('arm'), header.get('next_update_index')) != (SCHEMA, args.arm, header['step']):
            raise ValueError('Study resume checkpoint schema/arm/update differs')
        reconcile_resume_metrics(metrics_path, header, checkpoint_path=args.resume, checkpoint_sha256=file_sha256(args.resume))
        logged = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        validate_trajectory(header, logged, order, pair_ids=expected_ids,
                            prompt_hashes={pid: row['compiled_prompt_sha256'] for pid, row in bank.records.items()})
        del header
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    if not study_mode:
        # Legacy precomputation
        pipe.text_encoder.to('cuda')
        pipe.vae.to('cuda')
        missing_sources = missing_targets = 0
        pairs_to_cache = args.probe_indices if args.gradient_only else list(range(len(train)))
        for idx in pairs_to_cache:
            sample = train[idx]
            src_img = dataset_image_to_pil(sample['src_image'])
            if not cache._path(src_img, 'source').exists():
                cache.source(pipe, src_img)
                missing_sources += 1
            tgt_img = dataset_image_to_pil(sample['tgt_image'])
            if not cache._path(tgt_img, 'target').exists():
                cache.target(pipe, tgt_img)
                missing_targets += 1
        if not args.gradient_only:
            for idx in range(len(validation)):
                sample = validation[idx]
                src_img = dataset_image_to_pil(sample['src_image'])
                if not cache._path(src_img, 'source').exists():
                    cache.source(pipe, src_img)
                    missing_sources += 1
                tgt_img = dataset_image_to_pil(sample['tgt_image'])
                if not cache._path(tgt_img, 'target').exists():
                    cache.target(pipe, tgt_img)
                    missing_targets += 1
        if missing_sources or missing_targets:
            print(f"Precomputed {missing_sources} source conditions and {missing_targets} target latents on GPU", flush=True)
    pipe.text_encoder.to('cpu')
    torch.cuda.empty_cache()
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
    elif args.with_hand:
        init_hand_weights_if_zero(adapter)
    initialization_sha = saved['joint_initialization_sha256'] if args.resume else file_sha256(origin)
    if study_mode and initialization_sha != lock['initial_checkpoint_sha256']:
        raise ValueError('Study resume initialization differs from locked J256 checkpoint')
    adapter.joint_initialization_sha256 = initialization_sha
    if not args.gradient_only:
        validate_preflight(json.loads(args.gradient_report.read_text()), signature, initialization_sha)
    adapter.initialization_provenance = {
        'mode': 'resume' if args.resume else 'init_checkpoint',
        'checkpoint_path': str(origin), 'checkpoint_sha256': file_sha256(origin),
        'checkpoint_step': saved.get('step'), 'checkpoint_branch': saved.get('branch'),
        'optimizer_state_restored': bool(args.resume),
    }
    groups, names = joint_parameter_groups(
        adapter,
        body_scale=args.body_lr_scale,
        face_scale=args.face_lr_scale,
        hand_scale=args.hand_lr_scale,
        with_hand=args.with_hand,
    )
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
        if study_mode:
            validate_trajectory(saved, logged, order, pair_ids=expected_ids,
                                prompt_hashes={pid: row['compiled_prompt_sha256'] for pid, row in bank.records.items()})
        else:
            if len(logged) != start_step or sample_order != list(range(start_step)):
                raise ValueError('Resume metrics/checkpoint trajectory differs; inspect before continuing')
            if [row['pair_index'] for row in logged] != sample_order or [row['step'] for row in logged] != list(range(1, start_step + 1)):
                raise ValueError('Resume metric sequence differs from checkpoint')
    if study_mode:
        validate_update_bounds(start_step, args.stop_after_step)
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
        'controls': {'body': 1, 'face': 1, 'interaction': 0, 'hand': 1 if args.with_hand else 0, 'shared_detail_payload': False},
        'loss': {'body_flow': 1, 'face_roi': args.face_weight, 'hand_flow': args.hand_weight if args.with_hand else 0, 'identity': .10, 'perceptual': .05, 'outside_teacher': 0},
        'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16',
        'face_evaluator_range': 'normalized', 'sigma': 'uniform_0_1',
        'numerical_reproducibility': 'partial determinism: deterministic bilinear CUDA reductions; grid_sample/Flash SDP may retain small variation',
    }
    json_write(args.output / ('resume_manifest.json' if args.resume else 'run_manifest.json'), manifest)
    if args.gradient_only:
        rows = []
        for index in args.probe_indices:
            optimizer.zero_grad(set_to_none=True)
            probe_prompt = bank.prompt_for(pair_metadata(train.samples[index])["pair_id"]) if study_mode else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                row = run_joint_loss(pipe, adapter, cache, train, store, index, seed=24042 + index,
                                     prompt=probe_prompt,
                                     auxiliary=True, evaluators=evaluators, supervision=supervision,
                                     fixed_sigma=.2, face_weight=args.face_weight, check_auxiliary_gradients=True,
                                     with_hand=args.with_hand, hand_store=hand_store, hand_weight=args.hand_weight)
            row['group_grad_norms'] = joint_group_gradients(adapter, names)
            validate_group_gradients(row['group_grad_norms'], branches=('body', 'face', 'hand') if args.with_hand else ('body', 'face'))
            if row['auxiliary_eligible_faces'] and not all(value > 0 for value in row['auxiliary_gradient_norms'].values()):
                raise AssertionError('Auxiliary gradient path interrupted')
            assert_frozen_gradients(adapter, transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder)
            rows.append(row)
            print(json.dumps({'preflight_index': index, **row}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        high_prompt = bank.prompt_for(pair_metadata(train.samples[args.probe_indices[0]])["pair_id"]) if study_mode else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            high = run_joint_loss(pipe, adapter, cache, train, store, args.probe_indices[0], seed=24042,
                                  prompt=high_prompt,
                                  fixed_sigma=.8, face_weight=args.face_weight,
                                  with_hand=args.with_hand, hand_store=hand_store, hand_weight=args.hand_weight)
        high['group_grad_norms'] = joint_group_gradients(adapter, names)
        validate_group_gradients(high['group_grad_norms'], branches=('body', 'hand') if args.with_hand else ('body',))
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
        seed_all(args.seed + args.noise_offset)
    if start_step == 0:
        save(args.output / 'checkpoint_step_000000.pt', adapter, optimizer, args, step=0, signature=signature, sample_order=[])
        if not metrics_path.exists():
            metrics_path.touch()
    if args.stop_after_step == 0:
        all_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        json_write(args.output / 'summary.json', {
            'completed': False, 'stopped_at_checkpoint': True, 'steps': 0, 'max_updates': args.steps,
            'unique_pairs': 0, 'face_active_steps': 0, 'hand_active_steps': 0, 'auxiliary_steps': 0,
            'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True,
            'last_loss': None, 'active_seconds': 0.0,
        })
        return
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    end_step = args.stop_after_step
    for step in range(start_step, end_step):
        if study_mode:
            index = order[step]
            scale = warmup_scale(step)
            current_seed = noise_seed(step)
            prompt = bank.prompt_for(pair_metadata(train.samples[index])["pair_id"])
        else:
            index = step
            scale = min((step + 1) / args.warmup_steps, 1.) if args.warmup_steps else 1.
            current_seed = args.seed + args.noise_offset + step
            prompt = None

        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * scale
        with torch.autocast('cuda', dtype=torch.bfloat16):
            row = run_joint_loss(pipe, adapter, cache, train, store, index,
                                 seed=current_seed, prompt=prompt,
                                 auxiliary=args.aux_every > 0 and step % args.aux_every == 0,
                                 evaluators=evaluators, supervision=supervision, face_weight=args.face_weight,
                                 with_hand=args.with_hand, hand_store=hand_store, hand_weight=args.hand_weight)
        grads = joint_group_gradients(adapter, names)
        active_branches = ['body']
        if row['face_active']:
            active_branches.append('face')
        if row.get('hand_active'):
            active_branches.append('hand')
        validate_group_gradients(grads, branches=tuple(active_branches))
        norm = float(torch.nn.utils.clip_grad_norm_([parameter for parameter in adapter.parameters() if parameter.requires_grad], 1.))
        if not math.isfinite(norm) or norm <= 0 or not math.isfinite(row['loss']):
            raise FloatingPointError(f'Invalid loss/gradient: {row}, norm={norm}')
        if row['face_active'] and sum(value for key, value in grads.items() if key.startswith('face')) <= 0:
            raise AssertionError('Active Face received no gradient')
        if row.get('hand_active') and sum(value for key, value in grads.items() if key.startswith('hand')) <= 0:
            raise AssertionError('Active Hand received no gradient')
        assert_frozen_gradients(adapter, transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder,
                               **{f'evaluator_{index}': model for index, model in enumerate(evaluators)})
        optimizer.step()
        assert_fp32_training_state(adapter, optimizer)
        optimizer.zero_grad(set_to_none=True)
        sample_order.append(index)
        row.update(step=step + 1, group_grad_norms=grads, grad_norm=norm,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3)
        if study_mode:
            row.update(epoch=step // 512, epoch_position=step % 512, protocol=protocol, arm=args.arm)
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        print(json.dumps(row), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == end_step or step + 1 == args.steps:
            save(args.output / f'checkpoint_step_{step + 1:06d}.pt', adapter, optimizer, args,
                 step=step + 1, signature=signature, sample_order=sample_order)
            save(args.output / 'checkpoint_last.pt', adapter, optimizer, args,
                 step=step + 1, signature=signature, sample_order=sample_order)
    frozen_after = frozen_joint_digest(adapter, names)
    unchanged = versions == [parameter._version for parameter in pipe.transformer.parameters()]
    if not unchanged or frozen_before != frozen_after:
        raise AssertionError('Frozen model/Adapter parameters changed')
    all_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    actual_step = len(sample_order)
    prior_summary = {}
    summary_path = args.output / 'summary.json'
    if summary_path.is_file():
        try:
            prior_summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    prev_active = float(prior_summary.get('active_seconds', 0.0)) if args.resume else 0.0
    json_write(args.output / 'summary.json', {
        'completed': actual_step == args.steps,
        'stopped_at_checkpoint': actual_step < args.steps,
        'steps': actual_step,
        'max_updates': args.steps,
        'unique_pairs': len(set(sample_order)),
        'face_active_steps': sum(row['face_active'] for row in all_rows),
        'hand_active_steps': sum(row.get('hand_active', False) for row in all_rows),
        'auxiliary_steps': sum(row.get('auxiliary_eligible_faces', 0) > 0 for row in all_rows),
        'frozen_transformer_unchanged': unchanged,
        'frozen_adapter_unchanged': frozen_before == frozen_after,
        'last_loss': all_rows[-1]['loss'] if all_rows else None,
        'active_seconds': prev_active + (time.perf_counter() - started),
    })


if __name__ == '__main__':
    main()
