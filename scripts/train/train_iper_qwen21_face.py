#!/usr/bin/env python3
"""Train a source-identity Qwen21 face ROI branch over the frozen Body 20k adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline

from src.data.iper_dataset import IPERPoseDataset
from src.data.tiktok_dataset import load_pose_dataset, transform_box_for_crop
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.face_adapter import Qwen21FaceAdapter
from src.pose_control.qwen21.runtime import Qwen21IPERCache, transformer_inputs
from train_iper_qwen21_body import batch_one, precompute


ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--body-checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    parser.add_argument('--train-pairs', type=Path, default=FACE_ROOT / 'face_train_pairs.jsonl')
    parser.add_argument('--val-pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--dataset-type', choices=('auto', 'iper', 'tiktok_v3'), default='auto')
    parser.add_argument('--align-mode', choices=('square_crop', 'resize'), default='square_crop')
    parser.add_argument('--sampled-root', type=Path, default=ROOT / 'datasets/iPER/iper_sampled_6src64tgt')
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast')
    parser.add_argument('--tiktok-raw-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    parser.add_argument('--tiktok-assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--features', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--cache', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_iper_cache'))
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'face_train_3k')
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--aux-every', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-train-pairs', type=int, default=0)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--init-checkpoint', type=Path, help='Load Face weights only; start a new optimizer run')
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--core-lr', type=float, default=1e-5)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    return parser.parse_args()


class FaceFeatureStore:
    def __init__(self, root: Path):
        self.root = root
        self.loaded = {}

    def app(self, appearance: str):
        if appearance not in self.loaded:
            path = self.root / f'{appearance}.pt'
            if not path.exists():
                raise FileNotFoundError(path)
            self.loaded[appearance] = torch.load(path, map_location='cpu', weights_only=False)
        return self.loaded[appearance]

    @staticmethod
    def _resolve_box(meta, geometry):
        if not geometry:
            return torch.zeros(4, dtype=torch.float32), False
        box = torch.tensor(geometry['box'], dtype=torch.float32)
        norm_crop = meta.get('norm_crop')
        if norm_crop is not None and tuple(norm_crop) != (0.0, 0.0, 1.0, 1.0):
            box = transform_box_for_crop(box, norm_crop)
        box_valid = bool(
            geometry.get('geometry_valid', False)
            and float(box[2] - box[0]) > 0.02
            and float(box[3] - box[1]) > 0.02
        )
        return box, box_valid

    def usable(self, meta):
        if not (self.root / f"{meta['appearance']}.pt").exists():
            return False
        features = self.app(meta['appearance'])
        source = features['source'].get(meta['source']['stem'])
        geometry = features['target_geometry'].get(meta['target']['stem'])
        target = features['target_visibility'].get(meta['target']['stem'])
        _, box_valid = self._resolve_box(meta, geometry)
        return bool(source and source['valid'] and box_valid and target and target['visible'])

    def batch(self, metadata):
        embedding, crop, box, valid = [], [], [], []
        for item in metadata:
            features = self.app(item['appearance'])
            source = features['source'][item['source']['stem']]
            target = features['target_geometry'][item['target']['stem']]
            visible = features['target_visibility'][item['target']['stem']]['visible']
            resolved_box, box_valid = self._resolve_box(item, target)
            embedding.append(source['embedding'])
            crop.append(source['crop'])
            box.append(resolved_box)
            valid.append(bool(source['valid'] and box_valid and visible))
        return torch.stack(embedding), torch.stack(crop), torch.stack(box), torch.tensor(valid, dtype=torch.bool)


def crop_batch(images, boxes, side=112):
    axis = (torch.arange(side, device=images.device, dtype=images.dtype) + .5) / side
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    x0, y0, x1, y1 = [boxes[:, index, None, None] for index in range(4)]
    grid_x = x0 + x[None] * (x1 - x0)
    grid_y = y0 + y[None] * (y1 - y0)
    return F.grid_sample(images, torch.stack((grid_x * 2 - 1, grid_y * 2 - 1), dim=-1),
                         mode='bilinear', padding_mode='border', align_corners=False)


def decode_x0(pipe, tokens, resolution):
    latents = pipe._unpack_latents(tokens, resolution, resolution, pipe.vae_scale_factor).to(pipe.vae.dtype)
    # A differentiable full-size VAE decode exceeds one 85 GiB GPU at 1024.
    # Keep the flow objective at 1024, but compute the auxiliary face losses
    # from a 512 decode of the predicted latent grid.
    if resolution > 512:
        latents = F.interpolate(latents, size=(1, 512 // pipe.vae_scale_factor,
                                                512 // pipe.vae_scale_factor), mode='trilinear', align_corners=False)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    pixels = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    pixels = (pixels.float() * .5 + .5).clamp(0, 1)
    if pixels.shape[1] == 4:
        return pixels[:, :3] * pixels[:, 3:4] + (1 - pixels[:, 3:4])
    return pixels[:, :3]


def feature(evaluator, images):
    value = evaluator(images.float())
    if isinstance(value, (tuple, list)):
        value = value[-1]
    return value.flatten(1)


def loss_one(pipe, body, face, cache, store, dataset, index, *, seed, auxiliary, id_evaluator, perceptual_evaluator):
    device = next(face.parameters()).device
    batch = batch_one(dataset, index)
    meta = dataset.samples[index]
    embedding, crop, box, valid = store.batch([meta])
    grid = cache.resolution // 16
    tokens = grid * grid
    state = face.prepare(embedding, crop, box, valid, token_hw=(grid, grid))
    pair = cache.pair(pipe, batch, device)
    target = pair['target_latent']
    generator = torch.Generator(device=device).manual_seed(seed)
    sigma = torch.rand((1,), device=device, dtype=target.dtype, generator=generator)
    if auxiliary:
        sigma = sigma * .35
    noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    velocity = noise - target
    with torch.no_grad():
        geometry = body.encode_geometry(batch, (grid, grid))
    inputs = transformer_inputs(pair, noisy, sigma)
    body_prediction = None
    if auxiliary and bool(valid.item()):
        with torch.no_grad():
            with Qwen21ControlContext(pipe.transformer, body, geometry, sigma=sigma, target_tokens=tokens):
                body_prediction = pipe.transformer(**inputs)[0][:, -tokens:]
    with Qwen21ControlContext(pipe.transformer, body, geometry, sigma=sigma, target_tokens=tokens,
                              face_adapter=face, face_state=state, face_strength=1):
        prediction = pipe.transformer(**inputs)[0][:, -tokens:]
        face_mask = state.mask.to(prediction.device)
        human = F.interpolate(batch['human_mask'].to(device=device, dtype=torch.float32), size=(grid, grid),
                              mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        weight = 1 + 2 * human + 4 * face_mask
        squared = (prediction.float() - velocity.float()).square()
        flow_loss = (squared * weight).sum() / (weight.sum() * squared.shape[-1])
        identity_loss = flow_loss.new_zeros(())
        perceptual_loss = flow_loss.new_zeros(())
        outside_loss = flow_loss.new_zeros(())
        if auxiliary and bool(valid.item()):
            x0 = noisy - sigma[:, None, None] * prediction
            predicted_face = crop_batch(decode_x0(pipe, x0, cache.resolution), box.to(device), 112)
            target_pixels = (batch['tgt_image'].to(device).float() * .5 + .5).clamp(0, 1)
            target_face = crop_batch(target_pixels, box.to(device), 112)
            with torch.no_grad():
                target_id = feature(id_evaluator, target_face)
                target_perc = feature(perceptual_evaluator, target_face)
            generated_id = feature(id_evaluator, predicted_face)
            generated_perc = feature(perceptual_evaluator, predicted_face)
            identity_loss = (1 - F.cosine_similarity(generated_id, target_id, dim=-1)).mean()
            perceptual_loss = (generated_perc - target_perc).square().mean()
            assert body_prediction is not None
            outside = 1 - face_mask
            outside_loss = ((prediction.float() - body_prediction.float()).square() * outside).sum() / (
                outside.sum() * prediction.shape[-1]).clamp_min(1)
        total = flow_loss + .10 * identity_loss + .05 * perceptual_loss + .02 * outside_loss
        total.backward()
    return {key: float(value.detach()) for key, value in {
        'total': total, 'flow': flow_loss, 'id': identity_loss,
        'perceptual': perceptual_loss, 'outside': outside_loss,
    }.items()}


def save(path, face, optimizer, step, micro_step, args, best):
    payload = {'face_state_dict': face.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'step': step, 'micro_step': micro_step, 'args': vars(args), 'best_validation': best,
               'torch_rng_state': torch.random.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = arguments()
    if args.head_lr <= 0 or args.core_lr <= 0:
        raise ValueError('learning rates must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = load_pose_dataset(
        args.train_pairs,
        dataset_type=args.dataset_type,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        tiktok_raw_root=args.tiktok_raw_root,
        tiktok_assets_root=args.tiktok_assets_root,
        resolution=args.resolution,
        align_mode=args.align_mode,
        augment=False,
    )
    val = load_pose_dataset(
        args.val_pairs,
        dataset_type=args.dataset_type,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        tiktok_raw_root=args.tiktok_raw_root,
        tiktok_assets_root=args.tiktok_assets_root,
        resolution=args.resolution,
        align_mode=args.align_mode,
        augment=False,
    )
    store = FaceFeatureStore(args.features)
    usable = [index for index, meta in enumerate(train.samples) if store.usable(meta)]
    val_usable = [index for index, meta in enumerate(val.samples) if store.usable(meta)]
    if args.max_train_pairs:
        usable = usable[:args.max_train_pairs]
    if not usable or not val_usable:
        raise RuntimeError('no usable face training or validation pairs')
    print(f'usable train {len(usable)}/{len(train)}, validation {len(val_usable)}/{len(val)}', flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    pipe.transformer.eval().requires_grad_(False)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    pipe.text_encoder.eval().requires_grad_(False).to('cuda')
    pipe.vae.eval().requires_grad_(False).to('cuda')
    cache = Qwen21IPERCache(args.cache, resolution=args.resolution)
    precompute(pipe, cache, train, indices=usable)
    if args.val_count:
        precompute(pipe, cache, val, indices=val_usable[:args.val_count])
    pipe.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    pipe.transformer.to('cuda')
    body = Qwen21BodyAdapter().to('cuda', dtype=torch.float32).eval()
    body.load_state_dict(torch.load(args.body_checkpoint, map_location='cpu', weights_only=False)['adapter_state_dict'])
    body.requires_grad_(False)
    face = Qwen21FaceAdapter().to('cuda', dtype=torch.float32)
    if args.init_checkpoint:
        face.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)['face_state_dict'])
    id_evaluator = torch.jit.load(str(ROOT / 'models/face_eval/identity_encoder.ts'), map_location='cuda').eval()
    perceptual_evaluator = torch.jit.load(str(ROOT / 'models/face_eval/perceptual_encoder.ts'), map_location='cuda').eval()
    for evaluator in (id_evaluator, perceptual_evaluator):
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': list(face.zero_heads.parameters()), 'lr': args.head_lr},
        {'params': [parameter for name, parameter in face.named_parameters() if not name.startswith('zero_heads.')], 'lr': 0.0},
    ], weight_decay=.01)
    start_step, micro_step, best = 0, 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if int(saved.get('args', {}).get('resolution', 512)) != args.resolution:
            raise ValueError('cannot resume a checkpoint with a different resolution')
        face.load_state_dict(saved['face_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_step, micro_step, best = saved['step'], saved['micro_step'], saved['best_validation']
        torch.random.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    elif not args.init_checkpoint:
        sample_index = usable[0]
        sample = batch_one(train, sample_index)
        pair = cache.pair(pipe, sample, torch.device('cuda'))
        embedding, crop, box, valid = store.batch([train.samples[sample_index]])
        grid = args.resolution // 16
        tokens = grid * grid
        state = face.prepare(embedding, crop, box, valid, token_hw=(grid, grid))
        sigma = torch.tensor([.5], device='cuda', dtype=pair['target_latent'].dtype)
        inputs = transformer_inputs(pair, pair['target_latent'] * .5, sigma)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            geometry = body.encode_geometry(sample, (grid, grid))
            with Qwen21ControlContext(pipe.transformer, body, geometry, sigma=sigma, target_tokens=tokens):
                body_only = pipe.transformer(**inputs)[0][:, -tokens:]
            with Qwen21ControlContext(pipe.transformer, body, geometry, sigma=sigma, target_tokens=tokens,
                                      face_adapter=face, face_state=state, face_strength=1):
                face_zero = pipe.transformer(**inputs)[0][:, -tokens:]
        difference = float((body_only.float() - face_zero.float()).abs().max())
        if difference > 1e-3:
            raise AssertionError(f'zero Face branch changed Body output by {difference}')
        print(f'zero Face parity max_abs={difference}', flush=True)
    manifest = {'train_sha256': hashlib.sha256(args.train_pairs.read_bytes()).hexdigest(),
                'val_sha256': hashlib.sha256(args.val_pairs.read_bytes()).hexdigest(),
                'auxiliary_decode_resolution': min(args.resolution, 512),
                'body_checkpoint': str(args.body_checkpoint), 'args': {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    started = time.perf_counter()
    log = args.output / 'train_metrics.jsonl'
    for step in range(start_step, args.max_steps):
        if step == args.warmup_steps:
            optimizer.param_groups[1]['lr'] = args.core_lr
        optimizer.zero_grad(set_to_none=True)
        reports = []
        indices = []
        for _ in range(args.accumulation):
            epoch, offset = divmod(micro_step, len(usable))
            permutation = torch.randperm(len(usable), generator=torch.Generator().manual_seed(args.seed + epoch))
            index = usable[int(permutation[offset])]
            micro_step += 1
            indices.append(index)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reports.append(loss_one(pipe, body, face, cache, store, train, index,
                    seed=args.seed * 1000000 + micro_step, auxiliary=(step % args.aux_every == 0),
                    id_evaluator=id_evaluator, perceptual_evaluator=perceptual_evaluator))
        norm = float(torch.nn.utils.clip_grad_norm_(face.parameters(), 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError(f'nonfinite face gradient at step {step + 1}')
        optimizer.step()
        row = {'step': step + 1, 'micro_step': micro_step, 'pair_indices': indices,
               'loss': float(np.mean([x['total'] for x in reports])), 'grad_norm': norm,
               'flow_loss': float(np.mean([x['flow'] for x in reports])),
               'identity_loss': float(np.mean([x['id'] for x in reports])),
               'perceptual_loss': float(np.mean([x['perceptual'] for x in reports])),
               'outside_loss': float(np.mean([x['outside'] for x in reports])),
               'seconds': time.perf_counter() - started, 'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3}
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            # Frozen validation sample/order/noise; score is used only as a training-health proxy.
            face.eval()
            val_losses = []
            for index in val_usable[:args.val_count]:
                sample = batch_one(val, index)
                pair = cache.pair(pipe, sample, torch.device('cuda'))
                embedding, crop, box, valid = store.batch([val.samples[index]])
                grid = args.resolution // 16
                tokens = grid * grid
                state = face.prepare(embedding, crop, box, valid, token_hw=(grid, grid))
                sigma = torch.tensor([.25], device='cuda', dtype=pair['target_latent'].dtype)
                generator = torch.Generator(device='cuda').manual_seed(42000 + index)
                noise = torch.randn(pair['target_latent'].shape, device='cuda', dtype=pair['target_latent'].dtype, generator=generator)
                target = pair['target_latent']
                noisy = .75 * target + .25 * noise
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    geometry = body.encode_geometry(sample, (grid, grid))
                    with Qwen21ControlContext(pipe.transformer, body, geometry, sigma=sigma, target_tokens=tokens,
                                              face_adapter=face, face_state=state, face_strength=1):
                        prediction = pipe.transformer(**transformer_inputs(pair, noisy, sigma))[0][:, -tokens:]
                    val_losses.append(float((prediction.float() - (noise - target).float()).square().mean()))
            row['val_velocity_mse'] = float(np.mean(val_losses)) if val_losses else None
            face.train()
            if val_losses and row['val_velocity_mse'] < best:
                best = row['val_velocity_mse']
                save(args.output / 'checkpoint_best_loss.pt', face, optimizer, step + 1, micro_step, args, best)
            save(args.output / 'checkpoint_last.pt', face, optimizer, step + 1, micro_step, args, best)
            save(args.output / f'checkpoint_step_{step + 1:06d}.pt', face, optimizer,
                 step + 1, micro_step, args, best)
            print(f"step {step + 1} val_velocity_mse={row['val_velocity_mse']}", flush=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        if (step + 1) % args.log_every == 0 or step == start_step:
            print(json.dumps(row), flush=True)
    print(f'finished Face optimizer steps={args.max_steps}', flush=True)


if __name__ == '__main__':
    main()
