#!/usr/bin/env python3
"""Complete 30-step sampling for the preserved full controller and Qwen21."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.pose_control.qwen21_full.execution import (
    MODEL, RUN_ROOT, FaceConditionStore, QwenFullDataset, Qwen21IPERCache,
    batch_one, json_write, load_adapter_checkpoint, new_adapter, prepare, seed_all,
)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
from src.pose_control.qwen21_full.qwen_runtime import PROMPT, dataset_image_to_pil


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=RUN_ROOT / 'val_pairs.jsonl')
    p.add_argument('--dataset-type', choices=('auto', 'iper', 'tiktok_v3'), default='auto')
    p.add_argument('--tiktok-raw-root', type=Path, default=MODEL.parents[1] / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    p.add_argument('--tiktok-assets-root', type=Path, default=MODEL.parents[1] / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--features', type=Path, default=RUN_ROOT / 'face_cache')
    p.add_argument('--cache', type=Path, default=RUN_ROOT / 'native_cache')
    p.add_argument('--body-checkpoint', type=Path, default=RUN_ROOT / 'body_512_100/checkpoint_last.pt')
    p.add_argument('--face-checkpoint', type=Path, default=RUN_ROOT / 'face_512_100/checkpoint_last.pt')
    p.add_argument('--modes', nargs='+', choices=('initial', 'body', 'body_face'), default=['initial', 'body', 'body_face'])
    p.add_argument('--output', type=Path)
    p.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--progress-mode', choices=('step', 'sigma'), default='step')
    p.add_argument('--face-strength', type=float, default=1.0)
    p.add_argument('--face-scatter-grid', choices=('native', 'checkpoint'), default='native',
                   help='Native reproduces historical sampling; checkpoint maps Face residuals from its training grid.')
    p.add_argument('--use-kv-cache', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def opaque(image):
    image = image.convert('RGBA')
    return Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')


def image_metrics(generated, target, mask):
    # All pixel comparisons use a common 512 grid, including 1024 outputs.
    g = np.asarray(generated.resize((512, 512), Image.Resampling.LANCZOS)).astype(np.float32)
    t = np.asarray(target.resize((512, 512), Image.Resampling.LANCZOS)).astype(np.float32)
    m = torch.nn.functional.interpolate(mask.float().reshape(1, 1, *mask.shape[-2:]),
                                         size=(512, 512), mode='nearest')[0, 0].numpy()
    error = np.abs(g - t).mean(axis=-1)
    return {'psnr': float(peak_signal_noise_ratio(t, g, data_range=255)),
            'ssim': float(structural_similarity(t, g, channel_axis=2, data_range=255)),
            'human_mae': float((error * m).sum() / max(m.sum(), 1)),
            'background_mae': float((error * (1 - m)).sum() / max((1 - m).sum(), 1)),
            'metric_resolution': 512}


def main():
    args = arguments()
    args.output = args.output or RUN_ROOT / f'validation_{args.resolution}'
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / 'images'
    images.mkdir(exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    dataset = QwenFullDataset(args.pairs, resolution=args.resolution, dataset_type=args.dataset_type,
                              tiktok_raw_root=args.tiktok_raw_root, tiktok_assets_root=args.tiktok_assets_root)
    count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    store = FaceConditionStore(args.features)
    cache = Qwen21IPERCache(args.cache, resolution=args.resolution, prompt=args.prompt)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(true_cfg_scale=1.0, added_face_reference=False,
                  scope='validation engineering acceptance, not final test evaluation')
    config['pairs_sha256'] = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    config['checkpoint_sha256'] = {}
    for mode, checkpoint in (('body', args.body_checkpoint), ('body_face', args.face_checkpoint)):
        if mode in args.modes:
            with checkpoint.open('rb') as stream:
                config['checkpoint_sha256'][mode] = hashlib.file_digest(stream, 'sha256').hexdigest()
    config_path = args.output / 'config.json'
    if config_path.exists():
        previous_config = json.loads(config_path.read_text())
        previous_config.setdefault('face_scatter_grid', 'native')
        if previous_config != config:
            raise ValueError('Existing evaluation configuration differs; use a separate output directory')
    json_write(args.output / 'config.json', config)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    initial_state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    rows = []
    records = args.output / 'per_sample.jsonl'
    existing = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    completed = {(row['mode'], row['pair_index']): row for row in existing if Path(row['image']).exists()}
    for mode in args.modes:
        if mode == 'initial':
            adapter.load_state_dict(initial_state, strict=True)
        else:
            checkpoint_payload = load_adapter_checkpoint(adapter, args.body_checkpoint if mode == 'body' else args.face_checkpoint)
        face_reference_token_hw = None
        if args.face_scatter_grid == 'checkpoint' and mode == 'body_face':
            trained_resolution = checkpoint_payload.get('resolution')
            if trained_resolution not in (512, 1024):
                raise ValueError('Checkpoint-grid Face requires verified checkpoint training resolution')
            face_reference_token_hw = (trained_resolution // 16, trained_resolution // 16)
        for index in range(count):
            if (mode, index) in completed:
                rows.append(completed[(mode, index)])
                continue
            batch = batch_one(dataset, index)
            sample = dataset[index]
            meta = sample['pair_metadata']
            source = dataset_image_to_pil(sample['src_image'])
            target = dataset_image_to_pil(sample['tgt_image'])
            stem = f'{index:03d}_{meta["source_appearance"]}_{meta["source_stem"]}_{meta["target_stem"]}'
            source.save(images / f'{stem}_source.png')
            target.save(images / f'{stem}_target.png')
            with torch.no_grad():
                entry = cache.source(pipe, source)
                pair = {'source_latent': entry['reference_latent'].to('cuda'), 'resolution': args.resolution}
                prepared = prepare(adapter, batch, pair, [meta], store, with_face=mode == 'body_face')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                             face_strength=args.face_strength if mode == 'body_face' else 0,
                                             face_reference_token_hw=face_reference_token_hw,
                                             hand_strength=0, hand_mode='off', progress_mode=args.progress_mode,
                                             num_inference_steps=args.steps) as context:
                    result = pipe(prompt=args.prompt, image=[source], width=args.resolution, height=args.resolution,
                                  output_resolution=args.resolution, num_inference_steps=args.steps,
                                  true_cfg_scale=1.0, use_kv_cache=args.use_kv_cache,
                                  generator=torch.Generator(device='cuda').manual_seed(args.seed))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            generated = opaque(result.images[0])
            path = images / f'{stem}_{mode}.png'
            generated.save(path)
            metrics = None
            if meta['source_appearance'] == meta['target_appearance']:
                metrics = image_metrics(generated, target, sample['human_mask'])
            row = {'pair_index': index, 'mode': mode, 'image': str(path), 'source': str(images / f'{stem}_source.png'),
                   'target': str(images / f'{stem}_target.png'), 'metadata': meta,
                   'resolution': args.resolution, 'seconds': seconds,
                   'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3,
                   'face_valid': bool(prepared.face.face_valid.any()) if prepared.face is not None else None,
                   'control_forwards': context.forward_count, 'block_calls': context.call_counts,
                   'metrics': metrics, 'last_control_diagnostics': context.output.diagnostics}
            row['sampling_schedule'] = context.step_diagnostics
            row['injection_diagnostics'] = context.injection_diagnostics
            if context.forward_count != args.steps or any(v != args.steps for v in context.call_counts.values()):
                raise AssertionError('Incomplete full sampling or missing layer injection')
            with records.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('pair_index', 'mode', 'seconds', 'peak_cuda_gib', 'metrics')}), flush=True)
    summary = {}
    for mode in args.modes:
        selected = [row for row in rows if row['mode'] == mode]
        values = {'n': len(selected), 'seconds_mean': float(np.mean([row['seconds'] for row in selected])),
                  'peak_cuda_gib': max(row['peak_cuda_gib'] for row in selected)}
        with_metrics = [row for row in selected if row['metrics'] is not None]
        for key in ('psnr', 'ssim', 'human_mae', 'background_mae'):
            values[key] = float(np.mean([row['metrics'][key] for row in with_metrics])) if with_metrics else None
        summary[mode] = values
    json_write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
