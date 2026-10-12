#!/usr/bin/env python3
"""Paired Qwen21 Body 20k probe: source image with or without its own face crop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.runtime import PROMPT, dataset_image_to_pil


ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')
ROLE_PROMPT = (
    'Image 1 is the full source photo. Image 2 is a crop of the same person’s face from image 1. '
    'Change the person in image 1 to match the supplied target pose geometry. '
    'Preserve the identity and facial features from image 2, and preserve the clothing and background from image 1.'
)


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--inventory', type=Path, default=FACE_ROOT / 'face_val_inventory.jsonl')
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'reference_probe')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--modes', nargs='+', default=['body', 'body_crop_same', 'body_crop_role'])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def composite(image):
    rgba = image.convert('RGBA')
    return Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    options = args()
    allowed = {'body', 'body_crop_same', 'body_crop_role'}
    if not set(options.modes) <= allowed:
        raise ValueError(f'unsupported modes: {set(options.modes) - allowed}')
    if any(mode != 'body' for mode in options.modes) and 'body' not in options.modes:
        raise ValueError('body is required for unusable-face fallback')
    inventory = read_jsonl(options.inventory)
    dataset = IPERPoseDataset(
        str(options.pairs), str(ROOT / 'datasets/iPER/iper_sampled_6src64tgt'),
        str(ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast'), resolution=options.resolution, augment=False,
    )
    if len(dataset) != len(inventory):
        raise ValueError('pair and face inventory counts differ')
    options.output.mkdir(parents=True, exist_ok=True)
    image_dir = options.output / 'images'
    image_dir.mkdir(exist_ok=True)
    device = torch.device('cuda')
    pipe = QwenImage21Pipeline.from_pretrained(str(options.model), dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    adapter = Qwen21BodyAdapter().to(device=device, dtype=torch.float32).eval()
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    adapter.load_state_dict(saved['adapter_state_dict'])
    rows = []
    total = min(len(dataset), options.max_samples) if options.max_samples else len(dataset)
    for i in range(total):
        sample = dataset[i]
        record = inventory[i]
        if (record['appearance'], record['source_stem'], record['target_stem']) != (
            sample['appearance'], sample['source_stem'], sample['target_stem']
        ):
            raise AssertionError(f'pair mismatch at {i}')
        batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value] for name, value in sample.items()}
        source = dataset_image_to_pil(sample['src_image'])
        with torch.no_grad():
            grid = options.resolution // 16
            geometry = adapter.encode_geometry(batch, (grid, grid))
        face = None
        if record['face_usable']:
            with Image.open(record['face_crop_path']) as image:
                face = image.convert('RGB')
        modes = [('body', [source], PROMPT)]
        if face is not None:
            modes.extend([('body_crop_same', [source, face], PROMPT),
                          ('body_crop_role', [source, face], ROLE_PROMPT)])
        modes = [entry for entry in modes if entry[0] in options.modes]
        stem = f"{i:03d}_{sample['appearance']}_{sample['source_stem']}_{sample['target_stem']}"
        row = {'pair_idx': i, 'stem': stem, 'source_face_usable': bool(face is not None), 'results': {}}
        for name, images, prompt in modes:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(options.seed)
            kwargs = dict(prompt=prompt, image=images, width=options.resolution, height=options.resolution, output_resolution=options.resolution,
                          num_inference_steps=options.steps, true_cfg_scale=1.0, use_kv_cache=True,
                          generator=generator)
            with Qwen21ControlContext(pipe.transformer, adapter, geometry) as control:
                result = pipe(**kwargs)
            torch.cuda.synchronize()
            if any(count == 0 for count in control.call_counts.values()):
                raise RuntimeError('body control layer did not run')
            image_path = image_dir / f'{stem}_{name}.png'
            composite(result.images[0]).save(image_path)
            row['results'][name] = {'image': str(image_path), 'seconds': time.perf_counter() - started,
                                    'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3,
                                    'reference_count': len(images), 'prompt': prompt}
        if face is None:
            for name in ('body_crop_same', 'body_crop_role'):
                if name in options.modes:
                    row['results'][name] = {'fallback': 'body'}
        rows.append(row)
        with (options.output / 'per_sample.jsonl').open('w', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(entry, ensure_ascii=False) + '\n' for entry in rows))
        print(f'{i + 1}/{total} {stem} usable_face={face is not None}', flush=True)
    (options.output / 'config.json').write_text(json.dumps({
        'checkpoint': str(options.checkpoint), 'steps': options.steps, 'seed': options.seed,
        'resolution': options.resolution, 'true_cfg_scale': 1.0, 'use_kv_cache': True,
        'pairs': str(options.pairs), 'inventory': str(options.inventory), 'modes': options.modes,
        'role_prompt': ROLE_PROMPT, 'fixed_prompt': PROMPT,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
