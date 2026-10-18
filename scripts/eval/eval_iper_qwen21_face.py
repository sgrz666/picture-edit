#!/usr/bin/env python3
"""Sample the frozen Qwen21 Body checkpoint with a separately trained Face branch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from src.data.iper_dataset import IPERPoseDataset
from src.data.tiktok_dataset import load_pose_dataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.face_adapter import Qwen21FaceAdapter
from src.pose_control.qwen21.runtime import PROMPT, dataset_image_to_pil
from train_iper_qwen21_face import FaceFeatureStore
from eval_qwen21_face_references import ROLE_PROMPT, composite


ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--dataset-type', choices=('auto', 'iper', 'tiktok_v3'), default='auto')
    parser.add_argument('--align-mode', choices=('square_crop', 'resize'), default='square_crop')
    parser.add_argument('--sampled-root', type=Path, default=ROOT / 'datasets/iPER/iper_sampled_6src64tgt')
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast')
    parser.add_argument('--tiktok-raw-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    parser.add_argument('--tiktok-assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--features', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--body-checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'face_eval')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modes', nargs='+', default=['body_face', 'body_face_crop', 'body_face_gated'])
    return parser.parse_args()


def source_crop(original_path, bbox):
    with Image.open(original_path) as image:
        rgb = image.convert('RGB')
    x0, y0, x1, y1 = bbox
    width, height = rgb.size
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.8
    left = max(0, min(width - side, center_x - side / 2))
    top = max(0, min(height - side, center_y - side / 2))
    return rgb.crop((int(left), int(top), int(left + side), int(top + side))).resize((512, 512), Image.Resampling.LANCZOS)


def main():
    args = arguments()
    allowed = {'body_face', 'body_face_crop', 'body_face_gated'}
    if not set(args.modes) <= allowed:
        raise ValueError(f'unsupported modes: {set(args.modes) - allowed}')
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / 'images'
    image_dir.mkdir(exist_ok=True)
    dataset = load_pose_dataset(
        args.pairs,
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
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    body = Qwen21BodyAdapter().to('cuda', dtype=torch.float32).eval()
    body.load_state_dict(torch.load(args.body_checkpoint, map_location='cpu', weights_only=False)['adapter_state_dict'])
    face = Qwen21FaceAdapter().to('cuda', dtype=torch.float32).eval()
    face.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['face_state_dict'])
    rows = []
    total = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    for index in range(total):
        sample = dataset[index]
        meta = dataset.samples[index]
        batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value] for name, value in sample.items()}
        source = dataset_image_to_pil(sample['src_image'])
        embedding, crop, box, valid = store.batch([meta])
        grid = args.resolution // 16
        with torch.no_grad():
            state = face.prepare(embedding, crop, box, valid, token_hw=(grid, grid))
            geometry = body.encode_geometry(batch, (grid, grid))
        second = None
        if bool(valid.item()):
            features = store.app(meta['appearance'])['source'][meta['source']['stem']]
            original = Path(meta['source_rgb']) if 'source_rgb' in meta else (args.sampled_root / meta['appearance'] / meta['source']['rgb_1024'])
            second = source_crop(original, features['detection']['bbox'])
        stem = f"{index:03d}_{sample['appearance']}_{sample['source_stem']}_{sample['target_stem']}"
        row = {'pair_idx': index, 'stem': stem, 'face_valid': bool(valid.item()), 'modes': {}}
        for mode in args.modes:
            images = [source, second] if mode == 'body_face_crop' and second is not None else [source]
            prompt = ROLE_PROMPT if len(images) == 2 else PROMPT
            body_face_scale = .5 if mode == 'body_face_gated' and bool(valid.item()) else 1.0
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with Qwen21ControlContext(pipe.transformer, body, geometry, face_adapter=face,
                                      face_state=state, face_strength=1, body_face_scale=body_face_scale):
                result = pipe(prompt=prompt, image=images, width=args.resolution, height=args.resolution, output_resolution=args.resolution,
                              num_inference_steps=args.steps, true_cfg_scale=1.0,
                              use_kv_cache=True, generator=generator)
            torch.cuda.synchronize()
            path = image_dir / f'{stem}_{mode}.png'
            composite(result.images[0]).save(path)
            row['modes'][mode] = {'image': str(path), 'seconds': time.perf_counter() - started,
                                  'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3,
                                  'reference_count': len(images), 'body_face_scale': body_face_scale}
        rows.append(row)
        (args.output / 'per_sample.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in rows), encoding='utf-8')
        print(f'{index + 1}/{total} {stem} face_valid={bool(valid.item())}', flush=True)
    (args.output / 'config.json').write_text(json.dumps({
        'pairs': str(args.pairs), 'checkpoint': str(args.checkpoint), 'body_checkpoint': str(args.body_checkpoint),
        'steps': args.steps, 'seed': args.seed, 'modes': args.modes,
        'resolution': args.resolution, 'true_cfg_scale': 1.0, 'use_kv_cache': True,
        'single_image_prompt': PROMPT, 'two_image_prompt': ROLE_PROMPT,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
