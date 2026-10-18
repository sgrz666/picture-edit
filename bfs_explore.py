#!/usr/bin/env python3
"""Explore practical use of BFS Qwen-2.1 LoRAs on the project's iPER validation pairs.

Head swap (two-stage): stage 1 = frozen Body-adapter pose edit; stage 2 = BFS head LoRA
taking image1 = stage-1 result, image2 = face crop of A. Baselines: stage-1 output alone
and stage 2 without the LoRA (same inputs, repair prompt) to isolate the LoRA effect.
Body swap probe: image1 = B target frame, image2 = A full image (direction-reversed use).

Metrics follow the project protocol: antelopev2 identity cosine vs A and vs Target,
DWPose PCK@0.02 against Target BODY25 keypoints from v6_conditions.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import QwenImage21Pipeline

from src.data.tiktok_dataset import load_pose_dataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.runtime import PROMPT, dataset_image_to_pil
from eval_iper_four_arm import IdentityMetric, PoseMetric

ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')

BFS_HEAD_PROMPT = (
    'head_swap: start with image 1 as the base image, keeping its lighting, environment, and background. '
    'remove the head from image 1 completely and replace it with the head from image 2, '
    'strictly preserving the hair, eye color, nose structure from image 2. '
    'copy the direction of the eye, head rotation, micro expressions from image 1, high quality, sharp details, 4k'
)
REPAIR_PROMPT = (
    'Image 1 is an edited photo whose face is damaged. Image 2 is a crop of the same person\'s face '
    'from the original photo. Keep the pose, clothing, and background of image 1, and restore the '
    'face of the person to match image 2.'
)
BFS_BODY_PROMPT = (
    'body_swap: start with image 1 as the base image, keeping its lighting, environment, and background. '
    'replace the body from image 1 with the body from image 2, strictly preserving the clothing, '
    'body shape and proportions from image 2. '
    'copy the pose, direction of the eye, head rotation, micro expressions from image 1'
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--dataset-type', choices=('auto', 'iper', 'tiktok_v3'), default='auto')
    parser.add_argument('--sampled-root', type=Path, default=ROOT / 'datasets/iPER/iper_sampled_6src64tgt')
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast')
    parser.add_argument('--body-checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--bfs-dir', type=Path, default=Path('/home/shangguanrz/models/bfs'))
    parser.add_argument('--output', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_explore'))
    parser.add_argument('--max-pairs', type=int, default=6)
    parser.add_argument('--body-swap-pairs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_trusted_state(path: Path):
    """Load the project's own training artifacts; allowlist the path types they embed."""
    import pathlib
    from torch.serialization import safe_globals
    with safe_globals([pathlib.PosixPath, pathlib.PurePosixPath, pathlib.WindowsPath, pathlib.PureWindowsPath]):
        return torch.load(path, map_location='cpu', weights_only=True)


def detect_largest_face(identity: IdentityMetric, image_path: Path):
    bgr = cv2.cvtColor(np.asarray(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)
    boxes, landmarks = identity.det.detect(bgr)
    if len(boxes) == 0:
        return None
    index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 0]) * 0 + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    x0, y0, x1, y1 = [float(v) for v in boxes[index][:4]]
    return (x0, y0, x1, y1)


def face_crop(image_path: Path, bbox) -> Image.Image:
    rgb = Image.open(image_path).convert('RGB')
    width, height = rgb.size
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.8
    left = max(0, min(width - side, center_x - side / 2))
    top = max(0, min(height - side, center_y - side / 2))
    return rgb.crop((int(left), int(top), int(left + side), int(top + side))).resize((512, 512), Image.Resampling.LANCZOS)


def convert_bfs_state_dict(path: Path, gate_first: bool) -> dict:
    """Remap ComfyUI-native Qwen-2.1 LoRA keys to the diffusers 2.1 module names.

    ComfyUI fuses the SwiGLU projections into one `img_mlp.gate_up` Linear (out 24576 =
    gate 12288 + up 12288); diffusers splits them into `img_mlp.gate_layer` and
    `img_mlp.proj`. The shared lora_A is duplicated to both targets.
    """
    import re

    from safetensors.torch import load_file
    state = load_file(str(path))
    pattern = re.compile(r'diffusion_model\.transformer_blocks\.(\d+)\.img_mlp\.gate_up\.(lora_[AB])\.weight')
    converted = {}
    for key, value in state.items():
        match = pattern.match(key)
        if match:
            index, part = match.group(1), match.group(2)
            prefix = f'transformer.transformer_blocks.{index}.img_mlp.'
            if part == 'lora_A':
                converted[prefix + 'gate_layer.lora_A.weight'] = value
                converted[prefix + 'proj.lora_A.weight'] = value
            elif gate_first:
                converted[prefix + 'gate_layer.lora_B.weight'] = value[:12288]
                converted[prefix + 'proj.lora_B.weight'] = value[12288:]
            else:
                converted[prefix + 'proj.lora_B.weight'] = value[:12288]
                converted[prefix + 'gate_layer.lora_B.weight'] = value[12288:]
        else:
            converted['transformer.' + key.replace('diffusion_model.', '', 1)] = value
    return converted


def load_lora(pipe, path: Path, adapter_name: str, gate_first: bool = True) -> str:
    pipe.load_lora_weights(convert_bfs_state_dict(path, gate_first), adapter_name=adapter_name)
    attached = [n for n, _ in pipe.transformer.named_modules() if 'lora_A' in n]
    if not any('img_mlp.proj' in n for n in attached):
        raise RuntimeError(f'LoRA {path.name} did not attach to img_mlp.proj')
    if not any('attn.to_q' in n for n in attached):
        raise RuntimeError(f'LoRA {path.name} did not attach to attn.to_q')
    return f'converted (gate_first={gate_first}, attached_targets={len(attached)})'


def generate(pipe, prompt: str, images, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        image=images,
        width=args.resolution,
        height=args.resolution,
        output_resolution=args.resolution,
        num_inference_steps=args.steps,
        true_cfg_scale=1.0,
        use_kv_cache=True,
        generator=generator,
    )
    return result.images[0]


def main():
    args = arguments()
    (args.output / 'images').mkdir(parents=True, exist_ok=True)
    dataset = load_pose_dataset(
        args.pairs,
        dataset_type=args.dataset_type,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        align_mode='square_crop',
        augment=False,
    )
    total = min(len(dataset), args.max_pairs)
    print(f'dataset pairs: {len(dataset)}, using {total}', flush=True)

    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    body = Qwen21BodyAdapter().to('cuda', dtype=torch.float32).eval()
    body_checkpoint = load_trusted_state(args.body_checkpoint)
    body.load_state_dict(body_checkpoint['adapter_state_dict'])

    identity = IdentityMetric()
    pose_metric = PoseMetric()
    condition_cache: dict[str, dict] = {}

    def target_points_for(meta) -> np.ndarray:
        appearance = meta['appearance']
        if appearance not in condition_cache:
            condition_cache[appearance] = load_trusted_state(args.assets_root / appearance / 'v6_conditions.pt')
        conditions = condition_cache[appearance]
        target_stem = meta['target']['stem']
        return conditions['kps_25_coords'][conditions['stem_to_idx'][target_stem]].numpy()

    grid = args.resolution // 16
    records = []
    for index in range(total):
        sample = dataset[index]
        meta = dataset.samples[index]
        stem = f'{index:03d}_{sample["appearance"]}_{sample["source_stem"]}_{sample["target_stem"]}'
        source = dataset_image_to_pil(sample['src_image'])
        source_path = args.sampled_root / meta['appearance'] / meta['source']['rgb_1024']
        target_path = args.sampled_root / meta['appearance'] / meta['target']['rgb_1024']
        batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value] for name, value in sample.items()}
        record = {'stem': stem, 'appearance': meta['appearance'], 'images': {}, 'paths': {}}

        with torch.no_grad():
            geometry = body.encode_geometry(batch, (grid, grid))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad(), Qwen21ControlContext(pipe.transformer, body, geometry):
            body_image = generate(pipe, PROMPT, [source], args)
        record['images']['body'] = body_image
        record['seconds_body'] = time.perf_counter() - started
        body_image.save(args.output / 'images' / f'{stem}_body.png')

        bbox = detect_largest_face(identity, source_path)
        record['face_valid'] = bbox is not None
        crop = face_crop(source_path, bbox) if bbox is not None else None
        if crop is not None:
            crop.save(args.output / 'images' / f'{stem}_facecrop.png')
            started = time.perf_counter()
            no_lora_image = generate(pipe, REPAIR_PROMPT, [body_image, crop], args)
            record['seconds_stage2_nolora'] = time.perf_counter() - started
            record['images']['stage2_nolora'] = no_lora_image
            no_lora_image.save(args.output / 'images' / f'{stem}_stage2_nolora.png')

        record['target_points'] = target_points_for(meta).tolist()
        records.append(record)
        print(f'stage1 {index + 1}/{total} {stem} face_valid={record["face_valid"]}', flush=True)

    head_path = args.bfs_dir / 'bfs_head_v1.1_qwen_2.1.safetensors'
    head_load_notes = {}
    for variant, gate_first in (('bfs_head', True), ('bfs_head_rev', False)):
        head_load_notes[variant] = load_lora(pipe, head_path, 'bfs_head', gate_first=gate_first)
        pipe.set_adapters(['bfs_head'], adapter_weights=[1.0])
        print(f'head LoRA loaded via {head_load_notes[variant]}', flush=True)
        for record in records:
            crop_path = args.output / 'images' / f'{record["stem"]}_facecrop.png'
            if not crop_path.exists():
                continue
            body_image = Image.open(args.output / 'images' / f'{record["stem"]}_body.png')
            crop = Image.open(crop_path)
            started = time.perf_counter()
            head_image = generate(pipe, BFS_HEAD_PROMPT, [body_image, crop], args)
            record[f'seconds_{variant}'] = time.perf_counter() - started
            record['images'][variant] = head_image
            head_image.save(args.output / 'images' / f'{record["stem"]}_{variant}.png')
            print(f'head swap done ({variant}): {record["stem"]}', flush=True)
        pipe.unload_lora_weights()

    body_swap_records = []
    load_method_body = None
    body_swap_path = args.bfs_dir / 'bfs_body_swap_v1.0_qwen_2.1.safetensors'
    if args.body_swap_pairs > 0:
        load_method_body = load_lora(pipe, body_swap_path, 'bfs_body')
        pipe.set_adapters(['bfs_body'], adapter_weights=[1.0])
        print(f'body LoRA loaded via {load_method_body}', flush=True)
        for record in records[:args.body_swap_pairs]:
            source_path = args.sampled_root / record['appearance'] / dataset.samples[int(record['stem'][:3])]['source']['rgb_1024']
            target_path = args.sampled_root / record['appearance'] / dataset.samples[int(record['stem'][:3])]['target']['rgb_1024']
            scene = Image.open(target_path).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
            reference = Image.open(source_path).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
            started = time.perf_counter()
            swapped = generate(pipe, BFS_BODY_PROMPT, [scene, reference], args)
            image_path = args.output / 'images' / f'{record["stem"]}_bfs_body.png'
            swapped.save(image_path)
            body_swap_records.append({
                'stem': record['stem'],
                'image': str(image_path),
                'seconds': time.perf_counter() - started,
                'note': 'image1 = B target frame, image2 = A full frame; identity should follow A, pose/background follow B',
            })
            print(f'body swap done: {record["stem"]}', flush=True)
        pipe.unload_lora_weights()

    print('scoring...', flush=True)
    source_embeddings = {}
    target_embeddings = {}
    for record in records:
        source_path = args.sampled_root / record['appearance'] / dataset.samples[int(record['stem'][:3])]['source']['rgb_1024']
        target_path = args.sampled_root / record['appearance'] / dataset.samples[int(record['stem'][:3])]['target']['rgb_1024']
        source_embeddings[record['stem']] = identity.embedding(Image.open(source_path))
        target_embeddings[record['stem']] = identity.embedding(Image.open(target_path))

    target_points_by_stem = {r['stem']: np.asarray(r['target_points'], dtype=np.float32) for r in records}
    variants = ('body', 'stage2_nolora', 'bfs_head', 'bfs_head_rev')
    for record in records:
        record['metrics'] = {}
        for variant in variants:
            if variant not in record['images']:
                continue
            image = record['images'][variant]
            source_embedding = source_embeddings[record['stem']]
            generated_embedding = identity.embedding(image)
            entry = {
                'identity_cosine_vs_source': identity.cosine(source_embedding, generated_embedding) if source_embedding is not None else None,
                'identity_cosine_vs_target': identity.cosine(target_embeddings[record['stem']], generated_embedding),
            }
            entry.update(pose_metric.compare(pose_metric.detect(image), target_points_by_stem[record['stem']]))
            record['metrics'][variant] = entry
        record.pop('images', None)
        record.pop('target_points', None)

    for entry in body_swap_records:
        image = Image.open(entry['image'])
        stem = entry['stem']
        generated_embedding = identity.embedding(image)
        entry['identity_cosine_vs_source'] = identity.cosine(source_embeddings[stem], generated_embedding)
        entry['identity_cosine_vs_target'] = identity.cosine(target_embeddings[stem], generated_embedding)
        entry['pose_vs_target'] = pose_metric.compare(pose_metric.detect(image), target_points_by_stem[stem])

    summary = {}
    seconds_keys = {'body': 'seconds_body', 'stage2_nolora': 'seconds_stage2_nolora', 'bfs_head': 'seconds_bfs_head', 'bfs_head_rev': 'seconds_bfs_head_rev'}
    for variant in variants:
        identity_scores = [r['metrics'][variant]['identity_cosine_vs_source'] for r in records if variant in r.get('metrics', {}) and r['metrics'][variant]['identity_cosine_vs_source'] is not None]
        pck_scores = [r['metrics'][variant]['pose_pck_002'] for r in records if variant in r.get('metrics', {}) and r['metrics'][variant]['pose_pck_002'] is not None]
        seconds = [r[seconds_keys[variant]] for r in records if seconds_keys[variant] in r]
        summary[variant] = {
            'n_identity': len(identity_scores),
            'identity_mean': float(np.mean(identity_scores)) if identity_scores else None,
            'pck_mean': float(np.mean(pck_scores)) if pck_scores else None,
            'seconds_mean': float(np.mean(seconds)) if seconds else None,
        }
    source_target = [
        identity.cosine(source_embeddings[r['stem']], target_embeddings[r['stem']])
        for r in records
        if source_embeddings[r['stem']] is not None and target_embeddings[r['stem']] is not None
    ]
    summary['source_target_ceiling'] = {
        'n': len(source_target),
        'identity_mean': float(np.mean(source_target)) if source_target else None,
    }

    (args.output / 'results.json').write_text(json.dumps({
        'config': {
            'pairs': str(args.pairs), 'max_pairs': total, 'steps': args.steps, 'seed': args.seed,
            'resolution': args.resolution, 'body_checkpoint': str(args.body_checkpoint),
            'head_lora': 'bfs_head_v1.1_qwen_2.1.safetensors', 'body_lora': 'bfs_body_swap_v1.0_qwen_2.1.safetensors',
            'head_lora_load': head_load_notes, 'body_lora_load': load_method_body,
        },
        'summary': summary,
        'per_pair': records,
        'body_swap': body_swap_records,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
