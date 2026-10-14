#!/usr/bin/env python3
"""Cache source-only face identity/texture and target geometry for Qwen21 Face training."""

from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from insightface import model_zoo
from insightface.utils import face_align

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.precompute_iper_face_features_v65 import (
    compute_face_box, project_smplx_face72_to_rgb_normalized,
)
from scripts.precompute_tiktok_face_features_v65 import (
    project_tiktok_face72_to_rgb_normalized,
)


FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')
SAMPLED = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'
TIKTOK_RAW_ROOT = ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset'
TIKTOK_ASSETS_ROOT = ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native'


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', type=Path, default=FACE_ROOT / 'face_train_pairs.jsonl')
    parser.add_argument('--val-pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--extra-pairs', nargs='*', type=Path, default=[])
    parser.add_argument('--dataset-type', choices=('auto', 'iper', 'tiktok_v3'), default='auto')
    parser.add_argument('--sampled-root', type=Path, default=SAMPLED)
    parser.add_argument('--tiktok-raw-root', type=Path, default=TIKTOK_RAW_ROOT)
    parser.add_argument('--tiktok-assets-root', type=Path, default=TIKTOK_ASSETS_ROOT)
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--min-source-side', type=float, default=None)
    parser.add_argument('--min-target-side', type=float, default=None)
    parser.add_argument('--min-score', type=float, default=0.5)
    parser.add_argument('--limit-apps', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='cpu')
    parser.add_argument('--reverse', action='store_true')
    return parser.parse_args()


def detect(detector, rgb):
    boxes, landmarks = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if len(boxes) == 0:
        return None
    index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    box = boxes[index]
    return {'bbox': [float(v) for v in box[:4]], 'score': float(box[4]),
            'side': float(min(box[2] - box[0], box[3] - box[1])),
            'landmark': landmarks[index]}


def crop(rgb, bbox):
    x0, y0, x1, y1 = bbox
    height, width = rgb.shape[:2]
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.8
    left = max(0, min(width - side, center_x - side / 2))
    top = max(0, min(height - side, center_y - side / 2))
    patch = Image.fromarray(rgb).crop((int(left), int(top), int(left + side), int(top + side)))
    return torch.from_numpy(np.asarray(patch.resize((128, 128), Image.Resampling.LANCZOS)).copy()).permute(2, 0, 1)


def read_pairs(path: Path):
    if not path or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _resolve_tiktok_image(manifest_path: str | None, seq_dir: Path, stem: str) -> Path:
    if manifest_path:
        p = Path(manifest_path)
        if p.is_file():
            return p
    for ext in ('.png', '.jpg'):
        candidate = seq_dir / 'images' / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    return Path(manifest_path) if manifest_path else (seq_dir / 'images' / f'{stem}.png')


def main():
    import os
    args = options()
    by_app = defaultdict(lambda: {'is_tiktok': False, 'source': {}, 'target': {}})
    pair_files = [p for p in [args.train_pairs, args.val_pairs, *args.extra_pairs] if p and p.exists()]
    for pair in (pair for path in pair_files for pair in read_pairs(path)):
        is_tiktok = (
            args.dataset_type == 'tiktok_v3'
            or (args.dataset_type == 'auto' and ('sequence_id' in pair or 'source_rgb' in pair))
        )
        if is_tiktok:
            app = str(pair.get('sequence_id') or pair['appearance'])
            src_stem = str(pair['source_stem'] if 'source_stem' in pair else pair['source']['stem'])
            tgt_stem = str(pair['target_stem'] if 'target_stem' in pair else pair['target']['stem'])
            raw_seq = args.tiktok_raw_root / app
            src_path = _resolve_tiktok_image(pair.get('source_rgb'), raw_seq, src_stem)
            tgt_path = _resolve_tiktok_image(pair.get('target_rgb'), raw_seq, tgt_stem)
            by_app[app]['is_tiktok'] = True
            by_app[app]['source'][src_stem] = {'stem': src_stem, 'rgb_path': src_path}
            by_app[app]['target'][tgt_stem] = {'stem': tgt_stem, 'rgb_path': tgt_path}
        else:
            app = pair['appearance']
            by_app[app]['is_tiktok'] = False
            by_app[app]['source'][pair['source']['stem']] = pair['source']
            by_app[app]['target'][pair['target']['stem']] = pair['target']
    apps = sorted(by_app, reverse=args.reverse)
    if args.limit_apps:
        apps = apps[:args.limit_apps]
    args.output.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    use_cuda = args.device in ('auto', 'cuda') and 'CUDAExecutionProvider' in ort.get_available_providers()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    ctx_id = 0 if use_cuda else -1
    detector = model_zoo.get_model(str(ROOT / 'models/antelopev2/scrfd_10g_bnkps.onnx'), providers=providers)
    detector.prepare(ctx_id=ctx_id, input_size=(640, 640))
    recognizer = model_zoo.get_model(str(ROOT / 'models/antelopev2/glintr100.onnx'), providers=providers)
    recognizer.prepare(ctx_id=ctx_id)
    stats = {'apps': 0, 'sources': 0, 'source_valid': 0, 'targets': 0, 'target_visible': 0}
    for number, app in enumerate(apps, 1):
        destination = args.output / f'{app}.pt'
        if destination.exists():
            cached = torch.load(destination, map_location='cpu', weights_only=False)
            stats['apps'] += 1
            stats['sources'] += len(cached.get('source', {}))
            stats['source_valid'] += sum(int(v.get('valid', False)) for v in cached.get('source', {}).values())
            stats['targets'] += len(cached.get('target_visibility', {}))
            stats['target_visible'] += sum(int(v.get('visible', False)) for v in cached.get('target_visibility', {}).values())
            if number % 50 == 0 or number == len(apps):
                print(f'{number}/{len(apps)} {app} cached', flush=True)
            continue
        is_tiktok = by_app[app]['is_tiktok']
        min_source_side = args.min_source_side if args.min_source_side is not None else (36.0 if is_tiktok else 64.0)
        min_target_side = args.min_target_side if args.min_target_side is not None else (28.0 if is_tiktok else 48.0)
        if is_tiktok:
            assets_seq = args.tiktok_assets_root / app
            raw_seq = args.tiktok_raw_root / app
            meta = json.loads((assets_seq / 'smplestx/frame_map.json').read_text(encoding='utf-8'))['frames']
            stems = [str(f.get('stem', f"{f['index']+1:04d}")) for f in meta]
            params = torch.load(assets_seq / 'smplestx/params.pt', map_location='cpu', weights_only=False)
            projected = project_tiktok_face72_to_rgb_normalized(params, raw_seq_dir=raw_seq, stems=stems)
            expansion = 0.25
            original_resolution = [604, 1080]
        else:
            root = args.sampled_root / app
            meta = json.loads((root / 'smplestx/frame_map.json').read_text(encoding='utf-8'))['frames']
            params = torch.load(root / 'smplestx/params.pt', map_location='cpu', weights_only=False)
            projected = project_smplx_face72_to_rgb_normalized(params)
            expansion = 0.25
            original_resolution = 1024
        if len(meta) != len(projected):
            raise RuntimeError(f'{app}: SMPL-X/frame map count mismatch')
        target_boxes = {}
        for index, frame in enumerate(meta):
            stem = str(frame.get('stem', f"{frame['index']+1:04d}"))
            box, valid = compute_face_box(projected[index], expansion=expansion)
            target_boxes[stem] = {'box': box.tolist(), 'geometry_valid': bool(valid)}
        source_features = {}
        for stem, entry in by_app[app]['source'].items():
            path = Path(entry['rgb_path']) if is_tiktok else (args.sampled_root / app / entry['rgb_1024'])
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            found = detect(detector, rgb)
            valid = bool(found and found['score'] >= args.min_score and found['side'] >= min_source_side)
            if valid:
                aligned = face_align.norm_crop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), landmark=found['landmark'])
                feat = recognizer.get_feat(aligned).reshape(-1).astype(np.float32)
                feat /= max(float(np.linalg.norm(feat)), 1e-12)
                face_rgb = crop(rgb, found['bbox'])
            else:
                feat = np.zeros(512, dtype=np.float32)
                face_rgb = torch.zeros(3, 128, 128, dtype=torch.uint8)
            source_features[stem] = {'valid': valid, 'embedding': torch.from_numpy(feat).half(),
                                     'crop': face_rgb, 'detection': {key: value for key, value in found.items() if key != 'landmark'} if found else None}
            stats['sources'] += 1
            stats['source_valid'] += int(valid)
        target_visibility = {}
        for stem, entry in by_app[app]['target'].items():
            path = Path(entry['rgb_path']) if is_tiktok else (args.sampled_root / app / entry['rgb_1024'])
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            found = detect(detector, rgb)
            valid = bool(found and found['score'] >= args.min_score and found['side'] >= min_target_side)
            target_visibility[stem] = {'visible': valid,
                'detection': {key: value for key, value in found.items() if key != 'landmark'} if found else None}
            stats['targets'] += 1
            stats['target_visible'] += int(valid)
        payload = {'appearance': app, 'source': source_features, 'target_geometry': target_boxes,
                   'target_visibility': target_visibility,
                   'schema': 1, 'original_resolution': original_resolution,
                   'dataset': 'tiktok_v3' if is_tiktok else 'iper',
                   'thresholds': {'source_side': min_source_side, 'target_side': min_target_side,
                                  'score': args.min_score}}
        temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')
        torch.save(payload, temporary)
        temporary.replace(destination)
        stats['apps'] += 1
        print(f'{number}/{len(apps)} {app} sources={len(source_features)} targets={len(target_visibility)}', flush=True)
    (args.output / 'precompute_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
