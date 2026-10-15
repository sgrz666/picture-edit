#!/usr/bin/env python3
"""Fix the face-training appearance split and extract source-only face crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from insightface import model_zoo


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'datasets/iPER/iper_sampled_6src64tgt/splits'
SAMPLED = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'
DEFAULT_OUT = Path('/home/shangguanrz/nihaochen/qwen21_face')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-appearances', type=int, default=10)
    parser.add_argument('--pairs-per-appearance', type=int, default=5)
    parser.add_argument('--min-source-face', type=float, default=64.0, help='Minimum original-image box short side')
    parser.add_argument('--min-score', type=float, default=0.5)
    return parser.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def detect(detector, path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    boxes, _ = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if len(boxes) == 0:
        return rgb, None
    index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    box = boxes[index]
    return rgb, {
        'bbox': [float(v) for v in box[:4]],
        'score': float(box[4]),
        'short_side': float(min(box[2] - box[0], box[3] - box[1])),
        'num_boxes': int(len(boxes)),
    }


def crop_face(rgb, box):
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.8
    left = max(0, min(width - side, center_x - side / 2))
    top = max(0, min(height - side, center_y - side / 2))
    patch = Image.fromarray(rgb).crop((int(left), int(top), int(left + side), int(top + side)))
    return patch.resize((512, 512), Image.Resampling.LANCZOS)


def main():
    args = parse_args()
    train = read_jsonl(SPLITS / 'dev_train_pairs.jsonl')
    val = read_jsonl(SPLITS / 'dev_val_pairs.jsonl')
    appearances = sorted({row['appearance'] for row in train})
    selected = {row['appearance'] for row in val}
    if len(selected) != args.val_appearances:
        raise RuntimeError(f'existing dev_val split has {len(selected)} appearances, expected {args.val_appearances}')
    face_train = train
    candidates = defaultdict(list)
    for row in val:
        if row['appearance'] in selected:
            candidates[row['appearance']].append(row)
    face_val = []
    for appearance in sorted(selected):
        rows = sorted(candidates[appearance], key=lambda item: (item['source']['stem'], item['target']['stem']))
        if len(rows) < args.pairs_per_appearance:
            raise RuntimeError(f'appearance {appearance} has only {len(rows)} validation pairs')
        face_val.extend(random.Random(f'{args.seed}:{appearance}').sample(rows, args.pairs_per_appearance))
    if {row['appearance'] for row in face_train} & selected:
        raise AssertionError('face training and validation appearances overlap')
    if len(face_val) != args.val_appearances * args.pairs_per_appearance:
        raise AssertionError('wrong validation count')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'face_train_pairs.jsonl', face_train)
    write_jsonl(out / 'face_val_pairs.jsonl', face_val)
    detector = model_zoo.get_model(str(ROOT / 'models/antelopev2/scrfd_10g_bnkps.onnx'), providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1, input_size=(640, 640))
    face_by_source = {}
    crop_dir = out / 'source_face_crops'
    crop_dir.mkdir(exist_ok=True)
    inventory = []
    for index, row in enumerate(face_val):
        appearance = row['appearance']
        source = row['source']
        key = (appearance, source['stem'])
        if key not in face_by_source:
            path = SAMPLED / appearance / source['rgb_1024']
            rgb, detection = detect(detector, path)
            valid = bool(detection and detection['score'] >= args.min_score and detection['short_side'] >= args.min_source_face)
            crop_path = None
            if valid:
                crop_path = crop_dir / f'{appearance}_{source["stem"]}.png'
                crop_face(rgb, detection['bbox']).save(crop_path)
            face_by_source[key] = {
                'source_path': str(path), 'detection': detection, 'face_usable': valid,
                'face_crop_path': str(crop_path) if crop_path else None,
            }
        inventory.append({'pair_idx': index, 'appearance': appearance, 'source_stem': source['stem'],
                          'target_stem': row['target']['stem'], **face_by_source[key]})
    write_jsonl(out / 'face_val_inventory.jsonl', inventory)
    cross = []
    all_apps = sorted(selected)
    for index in range(min(8, len(all_apps))):
        source_row = next((row for row in face_val if row['appearance'] == all_apps[index] and
                           face_by_source[(row['appearance'], row['source']['stem'])]['face_usable']), None)
        target_app = next(app for app in all_apps[index + 1:] + all_apps[:index + 1]
                          if app.split('_')[0] != all_apps[index].split('_')[0])
        target_row = next(row for row in face_val if row['appearance'] == target_app)
        if source_row is not None:
            cross.append({'source_appearance': source_row['appearance'], 'source': source_row['source'],
                          'target_appearance': target_row['appearance'], 'target': target_row['target']})
    write_jsonl(out / 'cross_person_8_pairs.jsonl', cross)
    manifest = {
        'seed': args.seed, 'val_appearances': sorted(selected), 'train_appearances': appearances,
        'train_pairs': len(face_train), 'val_pairs': len(face_val), 'cross_person_pairs': len(cross),
        'usable_val_source_faces': sum(row['face_usable'] for row in inventory),
        'min_source_face_original_px': args.min_source_face, 'min_detector_score': args.min_score,
        'source_sha256': {name: hashlib.sha256((SPLITS / name).read_bytes()).hexdigest()
                          for name in ('dev_train_pairs.jsonl', 'dev_val_pairs.jsonl')},
    }
    (out / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
