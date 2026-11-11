"""CPU-only data/geometry and Qwen condition-layout contracts for this experiment."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DWPOSE_TO_COCO = np.asarray([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def letterbox(image, size):
    image = ImageOps.exif_transpose(image).convert('RGB')
    width, height = image.size
    scale = min(size[0] / width, size[1] / height)
    resized = (round(width * scale), round(height * scale))
    offset = ((size[0] - resized[0]) // 2, (size[1] - resized[1]) // 2)
    result = Image.new('RGB', tuple(size), 'black')
    result.paste(image.resize(resized, Image.Resampling.LANCZOS), offset)
    return result, {'original': [width, height], 'resized': list(resized),
                    'offset': list(offset), 'canvas': list(size)}


def transform_points(points, affine):
    return (np.asarray(points, dtype=np.float64) * np.asarray(affine['resized']) /
            np.asarray(affine['original']) + np.asarray(affine['offset']))


def pose_points(record, size):
    if 'valid_keypoint_mask' not in record:
        raise ValueError('DWPose record requires valid_keypoint_mask; raw SimCC scores are not probabilities')
    points = np.asarray(record['candidate'], dtype=np.float64)
    valid = np.asarray(record['valid_keypoint_mask'], dtype=bool)
    if points.shape != (134, 2) or valid.shape != (134,):
        raise ValueError('Expected DWPose 134 points and 134 validity flags')
    if record.get('coordinate_space', 'original_pixel') != 'original_pixel':
        raise ValueError('Expected original_pixel coordinates')
    valid &= np.isfinite(points).all(1) & (points >= 0).all(1)
    valid &= (points[:, 0] < size[0]) & (points[:, 1] < size[1])
    return points, valid


def render_skeleton(points, valid, size):
    # Same 134-point layout and OpenPose limb semantics as src/pose_edit/geometry.py.
    image = Image.new('RGB', tuple(size), 'black')
    draw = ImageDraw.Draw(image)
    palette = [(255, 50, 50), (255, 105, 30), (255, 165, 0), (255, 215, 0),
               (70, 220, 90), (0, 200, 180), (30, 144, 255), (65, 105, 225),
               (138, 43, 226), (220, 20, 160), (255, 70, 130), (255, 120, 90)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
             (2, 5), (8, 11), (0, 14), (14, 15), (0, 16), (16, 17)]
    def segment(a, b, color, width=3):
        if valid[a] and valid[b]:
            draw.line((tuple(points[a]), tuple(points[b])), fill=color, width=width)
    for i, (a, b) in enumerate(edges):
        segment(a, b, palette[i % len(palette)])
    for a, b in [(10, 21), (10, 23), (13, 18), (13, 20), (18, 19), (18, 20), (21, 22), (21, 23)]:
        segment(a, b, (255, 190, 40), 2)
    for start in (92, 113):
        for finger in range(5):
            first = start + 1 + finger * 4
            segment(start, first, palette[finger], 2)
            for joint in range(3):
                segment(first + joint, first + joint + 1, palette[finger], 2)
    for i in np.flatnonzero(valid):
        x, y = points[i]
        radius = 1 if 24 <= i < 92 else 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(240, 240, 240))
    return image


def audit_splits(splits):
    seen_ids, seen_groups, seen_frames = {}, {}, {}
    summary = {}
    for split, rows in splits.items():
        for row in rows:
            pair_id = row['pair_id']
            group = row.get('group_id', row.get('clip_id'))
            if pair_id in seen_ids:
                raise ValueError(f'duplicate/cross-split pair_id: {pair_id}')
            seen_ids[pair_id] = split
            for value, seen in [(group, seen_groups),
                                (row.get('source_frame_id'), seen_frames),
                                (row.get('target_frame_id'), seen_frames)]:
                if value is None:
                    continue
                if value in seen and seen[value] != split:
                    raise ValueError(f'cross-split video/frame: {value}')
                seen[value] = split
        summary[split] = {'pairs': len(rows), 'videos': len({r.get('group_id', r.get('clip_id')) for r in rows})}
    return summary


def select_pilot(rows, videos, targets_per_source, seed):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[row['group_id']][row['source_frame_id']][row['target_frame_id']] = row
    eligible = {g: {a: list(targets.values()) for a, targets in sources.items()
                    if len(targets) >= targets_per_source} for g, sources in grouped.items()}
    eligible = {g: sources for g, sources in eligible.items() if sources}
    if len(eligible) < videos:
        raise ValueError(f'Need {videos} videos with >= {targets_per_source} targets per A; found {len(eligible)}')
    rng = random.Random(seed)
    selected = []
    for group in rng.sample(sorted(eligible), videos):
        source = rng.choice(sorted(eligible[group]))
        candidates = sorted(eligible[group][source], key=lambda r: r['pair_id'])
        selected.extend(rng.sample(candidates, targets_per_source))
    return selected


def cache_key(row, identity):
    return digest({'conditions': row['condition_sha256'], 'target': row['target_sha256'],
                   'prompt': row['prompt'], 'identity': identity})


def verify_row_assets(row):
    if len(row['condition_images']) != 2 or len(row['condition_sha256']) != 2:
        raise ValueError('This experiment requires exactly two ordered condition images')
    for path, expected in zip([*row['condition_images'], row['target_image']],
                              [*row['condition_sha256'], row['target_sha256']]):
        if sha256(path) != expected:
            raise ValueError(f'Prepared image changed: {path}')


def joint_layout(condition_hw, target_hw, image_pad_slots):
    all_hw = [*condition_hw, target_hw]
    if not condition_hw or any(min(h, w) <= 0 or h % 2 or w % 2 for h, w in all_hw):
        raise ValueError('Latent dimensions must be positive and divisible by two')
    condition_tokens = sum(h * w for h, w in condition_hw)
    if int(image_pad_slots) != condition_tokens // 4:
        raise ValueError(f'VLM condition slots {image_pad_slots} != VAE slots {condition_tokens // 4}')
    target_tokens = target_hw[0] * target_hw[1]
    return {'img_shapes': [(1, h, w) for h, w in all_hw], 'condition_tokens': condition_tokens,
            'target_tokens': target_tokens, 'target_slots': target_tokens // 4}


def pose_stats(detected, reference, valid, size):
    reference = np.asarray(reference)[DWPOSE_TO_COCO]
    target_valid = np.asarray(valid, dtype=bool)[DWPOSE_TO_COCO]
    total = int(target_valid.sum())
    if detected is None:
        distances = np.asarray([])
        matched = 0
    else:
        xy, scores = detected
        good = target_valid & (np.asarray(scores) > 0.3) & np.isfinite(xy).all(1)
        distances = np.linalg.norm(np.asarray(xy)[good] - reference[good], axis=1) / float(np.hypot(*size))
        matched = int(good.sum())
    return {'pose_detected': detected is not None, 'target_joints': total, 'matched_joints': matched,
            'joint_recall': matched / total if total else None,
            'nme_diagonal': float(distances.mean()) if len(distances) else None,
            **{f'pck_{round(t*100):03d}_diagonal': float((distances < t).sum() / total) if total else None
               for t in (0.02, 0.05, 0.1)}}
