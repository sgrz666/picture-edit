#!/usr/bin/env python3
"""Measure torso-to-person-box height on the two-person set: is 0.31 a two-person constant too?

``TORSO_TO_PERSON_BOX_HEIGHT`` in ``src/pose_edit/geometry.py`` says a body's neck-to-hip-centre
torso is about 0.31 of the height of the box that encloses that person. The value was reverse-
engineered from five single-person pairs, and the single-person line uses it to decide how big a
transferred pose should be, so a wrong constant quietly mis-sizes every result it touches.

This script measures the same ratio on the Hi4D two-person set: 17 pairs, both people, both the
source frame (A) and the target frame (B) - up to 68 independent measurements. A's boxes are the
Hi4D person masks and A's landmarks come from the same DWPose ONNX the evaluation uses, run on
CPU; B's landmarks are the pair's own keypoint files and B's boxes come from the semantic masks in
the frame's geometry record.

Known difference from the single-person line, recorded in the output rather than hidden: there the
box is a detector box, here it is a silhouette mask. A mask hugs the body, so a mask-based ratio is
an upper bound on the detector-based one: if this measurement already comes out below 0.31, the
constant is high for bodies like these, and only a frame-clipped or padded detector box in the
single-person data could have produced 0.31 legitimately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_dual import detect_people, hip_centre, torso_length
from src.pose_edit.common import atomic_json, read_json, sha256_file

CONSTANT = 0.31


def mask_box(mask) -> list[float] | None:
    """xyxy box around the nonzero part of a person's semantic/mask image."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def ratio(points, box) -> dict:
    """One measurement: torso over box height, with the parts it was made of."""
    torso = torso_length(points) if points is not None else None
    height = (box[3] - box[1]) if box else None
    return {'torso_px': torso, 'box_height_px': height,
        'ratio': (torso / height) if torso and height else None}


def assign_to_box(people, box) -> int | None:
    """Pick the detected person whose hip centre falls inside this mask box."""
    hits = []
    for index, person in enumerate(people):
        centre = hip_centre(person['points'])
        if centre is None:
            continue
        if box[0] <= centre[0] <= box[2] and box[1] <= centre[1] <= box[3]:
            hits.append((index, float(np.linalg.norm(
                centre - np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])))))
    if not hits:
        return None
    return min(hits, key=lambda item: item[1])[0]


def source_measurements(pair_dir, pair, detected) -> list[dict]:
    entries = []
    for person in pair['A']['persons']:
        box = person['region'].get('bbox_xyxy')
        index = assign_to_box(detected, box) if box else None
        entry = ratio(detected[index]['points'] if index is not None else None, box)
        entry.update(side=person['hi4d_side'], frame='A',
            detector_slot=None if index is None else int(index))
        entries.append(entry)
    return entries


def target_measurements(pair_dir, pair) -> list[dict]:
    """B's landmarks are already the pair's ground truth; its box is the semantic mask."""
    geometry = np.load(pair['B']['geometry_original'], allow_pickle=True)
    entries = []
    for item in pair['B']['keypoints']:
        arrays = np.load(pair_dir / item['file'], allow_pickle=True)
        points = np.asarray(arrays['xy_native'], dtype=float)
        scores = np.asarray(arrays['scores'], dtype=float)
        keep = np.isfinite(points).all(axis=1) & (scores >= 0.5)
        visible = np.where(keep[:, None], points, np.nan)
        side = str(item['hi4d_side']).lower()
        key = f'semantic_{side}'
        if key not in geometry.files:
            key = 'semantic_A' if side == 'a' else 'semantic_B'
        entries.append(ratio(visible, mask_box(geometry[key] > 0))
            | {'side': item['hi4d_side'], 'frame': 'B'})
    return entries


def summarise(measurements) -> dict:
    values = [item['ratio'] for item in measurements if item['ratio']]
    by_frame = {frame: [item['ratio'] for item in measurements
        if item['frame'] == frame and item['ratio']] for frame in ('A', 'B')}
    return {'measured': len(values),
        'median': float(np.median(values)) if values else None,
        'mean': float(np.mean(values)) if values else None,
        'p25': float(np.percentile(values, 25)) if values else None,
        'p75': float(np.percentile(values, 75)) if values else None,
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'by_frame': {frame: {'n': len(items), 'median': float(np.median(items)) if items else None}
            for frame, items in by_frame.items()},
        'constant': CONSTANT,
        'ratio_to_constant': (float(np.median(values)) / CONSTANT) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--pairs', type=Path, nargs='*',
        default=[ROOT / 'pairs', ROOT / 'pairs_cross'])
    parser.add_argument('--out', type=Path, default=ROOT / 'ratios')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    directories = []
    for base in args.pairs:
        base = Path(base)
        if base.is_dir():
            directories += [item for item in sorted(base.iterdir())
                if (item / 'pair.json').is_file()]
    if args.limit:
        directories = directories[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    per_pair, measurements = [], []
    action_cache = {}
    for directory in directories:
        pair = read_json(directory / 'pair.json')
        with Image.open(directory / pair['A']['image']) as handle:
            rgb = np.asarray(handle.convert('RGB'))
        key = sha256_file(directory / pair['A']['image'])
        if key not in action_cache:
            # The same A frame can back several pairs; DWPose on CPU is the slow part here.
            action_cache[key] = detect_people(config, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        detected = action_cache[key]
        entries = source_measurements(directory, pair, detected) \
            + target_measurements(directory, pair)
        for entry in entries:
            entry['pair'] = pair['name']
            entry['action'] = f"{pair['manifest']['source_action']}->{pair['manifest']['target_action']}"
        per_pair.append({'pair': pair['name'], 'entries': entries})
        measurements += entries
        print(f"MEASURED {pair['name']} "
            + ' '.join(f"{item['frame']}{item['side']}="
                + (f"{item['ratio']:.3f}" if item['ratio'] else 'n/a')
                for item in entries), flush=True)
    summary = summarise(measurements)
    by_action = defaultdict(list)
    for item in measurements:
        if item['ratio']:
            by_action[item['action']].append(item['ratio'])
    summary['by_action'] = {action: {'n': len(values), 'median': float(np.median(values))}
        for action, values in sorted(by_action.items())}
    atomic_json(args.out / 'torso_ratio.json', {'summary': summary, 'pairs': per_pair})
    print('\nTORSO_OVER_BOX_HEIGHT ' + ' '.join(
        f'{key}={value:.3f}' if isinstance(value, float) else f'{key}={value}'
        for key, value in summary.items() if isinstance(value, (float, int)) or value is None))
    print('by_frame ' + str(summary['by_frame']))
    print('by_action ' + str(summary['by_action']))
    print('the box here is the Hi4D silhouette mask while the single-person line uses a detector '
        f'box; a mask hugs the body, so this ratio is an upper bound on the detector-based one - '
        f'if it sits below {CONSTANT}, the constant is high for bodies like these')


if __name__ == '__main__':
    main()
