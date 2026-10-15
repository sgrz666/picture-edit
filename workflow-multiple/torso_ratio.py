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
from src.pose_edit.appearance import mask_box
from src.pose_edit.common import atomic_json, read_json, sha256_file

CONSTANT = 0.31


def ratio(points, box) -> dict:
    """One measurement: torso over box height, with the parts it was made of."""
    torso = torso_length(points) if points is not None else None
    height = (box[3] - box[1]) if box else None
    return {'torso_px': torso, 'box_height_px': height,
        'box_xyxy': None if box is None else [float(value) for value in box],
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


def source_measurements(pair_dir, pair, detected, image_sha) -> list[dict]:
    entries = []
    for person in pair['A']['persons']:
        box = person['region'].get('bbox_xyxy')
        index = assign_to_box(detected, box) if box else None
        entry = ratio(detected[index]['points'] if index is not None else None, box)
        entry.update(side=person['hi4d_side'], frame='A', input_sha256=image_sha,
            detector_slot=None if index is None else int(index))
        entries.append(entry)
    return entries


def target_measurements(pair_dir, pair) -> list[dict]:
    """B's landmarks are already the pair's ground truth; its box is the semantic mask."""
    geometry = np.load(pair['B']['geometry_original'], allow_pickle=True)
    image_sha = sha256_file(pair_dir / pair['B']['image'])
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
            | {'side': item['hi4d_side'], 'frame': 'B', 'input_sha256': image_sha})
    return entries


def deduplicate(measurements) -> list[dict]:
    """One measurement per (image, frame, box): re-run directories and repeated source frames
    would otherwise count the same body several times and make the spread look tighter."""
    seen = {}
    for item in measurements:
        key = (item.get('input_sha256'), item.get('frame'),
            tuple(item.get('box_xyxy') or ()))
        seen.setdefault(key, item)
    return list(seen.values())


def summarise(measurements) -> dict:
    unique = deduplicate(measurements)
    values = [item['ratio'] for item in unique if item['ratio']]
    frames = sorted({item['frame'] for item in unique})
    by_frame = {frame: [item['ratio'] for item in unique
        if item['frame'] == frame and item['ratio']] for frame in frames}
    return {'measured': len(values), 'samples_raw': len(measurements),
        'unique_inputs': len(unique),
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


def single_person_measurements(config, runs_root, cache) -> list[dict]:
    """Measure the same ratio on the single-person line, with that line's own box.

    This is the check that has to happen before ``TORSO_TO_PERSON_BOX_HEIGHT`` is touched: the
    constant came from these pairs, so it is measured here against the box the pipeline actually
    used (``B_camera.json`` detections.A, in A's native pixels) rather than a mask.
    """
    entries = []
    for run in sorted(Path(runs_root).iterdir()):
        camera_path = run / 'geometry' / 'B_camera.json'
        image_path = run / 'inputs' / 'A_rgb.png'
        if not (camera_path.is_file() and image_path.is_file()):
            continue
        box = read_json(camera_path).get('detections', {}).get('A', {}).get('bbox_xyxy')
        if not box:
            continue
        key = sha256_file(image_path)
        if key not in cache:
            with Image.open(image_path) as handle:
                rgb = np.asarray(handle.convert('RGB'))
            cache[key] = detect_people(config, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        detected = cache[key]
        index = assign_to_box(detected, [float(value) for value in box])
        entry = ratio(detected[index]['points'] if index is not None else None,
            [float(value) for value in box])
        entry.update(pair=run.name, frame='A-single', side='A', input_sha256=key,
            detector_slot=None if index is None else int(index))
        entries.append(entry)
    return entries


def sizing_error(measurements) -> dict:
    """What the constant does to a body, in the units the framing uses.

    ``source_torso`` transfers a pose at ``CONSTANT * A_box_height`` and calls that the source
    person's torso. The measured torso is ``ratio * A_box_height``, so the body comes out
    ``CONSTANT / ratio`` of its true size - a per-pair multiplier, not an average error. A median
    near 1.0 would mean the constant is fine and only noisy; a median away from 1.0 means every
    pose drawn through that framing is systematically mis-sized.
    """
    factors = [CONSTANT / item['ratio'] for item in deduplicate(measurements) if item['ratio']]
    if not factors:
        return {'pairs': 0}
    return {'pairs': len(factors), 'median_multiplier': float(np.median(factors)),
        'p25': float(np.percentile(factors, 25)), 'p75': float(np.percentile(factors, 75)),
        'min': float(min(factors)), 'max': float(max(factors)),
        'within_10_percent': sum(1 for value in factors if 0.9 <= value <= 1.1) / len(factors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--pairs', type=Path, nargs='*',
        default=[ROOT / 'pairs', ROOT / 'pairs_cross'])
    parser.add_argument('--out', type=Path, default=ROOT / 'ratios')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--single-runs', type=Path, default=REPO_ROOT / 'workflow' / 'runs',
        help='also measure the single-person line, against the box that line itself recorded')
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
        entries = source_measurements(directory, pair, detected, key) \
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
    single = [] if not Path(args.single_runs).is_dir() \
        else single_person_measurements(config, args.single_runs, action_cache)
    for entry in single:
        print(f"SINGLE_MEASURED {entry['pair']} "
            + (f"{entry['ratio']:.3f}" if entry['ratio'] else f"n/a ({entry.get('reason', '')})"),
            flush=True)
    summary = summarise(measurements)
    by_action = defaultdict(list)
    for item in measurements + single:
        if item['ratio']:
            by_action[item.get('action', f"single:{item['pair']}")].append(item['ratio'])
    summary['by_action'] = {action: {'n': len(values), 'median': float(np.median(values))}
        for action, values in sorted(by_action.items())}
    summary['single_person_line'] = summarise(single)
    summary['both_lines_combined'] = summarise(measurements + single)
    summary['sizing_multiplier'] = {'single_person_line': sizing_error(single),
        'two_person_set': sizing_error(measurements)}
    for line, block in summary['sizing_multiplier'].items():
        if block.get('pairs'):
            print(f'SIZING {line}: pairs={block["pairs"]} '
                f'median={block["median_multiplier"]:.2f} '
                f'IQR={block["p25"]:.2f}-{block["p75"]:.2f} '
                f'range={block["min"]:.2f}-{block["max"]:.2f} '
                f'within_10pct={block["within_10_percent"]:.2f}')
    atomic_json(args.out / 'torso_ratio.json',
        {'summary': summary, 'pairs': per_pair, 'single_runs': single})
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
