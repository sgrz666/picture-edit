"""Finger-structure plausibility for the pose-transfer workflows, from keypoints only.

The contract (task/人物图像姿态编辑任务书_v3.docx) requires "手部无明显结构性错误" and scores hand
detail against a Qwen-Edit baseline, but every metric in this repo stops at hand *position*
(group MPJPE over the 42 hand slots). A hand can land exactly on the requested skeleton and still
be drawn as a mitten. This module answers the shape question, and needs no extra inference: it
reads the DWPose-134 arrays `evaluate.py` already writes.

Chain order is the COCO-WholeBody / OpenPose 21-slot hand: 0 wrist, thumb 1-4, index 5-8,
middle 9-12, ring 13-16, pinky 17-20 — the same split `src/pose_edit/geometry.py:111-112` uses
(`left_hand = [12, *range(25,45)]`, slots 92 and 113 are the two wrists).

What is **not** checked, and why (measured on 26 hands from this repo's own runs, 2026-10-14):
- radial monotonicity along a finger. A curled finger legitimately puts the tip nearer the wrist
  than the DIP; real guide arrays show exactly that (`right middle: [1.07, 1.39, 1.41, 1.38]`).
- per-segment length bands. Inter-segment ratios span 0.05–0.69 at p1–p99 on *valid* hands,
  because curvature and perspective foreshortening dominate at this hand size.
That same measurement is the real finding: **median palm length is 39.7 px at resolution 768**
(p5 18.1, p95 62.0). Detector noise on a 40 px palm is several pixels, which is a large fraction
of one phalanx, so finger-level judgement needs the hand enlarged first. Hence `MIN_PALM_PX` and
`hand_too_small_to_judge` below: excluding a hand we cannot measure is honest; scoring it is not.

The remaining bands are screening, not anatomy: they catch gross deformations
(fingertips fused into a mitten, a finger stretched past any real hand) and mark the image
`inspect`. Calibrate on hands with `palm_px >= MIN_PALM_PX` before treating them as a gate.
"""
from __future__ import annotations

import json
from pathlib import Path

HAND_SLOT_COUNT = 21
LEFT_HAND_START = 92
RIGHT_HAND_START = 113
DWSLOT_COUNT = 134

FINGERS = {'thumb': (1, 2, 3, 4), 'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12),
    'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}
NON_THUMB = ('index', 'middle', 'ring', 'pinky')

MIN_PALM_PX = 60.0            # below this the detector cannot resolve a phalanx
MIN_VALID_FRACTION = 0.5
FINGER_LENGTH = (0.45, 3.20)  # wrist->tip over palm; measured p5..p95 = 0.93..2.55
COLLAPSED_TIP_DISTANCE = 0.22  # fingertips closer than this share of palm = fused


def _distance(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def hand_points(points134, side):
    start = LEFT_HAND_START if side == 'left' else RIGHT_HAND_START
    return points134[start:start + HAND_SLOT_COUNT]


def palm_length(hand):
    """Wrist to the mean of the four non-thumb MCP joints; None when the palm is degenerate."""
    wrist = hand[0]
    heads = [hand[FINGERS[name][0]] for name in NON_THUMB]
    centre = (sum(point[0] for point in heads) / len(heads),
        sum(point[1] for point in heads) / len(heads))
    length = _distance(wrist, centre)
    return length if length > 1e-6 else None


def analyze_hand(hand, valid):
    """Per-finger measurements plus the flags worth looking at. Never raises on bad input."""
    result = {'fingers': {}, 'flags': [], 'measurable': True}
    valid = [bool(value) for value in valid]
    if sum(valid) < MIN_VALID_FRACTION * HAND_SLOT_COUNT:
        result['measurable'] = False
        result['flags'].append('hand_not_measurable')
        result['valid_ratio'] = sum(valid) / HAND_SLOT_COUNT
        return result
    palm = palm_length(hand)
    if not palm:
        result['measurable'] = False
        result['flags'].append('palm_degenerate')
        return result
    result['palm_px'] = round(palm, 2)
    if palm < MIN_PALM_PX:
        result['measurable'] = False
        result['flags'].append('hand_too_small_to_judge')
        return result
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        finger = {'length_ratio': round(_distance(hand[0], hand[tip]) / palm, 3)}
        if not all(valid[index] for index in (mcp, pip, dip, tip)):
            finger['skipped'] = 'joint_invalid'
            result['fingers'][name] = finger
            continue
        finger['segment_px'] = [round(_distance(hand[a], hand[b]), 1)
            for a, b in ((mcp, pip), (pip, dip), (dip, tip))]
        if not (FINGER_LENGTH[0] <= finger['length_ratio'] <= FINGER_LENGTH[1]):
            finger['flags'] = ['finger_length_out_of_range']
        else:
            finger['flags'] = []
        result['flags'].extend(f'{name}:{flag}' for flag in finger['flags'])
        result['fingers'][name] = finger
    scored = [name for name in FINGERS if 'segment_px' in result['fingers'][name]]
    for i, name_a in enumerate(scored):
        for name_b in scored[i + 1:]:
            if 'thumb' in (name_a, name_b):
                continue  # the thumb opposes the others; tip distance means nothing here
            index_a, index_b = FINGERS[name_a][3], FINGERS[name_b][3]
            if _distance(hand[index_a], hand[index_b]) / palm < COLLAPSED_TIP_DISTANCE:
                result['flags'].append(f'tips_collapsed:{name_a}+{name_b}')
    return result


def analyze(points134, valid134):
    points134 = [[float(x), float(y)] for x, y in points134]
    valid134 = [bool(value) for value in valid134]
    hands = {}
    for side in ('left', 'right'):
        start = LEFT_HAND_START if side == 'left' else RIGHT_HAND_START
        hands[side] = analyze_hand(hand_points(points134, side), valid134[start:start + 21])
    return {'hands': hands,
        'flag_count': sum(len(hands[side]['flags']) for side in hands),
        'measurable_hands': sum(1 for side in hands if hands[side]['measurable'])}


def compare(guide, generated):
    """Charge the generator only for what the requested skeleton did not already have."""
    new_flags = []
    for side in ('left', 'right'):
        if not generated['hands'][side]['measurable']:
            continue
        known = set(guide['hands'][side]['flags'])
        new_flags.extend(f'{side}:{flag}' for flag in generated['hands'][side]['flags']
            if flag not in known)
    measurable = any(guide['hands'][side]['measurable'] for side in ('left', 'right'))
    return {'guide': guide, 'generated': generated, 'new_in_generated': sorted(new_flags),
        'verdict': 'inputs_not_measurable' if not measurable
            else 'clean' if not new_flags else 'inspect'}


def distribution(records):
    """Percentiles of what the arrays actually contain, for calibrating the bands above."""
    import numpy as np
    palms = [hand['palm_px'] for record in records for hand in record['summary']['guide']['hands'].values()
        if hand.get('palm_px')]
    palms += [hand['palm_px'] for record in records
        for hand in record['summary']['generated']['hands'].values() if hand.get('palm_px')]
    if not palms:
        return {'hands': 0}
    small = sum(1 for value in palms if value < MIN_PALM_PX)
    return {'hands': len(palms),
        'palm_px_p5_p50_p95': [round(float(value), 1)
            for value in np.percentile(palms, [5, 50, 95])],
        'below_min_palm': small, 'below_min_palm_fraction': round(small / len(palms), 3)}


def run_on_run(run_dir, write=True):
    """Score one run from the evaluation arrays it already has. None when it has none."""
    npz = Path(run_dir) / 'evaluation' / 'final_keypoints.npz'
    if not npz.is_file():
        return None
    import numpy as np
    data = np.load(npz)
    summary = compare(analyze(data['guide_final_px'], data['guide_valid']),
        analyze(data['dwpose134_px'], data['valid']))
    record = {'run': Path(run_dir).name, 'source_npz': str(npz), 'summary': summary}
    if write:
        out = npz.parent / 'hand_plausibility.json'
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        record['written'] = str(out)
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=[], help='run directory (repeatable)')
    parser.add_argument('--all', action='store_true', help='every run with final_keypoints.npz')
    parser.add_argument('--workflow', type=Path,
        default=Path(__file__).resolve().parents[2] / 'workflow')
    parser.add_argument('--dry-run', action='store_true', help='print without writing json')
    parser.add_argument('--distribution', action='store_true',
        help='report palm-size percentiles instead of verdicts (calibration input)')
    args = parser.parse_args(argv)
    targets = [Path(item) for item in args.run]
    if args.all:
        targets = sorted(path.parent.parent for path in
            (args.workflow / 'runs').rglob('final_keypoints.npz'))
    records = []
    for run_dir in targets:
        record = run_on_run(run_dir, write=not args.dry_run)
        if record:
            records.append(record)
        else:
            print(f'SKIP {run_dir} (no evaluation/final_keypoints.npz)')
    if args.distribution:
        print(json.dumps(distribution(records), ensure_ascii=False, indent=2))
        return 0
    for record in records:
        summary = record['summary']
        print(f"{record['run']:34} {summary['verdict']:21} new={len(summary['new_in_generated']):2d} "
            f"guide_flags={summary['guide']['flag_count']:2d} "
            f"generated_flags={summary['generated']['flag_count']:2d}")
        for flag in summary['new_in_generated']:
            print(f'    {flag}')
    print(f'scored={len(records)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
