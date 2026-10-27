#!/usr/bin/env python3
"""Build the two-person test set out of the Hi4D pilot assets.

Every pair directory is self-contained: the two photographs, the two per-person DWPose-134
keypoint files copied next to them, and a ``pair.json`` that names every ground-truth asset
the evaluator may use plus the person ordering rules. Selection is deterministic, so
re-running the same command reproduces the same pairs and the same hashes.

Two flavours exist on purpose:

* ``same_identity`` - the Hi4D pairing itself: A and B show the same two actors from the same
  camera, so per-person ground truth is trustworthy and identity is trivially preserved.
* ``cross_pair`` - A taken from another pair's source frame: different actors and clothing,
  which is the only way to see appearance leakage, but there is no reliable correspondence
  between A's people and B's people, so evaluation falls back to best-of-two assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image

BODY_HIP_SLOTS = (8, 11)
MIN_VALID_BODY_POINTS = 12
ACTION_NUMBER = re.compile(r'\d+$')
# Two assignment costs closer than this fraction of image width are not a signal, they are noise.
PAIRING_AMBIGUITY = 0.01


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_rows(manifest):
    rows = []
    with Path(manifest).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if line.strip():
                row = json.loads(line)
                row['_row_index'] = index
                rows.append(row)
    return rows


def keypoints(path, score_threshold):
    with np.load(path) as data:
        xy = data['xy_native'].astype(float)
        scores = data['scores'].astype(float)
    valid = np.isfinite(xy).all(axis=1) & (scores >= score_threshold)
    hips = xy[list(BODY_HIP_SLOTS)].mean(axis=0) if valid[list(BODY_HIP_SLOTS)].all() else None
    return {'xy': xy, 'scores': scores, 'valid': valid,
        'valid_body_points': int(valid[:18].sum()),
        'hip_x': None if hips is None else float(hips[0])}


def region_of(mask_path):
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L')) > 127
    if not mask.any():
        return {'pixel_count': 0, 'centroid_xy': None, 'bbox_xyxy': None}
    ys, xs = np.nonzero(mask)
    return {'pixel_count': int(mask.sum()),
        'centroid_xy': [float(xs.mean()), float(ys.mean())],
        'bbox_xyxy': [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]}


def action_family(action):
    return ACTION_NUMBER.sub('', action) or action


def eligible(row, score_threshold):
    required = [row['source_rgb'], row['target_rgb'], row['source_person_mask_A'],
        row['source_person_mask_B'], row['target_dwpose_A'], row['target_dwpose_B']]
    if not all(Path(item).is_file() for item in required):
        return None
    try:
        persons = [keypoints(row['target_dwpose_A'], score_threshold),
            keypoints(row['target_dwpose_B'], score_threshold)]
    except (OSError, ValueError, KeyError):
        return None
    if any(person['valid_body_points'] < MIN_VALID_BODY_POINTS for person in persons):
        return None
    if any(person['hip_x'] is None for person in persons):
        return None
    return persons


def stratified_order(rows, persons_by_row):
    """Round-robin over (target action family, relation, contact bucket) groups."""
    groups = {}
    for row in rows:
        if row['_row_index'] not in persons_by_row:
            continue
        total = int(row.get('target_contact_vertex_count_total', 0))
        bucket = 'none' if total == 0 else ('light' if total <= 5 else 'heavy')
        key = (action_family(row['target_action']), row['pair_relation'], bucket)
        groups.setdefault(key, []).append(row)
    ordered, keys = [], sorted(groups)
    while any(groups[key] for key in keys):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def person_order(persons, sides):
    """Order the two people left to right by hip x, so skeleton 1/2 is reproducible.

    Interacting people overlap, so the two hip centres can be a pixel apart. The order stays
    deterministic, but `hip_x_separation` records how thin that decision actually is.
    """
    ranking = sorted(range(len(persons)), key=lambda i: persons[i]['hip_x'])
    ordered = [{'slot': slot + 1, 'hi4d_side': sides[i], 'hip_x': persons[i]['hip_x'],
        'valid_body_points': persons[i]['valid_body_points']}
        for slot, i in enumerate(ranking)]
    separation = abs(ordered[0]['hip_x'] - ordered[1]['hip_x'])
    for item in ordered:
        item['hip_x_separation'] = separation
    return ordered


def suggested_pairing(source_regions, persons):
    """Nearest horizontal centre in A to hip x in B; a hint for scoring, never a generation rule.

    When the two possible assignments cost about the same, claiming one of them would be
    invented precision, so the case is recorded as ``ambiguous`` and the evaluator must take
    the better of both assignments and say so.
    """
    a_xs = [region['centroid_xy'][0] for region in source_regions if region['centroid_xy']]
    b_xs = [person['hip_x'] for person in persons]
    if len(a_xs) != 2 or len(b_xs) != 2:
        return None
    width = max(max(a_xs), max(b_xs)) or 1.0
    matrix = [[abs(a / width - b / width) for b in b_xs] for a in a_xs]
    direct = matrix[0][0] + matrix[1][1]
    swapped = matrix[0][1] + matrix[1][0]
    margin = abs(direct - swapped)
    assignment = 'ambiguous' if margin < PAIRING_AMBIGUITY else (
        'direct' if direct < swapped else 'swapped')
    return {'assignment': assignment, 'cost_direct': direct, 'cost_swapped': swapped,
        'cost_margin': margin, 'ambiguous_below': PAIRING_AMBIGUITY,
        'basis': 'horizontal centre of the A person mask versus hip x of the B skeleton',
        'note': 'evaluation hint only; the generator is never told which person is which'}


def build_same_identity(row, out_dir, score_threshold, persons=None):
    persons = eligible(row, score_threshold) if persons is None else persons
    assert persons, row['pair']
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(row['source_rgb'], out_dir / 'A_source.png')
    shutil.copyfile(row['target_rgb'], out_dir / 'B_target.png')
    keypoint_files = []
    for slot, side in enumerate(('A', 'B'), start=1):
        target = out_dir / f'B_person_{slot}_keypoints.npz'
        shutil.copyfile(row[f'target_dwpose_{side}'], target)
        keypoint_files.append({'slot': slot, 'hi4d_side': side, 'file': target.name,
            'sha256': sha256_file(target)})
    source_regions = [region_of(row[f'source_person_mask_{side}']) for side in ('A', 'B')]
    ordered = person_order(persons, ['A', 'B'])
    with Image.open(row['target_rgb']) as image:
        b_size = list(image.size)
    with Image.open(row['source_rgb']) as image:
        a_size = list(image.size)
    pair = {
        'schema': 'dual_pose_pair_v1', 'kind': 'same_identity',
        'name': out_dir.name, 'built_at_unix': time.time(),
        'manifest': {'row_index': row['_row_index'], 'pair': row['pair'],
            'camera_id': row['camera_id'], 'source_action': row['source_action'],
            'target_action': row['target_action'], 'pair_relation': row['pair_relation'],
            'pair_type': row['pair_type'], 'source_frame_id': row['source_frame_id'],
            'target_frame_id': row['target_frame_id'],
            'target_contact_vertex_count_total': row.get('target_contact_vertex_count_total'),
            'asset_version': row.get('asset_version')},
        'A': {'image': 'A_source.png', 'sha256': sha256_file(out_dir / 'A_source.png'),
            'size': a_size, 'persons': [
                {'hi4d_side': side, 'region': region,
                    'mask_original': row[f'source_person_mask_{side}']}
                for side, region in zip(('A', 'B'), source_regions)]},
        'B': {'image': 'B_target.png', 'sha256': sha256_file(out_dir / 'B_target.png'),
            'size': b_size, 'slot_order': ordered, 'keypoints': keypoint_files,
            'smpl_original': row['target_smpl'],
            'geometry_original': row['target_geometry_npz'],
            'camera_original': row['target_camera'],
            'keypoint_source': 'Hi4D precomputed DWPose-134 per person'},
        'scoring': {'person_assignment': suggested_pairing(source_regions, persons),
            'score_threshold': score_threshold},
        'ground_truth_limits': {
            'hands_and_face': 'DWPose-134 supplies them; the SMPL npz is 24-joint SMPL only',
            'occlusion': ('the estimator guesses occluded landmarks; scores plus '
                'A_occluded_by_B/B_occluded_by_A in the geometry npz bound how far they can be trusted'),
            'appearance_contrast': ('A and B are the same two actors in the same camera, so this '
                'flavour cannot detect appearance leakage from B')},
    }
    (out_dir / 'pair.json').write_text(json.dumps(pair, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    (out_dir / 'GT_NOTE.md').write_text(
        f"# {out_dir.name}\n\n"
        f"- 类型：same_identity（Hi4D 原生配对，A 与 B 同演员同相机，只换动作）\n"
        f"- A：`{row['source_action']}` 帧 {row['source_frame_id']}；B：`{row['target_action']}` "
        f"帧 {row['target_frame_id']}，相机 {row['camera_id']}，relation `{row['pair_relation']}`\n"
        f"- 骨架 slot 顺序：按 B 中两人髖心 x 从左到右 → {ordered[0]['hi4d_side']}, "
        f"{ordered[1]['hi4d_side']}（Hi4D 侧标记）\n"
        f"- 逐人关键点：`{keypoint_files[0]['file']}`、`{keypoint_files[1]['file']}`"
        f"（DWPose-134，940×1280 原生坐标）\n"
        f"- 可用 GT：逐人掩码/深度/接触图、`A_occluded_by_B`、SMPL 参数与 8 相机内外参\n"
        f"- 不可用：手脸来自 SMPL（只有 24 关节）；被遮挡点的分数是检测器猜测；A 与 B 外观相同，"
        f"这一类对测不出外观泄漏\n", encoding='utf-8')
    return pair


def build_cross_pair(row_a, row_b, out_dir, score_threshold, persons_b):
    """A from one pair's source frame, B from another pair's target frame."""
    pair = build_same_identity(row_b, out_dir, score_threshold, persons_b)
    shutil.copyfile(row_a['source_rgb'], out_dir / 'A_source.png')
    source_regions = [region_of(row_a[f'source_person_mask_{side}']) for side in ('A', 'B')]
    with Image.open(row_a['source_rgb']) as image:
        a_size = list(image.size)
    pair['kind'] = 'cross_pair'
    pair['A'] = {'image': 'A_source.png', 'sha256': sha256_file(out_dir / 'A_source.png'),
        'size': a_size, 'origin_pair': row_a['pair'],
        'origin_action': row_a['source_action'],
        'persons': [{'hi4d_side': side, 'region': region,
            'mask_original': row_a[f'source_person_mask_{side}']}
            for side, region in zip(('A', 'B'), source_regions)]}
    pair['scoring']['person_assignment'] = {
        'assignment': 'best_of_two',
        'note': ('A and B come from different actor pairs, so no ground-truth correspondence '
            'exists; per-person metrics take the better of the two possible assignments and '
            'record which one was used')}
    pair['ground_truth_limits']['appearance_contrast'] = (
        'A and B show different actors, so identity preservation and appearance leakage are '
        'measurable; the cost is that per-person correspondence is not grounded')
    (out_dir / 'pair.json').write_text(json.dumps(pair, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    (out_dir / 'GT_NOTE.md').write_text((out_dir / 'GT_NOTE.md').read_text(encoding='utf-8')
        .replace('类型：same_identity（Hi4D 原生配对，A 与 B 同演员同相机，只换动作）',
            f"类型：cross_pair（A 取自 pair {row_a['pair']} 的源帧 "
            f"`{row_a['source_action']}`，B 取自 pair {row_b['pair']} 的目标帧 "
            f"`{row_b['target_action']}`）\n- 人物对应关系：无 GT，评测按两种分配取较优"),
        encoding='utf-8')
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--cross-count', type=int, default=5)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--max-per-action', type=int, default=2)
    parser.add_argument('--list', action='store_true', help='Print the selection, write nothing')
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    eligible_pairs, persons_by_index = [], {}
    for row in rows:
        persons = eligible(row, args.score_threshold)
        if persons:
            eligible_pairs.append(row)
            persons_by_index[row['_row_index']] = persons
    ordered = stratified_order(eligible_pairs, persons_by_index)
    chosen, per_action = [], {}
    for row in ordered:
        family = action_family(row['target_action'])
        if per_action.get(family, 0) >= args.max_per_action:
            continue
        chosen.append(row)
        per_action[family] = per_action.get(family, 0) + 1
        if len(chosen) >= args.count:
            break
    if args.list:
        for row in chosen:
            print(f"{row['pair']:8} {row['camera_id']:>3} {row['source_action']:12} -> "
                f"{row['target_action']:12} {row['pair_relation']:13} "
                f"contact={row.get('target_contact_vertex_count_total')}")
        print(f"selected {len(chosen)} of {len(eligible_pairs)} eligible rows "
            f"({len(rows)} in the manifest)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    built = []
    for row in chosen:
        name = (f"{action_family(row['target_action'])}_{row['pair']}_{row['target_frame_id']:03d}"
            f"_c{row['camera_id']}_r{row['_row_index']:03d}")
        built.append(build_same_identity(row, args.out / name, args.score_threshold,
            persons_by_index[row['_row_index']]))
    cross_dir = args.out.parent / 'pairs_cross'
    cross_dir.mkdir(parents=True, exist_ok=True)
    # A source frame comes from a row of a different pair folder, so actors and clothing differ.
    used_as_target = set()
    for row_b in chosen:
        if len([item for item in built if item['kind'] == 'cross_pair']) >= args.cross_count:
            break
        for row_a in chosen:
            if row_a['pair'] == row_b['pair'] or row_a['_row_index'] in used_as_target:
                continue
            used_as_target.add(row_a['_row_index'])
            name = (f"cross_{action_family(row_b['target_action'])}_{row_b['pair']}"
                f"_{row_b['target_frame_id']:03d}_r{row_b['_row_index']:03d}"
                f"_from_{row_a['pair']}")
            built.append(build_cross_pair(row_a, row_b, cross_dir / name, args.score_threshold,
                persons_by_index[row_b['_row_index']]))
            break
    print(f"built {len(built)} pair directories under {args.out} and {cross_dir}")
    for pair in built:
        print(f"  {pair['kind']:13} {pair['name']:38} A {pair['A']['size']} B {pair['B']['size']}")


if __name__ == '__main__':
    main()
