import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from benchmark import BODY_NAMES, SMPL_IDS, project_camera, save_new, sha256, validate_records


def mask_box(mask):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('Empty person mask')
    return [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]


def mask_ownership(points, masks):
    h, w = masks[0].shape
    result = []
    for person, joints in enumerate(points):
        codes = []
        for xy in joints:
            if not np.isfinite(xy).all():
                codes.append(-1)
                continue
            x, y = np.rint(xy).astype(int)
            if not (0 <= x < w and 0 <= y < h):
                codes.append(-1)
                continue
            own, other = bool(masks[person][y, x]), bool(masks[1 - person][y, x])
            codes.append(0 if own and not other else 1 if other and not own else -1)
        result.append(codes)
    return result


def category_from_boxes(boxes):
    a, b = np.asarray(boxes, float)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    areas = np.prod(np.asarray(boxes)[:, 2:] - np.asarray(boxes)[:, :2], axis=1)
    iou = float(intersection / (areas.sum() - intersection))
    iomin = float(intersection / areas.min())
    if iou >= .55 or iomin >= .7:
        category = 'severe'
    elif iou >= .28:
        category = 'hug'
    elif iou >= .02:
        category = 'light'
    else:
        category = 'separated'
    return category, iou, iomin


def temporally_separate(candidate, chosen, gap):
    return all(candidate['pair'] != other['pair'] or candidate['action'] != other['action'] or
               abs(candidate['frame_id'] - other['frame_id']) >= gap for other in chosen)


@lru_cache(maxsize=64)
def cameras(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def make_record(raw, subset, legacy=None, legacy_boxes=None):
    from PIL import Image
    image = Path(raw['rgb'])
    image_hash = sha256(image)
    with Image.open(image) as handle:
        size = list(handle.size)
    masks = []
    for key in ('person_mask_A', 'person_mask_B'):
        with Image.open(raw[key]) as handle:
            mask = np.asarray(handle.convert('L')) > 0
        if list(mask.shape) != size[::-1]:
            raise ValueError('Image/mask dimensions disagree')
        masks.append(mask)
    boxes = [mask_box(mask) for mask in masks]
    if legacy_boxes is not None:
        boxes = [person['bbox'] for person in legacy_boxes]
    camera = cameras(raw['camera_file'])
    camera_index = camera['ids'].tolist().index(raw['camera_id'])
    k = camera['intrinsics'][camera_index]
    if np.max(np.abs(camera['dist_coeffs'][camera_index])) > 1e-8:
        raise ValueError('Nonzero distortion requires explicit projection handling')
    with np.load(raw['official_smpl'], allow_pickle=False) as arrays:
        world = np.asarray(arrays['joints_3d'], float)
    xyz, xy = project_camera(world, k, camera['extrinsics'][camera_index])
    xy, body = xy[:, SMPL_IDS], xyz[:, SMPL_IDS]
    valid = np.isfinite(xy).all(axis=-1) & (xy[..., 0] >= 0) & (xy[..., 0] < size[0]) & (xy[..., 1] >= 0) & (xy[..., 1] < size[1])
    category, iou, iomin = category_from_boxes(boxes)
    result = {'id': image_hash[:16], 'image': str(image), 'image_sha256': image_hash,
              'size': size, 'subset': subset, 'split': raw['split'], 'pair': raw['pair'],
              'action': raw['action'], 'camera_id': raw['camera_id'], 'frame_id': raw['frame_id'],
              'category': legacy['category'] if legacy else category,
              'category_by_disjoint_box_overlap': category, 'bbox_iou': iou, 'bbox_iomin': iomin,
              'K': k.tolist(), 'gt_boxes': boxes, 'gt_xy': xy.tolist(), 'gt_xyz': body.tolist(),
              'gt_pelvis': xyz[:, 0].tolist(), 'gt_valid': valid.tolist(),
              'occlusion_proxy': mask_ownership(xy, masks),
              'mask_overlap_pixels': int(np.sum(masks[0] & masks[1])),
              'source': {key: raw[key] for key in ('official_smpl', 'camera_file', 'person_mask_A', 'person_mask_B')},
              'source_hashes': {key: sha256(raw[key]) for key in ('official_smpl', 'camera_file', 'person_mask_A', 'person_mask_B')}}
    if legacy:
        result['legacy_filename'] = legacy['filename']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', default='/home/shangguanrz/project/pic-edit')
    parser.add_argument('--additional-per-category', type=int, default=12)
    parser.add_argument('--min-frame-gap', type=int, default=15)
    args = parser.parse_args()
    target = args.run_dir / 'data/request.json'
    if target.exists():
        raise FileExistsError(target)
    raw = [json.loads(line) for line in args.raw_manifest.read_text().splitlines() if line.strip()]
    raw_by_path = {row['rgb']: row for row in raw}
    old_path = args.legacy_root / 'data/manifests/all_80.jsonl'
    old = [json.loads(line) for line in old_path.read_text().splitlines() if line.strip()]
    old_boxes = json.loads((args.legacy_root / 'data/gt_boxes/gt_boxes.json').read_text())
    old_groups = defaultdict(list)
    for row in old:
        original_hash = sha256(row['orig_rgb'])
        copied_hash = sha256(args.legacy_root / 'data/images' / row['filename'])
        if original_hash != copied_hash:
            raise ValueError('Legacy image differs from its source')
        old_groups[original_hash].append(row)
    priority = {'separated': 0, 'light': 1, 'hug': 2, 'severe': 3}
    selected = []
    for group in old_groups.values():
        row = max(group, key=lambda item: priority[item['category']])
        record = make_record(raw_by_path[row['orig_rgb']], 'legacy', row, old_boxes[row['filename']])
        record['legacy_category_memberships'] = [x['category'] for x in group]
        record['legacy_all_filenames'] = [x['filename'] for x in group]
        selected.append(record)
    candidates, skipped = defaultdict(list), []
    old_paths = {row['orig_rgb'] for row in old}
    for row in raw:
        if row['rgb'] in old_paths or not temporally_separate(row, selected, args.min_frame_gap):
            continue
        try:
            record = make_record(row, 'additional')
            if record['image_sha256'] in old_groups:
                continue
            if not np.asarray(record['gt_valid']).all():
                skipped.append({'image': row['rgb'], 'reason': 'offscreen_gt_joint'})
                continue
            candidates[record['category']].append(record)
        except (OSError, ValueError, KeyError) as error:
            skipped.append({'image': row['rgb'], 'reason': str(error)})
    for category in ('separated', 'light', 'hug', 'severe'):
        pool = sorted(candidates[category], key=lambda row: row['image_sha256'])
        counts = Counter()
        chosen = []
        while len(chosen) < args.additional_per_category:
            available = [row for row in pool if row not in chosen and
                         temporally_separate(row, selected + chosen, args.min_frame_gap)]
            if not available:
                raise ValueError(f'Insufficient distinct {category} frames: {len(chosen)} / {args.additional_per_category}')
            row = min(available, key=lambda item: (counts[item['pair']], counts[(item['pair'], item['action'])], item['image_sha256']))
            chosen.append(row)
            counts[row['pair']] += 1
            counts[(row['pair'], row['action'])] += 1
        selected.extend(chosen)
    validate_records(selected)
    request = {'schema': 'dual_pose_benchmark_v1', 'project_root': args.project_root,
               'out_dir': str(args.run_dir / 'output'), 'body_joint_names': list(BODY_NAMES),
               'records': selected,
               'sampling': {'legacy_rows': len(old), 'legacy_unique_images': len(old_groups),
                            'additional_per_category': args.additional_per_category,
                            'min_frame_gap_same_pair_action_all_cameras': args.min_frame_gap,
                            'raw_manifest': str(args.raw_manifest), 'raw_manifest_sha256': sha256(args.raw_manifest),
                            'legacy_manifest_sha256': sha256(old_path),
                            'category_basis': 'legacy labels preserved; additional disjoint bbox overlap bins, hug is moderate overlap not an action label',
                            'pool_sizes': {key: len(value) for key, value in candidates.items()}, 'skipped': skipped}}
    save_new(target, request)
    print(json.dumps({'request': str(target), 'records': len(selected),
                      'subsets': dict(Counter(x['subset'] for x in selected)),
                      'categories': dict(Counter(x['category'] for x in selected)),
                      'pairs': dict(Counter(x['pair'] for x in selected)),
                      'splits': dict(Counter(x['split'] for x in selected)),
                      'action_ids': len({(x['pair'], x['action']) for x in selected})}, indent=2))


if __name__ == '__main__':
    main()
