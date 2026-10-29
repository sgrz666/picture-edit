import argparse
import csv
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path
import runpy

import numpy as np

BODY_NAMES = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
SMPL_IDS = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)


def named_indices(meta):
    lookup = {item['name']: int(index) for index, item in meta.items()}
    if not set(BODY_NAMES) <= lookup.keys():
        raise ValueError('Missing named body landmarks')
    return [lookup[name] for name in BODY_NAMES]


def validate_records(records):
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs')
    if not ids:
        raise ValueError('Empty benchmark')


def project_camera(xyz, k, extrinsic):
    camera = np.asarray(xyz) @ np.asarray(extrinsic)[:, :3].T + np.asarray(extrinsic)[:, 3]
    homogeneous = camera @ np.asarray(k).T
    if np.any(homogeneous[..., 2] <= 0):
        raise ValueError('Ground truth behind the camera')
    return camera, homogeneous[..., :2] / homogeneous[..., 2:]


def validate_prediction(record, prediction, mode, request_sha256):
    expected = {'id': record['id'], 'mode': mode, 'image_sha256': record['image_sha256'],
                'request_sha256': request_sha256}
    if any(prediction.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Prediction provenance mismatch: {record["id"]} / {mode}')


def score_frame(record, prediction):
    gt = np.asarray(record['gt_xy'], float)
    valid = np.asarray(record['gt_valid'], bool)
    boxes = np.asarray(record['gt_boxes'], float)
    if gt.shape != (2, 12, 2) or valid.shape != (2, 12):
        raise ValueError('Expected two people and twelve named joints')
    scales = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
    declared = (prediction.get('provenance') or {}).get('absolute_3d_comparable')
    absolute_ok = bool(prediction.get('absolute_3d_comparable', declared
                                   if declared is not None else prediction.get('scene_camera_available')))
    if np.any(scales <= 0):
        raise ValueError('Degenerate ground truth boxes')
    people = prediction.get('instances', []) if prediction.get('status') == 'ok' else []
    points = []
    for person in people:
        if tuple(person['joint_names']) != BODY_NAMES:
            raise ValueError('Prediction landmark names do not match benchmark order')
        xy = np.asarray(person['xy'], float)
        if xy.shape != (12, 2):
            raise ValueError('Invalid prediction shape')
        points.append(xy)
    costs = np.zeros((2, len(points)))
    for g in range(2):
        for p, xy in enumerate(points):
            distance = np.linalg.norm(xy - gt[g], axis=1) / scales[g]
            distance = np.where(np.isfinite(distance), distance, 2.)
            costs[g, p] = float(distance[valid[g]].mean()) if valid[g].any() else 2.
    count = min(2, len(points))
    candidates = []
    for g_order in combinations(range(2), count):
        for p_order in permutations(range(len(points)), count):
            matches = list(zip(g_order, p_order))
            candidates.append((sum(costs[g, p] for g, p in matches), matches))
    candidates.sort(key=lambda item: item[0])
    matches = candidates[0][1] if candidates else []
    assignment = [None, None]
    distances, normalized, scores, visibility = [], [], [], []
    person_pck, root_errors, absolute_errors = [], [], []
    cross_count = 0
    discriminable, cross_discriminable = 0, 0
    for g, p in matches:
        assignment[g] = p
        xy = points[p]
        usable = valid[g] & np.isfinite(xy).all(axis=1)
        errors = np.linalg.norm(xy - gt[g], axis=1)
        norm = errors / scales[g]
        distances.extend(errors[usable].tolist())
        normalized.extend(norm[usable].tolist())
        scores.extend(np.asarray(people[p]['scores'], float)[usable].tolist())
        visibility.extend(np.asarray(record['occlusion_proxy'][g])[usable].tolist())
        person_pck.append(float(np.sum(usable & (norm < .10)) / max(int(valid[g].sum()), 1)))
        other_error = np.linalg.norm(xy - gt[1 - g], axis=1)
        separation = np.linalg.norm(gt[g] - gt[1 - g], axis=1)
        # A joint whose two persons sit within 10% of a body diagonal apart cannot tell
        # "reached the other body" from "imprecise", so those joints stay out of the rate.
        discriminable += int(np.sum(usable & (separation > .1 * scales[g])))
        cross = usable & valid[1 - g] & (other_error + .02 * scales[g] < errors) & (other_error < .1 * scales[1 - g])
        cross_count += int(np.sum(cross))
        cross_discriminable += int(np.sum(cross & (separation > .1 * scales[g])))
        if people[p].get('xyz') is not None and record.get('gt_xyz') is not None:
            xyz = np.asarray(people[p]['xyz'], float)
            reference = np.asarray(record['gt_xyz'][g], float)
            pelvis = np.asarray(people[p]['pelvis'], float)
            reference_pelvis = np.asarray(record['gt_pelvis'][g], float)
            if xyz.shape != (12, 3) or not np.isfinite(xyz).all():
                raise ValueError('Invalid predicted 3D joints')
            root_errors.extend((1000 * np.linalg.norm(
                xyz - pelvis - (reference - reference_pelvis), axis=1)).tolist())
            if absolute_ok:
                absolute_errors.extend((1000 * np.linalg.norm(xyz - reference, axis=1)).tolist())
    n_gt = int(valid.sum())
    supports_3d = prediction.get('mode', '').startswith(('multihmr', 'smplest')) or any(p.get('xyz') is not None for p in people)
    gt_3d_count = 24 if supports_3d and record.get('gt_xyz') is not None else 0
    gt_visibility = np.asarray(record['occlusion_proxy'])[valid]
    normalized = np.asarray(normalized, float)
    n_matched = len(normalized)
    hits05, hits10 = int(np.sum(normalized < .05)), int(np.sum(normalized < .1))
    vector_error = None
    if len(matches) == 2 and absolute_ok and record.get('gt_pelvis'):
        pa = np.asarray(people[assignment[0]]['pelvis'])
        pb = np.asarray(people[assignment[1]]['pelvis'])
        reference = np.asarray(record['gt_pelvis'])
        vector_error = float(1000 * np.linalg.norm((pb - pa) - (reference[1] - reference[0])))
    return {
        'id': record['id'], 'status': prediction.get('status', 'missing'),
        'error': prediction.get('error'), 'det_count': prediction.get('det_count', 0),
        'two_person_count_ok': prediction.get('det_count') == 2 and len(people) == 2,
        'matched_person_count': len(matches), 'assignment': assignment,
        'assignment_basis': 'minimum body error, unordered pose matching; not identity tracking',
        'assignment_margin': float(candidates[1][0] - candidates[0][0]) if len(candidates) > 1 else None,
        'gt_joint_count': n_gt, 'matched_joint_count': n_matched,
        'pck02_hits': int(np.sum(normalized < .02)), 'pck05_hits': hits05, 'pck10_hits': hits10,
        'mean_error_px': float(np.mean(distances)) if distances else None,
        'matched_pck02': float(np.mean(normalized < .02)) if n_matched else None,
        'matched_pck05': hits05 / n_matched if n_matched else None,
        'matched_pck10': hits10 / n_matched if n_matched else None,
        'e2e_pck02': int(np.sum(normalized < .02)) / n_gt if n_gt else None,
        'e2e_pck05': hits05 / n_gt if n_gt else None,
        'e2e_pck10': hits10 / n_gt if n_gt else None,
        'pair_pck10_at075': len(person_pck) == 2 and min(person_pck) >= .75,
        'cross_person_joint_candidates': cross_count,
        'cross_person_discriminable_joints': discriminable,
        'cross_person_discriminable_hits': cross_discriminable,
        'cross_person_rate_discriminable': cross_discriminable / discriminable if discriminable else None,
        'roi_assignment_disagrees': (assignment != [0, 1]) if 'gtbox' in prediction.get('mode', '') else None,
        'gt_visibility_counts': {str(code): int(np.sum(gt_visibility == code)) for code in (-1, 0, 1)},
        'gt_3d_joint_count': gt_3d_count, 'matched_3d_joint_count': len(root_errors),
        'root_pck150mm_hits': int(np.sum(np.asarray(root_errors) < 150)),
        'e2e_root_pck150mm': int(np.sum(np.asarray(root_errors) < 150)) / gt_3d_count if gt_3d_count else None,
        'root_mpjpe_mm': float(np.mean(root_errors)) if root_errors else None,
        'absolute_mpjpe_mm': float(np.mean(absolute_errors)) if absolute_errors else None,
        'pelvis_vector_error_mm': vector_error,
        'latency_ms': prediction.get('latency_ms'),
        'peak_allocated_gib': prediction.get('peak_allocated_gib'),
        'peak_reserved_gib': prediction.get('peak_reserved_gib'),
        'distances_px': distances, 'normalized_errors': normalized.tolist(),
        'scores': scores, 'occlusion_proxy': visibility,
    }


def summarize(rows):
    gt = sum(row['gt_joint_count'] for row in rows)
    matched = sum(row['matched_joint_count'] for row in rows)
    errors = [x for row in rows for x in row['distances_px']]
    norms = np.asarray([x for row in rows for x in row['normalized_errors']])
    vis = np.asarray([x for row in rows for x in row['occlusion_proxy']])
    result = {'frames': len(rows), 'ok': sum(x['status'] == 'ok' for x in rows),
              'missing_or_error': [x['id'] for x in rows if x['status'] != 'ok'],
              'two_people': sum(x['two_person_count_ok'] for x in rows),
              'gt_joints': gt, 'matched_joints': matched,
              'joint_coverage': matched / gt if gt else None,
              'mean_error_px': float(np.mean(errors)) if errors else None,
              'median_error_px': float(np.median(errors)) if errors else None,
              'pair_pck10_at075': sum(x['pair_pck10_at075'] for x in rows),
              'cross_person_joint_candidates': sum(x['cross_person_joint_candidates'] for x in rows),
              'cross_person_discriminable_joints': sum(x['cross_person_discriminable_joints'] for x in rows),
              'cross_person_rate_discriminable': (
                  sum(x['cross_person_discriminable_hits'] for x in rows) /
                  sum(x['cross_person_discriminable_joints'] for x in rows))
                  if sum(x['cross_person_discriminable_joints'] for x in rows) else None,
              'roi_assignment_disagrees': sum(x['roi_assignment_disagrees'] is True for x in rows)}
    for suffix in ('02', '05', '10'):
        hits = sum(x[f'pck{suffix}_hits'] for x in rows)
        result[f'e2e_pck{suffix}'] = hits / gt if gt else None
        result[f'matched_pck{suffix}'] = hits / matched if matched else None
    for label, code in [('own_mask', 0), ('other_mask', 1), ('ambiguous_or_outside_masks', -1)]:
        values = norms[vis == code]
        expected = sum(row['gt_visibility_counts'][str(code)] for row in rows)
        result[label] = {'gt_joints': expected, 'matched_joints': len(values),
                         'coverage': len(values) / expected if expected else None,
                         'matched_pck10': float(np.mean(values < .1)) if len(values) else None,
                         'e2e_pck10': int(np.sum(values < .1)) / expected if expected else None}
    gt3d = sum(row['gt_3d_joint_count'] for row in rows)
    matched3d = sum(row['matched_3d_joint_count'] for row in rows)
    result['gt_3d_joints'] = gt3d
    result['matched_3d_joints'] = matched3d
    result['joint_3d_coverage'] = matched3d / gt3d if gt3d else None
    result['e2e_root_pck150mm'] = sum(row['root_pck150mm_hits'] for row in rows) / gt3d if gt3d else None
    for key in ('root_mpjpe_mm', 'absolute_mpjpe_mm', 'pelvis_vector_error_mm', 'latency_ms',
                'peak_allocated_gib', 'peak_reserved_gib'):
        values = [x[key] for x in rows if x[key] is not None]
        result[key] = {'n': len(values), 'mean': float(np.mean(values)) if values else None,
                       'p95': float(np.percentile(values, 95)) if values else None}
    confidence = np.asarray([x for row in rows for x in row['scores']])
    result['confidence_sensitivity'] = {}
    for threshold in (.3, .5, .7):
        selected = confidence >= threshold
        result['confidence_sensitivity'][str(threshold)] = {
            'coverage_of_gt': int(selected.sum()) / gt if gt else None,
            'selected_pck10': float(np.mean(norms[selected] < .1)) if selected.any() else None}
    return result


def import_legacy(request, request_path, legacy_root, meta_path):
    meta = runpy.run_path(str(meta_path))['dataset_info']['keypoint_info']
    indices = named_indices(meta)
    imported = 0
    for record in request['records']:
        if record['subset'] != 'legacy':
            continue
        for old_mode, mode in [('auto', 'sapiens_detr030_cached'), ('gtbox', 'sapiens_gtbox_cached')]:
            source = Path(legacy_root) / 'output' / f'pose5b_{old_mode}' / (Path(record['legacy_filename']).stem + '.json')
            old = json.loads(source.read_text())
            instances = []
            for person in old['instances']:
                instances.append({'bbox': person['bbox'], 'xy': np.asarray(person['keypoints'])[indices].tolist(),
                                  'scores': np.asarray(person['keypoint_scores'])[indices].tolist(),
                                  'joint_names': list(BODY_NAMES), 'extra_keypoints': person['keypoints'],
                                  'extra_scores': person['keypoint_scores']})
            prediction = {'schema': 'dual_pose_prediction_v1', 'id': record['id'], 'mode': mode,
                          'image_sha256': record['image_sha256'], 'request_sha256': sha256(request_path),
                          'status': 'ok', 'instances': instances, 'det_count': old['det_count'],
                          'latency_ms': old['latency_ms'], 'scene_camera_available': False,
                          'peak_allocated_gib': None, 'peak_reserved_gib': None,
                          'provenance': {'source': str(source), 'source_sha256': sha256(source),
                                         'mapping': indices, 'cached_not_rerun': True,
                                         'det_fallback': old.get('det_fallback', False)}}
            target = Path(request['out_dir']) / mode / (record['id'] + '.json')
            save_new(target, prediction)
            imported += 1
    print(json.dumps({'imported_predictions': imported, 'correct_indices': indices}))


def evaluate(request, modes, destination, request_sha256):
    all_rows, summaries = [], {}
    metadata = {x['id']: x for x in request['records']}
    validate_records(request['records'])
    for mode in modes:
        rows = []
        relevant = [r for r in request['records'] if not mode.endswith('_cached') or r['subset'] == 'legacy']
        for record in relevant:
            path = Path(request['out_dir']) / mode / (record['id'] + '.json')
            prediction = json.loads(path.read_text()) if path.is_file() else {'status': 'missing', 'mode': mode}
            if path.is_file():
                validate_prediction(record, prediction, mode, request_sha256)
            row = score_frame(record, prediction)
            row.update(mode=mode, category=record['category'], subset=record['subset'],
                       pair=record['pair'], action=record['action'])
            rows.append(row)
        summaries[mode] = {'overall': summarize(rows)}
        for field in ('category', 'subset', 'pair'):
            summaries[mode][field] = {value: summarize([r for r in rows if r[field] == value])
                                      for value in sorted({r[field] for r in rows})}
        all_rows.extend(rows)
    paired = {}
    for a, b in combinations(modes, 2):
        aa = {r['id']: r for r in all_rows if r['mode'] == a}
        bb = {r['id']: r for r in all_rows if r['mode'] == b}
        ids = sorted(aa.keys() & bb.keys())
        delta = [aa[i]['e2e_pck10'] - bb[i]['e2e_pck10'] for i in ids]
        groups = {}
        for i, value in zip(ids, delta):
            groups.setdefault(metadata[i]['pair'], []).append(value)
        group_means = np.asarray([np.mean(values) for values in groups.values()])
        rng = np.random.default_rng(20261029)
        boot = [float(np.mean(rng.choice(group_means, len(group_means), replace=True))) for _ in range(1000)] if len(group_means) >= 2 else []
        paired[a + '_minus_' + b] = {'n': len(ids), 'actor_pairs': len(groups),
                                    'e2e_pck10_mean_delta': float(np.mean(delta)) if delta else None,
                                    'wins_ties_losses': [sum(x > 1e-9 for x in delta), sum(abs(x) <= 1e-9 for x in delta), sum(x < -1e-9 for x in delta)],
                                    'equal_actor_pair_mean_delta': float(group_means.mean()) if len(group_means) else None,
                                    'actor_cluster_bootstrap_95': np.percentile(boot, [2.5, 97.5]).tolist() if boot else None}
    result = {'schema': 'dual_pose_comparison_v1', 'records': len(metadata), 'body_joint_names': list(BODY_NAMES),
              'normalization': 'per-person GT bbox diagonal; unmatched GT joints count as end-to-end misses',
              'visibility': 'mask ownership proxy, not anatomical visibility ground truth',
              'summaries': summaries, 'paired': paired, 'rows': all_rows}
    destination = Path(destination)
    save_new(destination / 'results.json', result)
    fields = [k for k in all_rows[0] if k not in ('distances_px', 'normalized_errors', 'scores', 'occlusion_proxy')]
    with (destination / 'metrics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({mode: info['overall'] for mode, info in summaries.items()}, indent=2))
    if any(r['status'] != 'ok' for r in all_rows):
        raise SystemExit('Benchmark incomplete: missing/error predictions retained in the denominator')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import-legacy')
    imp.add_argument('--legacy-root', type=Path, required=True)
    imp.add_argument('--meta', type=Path, required=True)
    score = sub.add_parser('evaluate')
    score.add_argument('--modes', nargs='+', required=True)
    score.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    validate_records(request['records'])
    if args.command == 'import-legacy':
        import_legacy(request, args.request, args.legacy_root, args.meta)
    else:
        evaluate(request, args.modes, args.report, sha256(args.request))


if __name__ == '__main__':
    main()
