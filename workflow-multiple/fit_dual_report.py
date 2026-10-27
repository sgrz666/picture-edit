#!/usr/bin/env python3
"""WP7: measure how far a Multi-HMR fit is from the Hi4D ground truth before trusting it.

The generation line guides the editor with the pair's ground-truth DWPose-134 landmarks. Replacing
them with an estimator's output is only defensible once that estimator's own error has been
measured on the same photographs, because a wrong guide makes a wrong generation impossible to
attribute. For every pair this script fits the target image with Multi-HMR and reports, per person:

  * 2D PCK and MPJPE against the ground-truth DWPose landmarks, scored by the same function the
    generation evaluation uses and normalised by the same torso;
  * how far those DWPose landmarks sit from the Hi4D SMPL joints projected through the frame's own
    intrinsics - the floor no estimator can go below;
  * 3D absolute and root-aligned MPJPE in metres, and the inter-hip distance error;
  * whether pairing people by hip position recovers the pairing that minimises joint error.

Usage:

    add/envs/smplestx_bw/bin/python workflow-multiple/fit_dual_report.py [--limit 1] [--dry-run]

The fitting itself runs in the multi-HMR environment named by config.json.
"""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_dual import (BODY_SLOTS, PCK_THRESHOLDS, assign, cleaned, hip_centre,
    torso_length)
from src.pose_edit.common import atomic_json, read_json, sha256_file

# The slot table, the SMPL-24 names and the landmark-block builder live in dual_fitting so that
# this report and the guide that a fitted run draws cannot drift apart.
from dual_fitting import DWPOSE_TO_JOINT, SMPL24_NAMES, joint_indices, person_from_joints

EXTRINSIC_CONVENTIONS = ('world_to_camera', 'camera_to_world')
# DWPose's neck is the shoulder midpoint; SMPL-X's neck is a spine-top joint several centimetres
# above it. That is a convention offset, not an error, so the neck normalises the torso but is
# excluded from every error statistic.
ERROR_SLOTS = np.array([slot for slot in DWPOSE_TO_JOINT if slot != 1])
# GATE: what the guide has to satisfy before a fitted skeleton may replace the ground-truth
# landmarks. The pose criteria are relative to the measured floor - the disagreement between the
# DWPose landmarks and the Hi4D geometry projected through the frame's own intrinsics - because
# asking an estimator to agree with DWPose better than the true geometry does is not a question
# about the estimator. An absolute threshold calibrated on generation quality was tried first and
# is unreachable for any estimator: the ground truth itself does not meet it.
# The hip-gap term is decided against the projected Hi4D geometry, not against DWPose. On an
# embrace DWPose can place both hip centres on the same pixel (measured: 4.2-4.4 px where the fit
# and the projected geometry both report ~72 px), which made the DWPose-referenced term the only
# gate failure (mean 0.120) while the same term against projected geometry was 0.021 (max 0.058).
# The DWPose-referenced value is still reported, as a statement about the guide, not about the fit.
GATE = {'correspondence_rate': ('min', 0.90), 'person_count_rate': ('min', 0.90),
    'hip_gap_error_norm_gt': ('max', 0.10), 'torso_scale_ratio': ('range', (0.85, 1.20))}
DIAGNOSTIC = {'hip_gap_error_norm': 'fit vs DWPose hip gap; a disagreement here can be the '
    'landmarks being wrong (backhug), so it never decides the gate'}


def hip_gap_errors(gap):
    """(deciding error against projected geometry, diagnostic error against DWPose)."""
    return (gap or {}).get('error_norm_gt'), (gap or {}).get('error_norm')
FLOOR_SLACK_PCK = 0.05
FLOOR_SLACK_MPJPE = 1.25


def to_camera(points_world, extrinsics, convention):
    rotation = np.asarray(extrinsics[:3, :3], dtype=float)
    translation = np.asarray(extrinsics[:3, 3], dtype=float)
    points = np.asarray(points_world, dtype=float)
    if convention == 'world_to_camera':
        return points @ rotation.T + translation
    return (points - translation) @ rotation.T


def project(points_camera, intrinsics):
    k = np.asarray(intrinsics, dtype=float)
    points = np.asarray(points_camera, dtype=float)
    depth = np.clip(points[..., 2], 1e-3, None)
    return np.stack([k[0, 0] * points[..., 0] / depth + k[0, 2],
        k[1, 1] * points[..., 1] / depth + k[1, 2]], axis=-1)


def comparison(reference, candidate):
    """2D error of one person's landmarks, measured in another person's own torso frame."""
    reference_points, candidate_points = reference['points'], candidate['points']
    torso = torso_length(reference_points)
    finite = (np.isfinite(reference_points[ERROR_SLOTS]).all(axis=1)
        & np.isfinite(candidate_points[ERROR_SLOTS]).all(axis=1))
    if torso is None or not finite.any():
        return {'comparable': False, 'reason': 'no shared valid body landmark'}
    distance = np.linalg.norm(reference_points[ERROR_SLOTS] - candidate_points[ERROR_SLOTS],
        axis=1)
    normalised = distance[finite] / torso
    return {'comparable': True, 'joints': int(finite.sum()), 'torso_px': torso,
        'mpjpe_px': float(distance[finite].mean()), 'mpjpe_norm': float(normalised.mean()),
        **{f'pck@{value}': float((normalised <= value).mean()) for value in PCK_THRESHOLDS}}


def hip_gap(points_a, points_b):
    centre_a, centre_b = hip_centre(points_a), hip_centre(points_b)
    if centre_a is None or centre_b is None:
        return None
    return float(np.linalg.norm(np.asarray(centre_a) - np.asarray(centre_b)))


def geometry_checks(camera_joints, names):
    """A correctly ordered skeleton puts the pelvis at the hip midpoint and the neck at the collar
    midpoint, so these distances verify the stored joint order instead of assuming it."""
    lookup = {str(name): index for index, name in enumerate(names)}
    if not {'pelvis', 'left_hip', 'right_hip'} <= set(lookup):
        return None
    checks = {'pelvis_to_hip_mid_m': float(np.linalg.norm(
        camera_joints[lookup['pelvis']] - camera_joints[[lookup['left_hip'],
            lookup['right_hip']]].mean(axis=0)))}
    if {'neck', 'left_collar', 'right_collar'} <= set(lookup):
        checks['neck_to_collar_mid_m'] = float(np.linalg.norm(
            camera_joints[lookup['neck']] - camera_joints[[lookup['left_collar'],
                lookup['right_collar']]].mean(axis=0)))
    return checks


def load_pair(pair_dir, threshold):
    pair = read_json(pair_dir / 'pair.json')
    cameras = np.load(pair['B']['camera_original'], allow_pickle=True)
    ids = [int(value) for value in np.asarray(cameras['ids'])]
    camera_id = int(pair['manifest']['camera_id'])
    if camera_id not in ids:
        raise KeyError(f'camera {camera_id} not among {ids}')
    index = ids.index(camera_id)
    geometry = np.load(pair['B']['geometry_original'], allow_pickle=True)
    smpl = np.load(pair['B']['smpl_original'], allow_pickle=True)
    image = pair_dir / pair['B']['image']
    with Image.open(image) as handle:
        size = handle.size
    intrinsics = np.asarray(geometry['K_runtime'], dtype=float)
    if not np.allclose(intrinsics[:2, :3],
            np.asarray(cameras['intrinsics'], dtype=float)[index][:2, :3], atol=1.0):
        raise ValueError('runtime intrinsics disagree with the camera file')
    people = []
    for slot in pair['B']['slot_order']:
        record = next(item for item in pair['B']['keypoints']
            if int(item['slot']) == int(slot['slot']))
        arrays = np.load(pair_dir / record['file'], allow_pickle=True)
        if (int(arrays['width']), int(arrays['height'])) != size:
            raise ValueError(f"keypoint frame {int(arrays['width'])}x{int(arrays['height'])} is "
                f'not the image size {size}')
        points, _ = cleaned(np.asarray(arrays['xy_native'], dtype=float),
            np.asarray(arrays['scores'], dtype=float), threshold, size)
        people.append({'slot': int(slot['slot']), 'hi4d_side': slot['hi4d_side'],
            'points': points})
    return {'name': pair_dir.name, 'kind': pair['kind'], 'dir': pair_dir, 'image': image,
        'image_sha256': sha256_file(image), 'size': size, 'camera_id': camera_id,
        'intrinsics': intrinsics,
        'extrinsics': np.asarray(cameras['extrinsics'], dtype=float)[index],
        'joints_world': np.asarray(smpl['joints_3d'], dtype=float), 'gt_people': people}


def ordered_by_error(people, reference_people):
    """Pair one set of people with another by joint error rather than by hip position.

    In an embrace the two hip centres can be a handful of pixels apart, so a position-based
    pairing can silently swap the people; the floor comparison would then measure person A
    against person B's landmarks and report a huge disagreement.
    """
    candidates = []
    for permutation in permutations(range(len(people)), r=len(reference_people)):
        cost = sum(comparison(reference_people[slot], people[index])
            .get('mpjpe_px', float('inf')) for slot, index in enumerate(permutation))
        candidates.append((cost, [int(value) for value in permutation]))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    best_cost, best = candidates[0]
    margin = candidates[1][0] - best_cost if len(candidates) > 1 else None
    if not np.isfinite(best_cost):
        return None, None
    return best, {'cost_px': best_cost, 'cost_margin_to_next': margin,
        'basis': 'summed body-landmark error, both permutations tried'}


def project_ground_truth(record):
    """Project the Hi4D SMPL joints into the photograph.

    Nothing in the asset records says which way the stored extrinsics act, and guessing wrong
    would move the whole reference frame silently, so the convention is chosen by how well the
    projection lands on the DWPose landmarks drawn on that same photograph. The projected people
    are then ordered to match the guide's slots.
    """
    best = None
    cost_by_convention = {}
    for convention in EXTRINSIC_CONVENTIONS:
        camera = [to_camera(joints, record['extrinsics'], convention)
            for joints in record['joints_world']]
        people = [person_from_joints(project(one, record['intrinsics']), SMPL24_NAMES,
            record['size']) for one in camera]
        mapping, detail = ordered_by_error(people, record['gt_people'])
        if mapping is None:
            cost_by_convention[convention] = None
            continue
        cost_by_convention[convention] = detail['cost_px']
        if best is None or detail['cost_px'] < best['cost_px']:
            best = {'convention': convention, 'cost_px': detail['cost_px'],
                'assignment': detail,
                'people': [people[mapping[slot]] for slot in range(len(mapping))],
                'camera_joints': [camera[mapping[slot]] for slot in range(len(mapping))]}
    if best is None:
        return None
    best['cost_px_by_convention'] = cost_by_convention
    best['checks'] = [geometry_checks(one, SMPL24_NAMES) for one in best['camera_joints']]
    best['floor'] = [comparison(gt, projected)
        for gt, projected in zip(record['gt_people'], best['people'])]
    return best


def run_fits(records, config, out_dir, det_thresh, force):
    """Ask the multi-HMR worker only for pairs without a fit of the current image."""
    manifest_path = out_dir / 'fit_manifest.json'
    cached = {}
    if manifest_path.is_file() and not force:
        cached = {item['name']: item for item in read_json(manifest_path).get('records', [])
            if item.get('failure') is None and item.get('npz')}
    by_name = dict(cached)
    todo = []
    for record in records:
        usable = (record['name'] in cached and (out_dir / cached[record['name']]['npz']).is_file()
            and cached[record['name']].get('image_sha256') == record['image_sha256'])
        if usable:
            continue
        todo.append(record)
        by_name.pop(record['name'], None)
    if not todo:
        return by_name, None
    request_path = out_dir / 'fit_request.json'
    atomic_json(request_path, {'project_root': str(REPO_ROOT), 'out_dir': str(out_dir),
        'det_thresh': det_thresh, 'images': [{'name': record['name'], 'image': str(record['image']),
            'K_native': record['intrinsics'].tolist()} for record in todo]})
    log_path = out_dir / 'fit_worker.log'
    print(f'MULTIHMR_FIT {len(todo)} image(s), log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.run([config['multihmr_python'], '-u', str(ROOT / 'fit_dual_worker.py'),
            '--request', str(request_path)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
        raise RuntimeError(f'fit_dual_worker.py exited {process.returncode}.\n{tail}')
    manifest = read_json(manifest_path)
    refreshed = {item['name']: item for item in manifest['records']}
    merged = {name: item for name, item in cached.items() if name not in refreshed}
    merged.update(refreshed)
    return merged, manifest['provenance']


def score_pair(record, fit, out_dir):
    if fit is None:
        return {'name': record['name'], 'kind': record['kind'], 'available': False,
            'reason': 'the fit worker reported no result for this pair'}
    if fit.get('failure'):
        return {'name': record['name'], 'kind': record['kind'], 'available': False,
            'reason': fit['failure']}
    arrays = np.load(out_dir / fit['npz'], allow_pickle=True)
    names = [str(value) for value in arrays['joint_names']]
    j2d, j3d = np.asarray(arrays['j2d_native'], float), np.asarray(arrays['j3d'], float)
    fitted = [person_from_joints(j2d[index], names, record['size']) for index in range(len(j2d))]
    result = {'name': record['name'], 'kind': record['kind'], 'available': True,
        'person_count': int(len(fitted)), 'reproject_median_px': fit.get('reproject_median_px'),
        'people': [], 'correspondence': None, 'hip_gap': None, 'error_3d': None}
    mapping, detail = assign(fitted, [person['points'] for person in record['gt_people']],
        0.5, record['size'])
    result['assignment'] = detail
    if mapping is None:
        result['reason'] = 'no hip-centre based assignment was possible'
        return result
    for slot, index in enumerate(mapping):
        person = {'gt_slot': record['gt_people'][slot]['slot'],
            'hi4d_side': record['gt_people'][slot]['hi4d_side'], 'fit_index': int(index),
            'vs_dwpose': comparison(record['gt_people'][slot], fitted[index])}
        reference_torso = torso_length(record['gt_people'][slot]['points'])
        fit_torso = torso_length(fitted[index]['points'])
        # A fit drawn at the wrong pixel scale (a padding or intrinsics mistake, not an estimator
        # error) shows up here as 0.7x or 1.4x before it can be misread as a bad pose estimate.
        person['torso_scale_ratio'] = (fit_torso / reference_torso) if reference_torso and fit_torso else None
        if record.get('ground_truth'):
            # Where the two references themselves disagree, this says which one the fit follows;
            # without it a bad ground-truth landmark set would look like a bad estimator.
            person['vs_smpl_projected'] = comparison(
                record['ground_truth']['people'][slot], fitted[index])
        result['people'].append(person)
    if len(fitted) == 2:
        result['hip_gap'] = {'dwpose_px': hip_gap(record['gt_people'][0]['points'],
            record['gt_people'][1]['points']),
            'fit_px': hip_gap(fitted[mapping[0]]['points'], fitted[mapping[1]]['points'])}
        if result['hip_gap']['dwpose_px'] and result['hip_gap']['fit_px']:
            error = abs(result['hip_gap']['fit_px'] - result['hip_gap']['dwpose_px'])
            result['hip_gap']['error_px'] = error
            result['hip_gap']['error_relative'] = error / result['hip_gap']['dwpose_px']
            # In an embrace the two ground-truth hip centres can sit a few pixels apart, which
            # makes a relative error explode on a near-zero denominator; the torso-normalised
            # error is the one the gate uses.
            torsos = [value for value in
                (torso_length(person['points']) for person in record['gt_people']) if value]
            if torsos:
                result['hip_gap']['norm_torso_px'] = float(np.mean(torsos))
                result['hip_gap']['error_norm'] = error / result['hip_gap']['norm_torso_px']
        costs = [sum(comparison(record['gt_people'][slot], fitted[permutation[slot]])
            .get('mpjpe_px', float('inf')) for slot in range(2))
            for permutation in ([0, 1], [1, 0])]
        hip_assignment = [int(value) for value in mapping]
        best_assignment = [0, 1] if costs[0] <= costs[1] else [1, 0]
        result['correspondence'] = {'hip_assignment': hip_assignment,
            'error_assignment': best_assignment, 'correct': hip_assignment == best_assignment,
            'cost_hip_px': costs[0] if hip_assignment == [0, 1] else costs[1],
            'cost_best_px': min(costs), 'ambiguous': (detail or {}).get('ambiguous')}
        # Which reference the fit follows matters where the two disagree: in a backhug the DWPose
        # landmarks can put both hip centres on the same pixel, which the true geometry does not.
        # Without this column a fit that agrees with the geometry looks like the failure.
        if record.get('ground_truth'):
            result['hip_gap']['projected_gt_px'] = hip_gap(
                record['ground_truth']['people'][0]['points'],
                record['ground_truth']['people'][1]['points'])
            gap = result['hip_gap']
            if gap.get('norm_torso_px') and gap['projected_gt_px'] is not None:
                gap['error_norm_gt'] = abs(gap['fit_px'] - gap['projected_gt_px']) / gap['norm_torso_px']
            result['error_3d'] = []
            indices, smpl_indices = joint_indices(names), joint_indices(SMPL24_NAMES)
            pelvis_fit, pelvis_smpl = names.index('pelvis'), SMPL24_NAMES.index('pelvis')
            shared = [(smpl_indices[slot], indices[slot]) for slot in sorted(DWPOSE_TO_JOINT)]
            for slot, index in enumerate(mapping):
                gt_camera, fit_camera = record['ground_truth']['camera_joints'][slot], j3d[index]
                offset_gt, offset_fit = gt_camera[pelvis_smpl], fit_camera[pelvis_fit]
                pairs = [(gt_camera[a], fit_camera[b]) for a, b in shared]
                entry = {
                    'mpjpe3d_absolute_m': float(np.mean(
                        [np.linalg.norm(a - b) for a, b in pairs])),
                    'mpjpe3d_root_aligned_m': float(np.mean([np.linalg.norm(
                        (a - offset_gt) - (b - offset_fit)) for a, b in pairs])),
                    'hip_gap_m': {'ground_truth': float(np.linalg.norm(
                        gt_camera[smpl_indices[8]] - gt_camera[smpl_indices[11]])),
                        'fit': float(np.linalg.norm(
                            fit_camera[indices[8]] - fit_camera[indices[11]]))}}
                entry['hip_gap_m']['error_m'] = abs(
                    entry['hip_gap_m']['ground_truth'] - entry['hip_gap_m']['fit'])
                result['error_3d'].append(entry)
    return result


def mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return float(np.mean(values)) if values else None


def aggregate(results, records):
    scored = [item for item in results if item.get('available') and item.get('people')]
    fits = [person['vs_dwpose'] for item in scored for person in item['people']]
    floors = [floor for record in records if record.get('ground_truth')
        for floor in record['ground_truth']['floor']]
    keys = ['pck@0.1', 'pck@0.2', 'pck@0.5', 'mpjpe_px', 'mpjpe_norm']
    summary = {'pairs_total': len(results), 'pairs_scored': len(scored),
        'unavailable': [item for item in results if not item.get('available')],
        'fit_vs_dwpose': {key: mean([person.get(key) for person in fits]) for key in keys},
        'fit_vs_smpl_projected': {key: mean([person['vs_smpl_projected'].get(key)
            for item in scored for person in item['people']
            if person.get('vs_smpl_projected')]) for key in keys},
        'torso_scale_ratio': mean([person.get('torso_scale_ratio') for item in scored
            for person in item['people']]),
        'floor_dwpose_vs_smpl': {key: mean([person.get(key) for person in floors]) for key in keys},
        'person_count_rate': mean([1.0 if item['person_count'] == 2 else 0.0
            for item in results if item.get('available')]),
        'correspondence_rate': mean([1.0 if (item.get('correspondence') or {}).get('correct')
            else 0.0 for item in scored]),
        'hip_gap_error_relative': mean([(item.get('hip_gap') or {}).get('error_relative')
            for item in scored]),
        'hip_gap_error_norm': mean([hip_gap_errors(item.get('hip_gap'))[1] for item in scored]),
        'hip_gap_error_norm_gt': mean([hip_gap_errors(item.get('hip_gap'))[0] for item in scored]),
        'mpjpe3d_absolute_m': mean([person['mpjpe3d_absolute_m'] for item in scored
            for person in (item.get('error_3d') or [])]),
        'mpjpe3d_root_aligned_m': mean([person['mpjpe3d_root_aligned_m'] for item in scored
            for person in (item.get('error_3d') or [])]),
        'hip_gap_error_m': mean([person['hip_gap_m']['error_m'] for item in scored
            for person in (item.get('error_3d') or [])]),
        'reproject_median_px': mean([item.get('reproject_median_px') for item in results
            if item.get('available')]),
        'conventions': sorted({(record.get('ground_truth') or {}).get('convention')
            for record in records if record.get('ground_truth')}),
        'geometry_checks': [check for record in records if record.get('ground_truth')
            for check in record['ground_truth']['checks']]}
    measured = {'correspondence_rate': summary['correspondence_rate'],
        'person_count_rate': summary['person_count_rate'],
        'hip_gap_error_norm_gt': summary['hip_gap_error_norm_gt'],
        'torso_scale_ratio': summary['torso_scale_ratio']}
    verdict = {}
    for key, (direction, limit) in GATE.items():
        value = measured[key]
        if value is None:
            passed = False
        elif direction == 'min':
            passed = value >= limit
        elif direction == 'max':
            passed = value <= limit
        else:
            passed = limit[0] <= value <= limit[1]
        verdict[key] = {'measured': value, 'limit': list(limit) if isinstance(limit, tuple) else limit,
            'direction': direction, 'pass': bool(passed)}
    for key, note in DIAGNOSTIC.items():
        print(f'DIAGNOSTIC {key}={number(summary.get(key))} -- {note}')
    fit, floor = summary['fit_vs_dwpose'], summary['floor_dwpose_vs_smpl']
    limits = {'pck@0.2': (floor['pck@0.2'] - FLOOR_SLACK_PCK if floor['pck@0.2'] is not None else None,
            'min', f'floor {number(floor["pck@0.2"])} - {FLOOR_SLACK_PCK}'),
        'mpjpe_norm': (floor['mpjpe_norm'] * FLOOR_SLACK_MPJPE if floor['mpjpe_norm'] is not None else None,
            'max', f'floor {number(floor["mpjpe_norm"])} x {FLOOR_SLACK_MPJPE}')}
    for key, (limit, direction, stated) in limits.items():
        value = fit[key]
        verdict[f'fit_vs_dwpose.{key}'] = {'measured': value, 'limit': limit,
            'direction': direction, 'basis': stated,
            'pass': bool(value is not None and limit is not None
                and (value >= limit if direction == 'min' else value <= limit))}
    verdict['all_pass'] = all(item['pass'] for key, item in verdict.items() if key != 'all_pass')
    summary['gate'] = verdict
    return summary


def number(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def dumps(value, indent=None):
    return json.dumps(value, indent=indent, default=float, ensure_ascii=False)


def print_table(results, records):
    by_name = {record['name']: record for record in records}
    header = (f'{"pair":40} {"kind":14} ppl  pck.1 pck.2 mpjp_n gap_px gap_rel corr '
        f'abs3d_m gap3d_m floor.2 reproj_px')
    print(header)
    print('-' * len(header))
    for item in results:
        record = by_name.get(item['name'])
        if not item.get('available'):
            print(f'{item["name"]:40} {item["kind"]:14} -- {item.get("reason", "")}')
            continue
        people = item['people']
        floor = (record or {}).get('ground_truth', {}).get('floor', [])
        row = {'pair': item['name'], 'kind': item['kind'],
            'ppl': item['person_count'],
            'pck.1': mean([person['vs_dwpose'].get('pck@0.1') for person in people]),
            'pck.2': mean([person['vs_dwpose'].get('pck@0.2') for person in people]),
            'mpjp_n': mean([person['vs_dwpose'].get('mpjpe_norm') for person in people]),
            'gap_px': (item.get('hip_gap') or {}).get('dwpose_px'),
            'gap_rel': (item.get('hip_gap') or {}).get('error_relative'),
            'corr': 1.0 if (item.get('correspondence') or {}).get('correct') else 0.0,
            'abs3d_m': mean([person['mpjpe3d_absolute_m'] for person in (item.get('error_3d') or [])]),
            'gap3d_m': mean([person['hip_gap_m']['error_m'] for person in (item.get('error_3d') or [])]),
            'floor.2': mean([person.get('pck@0.2') for person in floor]),
            'reproj_px': item.get('reproject_median_px')}
        print(f'{row["pair"]:40} {str(row["kind"]):14} {row["ppl"]:>3} '
            f'{number(row["pck.1"]):>5} {number(row["pck.2"]):>5} {number(row["mpjp_n"]):>6} '
            f'{number(row["gap_px"], 1):>6} {number(row["gap_rel"]):>7} {number(row["corr"], 1):>4} '
            f'{number(row["abs3d_m"]):>7} {number(row["gap3d_m"]):>7} {number(row["floor.2"]):>7} '
            f'{number(row["reproj_px"], 2):>9}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--pairs', type=Path, nargs='*',
        default=[ROOT / 'pairs', ROOT / 'pairs_cross'])
    parser.add_argument('--out', type=Path, default=ROOT / 'fits')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--det-thresh', type=float, default=0.2)
    parser.add_argument('--force', action='store_true', help='refit even when a cached fit exists')
    parser.add_argument('--dry-run', action='store_true',
        help='load the pairs and check the ground-truth projection only')
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
    records, skipped = [], []
    for directory in directories:
        try:
            record = load_pair(directory, config['keypoint_score_threshold'])
            record['ground_truth'] = project_ground_truth(record)
            records.append(record)
        except Exception as error:
            skipped.append({'pair': directory.name,
                'reason': f'{type(error).__name__}: {error}'})
    print(f'PAIRS_LOADED {len(records)}, skipped {len(skipped)}', flush=True)
    for item in skipped:
        print(f'  SKIP {item["pair"]}: {item["reason"]}', flush=True)
    for record in records:
        if not record['ground_truth']:
            print(f'  NO_PROJECTION {record["name"]}', flush=True)
    if args.dry_run:
        for record in records:
            projection = record['ground_truth'] or {}
            print(dumps({'pair': record['name'], 'size': list(record['size']),
                'convention': projection.get('convention'),
                'cost_px_by_convention': projection.get('cost_px_by_convention'),
                'floor': projection.get('floor'), 'checks': projection.get('checks'),
                'dwpose_hip_gap_px': hip_gap(*[person['points'] for person in record['gt_people']])}))
        return
    fits, provenance = run_fits(records, config, args.out, args.det_thresh, args.force)
    results = []
    for record in records:
        try:
            results.append(score_pair(record, fits.get(record['name']), args.out))
        except Exception as error:  # one unreadable fit must not lose the other sixteen
            results.append({'name': record['name'], 'kind': record['kind'], 'available': False,
                'reason': f'scoring failed: {type(error).__name__}: {error}'})
    summary = aggregate(results, records)
    print_table(results, records)
    print('\nFIT_VS_GROUND_TRUTH ' + dumps(summary, indent=2))
    print('\nGATE')
    for key, item in summary['gate'].items():
        if key == 'all_pass':
            continue
        print(f'  {"PASS" if item["pass"] else "FAIL"} {key} {item["direction"]} {item["limit"]} '
            f'measured {number(item["measured"])}')
    print(f'  VERDICT {"PASS" if summary["gate"]["all_pass"] else "FAIL"}: '
        + ('the fitted skeleton is accurate enough to guide generation'
            if summary['gate']['all_pass']
            else 'keep the ground-truth guide; a fit this noisy cannot be attributed separately'))
    atomic_json(args.out / 'fit_dual_report.json',
        {'provenance': provenance, 'skipped': skipped, 'summary': summary, 'results': results})


if __name__ == '__main__':
    main()
