#!/usr/bin/env python3
"""Re-estimate the pose of a generated image and score it against the guide that was requested.

Run with the geometry interpreter, e.g.
    $(python -c "import json;print(json.load(open('config.json'))['geometry_python'])") evaluate.py --run hard_01

The score answers one question only: did the generator move the body to where the skeleton
guide asked? Identity, garment and scene preservation are not measured here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from pose_transfer.common import (atomic_json, finish_stage, make_request, model_signature,
    read_json, sha256_file, stage_reasons)
from pose_transfer.estimator import estimate_single, project, reconstruct
from pose_transfer.geometry import render_poseguide, smplx137_to_dwpose134, valid_points

ROOT = Path(__file__).resolve().parent
ESTIMATOR_SOURCE_FILES = ['main/inference.py', 'main/base.py', 'models/SMPLest_X.py',
    'human_models/human_models.py', 'scripts/run_smplestx_params_only.py']
CODE = ['evaluate.py', 'pose_transfer/common.py', 'pose_transfer/geometry.py',
    'pose_transfer/estimator.py']
GROUPS = {'body': slice(0, 18), 'feet': slice(18, 24), 'face': slice(24, 92),
    'left_hand': slice(92, 113), 'right_hand': slice(113, 134)}
PCK_THRESHOLDS = (0.2, 0.5)


def group_metrics(errors, torso):
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {'joints': 0}
    normalised = finite / torso
    return {'joints': int(finite.size),
        'mpjpe_px': float(finite.mean()),
        'mpjpe_norm': float(normalised.mean()),
        'p95_norm': float(np.percentile(normalised, 95)),
        **{f'pck@{threshold}': float((normalised <= threshold).mean())
            for threshold in PCK_THRESHOLDS},
        'worst_joint_offset_px': float(finite.max())}


def torso_scale(guide, valid):
    """Neck-to-hip-centre length of the requested pose; the usual PCK normalisation."""
    hips = np.nanmean(guide[[8, 11]], axis=0)
    if valid[1] and np.isfinite(hips).all():
        length = float(np.linalg.norm(guide[1] - hips))
        if length > 1:
            return length
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run directory name under workflow/runs')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = read_json(ROOT / 'config.json')
    run_dir = ROOT / 'runs' / args.run
    generation = run_dir / 'generation'
    geometry = run_dir / 'geometry'
    final_image = generation / 'final_image.png'
    keypoints = geometry / 'B_keypoints.npz'
    for required in (final_image, keypoints, generation / 'generation.json'):
        if not required.is_file():
            raise FileNotFoundError(f'{args.run} has no {required.name}; run the workflow first')

    smpl_root = Path(config['smplest_root'])
    settings = {key: config[key] for key in
        ('smplest_root', 'smplest_checkpoint', 'person_detection_confidence')}
    identity = {'settings': settings,
        'inputs_sha256': {'final_image': sha256_file(final_image),
            'B_keypoints': sha256_file(keypoints)},
        'model_identity': {'checkpoint': model_signature(
                smpl_root / 'pretrained_models' / config['smplest_checkpoint']),
            'detector': model_signature(smpl_root / 'pretrained_models/yolov8x.pt'),
            'human_models': model_signature(smpl_root / 'human_models/human_model_files/smplx'),
            'estimator_source': {name: sha256_file(smpl_root / name)
                for name in ESTIMATOR_SOURCE_FILES}},
        'code': {name: sha256_file(ROOT / name) for name in CODE}}
    request = make_request('evaluate', identity, run_dir,
        {'run_dir': str(run_dir), 'final_image': final_image, 'B_keypoints': keypoints})
    stage_dir = run_dir / 'evaluation'
    record = stage_dir / 'stage.json'
    if not args.force:
        reasons = stage_reasons(record, request, run_dir)
        if not reasons:
            print(f'EVALUATE CACHE_HIT {args.run}', flush=True)
            return
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    from ultralytics import YOLO
    detector = YOLO(str(smpl_root / 'pretrained_models/yolov8x.pt'))
    detection = detector.predict(source=str(final_image), classes=[0],
        conf=settings['person_detection_confidence'], device='cuda', verbose=False)[0]
    persons = int(len(detection.boxes))
    del detector, detection

    params, staging = estimate_single(final_image, stage_dir, settings, smpl_root)
    saved_params = stage_dir / 'final_smplx.pt'
    shutil.copyfile(staging['estimator_params'], saved_params)
    joints_cam, mesh_error = reconstruct(params, smpl_root)
    projected, camera_checks = project(params, joints_cam)
    estimated = smplx137_to_dwpose134(projected)

    arrays = np.load(keypoints)
    guide_canvas = arrays['keypoints_generation_px'].astype(float)
    guide_valid = arrays['valid_generation'].astype(bool)
    preprocess = read_json(run_dir / 'inputs' / 'source_preprocess.json')
    offset = np.asarray(preprocess['content_box'][:2], dtype=float)
    with Image.open(final_image) as image:
        final_size = image.size
    guide = guide_canvas - offset
    guide_valid &= valid_points(guide, *final_size)
    estimated_valid = valid_points(estimated, *final_size)

    scale = torso_scale(guide, guide_valid)
    if scale is None:
        scale = float(np.hypot(*final_size)) / 4
    both = guide_valid & estimated_valid
    distances = np.linalg.norm(guide - estimated, axis=1)
    distances[~both] = np.nan
    groups = {name: group_metrics(distances[block], scale) for name, block in GROUPS.items()}
    body = groups['body']
    hands = [groups['left_hand'].get('mpjpe_norm'), groups['right_hand'].get('mpjpe_norm')]
    hands = [value for value in hands if value is not None]
    summary = {'body_pck@0.2': body.get('pck@0.2'), 'body_mpjpe_norm': body.get('mpjpe_norm'),
        'hand_mpjpe_norm': float(np.mean(hands)) if hands else None,
        'face_mpjpe_norm': groups['face'].get('mpjpe_norm'),
        'persons_detected': float(persons)}
    evaluation = {'schema': 'single_person_pose_evaluation_v1', 'run': args.run,
        'final_image': str(final_image), 'final_size': list(final_size),
        'guide_offset_from_canvas': offset.tolist(),
        'torso_normalisation_px': scale, 'persons_detected': persons,
        'joints_compared': int(both.sum()), 'joints_requested': int(guide_valid.sum()),
        'joints_estimated': int(estimated_valid.sum()),
        'summary': summary, 'groups': groups,
        'camera_checks': camera_checks, 'mesh_max_error_m': mesh_error,
        'estimator_staging': staging,
        'limits': 'Pose fidelity only: it measures where the generated body landed relative to '
            'the requested skeleton, using the same estimator for both sides. It says nothing '
            'about identity, clothing or scene preservation. A guide joint that the estimator '
            'placed outside the frame is excluded rather than scored as an error. When '
            'persons_detected is greater than 1 the estimator picks its own person in the '
            'generated image, which is not guaranteed to be the subject chosen with --subject.'}
    atomic_json(stage_dir / 'eval.json', evaluation)
    np.savez_compressed(stage_dir / 'final_keypoints.npz', dwpose134_px=estimated,
        valid=estimated_valid, guide_final_px=guide, guide_valid=guide_valid,
        distances_px=distances, smplx137_cam=joints_cam)
    guide_layer = render_poseguide(np.where(guide_valid[:, None], guide, np.nan), final_size)
    estimated_layer = render_poseguide(np.where(estimated_valid[:, None], estimated, np.nan),
        final_size)
    with Image.open(final_image) as image:
        base = image.convert('RGB')
    overlay = Image.blend(Image.blend(base, guide_layer, 0.35), estimated_layer, 0.35)
    overlay.save(stage_dir / 'eval_overlay.png')
    outputs = {'eval': stage_dir / 'eval.json', 'final_keypoints': stage_dir / 'final_keypoints.npz',
        'final_smplx': saved_params, 'overlay': stage_dir / 'eval_overlay.png'}
    finish_stage(record, request, outputs,
        {'seconds': time.perf_counter() - started, 'summary': summary}, run_dir)
    print('EVALUATE_COMPLETE ' + ' '.join(f'{key}={value:.3f}'
        for key, value in summary.items() if value is not None), flush=True)


if __name__ == '__main__':
    main()
