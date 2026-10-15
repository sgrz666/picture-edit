#!/usr/bin/env python3
"""Estimate B and export camera-consistent 3D joints and projected pose guides."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

# The repo-level shared package lives one directory up from this workflow.
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import atomic_json, finish_stage, read_json, runtime_paths
from src.pose_edit.estimator import estimate_single, project, reconstruct
from src.pose_edit.geometry import (align_to_source_bbox, choose_subject, render_poseguide,
    smplx137_to_dwpose134, valid_points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    settings, paths = request['identity']['settings'], runtime_paths(request)
    run_dir = Path(paths['run_dir'])
    stage_dir = run_dir / 'geometry'
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    smpl_root = Path(settings['smplest_root'])
    from ultralytics import YOLO

    detector = YOLO(str(smpl_root / 'pretrained_models/yolov8x.pt'))
    subject_rule = settings.get('subject', 'auto')
    detections = {}
    for role in ('A', 'B'):
        result = detector.predict(source=paths[f'{role}_rgb'], classes=[0],
            conf=settings['person_detection_confidence'], device='cuda', verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        chosen = choose_subject(boxes, confidence, subject_rule)
        detections[role] = {'bbox_xyxy': boxes[chosen].tolist(), 'score': float(confidence[chosen]),
            'persons_detected': int(len(boxes)), 'subject_rule': subject_rule,
            'subject_index': int(chosen), 'all_boxes_xyxy': boxes.tolist(),
            'all_scores': [float(value) for value in confidence]}
    del detector, result
    torch.cuda.empty_cache()

    params, staging = estimate_single(paths['B_rgb'], stage_dir, settings, smpl_root)
    saved_params = stage_dir / 'B_smplx.pt'
    shutil.copyfile(staging['estimator_params'], saved_params)
    joints_cam, mesh_max_error = reconstruct(params, smpl_root)
    projected, camera_checks = project(params, joints_cam)
    projection_error = camera_checks['projection_max_error_px']
    focal = np.asarray(camera_checks['focal_xy'])
    principal = np.asarray(camera_checks['principal_xy'])

    points = smplx137_to_dwpose134(projected)
    points_3d = smplx137_to_dwpose134(joints_cam)
    # The estimator runs its own detection internally, so confirm it fitted the person we chose.
    subject_box = np.asarray(detections['B']['bbox_xyxy'], dtype=float)
    finite = np.isfinite(points).all(axis=1)
    centroid = points[finite].mean(axis=0) if finite.any() else np.array([np.nan, np.nan])
    margin = 0.15 * np.array([subject_box[2] - subject_box[0], subject_box[3] - subject_box[1]])
    inside_subject = bool(np.isfinite(centroid).all()
        and subject_box[0] - margin[0] <= centroid[0] <= subject_box[2] + margin[0]
        and subject_box[1] - margin[1] <= centroid[1] <= subject_box[3] + margin[1])
    if not inside_subject:
        raise RuntimeError(f'the estimator fitted a person outside the selected subject box '
            f'{subject_box.tolist()}: joint centroid {centroid.tolist()}. --subject cannot '
            'redirect the estimator itself, so with several people in B the pose would belong '
            'to somebody else')
    source_meta = read_json(paths['source_preprocess'])
    bbox_a = np.asarray(detections['A']['bbox_xyxy'])
    sx, sy = source_meta['scale_xy']
    ox, oy = source_meta['offset_xy']
    bbox_a = bbox_a * np.array([sx, sy, sx, sy]) + np.array([ox, oy, ox, oy])
    aligned, transform = align_to_source_bbox(points, detections['B']['bbox_xyxy'], bbox_a,
        settings.get('pose_framing', 'source_person'))
    with Image.open(paths['B_rgb']) as image:
        b_image = image.convert('RGB')
    with Image.open(paths['A_reference']) as image:
        a_image = image.convert('RGB')
    native_valid = valid_points(points, *b_image.size)
    # Never move a joint outside the actual B frame into view through alignment.
    aligned[~native_valid] = np.nan
    guide_native = render_poseguide(points, b_image.size)
    guide = render_poseguide(aligned, a_image.size)
    guide_native.save(stage_dir / 'B_poseguide_native.png')
    guide.save(stage_dir / 'B_3d_dwpose.png')
    Image.blend(b_image, guide_native, 0.45).save(stage_dir / 'B_poseguide_overlay.png')
    Image.blend(a_image, guide, 0.45).save(stage_dir / 'A_poseguide_alignment.png')
    np.savez_compressed(stage_dir / 'B_keypoints.npz', smplx137_cam=joints_cam,
        dwpose134_cam=points_3d, keypoints_B_px=points, keypoints_generation_px=aligned,
        valid_B=native_valid, valid_generation=valid_points(aligned, *a_image.size),
        source_person_bbox_canvas=bbox_a.tolist())
    camera = {'focal_xy': focal.tolist(), 'principal_xy': principal.tolist(),
        'B_size': list(b_image.size), 'generation_canvas': list(a_image.size),
        'B_to_generation': transform, 'source_letterbox': source_meta,
        'coordinate_system': 'camera XYZ; pixels x-right y-down; projection uses Z+1e-4',
        'detections': detections, 'mesh_max_error_m': mesh_max_error,
        'projection_max_error_px': projection_error, 'valid_B_joints': int(native_valid.sum()),
        'visibility_policy': 'finite positive depth and in-frame only; not an occlusion confidence',
        'estimated_hidden_joints': True, 'native_estimator_input': staging['staged_input'],
        'native_estimator_params': staging['estimator_params'],
        'estimator_root': staging['estimator_root'],
        'subject_consistency': {'B_joint_centroid_px': centroid.tolist(),
            'inside_selected_box': inside_subject}}
    atomic_json(stage_dir / 'B_camera.json', camera)
    outputs = {'B_smplx': saved_params, 'B_keypoints': stage_dir / 'B_keypoints.npz',
        'B_camera': stage_dir / 'B_camera.json', 'B_3d_dwpose': stage_dir / 'B_3d_dwpose.png',
        'B_poseguide_native': stage_dir / 'B_poseguide_native.png',
        'B_overlay': stage_dir / 'B_poseguide_overlay.png',
        'A_alignment': stage_dir / 'A_poseguide_alignment.png'}
    finish_stage(stage_dir / 'stage.json', request, outputs,
        {'seconds': time.perf_counter() - started, 'camera_checks': camera,
         'poseguide_kind': 'SMPL-X 3D joints projected to DWPose-style 2D RGB guide'}, run_dir)
    print(f'GEOMETRY_COMPLETE {stage_dir}', flush=True)


if __name__ == '__main__':
    main()
