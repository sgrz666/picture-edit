"""The single way this workflow calls the vendored SMPLest-X estimator.

Both the pose-transfer geometry stage and the evaluation stage estimate one image, so the
sandboxing, staging, parameter validation and camera reconstruction live here once.
"""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image
import torch

from src.pose_edit.common import create_estimator_sandbox
from src.pose_edit.geometry import project_camera

REQUIRED_PARAMETER_KEYS = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
    'jaw_pose', 'betas', 'expression', 'mesh_cam', 'joint_proj_model', 'transl',
    'focal_length_xy', 'princpt']
MESH_TOLERANCE_M = 1e-4
PROJECTION_TOLERANCE_PX = 0.25


def estimate_single(image_path, work_dir, settings, smpl_root):
    """Estimate one image in a run-local sandbox and return validated parameters.

    The shared estimator checkout is only read from: its root-sensitive entry points are
    copied next to the run directory, while immutable weights stay symlinked.
    """
    job = f'workflow_{uuid.uuid4().hex[:16]}'
    smpl_root = Path(smpl_root)
    estimator_root = create_estimator_sandbox(smpl_root, Path(work_dir) / 'estimator' / job)
    input_dir = estimator_root / 'demo/input_frames' / job
    output_dir = estimator_root / 'demo/output_frames' / job
    input_dir.mkdir(parents=True, exist_ok=False)
    staged = input_dir / '000001.jpg'
    with Image.open(image_path) as image:
        image.convert('RGB').save(staged, quality=100, subsampling=0)
    command = [sys.executable, str(estimator_root / 'scripts/run_smplestx_params_only.py'),
        '--num_gpus', '1', '--file_name', job, '--ckpt_name', settings['smplest_checkpoint'],
        '--start', '1', '--end', '1']
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(estimator_root)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['SMPLESTX_DET_CONF'] = str(settings['person_detection_confidence'])
    for key in ('CUDA_HOME', 'CUDA_PATH', 'CUDACXX', 'LD_LIBRARY_PATH'):
        environment.pop(key, None)
    subprocess.run(command, cwd=estimator_root, env=environment, check=True)
    params_path = output_dir / f'{job}_params.pt'
    if not params_path.is_file():
        raise RuntimeError(f'SMPLest-X returned no parameters for {image_path}')
    params = torch.load(params_path, map_location='cpu', weights_only=False)
    for key in REQUIRED_PARAMETER_KEYS:
        if key not in params or params[key].shape[0] != 1 or not torch.isfinite(params[key]).all():
            raise RuntimeError(f'Invalid single-frame SMPL-X parameter: {key}')
    return params, {'estimator_root': str(estimator_root), 'staged_input': str(staged),
        'estimator_params': str(params_path)}


def reconstruct(params, smpl_root):
    """Rebuild camera-space joints with the estimator's own neutral SMPL-X model."""
    smpl_root = Path(smpl_root)
    if str(smpl_root) not in sys.path:
        sys.path.insert(0, str(smpl_root))
    from human_models.human_models import SMPLX
    smpl_model = SMPLX(str(smpl_root / 'human_models/human_model_files'))
    layer = smpl_model.layer['neutral'].cpu().eval()
    with torch.inference_mode():
        body = layer(betas=params['betas'], body_pose=params['body_pose'],
            global_orient=params['global_orient'], left_hand_pose=params['left_hand_pose'],
            right_hand_pose=params['right_hand_pose'], jaw_pose=params['jaw_pose'],
            expression=params['expression'], transl=params['transl'],
            leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3))
    joints_cam = body.joints[0, list(smpl_model.joint_idx)].numpy()
    mesh_max_error = float((body.vertices - params['mesh_cam']).abs().max())
    if mesh_max_error > MESH_TOLERANCE_M:
        raise RuntimeError(f'Reconstructed mesh differs from exported camera-space mesh by '
            f'{mesh_max_error}m')
    return joints_cam, mesh_max_error


def project(params, joints_cam):
    """Project reconstructed joints into image pixels and cross-check the estimator's own crop."""
    focal = params['focal_length_xy'][0].numpy()
    principal = params['princpt'][0].numpy()
    projected = project_camera(joints_cam, focal, principal)
    crop_points = params['joint_proj_model'][0].numpy()
    crop_width, crop_height = focal * np.array([192 / 5000, 256 / 5000])
    original_projection = np.stack([
        principal[0] - crop_width / 2 + crop_points[:, 0] / 12 * crop_width,
        principal[1] - crop_height / 2 + crop_points[:, 1] / 16 * crop_height], axis=1)
    projection_error = float(np.abs(projected - original_projection).max())
    if not np.isfinite(projection_error) or projection_error > PROJECTION_TOLERANCE_PX:
        raise RuntimeError(f'Camera projection inconsistent with estimator by '
            f'{projection_error}px')
    return projected, {'focal_xy': focal.tolist(), 'principal_xy': principal.tolist(),
        'projection_max_error_px': projection_error}
