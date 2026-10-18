"""Reproducible inventory of existing tools and Hi4D annotation schemas."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile
import numpy as np
from geometry import associate, project


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/shangguanrz/project/pic-edit')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    paths = {
        'multihmr_code': root / 'add/tools/multi-hmr/demo_smplx_only.py',
        'multihmr_weights': root / 'add/tools/multi-hmr/models/multiHMR/multiHMR_896_L.pt',
        'smplestx_weights': root / 'add/tools/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar',
        'neutral_smplx': root / 'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',
        'male_smplx': root / 'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_MALE.npz',
        'female_smplx': root / 'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_FEMALE.npz',
        'dwpose_code': root / 'add/tools/dwpose_mimicmotion_official/dwpose/wholebody.py',
        'dwpose_weights': root / 'add/tools/dwpose_mimicmotion_official/models/dw-ll_ucoco_384.onnx',
    }
    tools = {k: {'path': str(v), 'exists': v.is_file(), 'bytes': v.stat().st_size if v.exists() else None} for k, v in paths.items()}
    for k in ['multihmr_code', 'neutral_smplx', 'dwpose_code', 'dwpose_weights']:
        tools[k]['sha256'] = sha256(paths[k])
    counts, candidates = {}, []
    for sub in ['datasets/Hi4D_pilot_v1', 'experiments', 'research', 'caches']:
        scope = root / sub
        n = 0
        if not scope.exists():
            continue
        for parent, dirs, files in os.walk(scope):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'checkpoints', 'models', 'hi4d_smplx_pilot_20261011']]
            for name in files:
                if not name.endswith('.npz'):
                    continue
                n += 1
                path = Path(parent) / name
                try:
                    with zipfile.ZipFile(path) as z:
                        keys = [Path(k).stem for k in z.namelist()]
                except zipfile.BadZipFile:
                    continue
                if any(k in keys for k in ['left_hand_pose', 'right_hand_pose', 'smplx_body_pose', 'rotvec', 'keypoints3d', 'joints3d', 'keypoints_3d']):
                    candidates.append({'path': str(path), 'fields': keys})
        counts[sub] = n
    conversion = []
    for sub in ['add/tools', 'add/models', 'third_party', 'archive']:
        for parent, dirs, files in os.walk(root / sub):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'envs']]
            for name in files:
                if any(s in name.lower() for s in ['deftrafo', 'smpl2smplx', 'smplx_mask_ids', 'smplx2smpl']):
                    conversion.append(str(Path(parent) / name))
    base = root / 'datasets/Hi4D_pilot_v1'
    smpl_path = base / 'raw/pair14/hug14/smpl/000062.npz'
    smpl = np.load(smpl_path)
    cam = np.load(base / 'raw/pair14/hug14/cameras/rgb_cameras.npz')
    dwpose_schemas, xy, scores = [], [], []
    for actor in ['A', 'B']:
        path = base / f'final_assets_v4_2/targets/pair14/hug14/cam4/000062/dwpose_{actor}.npz'
        z = np.load(path)
        dwpose_schemas.append({'path': str(path), 'fields': {k: list(z[k].shape) for k in z.files}, 'generator': str(z['generator'])})
        xy.append(z['xy_native']); scores.append(z['scores'])
    truth = np.stack([project(j[[17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]], cam['intrinsics'][0], cam['extrinsics'][0]) for j in smpl['joints_3d']])
    _, errors = associate(np.stack(xy)[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], np.stack(scores)[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], truth, max_error=1000)
    report = {'tools': tools, 'npz_header_counts': counts, 'existing_smplx_or_3d_wholebody_candidates': candidates, 'conversion_asset_matches': conversion, 'smpl_schema': {'path': str(smpl_path), 'fields': {k: list(smpl[k].shape) for k in smpl.files}}, 'existing_dwpose_schema': dwpose_schemas, 'existing_dwpose_body_alignment_error_px': errors.tolist(), 'existing_dwpose_A_B_body_mean_distance_px': float(np.linalg.norm(xy[0][:18] - xy[1][:18], axis=-1).mean()), 'notes': ['Scope is specified directories; it is not a claim about every disk or unopened compressed file', 'Existing DWPose stores 2D xy and scores; SMPL joints_3d is only a 24-joint body skeleton', 'The current pilot output is excluded from existing-asset searches', 'Large A/B reprojection error flags a possible binding or occlusion issue; this alone is not proof of mislabeled identity']}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'counts': counts, 'existing_candidates': len(candidates), 'conversion_assets': conversion, 'old_dwpose_body_error_px': errors.tolist()}, indent=2))


if __name__ == '__main__':
    main()
