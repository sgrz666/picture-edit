#!/usr/bin/env python3
"""Offline Hi4D V3 cache exporter. Explicit execution only; no estimator or optimizer.

Use a separate SMPL/nvdiffrast/FCL environment. All outputs must be new /data paths.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import numpy as np
from PIL import Image

from src.pose_control.qwen21_full.pose_cache import sha256
from src.pose_control.qwen21_full.body_pose import JOINT_NAMES, PARENTS_22, ROTATION_CONVENTION
from src.pose_control.qwen21_full.dual_hi4d_geometry import (CACHE_SCHEMA, body21_from_smpl,
    part_distances, IndependentMeshRasterizer, decode_contact)

ROOT = Path(__file__).resolve().parents[2]


def _rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def _source(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def _dependency_versions():
    result = {}
    for name in ('python-fcl', 'numpy', 'torch', 'smplx', 'nvdiffrast'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'distribution_metadata_unavailable'
    return result


def _contact_raster(people, indices, k, hw):
    h, w = hw
    out = np.zeros((2, h, w), np.float32)
    for p in range(2):
        vertices = np.unique(np.concatenate([indices[indices[:, 0] == p, 1], indices[indices[:, 2] == p, 3]]))
        points = people[p]['mesh_cam'][vertices] @ k.T
        uv = points[:, :2]/points[:, 2:]
        for x, y in uv:
            x0, x1 = max(0, int(x)-6), min(w, int(x)+7)
            y0, y1 = max(0, int(y)-6), min(h, int(y)+7)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            patch = np.exp(-((xx+.5-x)**2+(yy+.5-y)**2)/8)
            out[p, y0:y1, x0:x1] = np.maximum(out[p, y0:y1, x0:x1], patch)
    return out


def export_frame(frame, target, model_root, rasterizer, asset_max_side):
    required = ('official_smpl', 'meta', 'camera_file', 'rgb', 'pair', 'action', 'frame_id', 'camera_id')
    if any(k not in frame for k in required):
        raise ValueError('Raw frames manifest needs explicit SMPL/meta/camera/RGB/identity fields')
    with np.load(frame['official_smpl'], allow_pickle=False) as f:
        params = dict(f)
    with np.load(frame['meta'], allow_pickle=False) as f:
        meta = dict(f)
    with np.load(frame['camera_file'], allow_pickle=False) as f:
        cameras = dict(f)
    camera_index = list(cameras['ids']).index(frame['camera_id'])
    extr, k = cameras['extrinsics'][camera_index], cameras['intrinsics'][camera_index]
    if not np.isfinite(extr).all() or not np.allclose(extr[:, :3] @ extr[:, :3].T, np.eye(3), atol=1e-6):
        raise ValueError('Invalid world-to-camera rigid transform')
    if np.abs(cameras['dist_coeffs'][camera_index]).max() > 1e-12:
        raise ValueError('Nonzero distortion requires an explicitly rectified raw-frame manifest; no implicit pinhole approximation')
    with Image.open(frame['rgb']) as image:
        original_hw = (image.height, image.width)
    ratio = min(1., asset_max_side/max(original_hw))
    asset_hw = tuple(max(1, round(x*ratio)) for x in original_hw)
    affine = np.diag([original_hw[1]/asset_hw[1], original_hw[0]/asset_hw[0], 1.])
    asset_k = np.linalg.inv(affine) @ k
    models = [model_root/f"SMPL_{str(g).upper()}.pkl" for g in meta['genders']]
    people = [body21_from_smpl(params, meta, p, models[p], extr, k, original_hw) for p in range(2)]
    renders = [rasterizer.render(p, asset_k, asset_hw) for p in people]
    distance, witnesses, valid, witness_valid, intersections = part_distances(people)
    height = np.mean([p['body_height'] for p in people])
    indices, contact_distance = decode_contact(params['contact'], params['verts'], height)
    contact = np.zeros((14, 14, 4), np.float32)
    for p, a, q, b in indices:
        pa, pb = people[p]['vertex_part'][a], people[q]['vertex_part'][b]
        i, j = (pa, pb) if p == 0 else (pb, pa)
        contact[i, j, :2] = 1  # observed positive + known; absence remains unknown
    key = f"hi4d:{frame['pair']}:{frame['action']}:{frame['camera_id']}:{frame['frame_id']}"
    sources = {name: _source(frame[field]) for name, field in (('params', 'official_smpl'), ('meta', 'meta'), ('camera', 'camera_file'))}
    sources.update({f'model_{p}': _source(path) for p, path in enumerate(models)})
    sources['fk_implementation'] = _source(ROOT/'src/pose_control/qwen21_full/dual_hi4d_geometry.py')
    metadata = {'schema': CACHE_SCHEMA, 'record_id': key, 'person_ids': [f"{frame['pair']}:0", f"{frame['pair']}:1"],
        'pair': frame['pair'], 'action': frame['action'], 'frame_id': frame['frame_id'],
        'camera_id': f"hi4d:{frame['pair']}:{frame['action']}:{frame['camera_id']}",
        'scale_group_id': f"hi4d:{frame['pair']}:release_model_scale", 'original_hw': list(original_hw),
        'asset_hw': list(asset_hw), 'asset_to_original': affine.tolist(), 'units': 'model_unit', 'scale_status': 'model_scale',
        'model_family_indices': [1, 1], 'genders': [p['gender'] for p in people],
        'mesh_errors': [p['mesh_error'] for p in people], 'joint_errors': [p['joint_error'] for p in people],
        'camera_mesh_errors': [p['camera_mesh_error'] for p in people], 'sources': sources,
        'joint_order': ['pelvis', *JOINT_NAMES], 'parents_22': list(PARENTS_22), 'rotation_convention': ROTATION_CONVENTION,
        'vertex_counts': [len(p['mesh_cam']) for p in people],
        'contact_encoding': 'one_based_zero_no_record', 'part_partition': 'argmax_face_sum_lbs_14',
        'normal_basis': 'camera_cv', 'distance_backend': 'python-fcl_triangle_bvh',
        'body_source': 'paired_target_frame', 'rgb': _source(frame['rgb'])}
    metadata['dependencies'] = _dependency_versions()
    arrays = {name: np.stack([p[name] for p in people]) for name in ('body_pose', 'joints_cam', 'global26', 'root_rotation')}
    arrays['body_height'] = np.asarray([p['body_height'] for p in people], np.float32)
    arrays.update({name: np.stack([r[name] for r in renders]) for name in ('z', 'valid', 'normal', 'part')})
    arrays.update(K=k.astype(np.float32), part_distance=distance, part_distance_valid=valid,
                  closest_points=witnesses, closest_points_valid=witness_valid, surface_intersection=intersections,
                  part_contact=contact, contact_indices=indices, contact_distance=contact_distance,
                  contact_raster=_contact_raster(people, indices, asset_k, asset_hw),
                  metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    name = hashlib.sha256(key.encode()).hexdigest()[:24]+'.npz'
    path = target/name
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return {'path': str(path), 'sha256': sha256(path), 'metadata': metadata}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--frames-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-root', type=Path, default=ROOT/'add/tools/SMPLest-X/human_models/human_model_files/smpl')
    p.add_argument('--asset-max-side', type=int, default=1024)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new cache directory')
    target = args.output.resolve()
    if not target.is_relative_to('/data/shangguanrz') or target == Path('/data/shangguanrz'):
        raise ValueError('Cache must be a dedicated new /data/shangguanrz directory')
    if args.asset_max_side <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError('Positive asset side/limit required')
    # Fail before creating outputs if a dependency/model is absent. No package install here.
    import fcl
    rasterizer = IndependentMeshRasterizer()
    frames = {(f['pair'], f['action'], int(f['camera_id']), int(f['frame_id'])): f for f in _rows(args.frames_manifest)}
    pairs = _rows(args.pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    geometries = args.output/'geometry'
    geometries.mkdir()
    seen = {}
    inputs = {'pairs_sha256': sha256(args.pairs), 'frames_sha256': sha256(args.frames_manifest), 'tool_sha256': sha256(__file__)}
    with (args.output/'records.jsonl').open('x', encoding='utf8') as out:
        for i, row in enumerate(pairs):
            records, raw_frames = [], []
            for side in ('source', 'target'):
                action = row.get(side+'_action', row.get('action'))
                camera = row.get(side+'_camera_id', row.get('camera_id'))
                key = row['pair'], action, int(camera), int(row[side+'_frame_id'])
                if key not in frames:
                    raise ValueError(f'Raw frame not explicitly declared in frames manifest: {key}')
                frame = frames[key]
                if key not in seen:
                    seen[key] = export_frame(frame, geometries, args.model_root, rasterizer, args.asset_max_side)
                records.append(seen[key]); raw_frames.append(frame)
            src, dst = records
            source_masks = [_source(raw_frames[0][f'person_mask_{letter}']) for letter in ('A', 'B')]
            record = {'schema': 'native_dual_pair_v3', 'pair_id': row.get('pair_id', f"hi4d:{i}:{src['metadata']['record_id']}->{dst['metadata']['record_id']}"),
                'source_geometry': {'path': src['path'], 'sha256': src['sha256']},
                'target_geometry': {'path': dst['path'], 'sha256': dst['sha256']},
                'source_rgb': src['metadata']['rgb'], 'target_rgb': dst['metadata']['rgb'], 'source_masks': source_masks,
                'source_person_ids': src['metadata']['person_ids'], 'target_person_ids': dst['metadata']['person_ids'],
                'source_indices': [0, 1], 'kind': 'camera_z', 'pairing': 'same_subject_pair_target_frame', **inputs}
            if record['source_person_ids'] != record['target_person_ids']:
                raise ValueError('Source/target subject pair differs; no implicit retargeting')
            out.write(json.dumps(record, ensure_ascii=False)+'\n')
    (args.output/'README.md').write_text('# Native Dual V3 缓存\n\n'
        f'{len(pairs)} 个同源双人 pair，{len(seen)} 个独立帧缓存。模型尺度，不自动宣称米制。\n'
        'geometry/ 保存原始非学习几何；records.jsonl 绑定来源 SHA、人物与相机。未训练或迁移权重。\n', encoding='utf8')


if __name__ == '__main__':
    main()
