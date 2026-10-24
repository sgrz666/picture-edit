"""Versioned, provenance-bound offline FK records and canonical camera semantics."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch
from src.data.canvas_contract import CanvasSpec
from .body_pose import (POSE_SCHEMA, JOINT_NAMES, PARENTS_22, ROTATION_CONVENTION,
                        axis_angle_to_matrix, matrix_to_rotation_6d)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_id(dataset, sequence, role, stem):
    if dataset not in ('tiktok', 'iper'):
        raise ValueError('Only SMPL-X TikTok/iPER records are supported; SMPL69 is not body63')
    return ':'.join((dataset, str(sequence), str(role), str(stem)))


def canonical_global(params, index, original_hw, *, dataset):
    if dataset not in ('tiktok', 'iper') or params['body_pose'].shape[1:] != (63,):
        raise ValueError('Require SMPL-X body_pose63; SMPL69 cannot be truncated')
    raw = dataset == 'iper' or any(k in params for k in ('focal_length_xy_raw', 'princpt_raw', 'transl_raw'))
    suffix = '_raw' if raw else ''
    required = ('focal_length_xy' + suffix, 'princpt' + suffix, 'transl' + suffix)
    if any(k not in params for k in required):
        raise ValueError('Raw camera requires matching raw intrinsics AND transl_raw')
    focal, principal, translation = (params[k][index].float().reshape(-1) for k in required)
    if focal.shape != (2,) or principal.shape != (2,) or translation.shape != (3,) or (focal <= 0).any():
        raise ValueError('Invalid camera fields')
    h, w = original_hw
    rot = matrix_to_rotation_6d(axis_angle_to_matrix(params['global_orient'][index].float().reshape(3)))
    camera = torch.cat((focal, principal)) / torch.tensor([w, h, w, h])
    value = torch.cat((params['betas'][index].float().reshape(10), rot, translation, camera, translation))
    if value.shape != (26,) or not torch.isfinite(value).all():
        raise ValueError('Global26 must be finite')
    return value, {'selection': 'raw' if raw else 'standard', 'focal_xy': focal.tolist(),
                   'principal_xy': principal.tolist(), 'translation': translation.tolist(),
                   'translation_field': 'transl' + suffix, 'coordinates': 'camera_xyz'}


def project_camera(joints, focal, principal, original_hw):
    z = joints[..., 2]
    safe_z = torch.where(z.abs() > 1e-8, z, torch.ones_like(z))
    uv = joints[..., :2] / safe_z[..., None] * focal + principal
    h, w = original_hw
    valid = (z > 1e-8) & torch.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    # Behind-camera and off-canvas coordinates are retained, never clamped.
    return uv, valid


class BodyPoseRecords:
    """Read only raw, non-learned geometry. Canvas is applied exactly once here."""
    def __init__(self, manifest):
        self.path = Path(manifest)
        self.records = {}
        self._verified = {}
        for line in self.path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['record_id']
            if key in self.records:
                raise ValueError('Duplicate pose record ID')
            p = Path(row['path'])
            row['path'] = str(p if p.is_absolute() else self.path.parent / p)
            self.records[key] = row
        if not self.records:
            raise ValueError('Empty pose manifest')

    def verify_source(self, path, expected):
        p = Path(path)
        stat = p.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = self._verified.get(str(p))
        if previous != (signature, expected):
            if sha256(p) != expected:
                raise ValueError(f'Pose source fingerprint changed: {p}')
            self._verified[str(p)] = (signature, expected)

    def raw(self, key):
        if key not in self.records:
            raise ValueError(f'Pose manifest missing {key}')
        row = self.records[key]
        self.verify_source(row['path'], row['sha256'])
        data = torch.load(row['path'], map_location='cpu', weights_only=True)
        meta = data.get('metadata', {})
        if meta.get('schema') != POSE_SCHEMA:
            raise ValueError('Pose schema missing/wrong; legacy cache cannot be used as v2')
        if (meta.get('record_id') != key or meta.get('joint_names') != list(JOINT_NAMES)
                or meta.get('parents_22') != list(PARENTS_22)
                or meta.get('rotation_convention') != ROTATION_CONVENTION
                or meta.get('projection_space') != 'original_camera_pixels'
                or not 0 <= meta.get('mesh_max_abs_error', float('inf')) <= 1e-4):
            raise ValueError('Pose joint/camera/FK provenance differs')
        required_sources = {'params', 'frame_map', 'rgb', 'legacy_conditions', 'normal', 'smpl_mask', 'part',
                            'depth_vis', 'model', 'fk_implementation'}
        if not isinstance(meta.get('sources'), Mapping) or not required_sources.issubset(meta['sources']):
            raise ValueError('Pose provenance requires model/parameter/camera/condition source fingerprints')
        for name, item in meta['sources'].items():
            self.verify_source(item['path'], item['sha256'])
        shapes = {'body_pose': (63,), 'joints_cam': (22, 3), 'joint_uv': (21, 2),
                  'fk_valid': (22,), 'projection_valid': (21,), 'smplx_global': (26,), 'body_height': ()}
        for name, shape in shapes.items():
            value = data.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f'Pose cache tensor differs: {name}')
            if name in ('fk_valid', 'projection_valid') and value.dtype != torch.bool:
                raise ValueError('Pose masks must be bool')
        if data['body_height'] <= 0 or not data['fk_valid'].all():
            raise ValueError('Offline FK must be complete; no missing-Pose fallback')
        if (data['projection_valid'] & ~data['fk_valid'][1:]).any():
            raise ValueError('Projection requires FK')
        return data

    def read(self, key, canvas_record):
        data = self.raw(key)
        spec = CanvasSpec.from_record(canvas_record)
        meta = data['metadata']
        if tuple(meta['original_hw']) != spec.original_hw:
            raise ValueError('Pose camera/RGB original dimensions differ; possible double scaling')
        uv = data['joint_uv'].clone()
        scale = uv.new_tensor(spec.scale_xy)
        uv = uv * scale + uv.new_tensor(spec.offset_xy)
        x0, y0, x1, y1 = spec.valid_rect
        projected = data['projection_valid'] & (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        bbox = meta['smpl_mask_bbox_original_xyxy']
        bbox_height = (bbox[3] - bbox[1]) * spec.scale_xy[1]
        result = {name: data[name].clone() for name in ('body_pose', 'joints_cam', 'fk_valid', 'body_height')}
        result.update(joint_uv=uv, projection_valid=projected,
                      smplx_global=spec.global_camera(data['smplx_global']),
                      smpl_bbox_height=torch.tensor(bbox_height, dtype=torch.float32),
                      pose_provenance={'record_id': key, 'record_sha256': self.records[key]['sha256'],
                                       'schema': POSE_SCHEMA, 'canvas_id': spec.fingerprint(),
                                       'camera': meta['camera'], 'sources': meta['sources'],
                                       'condition_to_original': meta['condition_to_original']})
        return result
