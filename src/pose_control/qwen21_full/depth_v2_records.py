"""Pair/pose-bound Depth V2 manifest reader. All geometry is prepared before CNNs."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from .depth_condition import canvas_identity
from .depth_condition_v2 import (DepthConditionV2, DepthFusionContext, CONDITION_SCHEMA, CONTEXT_SCHEMA,
                                 FEATURE_SCHEMA, DETAIL_CHANNELS)
from .depth_geometry import (DepthGeometryConfig, full_frame_affine, resize_native, onto_canvas,
                              source_features, context_to_canvas)
from .depth_geometry import camera_rays
from .pose_cache import sha256

RECORD_SCHEMA = 'qwen21_native_depth_record_v2'
NORMAL_BASES = ('view_xright_yup_ztoward', 'camera_cv', 'unknown')


class NativeDepthV2Dataset(torch.utils.data.Dataset):
    """Optional DataLoader wrapper; the existing explicit-record path also works."""
    def __init__(self, dataset, records):
        self.dataset, self.depth_records = dataset, records

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample['depth_condition_v2'] = self.depth_records.condition(sample)
        return sample


def camera_fingerprint(camera):
    return hashlib.sha256(json.dumps(camera, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _image(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def _mask(path):
    data = np.load(path, allow_pickle=False) if Path(path).suffix == '.npy' else _image(path, 'L')
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('Depth validity must be an independent H,W asset')
    return torch.from_numpy(data > 0)[None, None]


class NativeDepthV2Records:
    def __init__(self, path, *, pose_records, mode='auto', geometry_config=None):
        if mode not in ('auto', 'camera_z', 'relative_png'):
            raise ValueError('Depth V2 mode must be auto/camera_z/relative_png')
        if pose_records is None:
            raise ValueError('Depth V2 requires the same verified Body21 pose record store')
        self.path, self.mode = Path(path), mode
        self.pose_records = pose_records
        self.config = geometry_config or DepthGeometryConfig()
        self.records, self.fingerprints, self._assets = {}, {}, {}
        for line in self.path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('pair_id')
            if not isinstance(key, str) or not key or key in self.records:
                raise ValueError('Depth V2 manifest requires unique pair_id')
            if row.get('schema') != RECORD_SCHEMA or row.get('kind') not in ('camera_z', 'relative_png'):
                raise ValueError('Depth V2 requires an explicit versioned per-record data kind')
            if mode != 'auto' and row['kind'] != mode:
                raise ValueError('Manifest kind differs from the explicit selected mode')
            for name in ('target_stem', 'pose_record_id', 'pose_record_sha256', 'camera_sha256',
                         'source_params_sha256', 'source_model_sha256', 'camera_id', 'scale_group_id'):
                if not isinstance(row.get(name), str) or not row[name]:
                    raise ValueError(f'Depth V2 manifest requires {name}')
            if row.get('normal_basis') not in NORMAL_BASES:
                raise ValueError('Normal basis must be declared, including unknown')
            if row.get('units') not in ('unknown', 'model_unit', 'metre') or row.get('scale_status') not in ('unknown', 'model_scale', 'calibrated'):
                raise ValueError('Depth scale semantics must be explicit')
            for key_path, key_sha in (('depth_path', 'depth_sha256'), ('valid_path', 'valid_sha256'),
                                      ('confidence_path', 'confidence_sha256')):
                if key_path == 'confidence_path' and row.get(key_path) is None:
                    continue
                if not isinstance(row.get(key_sha), str) or len(row[key_sha]) != 64:
                    raise ValueError(f'Depth V2 requires {key_sha}')
                p = Path(row[key_path])
                row[key_path] = str(p if p.is_absolute() else self.path.parent / p)
            if row.get('confidence_path') and row.get('confidence_semantics') != 'provided_quality':
                raise ValueError('Provided confidence needs explicit quality semantics, not inferred uncertainty')
            self.records[key] = row
        if not self.records:
            raise ValueError('Empty Depth V2 manifest')

    def _verify(self, path, expected):
        p = Path(path)
        stat = p.stat()
        signature = (stat.st_size, stat.st_mtime_ns, expected)
        if self._assets.get(str(p)) != signature:
            if sha256(p) != expected:
                raise ValueError(f'Depth V2 asset fingerprint differs: {p}')
            self._assets[str(p)] = signature

    def _context(self, pose, row, native_hw, canvas_id):
        meta = pose['metadata']
        original_hw = tuple(meta['original_hw'])
        for name in ('normal', 'part', 'smpl_mask'):
            transform = meta['condition_to_original'][name]
            if tuple(transform['original_hw']) != original_hw:
                raise ValueError('Normal/Part/SMPL mask belongs to a different original frame')
            sx, sy = transform['scale_xy']
            full_frame_affine(transform['source_hw'], original_hw, [[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        sources = meta['sources']
        mask_array = _image(sources['smpl_mask']['path'], 'L')
        normals = _image(sources['normal']['path'], 'RGB')
        labels = _image(sources['part']['path'], 'L')
        for name, array in (('smpl_mask', mask_array), ('normal', normals), ('part', labels)):
            if tuple(array.shape[:2]) != tuple(meta['condition_to_original'][name]['source_hw']):
                raise ValueError('Spatial asset grid differs from its verified source transform')
        mask = resize_native(torch.from_numpy(mask_array > 0)[None, None].float(), native_hw) > .5
        normal = torch.from_numpy(normals).permute(2, 0, 1)[None].float() / 127.5 - 1
        normal = resize_native(normal, native_hw)
        if row['normal_basis'] == 'view_xright_yup_ztoward':
            normal = normal * normal.new_tensor([1., -1., -1.])[None, :, None, None]
        normal_valid = mask & (torch.linalg.vector_norm(normal, dim=1, keepdim=True) > 1e-6)
        if row['normal_basis'] == 'unknown':
            normal_valid = torch.zeros_like(mask)
        normal = torch.where(normal_valid, F.normalize(normal, dim=1, eps=1e-6), 0.)
        labels = resize_native(torch.from_numpy(labels.astype(np.float32))[None, None], native_hw)[0, 0].long()
        if ((labels < 0) | (labels > 14)).any():
            raise ValueError('Part labels must be 0(background)..14')
        part = F.one_hot(labels, 15)[..., 1:].permute(2, 0, 1)[None].float() * mask
        part_valid = mask & (labels[None, None] > 0)
        camera = meta['camera']
        fx, fy = camera['focal_xy']
        cx, cy = camera['principal_xy']
        k = torch.tensor([[[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]], dtype=torch.float32)
        same_group = row['scale_group_id'] == row['pose_record_id']
        bound = (row['kind'] == 'camera_z' and row['units'] == 'model_unit'
                 and row['scale_status'] == 'model_scale' and same_group)
        metric = row['kind'] == 'camera_z' and row['units'] == 'metre' and row['scale_status'] == 'calibrated'
        factor = 1.
        if metric:
            calibration = row.get('calibration', {})
            factor = calibration.get('metres_per_model_unit', 0.)
            if (not isinstance(factor, (int, float)) or not np.isfinite(factor) or factor <= 0
                    or calibration.get('pose_record_sha256') != row['pose_record_sha256'] or not same_group):
                raise ValueError('Metric Z requires explicit same-pose metres_per_model_unit calibration')
            bound = True
        ctx_meta = {'schema': CONTEXT_SCHEMA, 'canvas_id': canvas_id, 'pose_record_id': row['pose_record_id'],
                    'camera_sha256': row['camera_sha256'], 'units': row['units'], 'scale_status': row['scale_status'],
                    'metres_per_model_unit': factor if metric else None,
                    'normal_basis': 'unknown' if row['normal_basis'] == 'unknown' else 'camera_cv'}
        return DepthFusionContext(k, pose['joints_cam'][0, 2].float().reshape(1) * factor,
                                  pose['body_height'].float().reshape(1) * factor,
                                  normal, normal_valid, part, part_valid, mask, torch.tensor([bound]),
                                  torch.tensor([metric]), (ctx_meta,)).validate()

    def condition(self, sample):
        pair = sample.get('pair_metadata', {})
        key = pair.get('pair_id')
        if key not in self.records:
            raise ValueError(f'Depth V2 manifest missing pair {key!r}')
        row = self.records[key]
        if pair.get('cross_person') or sample.get('geometry_retargeted') or row['target_stem'] != pair.get('target_stem'):
            raise ValueError('Depth V2 requires same-target-frame paired geometry')
        pose_ref = sample.get('pose_provenance', {})
        if (pose_ref.get('record_id') != row['pose_record_id']
                or pose_ref.get('record_sha256') != row['pose_record_sha256']):
            raise ValueError('Depth and Body21 pose records differ')
        pose = self.pose_records.raw(row['pose_record_id'])
        pm = pose['metadata']
        if (pm['stem'] != row['target_stem'] or pm['sequence_id'] != pair.get('target_appearance')
                or pm['sources']['params']['sha256'] != row['source_params_sha256']
                or pm['sources']['model']['sha256'] != row['source_model_sha256']
                or camera_fingerprint(pm['camera']) != row['camera_sha256']):
            raise ValueError('Depth/mesh/model/camera provenance differs')
        if pm['dataset'] == 'iper':
            expected_role = sample.get('target_role', row.get('target_role'))
            if row.get('target_role') != pm['role'] or expected_role != pm['role']:
                raise ValueError('Depth V2 iPER role must match the exact FK frame')
        for path, digest in (('depth_path', 'depth_sha256'), ('valid_path', 'valid_sha256'),
                             ('confidence_path', 'confidence_sha256')):
            if row.get(path) is not None:
                self._verify(row[path], row[digest])
        valid = _mask(row['valid_path'])
        z = None
        if row['kind'] == 'camera_z':
            array = np.load(row['depth_path'], allow_pickle=False)
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
                raise ValueError('Camera-Z requires a floating H,W NPY')
            z = torch.from_numpy(array.copy()).float()[None, None]
            if z.shape != valid.shape or not torch.isfinite(z[valid]).all() or (z[valid] <= 0).any():
                raise ValueError('Valid Camera-Z must be finite positive on an independent mask')
            z = torch.where(valid, z, 0.)
            values = z[valid]
            lo, hi = (float(values.min()), float(values.max())) if values.numel() else (0., 0.)
            relative = torch.where(valid, (hi - z) / (hi - lo) if hi > lo + 1e-8 else .5, 0.)
            bounds = [lo, hi]
        else:
            with Image.open(row['depth_path']) as image:
                if image.format != 'PNG':
                    raise ValueError('Relative Depth must be explicitly declared grayscale PNG')
                rgb = np.asarray(image.convert('RGB')).copy()
            if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
                raise ValueError('Relative Depth PNG must have equal grayscale channels')
            relative = torch.from_numpy(rgb[..., 0].copy()).float()[None, None] / 255
            if relative.shape != valid.shape or row.get('polarity') not in ('near_one', 'near_zero'):
                raise ValueError('Relative Depth requires matching valid grid and explicit polarity')
            relative = relative if row['polarity'] == 'near_one' else 1 - relative
            relative = torch.where(valid, relative, 0.)
            bounds = None
        spec = CanvasSpec.from_record(sample['target_canvas'])
        if tuple(pm['original_hw']) != spec.original_hw:
            raise ValueError('Depth camera and RGB original frame differ')
        if pose_ref.get('canvas_id') != spec.fingerprint():
            raise ValueError('Body21 pose/canvas fingerprint differs from the requested Depth canvas')
        affine = full_frame_affine(relative.shape[-2:], spec.original_hw, row.get('asset_to_original'))
        identity = canvas_identity(sample['target_canvas'])
        ctx_original = self._context(pose, row, relative.shape[-2:], identity)
        # Root/height/K/Normal share the same source mesh; no transform of vector components on resize.
        ctx_asset = replace(ctx_original, intrinsics=torch.linalg.inv(affine)[None] @ ctx_original.intrinsics)
        valid = valid & ctx_asset.smpl_mask
        relative = torch.where(valid, relative, 0.)
        if z is not None:
            z = torch.where(valid, z, 0.)
            values = z[valid]
            lo, hi = (float(values.min()), float(values.max())) if values.numel() else (0., 0.)
            relative = torch.where(valid, (hi - z) / (hi - lo) if hi > lo + 1e-8 else .5, 0.)
            bounds = [lo, hi]
        confidence_present = torch.tensor([row.get('confidence_path') is not None])
        if confidence_present[0]:
            data = np.load(row['confidence_path'], allow_pickle=False)
            confidence = torch.from_numpy(data.copy()).float()[None, None]
            if confidence.shape != valid.shape or not torch.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
                raise ValueError('Confidence must be an explicitly supplied finite 0..1 H,W NPY')
        else:
            confidence = torch.ones_like(relative)
        confidence = torch.where(valid, confidence, 0.)
        detail, geometry_valid = source_features(relative, valid, z, ctx_asset, confidence, confidence_present,
                                                self.config, asset_to_original=affine, original_hw=spec.original_hw)
        channels = [onto_canvas(detail[:, i:i+1], spec, edge=i in (3, 4, 9)) for i in range(len(DETAIL_CHANNELS))]
        detail_canvas = torch.cat(channels, 1)
        canvas_valid = spec.valid_mask(dtype=torch.float32)[None].bool()
        ctx_canvas = context_to_canvas(ctx_original, spec)
        support = onto_canvas(valid.float(), spec, valid=valid) > 0
        support &= canvas_valid & ctx_canvas.smpl_mask
        relative_canvas = torch.where(support, onto_canvas(relative, spec, valid=valid), 0.).clamp(0, 1)
        zvalid = torch.zeros_like(support) if z is None else ((onto_canvas(valid.float(), spec) > .5) & support)
        zcanvas = None if z is None else torch.where(zvalid, onto_canvas(z, spec), 0.)
        gv = (onto_canvas(geometry_valid.float(), spec) > .5) & zvalid
        for i in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22):
            detail_canvas[:, i:i+1] *= gv
        detail_canvas[:, 10:13] = torch.where(gv.expand(-1, 3, -1, -1), camera_rays(ctx_canvas.intrinsics, spec.canvas_hw), 0.)
        confidence_canvas = torch.where(support, onto_canvas(confidence, spec), 0.)
        detail_canvas[:, 15:16] = support.float()
        detail_canvas[:, 16:17] = gv.float()
        detail_canvas[:, 17:18] = confidence_canvas
        detail_canvas[:, 18:19] = confidence_present[:, None, None, None] * support
        metadata = {**deepcopy(row), 'schema': CONDITION_SCHEMA, 'feature_schema': FEATURE_SCHEMA,
                    'detail_channels': list(DETAIL_CHANNELS), 'geometry_config': asdict(self.config),
                    'source_id': key, 'canvas_id': identity, 'pose_canvas_id': spec.fingerprint(), 'polarity': 'near_one', 'bounds': bounds,
                    'input_polarity': row.get('polarity'), 'metric': bool(ctx_canvas.metric_available[0]),
                    'geometry_disabled_reason': None if gv.any() else ('relative_png' if z is None else 'no_bound_valid_geometry'),
                    'asset_hw': list(relative.shape[-2:]), 'original_hw': list(spec.original_hw),
                    'canvas': spec.metadata(), 'pose_sources': deepcopy(pm['sources'])}
        signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if (key, identity) in self.fingerprints and self.fingerprints[key, identity] != signature:
            raise ValueError('Depth V2 source or preprocessing configuration changed during runtime')
        self.fingerprints[key, identity] = signature
        condition = DepthConditionV2(relative_canvas, support, canvas_valid, support.flatten(1).any(1), zcanvas,
                                     zvalid, gv, confidence_canvas, confidence_present, ctx_canvas, detail_canvas, (metadata,))
        condition.validate(target_token_hw=sample['target_token_hw'], expected_canvas_ids=(identity,))
        if not torch.equal(canvas_valid[0], sample['valid_image_mask'].bool()):
            raise ValueError('Depth V2 and RGB canvas validity differ')
        return condition
