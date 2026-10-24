"""Typed single-person depth/context; no image-derived fake Camera-Z or confidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace

import torch

from .conditions import _to_preserving_discrete_dtype

CONDITION_SCHEMA = 'qwen21_native_depth_condition_v2'
CONTEXT_SCHEMA = 'qwen21_depth_fusion_context_v1'
FEATURE_SCHEMA = 'qwen21_depth_geometry_features_v1'
DETAIL_CHANNELS = (
    'relative_dx', 'relative_dy', 'relative_laplacian', 'relative_jump', 'silhouette',
    'body_z', 'body_z_dx', 'body_z_dy', 'body_z_laplacian', 'body_z_jump',
    'ray_x', 'ray_y', 'ray_z', 'normal_cosine', 'normal_relation_valid',
    'depth_valid', 'geometry_valid', 'confidence', 'confidence_available',
    'root_distance', 'calibrated_height', 'metric_available', 'camera_available',
)
GATE_CHANNELS = (
    *(f'part_{i}' for i in range(14)), 'normal_x', 'normal_y', 'normal_z',
    'depth_coverage', 'smpl_coverage', 'geometry_coverage', 'normal_coverage', 'part_coverage',
    'normal_cosine', 'normal_relation_coverage', 'confidence', 'confidence_available',
    'silhouette', 'depth_jump', 'metric_available',
)


class _TensorRecord:
    def to(self, *args, **kwargs):
        # The entire depth frontend is FP32, including deterministic geometry.
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = (_to_fp32(value, *args, **kwargs) if value.is_floating_point()
                         else _to_preserving_discrete_dtype(value, *args, **kwargs))
            elif isinstance(value, _TensorRecord):
                value = value.to(*args, **kwargs)
            elif field.name == 'metadata':
                value = deepcopy(value)
            values[field.name] = value
        return type(self)(**values)

    def index_select(self, indices):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError('Depth selection requires int64 [B] indices')
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.index_select(0, indices.to(value.device))
            elif isinstance(value, _TensorRecord):
                value = value.index_select(indices)
            elif field.name == 'metadata':
                value = tuple(deepcopy(value[i]) for i in indices.cpu().tolist())
            values[field.name] = value
        return type(self)(**values)

    @classmethod
    def stack(cls, records):
        if not records or any(type(r) is not cls for r in records):
            raise ValueError('Cannot stack empty/mixed depth record types')
        values = {}
        for field in fields(records[0]):
            items = [getattr(r, field.name) for r in records]
            if field.name == 'metadata':
                value = tuple(deepcopy(meta) for item in items for meta in item)
            elif isinstance(items[0], _TensorRecord):
                value = type(items[0]).stack(items)
            elif all(v is None for v in items):
                value = None
            else:
                # Missing Camera-Z is storage-only zero padding, declared by its own mask.
                if any(v is None for v in items):
                    if field.name != 'camera_z':
                        raise ValueError(f'Missing stack field: {field.name}')
                    items = [torch.zeros_like(r.relative) if v is None else v for r, v in zip(records, items)]
                if any(v.shape[1:] != items[0].shape[1:] for v in items):
                    raise ValueError('Bucket depth batches by exact canvas; never resize in collate')
                value = torch.cat(items, dim=0)
            values[field.name] = value
        return cls(**values).validate()


def _to_fp32(value, *args, **kwargs):
    """Ignore requested float dtype BEFORE conversion, avoiding BF16 round trips."""
    device = kwargs.get('device', value.device)
    non_blocking, copy = kwargs.get('non_blocking', False), kwargs.get('copy', False)
    flags = 0
    if args:
        first = args[0]
        if torch.is_tensor(first):
            device, flags = first.device, 1
        elif isinstance(first, torch.dtype):
            flags = 1
        elif isinstance(first, (str, int, torch.device)):
            device = first
            flags = 2 if len(args) > 1 and (args[1] is None or isinstance(args[1], torch.dtype)) else 1
        if len(args) > flags:
            non_blocking = args[flags]
        if len(args) > flags + 1:
            copy = args[flags + 1]
    options = {'device': value.device if device is None else device, 'dtype': torch.float32,
               'non_blocking': bool(non_blocking), 'copy': bool(copy)}
    if 'memory_format' in kwargs:
        options['memory_format'] = kwargs['memory_format']
    return value.to(**options)


def _tensor(value, shape, device, name, *, boolean=False):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.device != device:
        raise ValueError(f'{name} requires shape {tuple(shape)} on {device}')
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f'{name} must be bool')
    elif value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite FP32')


@dataclass(frozen=True)
class DepthFusionContext(_TensorRecord):
    intrinsics: torch.Tensor
    root_z: torch.Tensor
    body_height: torch.Tensor
    normal_cam: torch.Tensor
    normal_valid: torch.Tensor
    part_onehot: torch.Tensor
    part_valid: torch.Tensor
    smpl_mask: torch.Tensor
    camera_available: torch.Tensor
    metric_available: torch.Tensor
    metadata: tuple[dict, ...]

    def validate(self):
        if self.normal_cam.ndim != 4:
            raise ValueError('Normal context must be [B,3,H,W]')
        b, _, h, w = self.normal_cam.shape
        device = self.normal_cam.device
        for name, shape in (('intrinsics', (b, 3, 3)), ('root_z', (b,)), ('body_height', (b,)),
                            ('normal_cam', (b, 3, h, w)), ('part_onehot', (b, 14, h, w))):
            _tensor(getattr(self, name), shape, device, name)
        for name, shape in (('normal_valid', (b, 1, h, w)), ('part_valid', (b, 1, h, w)),
                            ('smpl_mask', (b, 1, h, w)), ('camera_available', (b,)), ('metric_available', (b,))):
            _tensor(getattr(self, name), shape, device, name, boolean=True)
        if len(self.metadata) != b or any(m.get('schema') != CONTEXT_SCHEMA for m in self.metadata):
            raise ValueError('Missing/version-mismatched geometry context metadata')
        k = self.intrinsics
        if (not torch.allclose(k[:, 2], k.new_tensor([0., 0., 1.]).expand(b, -1))
                or (k[:, 0, 0] <= 0).any() or (k[:, 1, 1] <= 0).any()
                or (k[:, 0, 1].abs() > 1e-6).any() or (k[:, 1, 0].abs() > 1e-6).any()):
            raise ValueError('Only declared zero-skew pinhole intrinsics are supported')
        if (self.camera_available & ((self.root_z <= 0) | (self.body_height <= 0))).any():
            raise ValueError('Available geometry requires positive FK root and same-scale height')
        if (self.metric_available & ~self.camera_available).any():
            raise ValueError('Metric availability requires bound camera geometry')
        if ((self.part_onehot < 0) | (self.part_onehot > 1)).any() or (self.part_onehot.sum(1) > 1.00001).any():
            raise ValueError('Part must be 14 nonoverlapping channels with zero background')
        if ((self.normal_valid | self.part_valid) & ~self.smpl_mask).any():
            raise ValueError('Normal/Part validity must stay on same-source SMPL support')
        if (self.normal_cam.masked_select(~self.normal_valid.expand_as(self.normal_cam)) != 0).any():
            raise ValueError('Unknown/invalid normal must have zero context values')
        for i, meta in enumerate(self.metadata):
            if meta.get('normal_basis') not in ('camera_cv', 'unknown'):
                raise ValueError('Normal basis must be converted to camera_cv or explicitly unavailable')
            if meta['normal_basis'] == 'unknown' and self.normal_valid[i].any():
                raise ValueError('Unknown normal basis cannot be used in geometry fusion')
            if self.metric_available[i] and not (meta.get('units') == 'metre' and meta.get('scale_status') == 'calibrated'):
                raise ValueError('Model scale is not calibrated metric scale')
        return self


@dataclass(frozen=True)
class DepthConditionV2(_TensorRecord):
    relative: torch.Tensor
    valid: torch.Tensor
    canvas_valid: torch.Tensor
    available: torch.Tensor
    camera_z: torch.Tensor | None
    camera_z_valid: torch.Tensor
    geometry_valid: torch.Tensor
    confidence: torch.Tensor
    confidence_available: torch.Tensor
    context: DepthFusionContext
    detail_features: torch.Tensor
    metadata: tuple[dict, ...]

    @property
    def geometry_available(self):
        return self.geometry_valid.flatten(1).any(1)

    def validate(self, *, target_token_hw=None, expected_canvas_ids=None):
        if self.relative.ndim != 4:
            raise ValueError('DepthConditionV2 requires [B,1,H,W]')
        b, _, h, w = self.relative.shape
        device = self.relative.device
        shape = (b, 1, h, w)
        _tensor(self.relative, shape, device, 'relative')
        _tensor(self.confidence, shape, device, 'confidence')
        _tensor(self.detail_features, (b, len(DETAIL_CHANNELS), h, w), device, 'detail_features')
        for name in ('valid', 'canvas_valid', 'camera_z_valid', 'geometry_valid'):
            _tensor(getattr(self, name), shape, device, name, boolean=True)
        for name in ('available', 'confidence_available'):
            _tensor(getattr(self, name), (b,), device, name, boolean=True)
        if h % 16 or w % 16 or h // 16 * (w // 16) % 4:
            raise ValueError('Depth V2 requires the exact Qwen target grid')
        if target_token_hw is not None and tuple(target_token_hw) != (h // 16, w // 16):
            raise ValueError('Depth V2/Geometry canvas differs; no implicit resize')
        if not torch.equal(self.available, self.valid.flatten(1).any(1)):
            raise ValueError('Depth availability must describe actual valid support')
        if ((self.valid & ~self.canvas_valid) | (self.camera_z_valid & ~self.valid)
                | (self.geometry_valid & ~self.camera_z_valid)).any():
            raise ValueError('Inconsistent independent depth/Camera-Z/geometry masks')
        for name in ('relative', 'confidence'):
            value = getattr(self, name)
            if ((value < 0) | (value > 1)).any() or (value[~self.valid] != 0).any():
                raise ValueError(f'{name} must be 0..1, zero outside depth support')
        self.context.validate()
        if self.context.normal_cam.shape != (b, 3, h, w) or self.context.normal_cam.device != device:
            raise ValueError('Depth and fusion context must share one canvas/device')
        if (self.geometry_valid & ~self.context.camera_available[:, None, None, None]).any():
            raise ValueError('Physical channels need bound same-scale camera/root/height')
        physical = self.detail_features[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22]]
        if (physical.masked_select(~self.geometry_valid.expand_as(physical)) != 0).any():
            raise ValueError('Unavailable Camera-Z geometry channels must be explicitly zero')
        if (not torch.equal(self.detail_features[:, 15:16], self.valid.float())
                or not torch.equal(self.detail_features[:, 16:17], self.geometry_valid.float())
                or not torch.equal(self.detail_features[:, 17:18], self.confidence)
                or not torch.equal(self.detail_features[:, 18:19], self.confidence_available[:, None, None, None] * self.valid.float())
                or not torch.equal(self.detail_features[:, 21:22], (self.geometry_valid & self.context.metric_available[:, None, None, None]).float())
                or not torch.equal(self.detail_features[:, 22:23], self.geometry_valid.float())):
            raise ValueError('Prepared features disagree with typed validity/confidence')
        if self.camera_z is None:
            if self.camera_z_valid.any():
                raise ValueError('Cannot claim Camera-Z without Camera-Z data')
        else:
            _tensor(self.camera_z, shape, device, 'camera_z')
            if (self.camera_z[self.camera_z_valid] <= 0).any() or (self.camera_z[~self.camera_z_valid] != 0).any():
                raise ValueError('Camera-Z must be positive on its own support; never filled from PNG')
        if len(self.metadata) != b:
            raise ValueError('Depth provenance must carry one record per row')
        for i, meta in enumerate(self.metadata):
            if (meta.get('schema') != CONDITION_SCHEMA or meta.get('feature_schema') != FEATURE_SCHEMA
                    or meta.get('detail_channels') != list(DETAIL_CHANNELS)
                    or meta.get('kind') not in ('camera_z', 'relative_png') or meta.get('polarity') != 'near_one'):
                raise ValueError('Depth V2 schema/channel order/polarity differs')
            for key in ('source_id', 'canvas_id', 'pose_canvas_id', 'camera_id', 'pose_record_id', 'pose_record_sha256'):
                if not isinstance(meta.get(key), str) or not meta[key]:
                    raise ValueError(f'Depth V2 requires explicit {key}')
            if meta['kind'] == 'relative_png' and (self.camera_z_valid[i].any() or self.context.camera_available[i]):
                raise ValueError('PNG must disable Camera-Z geometry, including camera channels')
            if meta['canvas_id'] != self.context.metadata[i].get('canvas_id'):
                raise ValueError('Depth/context canvas provenance differs')
            if meta['pose_record_id'] != self.context.metadata[i].get('pose_record_id'):
                raise ValueError('Depth/context person provenance differs')
        if expected_canvas_ids is not None and tuple(m['canvas_id'] for m in self.metadata) != tuple(expected_canvas_ids):
            raise ValueError('Depth V2 and RGB canvas provenance differ')
        return self
