"""V6.1 spatial contracts, independent of model weights and RGB supervision."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Sequence

import torch
from torch.nn import functional as F

from ..conditions import _to_preserving_discrete_dtype
from ..depth_condition_v2 import _to_fp32
from ..native_v6.contracts import PoseGeometry, PoseLinkConfig

SPATIAL_SCHEMA = 'native_pose_spatial_v61'


@dataclass(frozen=True)
class PoseLinkConfigV61(PoseLinkConfig):
    geometry_key_mask: bool = True
    edit_region_gate: bool = True
    multiscale_tokens: bool = True
    global_token_budget: int = 128
    joint_token_budget: int = 64
    local_token_budget: int = 64
    support_expand_px: int = 16
    edit_expand_px: int = 16
    feather_px: int = 8

    def __post_init__(self):
        super().__post_init__()
        if self.max_pose_tokens_per_reference > 256:
            raise ValueError('V6.1 reference token budget cannot exceed 256')
        for name in ('geometry_key_mask', 'edit_region_gate', 'multiscale_tokens'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' requires bool')
        budgets = ('global_token_budget', 'joint_token_budget', 'local_token_budget')
        for name in (*budgets, 'support_expand_px', 'edit_expand_px', 'feather_px'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + ' requires a nonnegative integer')
        total = sum(getattr(self, name) for name in budgets)
        if total <= 0 or total > self.max_pose_tokens_per_reference:
            raise ValueError('token budget sum must be positive and <= max_pose_tokens_per_reference')


@dataclass(frozen=True)
class PoseSpatialConditionV61:
    """Raw target-Canvas spatial data. Gate is conditional on valid pixels.

    The PoseLink output still applies V6 coverage exactly once. In particular,
    edit_gate is avg(valid * gate) / avg(valid), not avg(valid * gate).
    Local order is left_hand, right_hand, left_foot, right_foot, face.
    """
    key_support: torch.Tensor             # FP32 [B,2,H,W], already masked by valid_pixels
    valid_pixels: torch.Tensor            # FP32 [B,1,H,W]
    edit_gate: torch.Tensor               # FP32 [B,N], conditional mean
    joint_uv: torch.Tensor                # FP32 [B,2,21,2], normalized target Canvas
    joint_valid: torch.Tensor             # bool [B,2,21]
    local_boxes: torch.Tensor             # FP32 [B,2,5,4], normalized target Canvas
    local_valid: torch.Tensor             # bool [B,2,5]
    canvas_ids: tuple[str, ...]
    metadata: tuple[dict, ...]

    @property
    def batch_size(self) -> int:
        return self.key_support.shape[0]

    @property
    def target_token_hw(self) -> tuple[int, int]:
        return tuple(int(x) // 16 for x in self.key_support.shape[-2:])

    def validate(self, geometry: PoseGeometry | None = None) -> 'PoseSpatialConditionV61':
        if not torch.is_tensor(self.key_support) or self.key_support.ndim != 4:
            raise ValueError('key_support requires FP32 [B,2,H,W]')
        b, _, h, w = self.key_support.shape
        if b <= 0 or h <= 0 or w <= 0 or h % 16 or w % 16:
            raise ValueError('spatial pixel grid must be a positive multiple of 16')
        device = self.key_support.device
        n = h * w // 256
        shapes = {'key_support': (b, 2, h, w), 'valid_pixels': (b, 1, h, w),
            'edit_gate': (b, n), 'joint_uv': (b, 2, 21, 2), 'local_boxes': (b, 2, 5, 4)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.device != device
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError(f'{name} requires finite floating {shape} on {device}')
            if name in ('key_support', 'valid_pixels', 'edit_gate') and value.dtype != torch.float32:
                raise ValueError(name + ' must remain FP32')
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(name + ' must be in [0,1]')
        for name, shape in (('joint_valid', (b, 2, 21)), ('local_valid', (b, 2, 5))):
            value = getattr(self, name)
            if (not torch.is_tensor(value) or tuple(value.shape) != shape
                    or value.dtype != torch.bool or value.device != device):
                raise ValueError(f'{name} requires bool {shape} on {device}')
        if bool((self.key_support > self.valid_pixels + 1e-6).any()):
            raise ValueError('key_support exceeds valid_pixels')
        if bool((self.joint_valid & (self.joint_uv >= 1).any(-1)).any()):
            raise ValueError('valid joint_uv must be strictly inside the target Canvas')
        if bool((self.local_valid & (self.local_boxes[..., 2:] <= self.local_boxes[..., :2]).any(-1)).any()):
            raise ValueError('valid local_boxes require positive area')
        coverage = F.avg_pool2d(self.valid_pixels, 16).flatten(1)
        if bool(((coverage == 0) & (self.edit_gate != 0)).any()):
            raise ValueError('edit_gate must be zero on uncovered tokens')
        if (not isinstance(self.canvas_ids, tuple) or len(self.canvas_ids) != b
                or any(not isinstance(value, str) or not value for value in self.canvas_ids)):
            raise ValueError('canvas_ids requires one nonempty string per sample')
        if (not isinstance(self.metadata, tuple) or len(self.metadata) != b
                or any(not isinstance(value, dict) for value in self.metadata)):
            raise ValueError('metadata requires one dictionary per sample')
        if any(value.get('schema') != SPATIAL_SCHEMA or value.get('target_canvas_id') != canvas
               for value, canvas in zip(self.metadata, self.canvas_ids)):
            raise ValueError('V6.1 spatial metadata schema/Canvas differs')
        if geometry is not None:
            geometry.validate()
            if geometry.target_token_hw != self.target_token_hw or geometry.batch_size != b:
                raise ValueError('spatial and geometry grids differ')
            if geometry.geometry.device != device:
                raise ValueError('spatial and geometry device differ')
            if geometry.canvas_ids != self.canvas_ids:
                raise ValueError('spatial and geometry canvas_ids differ')
            if not torch.allclose(geometry.coverage.float(), coverage, atol=1e-6, rtol=1e-6):
                raise ValueError('spatial valid_pixels and geometry coverage differ')
            inactive = ~geometry.person_valid
            if (bool((self.key_support * inactive[..., None, None]).any())
                    or bool((self.joint_valid & inactive[..., None]).any())
                    or bool((self.local_valid & inactive[..., None]).any())):
                raise ValueError('inactive geometry people cannot have spatial support')
        return self

    def to(self, *args, **kwargs) -> 'PoseSpatialConditionV61':
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                if value.is_floating_point():
                    value = _to_fp32(value, *args, **kwargs)
                else:
                    value = _to_preserving_discrete_dtype(value, *args, **kwargs)
            elif field.name == 'metadata':
                value = deepcopy(value)
            values[field.name] = value
        return type(self)(**values)

    def index_select(self, indices: torch.Tensor) -> 'PoseSpatialConditionV61':
        if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError('batch indices require 1D int64')
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.index_select(0, indices.to(value.device))
            else:
                value = tuple(deepcopy(value[i]) for i in indices.cpu().tolist())
            values[field.name] = value
        return type(self)(**values)

    @classmethod
    def stack(cls, records: Sequence['PoseSpatialConditionV61']) -> 'PoseSpatialConditionV61':
        if not records or any(type(value) is not cls for value in records):
            raise ValueError('cannot stack empty or mixed spatial records')
        for value in records:
            value.validate()
        if any(value.target_token_hw != records[0].target_token_hw for value in records):
            raise ValueError('bucket spatial records by exact target grid')
        if any(value.key_support.device != records[0].key_support.device for value in records):
            raise ValueError('stack spatial records on the same device')
        values = {}
        for field in fields(records[0]):
            first = getattr(records[0], field.name)
            if torch.is_tensor(first):
                values[field.name] = torch.cat([getattr(value, field.name) for value in records], 0)
            else:
                values[field.name] = tuple(deepcopy(item) for value in records for item in getattr(value, field.name))
        return cls(**values).validate()
