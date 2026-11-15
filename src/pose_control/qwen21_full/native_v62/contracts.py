"""V6.2 routing and conditional spatial multipliers; no model weights."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import math
from typing import Sequence

import torch

from ..conditions import _to_preserving_discrete_dtype
from ..depth_condition_v2 import _to_fp32
from ..native_v6.contracts import PoseGeometry
from ..native_v61.contracts import PoseLinkConfigV61

ROUTING_SCHEMA = 'native_pose_routing_v62'
GATE_SCHEMA = 'native_branch_spatial_gates_v62'


@dataclass(frozen=True)
class PoseLinkConfigV62(PoseLinkConfigV61):
    person_routing: bool = True
    body_spatial_gate: bool = True
    interaction_spatial_gate: bool = True
    owner_threshold: float = .75
    routing_epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        for name in ('person_routing', 'body_spatial_gate', 'interaction_spatial_gate'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' requires bool')
        for name in ('owner_threshold', 'routing_epsilon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(name + ' requires a finite number')
        if not .5 < self.owner_threshold <= 1:
            raise ValueError('owner_threshold must be in (.5,1]')
        if self.routing_epsilon <= 0:
            raise ValueError('routing_epsilon must be positive')


def _grid(record, b, n, schema, device, geometry):
    hw = record.target_token_hw
    if (not isinstance(hw, tuple) or len(hw) != 2 or any(type(x) is not int or x <= 0 for x in hw)
            or b <= 0 or n != hw[0] * hw[1]):
        raise ValueError('record requires a positive exact target_token_hw')
    if (not isinstance(record.canvas_ids, tuple) or len(record.canvas_ids) != b
            or any(not isinstance(v, str) or not v for v in record.canvas_ids)):
        raise ValueError('canvas_ids requires one nonempty string per sample')
    if (not isinstance(record.metadata, tuple) or len(record.metadata) != b
            or any(not isinstance(m, dict) or m.get('schema') != schema
                or m.get('target_canvas_id') != c for m, c in zip(record.metadata, record.canvas_ids))):
        raise ValueError('V6.2 metadata schema/Canvas differs')
    if geometry is not None:
        if not isinstance(geometry, PoseGeometry):
            raise ValueError('geometry requires typed PoseGeometry')
        geometry.validate()
        if geometry.batch_size != b or geometry.target_token_hw != hw:
            raise ValueError('record and geometry grids differ')
        if geometry.geometry.device != device:
            raise ValueError('record and geometry device differ')
        if geometry.canvas_ids != record.canvas_ids:
            raise ValueError('record and geometry canvas_ids differ')


def _tensor(value, shape, dtype, device, name):
    if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype or value.device != device):
        raise ValueError(f'{name} requires {dtype} {shape} on {device}')
    if dtype == torch.float32 and (not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any())):
        raise ValueError(name + ' requires finite FP32 values in [0,1]')


class _ConditionRecord:
    @property
    def batch_size(self):
        return self.body.shape[0] if hasattr(self, 'body') else self.query_person_ids.shape[0]

    def to(self, *args, **kwargs):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = (_to_fp32(value, *args, **kwargs) if value.is_floating_point()
                    else _to_preserving_discrete_dtype(value, *args, **kwargs))
            elif field.name == 'metadata':
                value = deepcopy(value)
            values[field.name] = value
        return type(self)(**values)

    def index_select(self, indices):
        if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError('batch indices require 1D int64')
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.index_select(0, indices.to(value.device))
            elif field.name != 'target_token_hw':
                value = tuple(deepcopy(value[i]) for i in indices.cpu().tolist())
            values[field.name] = value
        return type(self)(**values)

    @classmethod
    def stack(cls, records: Sequence):
        if not records or any(type(r) is not cls for r in records):
            raise ValueError('cannot stack empty or mixed V6.2 records')
        for record in records:
            record.validate()
        if any(r.target_token_hw != records[0].target_token_hw for r in records):
            raise ValueError('bucket V6.2 records by exact target grid')
        values = {}
        for field in fields(records[0]):
            first = getattr(records[0], field.name)
            if torch.is_tensor(first):
                if any(getattr(r, field.name).device != first.device for r in records):
                    raise ValueError('stack V6.2 records on the same device')
                values[field.name] = torch.cat([getattr(r, field.name) for r in records], 0)
            elif field.name == 'target_token_hw':
                values[field.name] = first
            else:
                values[field.name] = tuple(deepcopy(v) for r in records for v in getattr(r, field.name))
        return cls(**values).validate()


@dataclass(frozen=True)
class RoutingConditionV62(_ConditionRecord):
    query_person_ids: torch.Tensor         # int64 [B,N], -1 means uncertain
    part_tokens: torch.Tensor              # FP32 [B,2,N,14], raw area fractions
    contact_positive: torch.Tensor         # bool [B,14,14], person0 -> person1
    contact_forbidden: torch.Tensor        # bool [B,14,14], forbidden always vetoes
    person_valid: torch.Tensor             # bool [B,2]
    target_token_hw: tuple[int, int]
    canvas_ids: tuple[str, ...]
    person_ids: tuple[tuple[str, str], ...]
    metadata: tuple[dict, ...]

    @property
    def query_parts(self):
        return self.part_tokens.transpose(1, 2)

    def validate(self, geometry: PoseGeometry | None = None):
        if not torch.is_tensor(self.query_person_ids) or self.query_person_ids.ndim != 2:
            raise ValueError('query_person_ids requires int64 [B,N]')
        b, n = self.query_person_ids.shape
        device = self.query_person_ids.device
        _grid(self, b, n, ROUTING_SCHEMA, device, geometry)
        for name, shape, dtype in (('query_person_ids', (b, n), torch.long),
                ('part_tokens', (b, 2, n, 14), torch.float32), ('contact_positive', (b, 14, 14), torch.bool),
                ('contact_forbidden', (b, 14, 14), torch.bool), ('person_valid', (b, 2), torch.bool)):
            _tensor(getattr(self, name), shape, dtype, device, name)
        if bool(((self.query_person_ids < -1) | (self.query_person_ids > 1)).any()):
            raise ValueError('query_person_ids must be -1, 0 or 1')
        valid_owner = self.person_valid.gather(1, self.query_person_ids.clamp_min(0))
        if bool(((self.query_person_ids >= 0) & ~valid_owner).any()):
            raise ValueError('inactive person cannot own queries')
        if bool((self.part_tokens * ~self.person_valid[:, :, None, None]).any()):
            raise ValueError('inactive person cannot have part evidence')
        if bool((self.part_tokens.sum(-1) > 1.00001).any()):
            raise ValueError('part_tokens must have nonoverlapping area fractions')
        if (not isinstance(self.person_ids, tuple) or len(self.person_ids) != b
                or any(not isinstance(ids, tuple) or len(ids) != 2 or any(not isinstance(s, str) for s in ids)
                    for ids in self.person_ids)):
            raise ValueError('person_ids requires two stable ID strings per sample')
        for row, ids in enumerate(self.person_ids):
            missing_disabled = (ids == ('', '') and self.metadata[row].get('person_ids_status') in
                ('missing_routing_disabled','not_consumed_routing_disabled')
                and self.metadata[row].get('routing_enabled') is False and bool(self.query_person_ids[row].eq(-1).all()))
            if missing_disabled:
                continue
            active = [ids[p] for p in range(2) if self.person_valid[row, p]]
            if any(not s or s == '__inactive__' for s in active) or len(set(active)) != len(active):
                raise ValueError('active person_ids must be stable nonempty distinct IDs')
        if geometry is not None:
            if not torch.equal(self.person_valid, geometry.person_valid):
                raise ValueError('routing and geometry person_valid differ')
            uncovered = geometry.coverage == 0
            if bool((uncovered & (self.query_person_ids != -1)).any()) or bool((self.part_tokens * uncovered[:, None, :, None]).any()):
                raise ValueError('uncovered queries cannot have owner or part evidence')
        return self


@dataclass(frozen=True)
class BranchSpatialGatesV62(_ConditionRecord):
    body: torch.Tensor                    # FP32 [B,N], conditional multiplier
    interaction: torch.Tensor             # FP32 [B,N], relation_valid stays outside
    target_token_hw: tuple[int, int]
    canvas_ids: tuple[str, ...]
    metadata: tuple[dict, ...]

    def validate(self, geometry: PoseGeometry | None = None):
        if not torch.is_tensor(self.body) or self.body.ndim != 2:
            raise ValueError('body requires FP32 [B,N]')
        b, n = self.body.shape
        _grid(self, b, n, GATE_SCHEMA, self.body.device, geometry)
        for name in ('body', 'interaction'):
            _tensor(getattr(self, name), (b, n), torch.float32, self.body.device, name)
        return self
