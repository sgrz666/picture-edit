"""PoseLink 配置与 Source Identity 融合前的纯几何记录。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Sequence

import torch

from ..conditions import _to_preserving_discrete_dtype


@dataclass(frozen=True)
class PoseLinkConfig:
    hidden_dim: int = 4096
    width: int = 512
    heads: int = 8
    injection_layers: tuple[int, ...] = (5, 10, 15, 21, 26, 31)
    use_geometry: bool = True
    use_spatial: bool = True
    use_role: bool = True
    use_contact: bool = True
    use_interaction: bool = True
    query_chunk_size: int = 128
    max_pose_tokens_per_reference: int = 256

    def __post_init__(self):
        for name in ("hidden_dim", "width", "heads", "query_chunk_size", "max_pose_tokens_per_reference"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (not isinstance(self.injection_layers, tuple) or not self.injection_layers
                or any(type(i) is not int or i < 0 for i in self.injection_layers)
                or len(set(self.injection_layers)) != len(self.injection_layers)):
            raise ValueError("injection_layers must contain distinct nonnegative integers")
        for name in ("use_geometry", "use_spatial", "use_role", "use_contact", "use_interaction"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires bool")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["injection_layers"] = list(self.injection_layers)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "PoseLinkConfig":
        value = dict(value)
        if "injection_layers" in value:
            value["injection_layers"] = tuple(value["injection_layers"])
        return cls(**value)


@dataclass(frozen=True)
class PoseGeometry:
    """几何分辨率是 target_token_hw；projected 不受可见 owner 遮挡裁剪。

    contact 只表示有正证据的局部接触支持；未知接触填零，不推断全局接触。
    interaction 可独立消融，与 geometry 一样必须来自 Identity 融合前。
    """
    geometry: torch.Tensor                 # [B,N,512]
    person_tokens: torch.Tensor            # [B,2,N,512], raw reasoner output
    owner: torch.Tensor                    # [B,N,2], visible soft ownership
    projected: torch.Tensor                # [B,N,2], independent projected support
    coverage: torch.Tensor                 # [B,N], valid canvas fraction
    contact: torch.Tensor                  # [B,N,2], local positive contact support
    target_token_hw: tuple[int, int]
    person_valid: torch.Tensor             # bool [B,2]
    canvas_ids: tuple[str, ...]
    interaction: torch.Tensor | None = None  # [B,N,512]

    @property
    def batch_size(self) -> int:
        return self.geometry.shape[0]

    def validate(self) -> "PoseGeometry":
        if (not isinstance(self.target_token_hw, tuple) or len(self.target_token_hw) != 2
                or any(type(i) is not int or i <= 0 for i in self.target_token_hw)):
            raise ValueError("target_token_hw must contain positive integers")
        if not isinstance(self.geometry, torch.Tensor) or self.geometry.ndim != 3:
            raise ValueError("geometry requires floating [B,N,512]")
        b, n, _ = self.geometry.shape
        if b <= 0 or n != self.target_token_hw[0] * self.target_token_hw[1]:
            raise ValueError("geometry and target_token_hw differ")
        device = self.geometry.device
        shapes = {"geometry": (b, n, 512), "person_tokens": (b, 2, n, 512),
                  "owner": (b, n, 2), "projected": (b, n, 2),
                  "coverage": (b, n), "contact": (b, n, 2)}
        if self.interaction is not None:
            shapes["interaction"] = (b, n, 512)
        for name, shape in shapes.items():
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or value.device != device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} requires finite floating {shape} on {device}")
            if name in ("owner", "projected", "coverage", "contact"):
                if bool(((value < 0) | (value > 1)).any()):
                    raise ValueError(f"{name} must be in [0,1]")
        if bool((self.owner.sum(-1) > 1.00001).any()):
            raise ValueError("visible owner must not duplicate ownership")
        if (not isinstance(self.person_valid, torch.Tensor) or self.person_valid.dtype != torch.bool
                or self.person_valid.shape != (b, 2) or self.person_valid.device != device):
            raise ValueError("person_valid requires bool [B,2] on the geometry device")
        if (not isinstance(self.canvas_ids, tuple) or len(self.canvas_ids) != b
                or any(not isinstance(value, str) or not value for value in self.canvas_ids)):
            raise ValueError("canvas_ids requires one nonempty string per sample")
        return self

    def to(self, *args, **kwargs) -> "PoseGeometry":
        return type(self)(**{
            field.name: (_to_preserving_discrete_dtype(value, *args, **kwargs)
                         if isinstance(value, torch.Tensor) else value)
            for field in fields(self) for value in (getattr(self, field.name),)
        })

    def index_select(self, indices: torch.Tensor) -> "PoseGeometry":
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("batch indices require 1D int64")
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.index_select(0, indices.to(value.device))
            elif field.name == "canvas_ids":
                value = tuple(value[i] for i in indices.tolist())
            values[field.name] = value
        return type(self)(**values)

    @classmethod
    def stack(cls, records: Sequence["PoseGeometry"]) -> "PoseGeometry":
        """沿 batch 拼接，同一 bucket 必须具有相同网格和 interaction 开关。"""
        if not records or any(type(value) is not cls for value in records):
            raise ValueError("cannot stack empty or mixed PoseGeometry records")
        for value in records:
            value.validate()
        if any(value.target_token_hw != records[0].target_token_hw for value in records):
            raise ValueError("bucket PoseGeometry by target grid")
        if any((value.interaction is None) != (records[0].interaction is None) for value in records):
            raise ValueError("bucket PoseGeometry by interaction availability")
        values = {}
        for field in fields(records[0]):
            first = getattr(records[0], field.name)
            if isinstance(first, torch.Tensor):
                values[field.name] = torch.cat([getattr(value, field.name) for value in records], dim=0)
            elif field.name == "canvas_ids":
                values[field.name] = tuple(item for value in records for item in value.canvas_ids)
            else:
                values[field.name] = first
        return cls(**values).validate()
