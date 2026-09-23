from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..conditions import _to_preserving_discrete_dtype
from ..detail.hand import HandDetailCondition


def _transform(obj, function):
    return type(obj)(**{field.name: function(getattr(obj, field.name)) for field in fields(obj)})


@dataclass
class HandFineCondition(HandDetailCondition):
    """Target-only local geometry; no target RGB is required at inference."""

    mesh_bps: torch.Tensor
    mesh_valid: torch.Tensor
    depth: torch.Tensor
    visibility: torch.Tensor

    @classmethod
    def from_legacy(cls, value: HandDetailCondition) -> "HandFineCondition":
        value.validate()
        b = value.batch_size
        return cls(
            **{field.name: getattr(value, field.name) for field in fields(HandDetailCondition)},
            mesh_bps=value.hand_pose.new_zeros(b, 2, 2, 1024),
            mesh_valid=torch.zeros(b, 2, 2, dtype=torch.bool, device=value.device),
            depth=value.hand_pose.new_zeros(b, 2, 2, 1, 16, 16),
            visibility=value.region_valid.to(value.hand_pose.dtype),
        )

    def validate(self) -> "HandFineCondition":
        super().validate()
        b = self.batch_size
        expected = {"mesh_bps": (b, 2, 2, 1024), "depth": (b, 2, 2, 1, 16, 16), "visibility": (b, 2, 2)}
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != self.hand_pose.dtype or value.device != self.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite {shape} and match hand dtype/device")
        if self.mesh_valid.shape != (b, 2, 2) or self.mesh_valid.dtype != torch.bool or self.mesh_valid.device != self.device:
            raise ValueError("mesh_valid must be bool [B,2,2] on the hand device")
        if torch.any((self.visibility < 0) | (self.visibility > 1)):
            raise ValueError("visibility must lie in [0,1]")
        return self

    def to(self, *args, **kwargs):
        return _transform(self, lambda x: _to_preserving_discrete_dtype(x, *args, **kwargs))

    def index_select(self, indices):
        return _transform(self, lambda x: x.index_select(0, indices))

    def expand_to_batch(self, batch_size):
        if batch_size <= 0 or batch_size % self.batch_size:
            raise ValueError("invalid expanded batch size")
        factor = batch_size // self.batch_size
        return _transform(self, lambda x: x.repeat((factor,) + (1,) * (x.ndim - 1))).validate()


@dataclass
class HandReferenceFeatures:
    """Source-hand DINOv2 patch cache; [B,person,side,reference,patch,1536]."""

    dino_patches: torch.Tensor
    reference_valid: torch.Tensor

    @property
    def batch_size(self):
        return self.dino_patches.shape[0]

    def validate(self):
        patches = self.dino_patches
        if patches.ndim != 6 or patches.shape[1:3] != (2, 2) or patches.shape[-1] != 1536 or not 1 <= patches.shape[3] <= 3:
            raise ValueError("dino_patches must have shape [B,2,2,R,N,1536], R<=3")
        if self.reference_valid.shape != patches.shape[:4] or self.reference_valid.dtype != torch.bool:
            raise ValueError("reference_valid must be bool [B,2,2,R]")
        if patches.device != self.reference_valid.device or not torch.isfinite(patches).all():
            raise ValueError("reference tensors must be finite and colocated")
        return self

    def to(self, *args, **kwargs):
        return _transform(self, lambda x: _to_preserving_discrete_dtype(x, *args, **kwargs))

    def index_select(self, indices):
        return _transform(self, lambda x: x.index_select(0, indices))


@dataclass
class PreparedHandConditioning:
    spatial_features: torch.Tensor  # [B,2,2,256,D]
    geometry_tokens: torch.Tensor  # [B,2,2,4,D]
    appearance_tokens: torch.Tensor  # [B,2,2,8,D]
    hand_masks: torch.Tensor  # [B,2,2,16,16]
    target_boxes: torch.Tensor
    hand_valid: torch.Tensor
    visibility: torch.Tensor

    @property
    def batch_size(self):
        return self.spatial_features.shape[0]

    @property
    def device(self):
        return self.spatial_features.device

    def validate(self):
        b, _, _, _, dim = self.spatial_features.shape
        for name, shape in {
            "spatial_features": (b, 2, 2, 256, dim),
            "geometry_tokens": (b, 2, 2, 4, dim),
            "appearance_tokens": (b, 2, 2, 8, dim),
            "hand_masks": (b, 2, 2, 16, 16),
            "target_boxes": (b, 2, 2, 4),
            "visibility": (b, 2, 2),
        }.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.hand_valid.shape != (b, 2, 2) or self.hand_valid.dtype != torch.bool:
            raise ValueError("hand_valid must be bool [B,2,2]")
        return self

    def to(self, *args, **kwargs):
        return _transform(self, lambda x: _to_preserving_discrete_dtype(x, *args, **kwargs))

    def index_select(self, indices):
        return _transform(self, lambda x: x.index_select(0, indices))
