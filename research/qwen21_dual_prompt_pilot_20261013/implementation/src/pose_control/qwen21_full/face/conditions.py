from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..conditions import _to_preserving_discrete_dtype


def _index_selected(instance, indices: torch.Tensor):
    if indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional int64 tensor")
    return type(instance)(
        **{
            item.name: getattr(instance, item.name).index_select(0, indices)
            for item in fields(instance)
        }
    )


def _moved(instance, *args, **kwargs):
    return type(instance)(
        **{
            item.name: _to_preserving_discrete_dtype(
                getattr(instance, item.name), *args, **kwargs
            )
            for item in fields(instance)
        }
    )


def _expanded(instance, batch_size: int):
    current = instance.batch_size
    if current == 0:
        raise ValueError("batch must be non-empty before expansion")
    if batch_size <= 0 or batch_size % current:
        raise ValueError(
            "expanded batch size must be a positive multiple of the current batch"
        )
    repeats = batch_size // current
    return type(instance)(
        **{
            item.name: getattr(instance, item.name).repeat(
                (repeats,) + (1,) * (getattr(instance, item.name).ndim - 1)
            )
            for item in fields(instance)
        }
    ).validate()


def _validate_same_float_dtype_device(
    named_tensors: dict[str, torch.Tensor],
) -> tuple[torch.dtype, torch.device]:
    first_name, first = next(iter(named_tensors.items()))
    if not first.is_floating_point():
        raise ValueError(f"{first_name} must use a floating dtype")
    for name, value in named_tensors.items():
        if not value.is_floating_point():
            raise ValueError(f"{name} must use a floating dtype")
        if value.dtype != first.dtype:
            raise ValueError("all floating tensors must use the same dtype")
        if value.device != first.device:
            raise ValueError("all tensors must be on the same device")
        if not torch.isfinite(value).all():
            raise ValueError("floating tensors must contain only finite values")
    return first.dtype, first.device


def _validate_normalized_boxes(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    name: str,
) -> None:
    if torch.any((boxes < 0) | (boxes > 1)):
        raise ValueError(f"{name} must contain normalized xyxy boxes in [0,1]")
    positive = (boxes[..., 0] < boxes[..., 2]) & (
        boxes[..., 1] < boxes[..., 3]
    )
    if valid.any() and not positive[valid].all():
        raise ValueError(f"valid {name} must contain positive-area xyxy boxes")


@dataclass
class FaceFineCondition:
    """Fine face geometry and two-person source/target bindings."""

    landmarks: torch.Tensor
    jaw_pose: torch.Tensor
    expression: torch.Tensor
    source_boxes: torch.Tensor
    target_boxes: torch.Tensor
    face_valid: torch.Tensor
    source_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def device(self) -> torch.device:
        return self.landmarks.device

    def validate(self) -> "FaceFineCondition":
        batch_size = self.batch_size
        if batch_size == 0:
            raise ValueError("face condition batch must be non-empty")
        expected_shapes = {
            "landmarks": (batch_size, 2, 72, 3),
            "jaw_pose": (batch_size, 2, 3),
            "expression": (batch_size, 2, 10),
            "source_boxes": (batch_size, 2, 4),
            "target_boxes": (batch_size, 2, 4),
        }
        floating = {}
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            floating[name] = value
        _, device = _validate_same_float_dtype_device(floating)
        if tuple(self.face_valid.shape) != (batch_size, 2) or self.face_valid.dtype != torch.bool:
            raise ValueError("face_valid must have shape [B,2] and dtype bool")
        if (
            tuple(self.source_indices.shape) != (batch_size, 2)
            or self.source_indices.dtype != torch.long
        ):
            raise ValueError("source_indices must have shape [B,2] and dtype int64")
        if self.face_valid.device != device or self.source_indices.device != device:
            raise ValueError("all tensors must be on the same device")
        if torch.any((self.landmarks[..., 2] < 0) | (self.landmarks[..., 2] > 1)):
            raise ValueError("landmark confidence must be in [0,1]")
        _validate_normalized_boxes(
            self.source_boxes, self.face_valid, name="source_boxes"
        )
        _validate_normalized_boxes(
            self.target_boxes, self.face_valid, name="target_boxes"
        )
        valid_indices = self.source_indices[self.face_valid]
        if valid_indices.numel() and torch.any(
            (valid_indices < 0) | (valid_indices > 15)
        ):
            raise ValueError("valid source_indices must be in [0, 15]")
        both_valid = self.face_valid.all(dim=1)
        duplicates = self.source_indices[:, 0] == self.source_indices[:, 1]
        if torch.any(both_valid & duplicates):
            raise ValueError(
                "valid people must use unique source_indices within each sample"
            )
        return self

    def to(self, *args, **kwargs) -> "FaceFineCondition":
        return _moved(self, *args, **kwargs)

    def index_select(self, indices: torch.Tensor) -> "FaceFineCondition":
        return _index_selected(self, indices)

    def expand_to_batch(self, batch_size: int) -> "FaceFineCondition":
        return _expanded(self, batch_size)


@dataclass
class FaceReferenceFeatures:
    """One to three precomputed ArcFace and DINO references per person."""

    arcface: torch.Tensor
    dino_patches: torch.Tensor
    reference_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.arcface.shape[0])

    @property
    def reference_count(self) -> int:
        return int(self.arcface.shape[2])

    @property
    def device(self) -> torch.device:
        return self.arcface.device

    def validate(self) -> "FaceReferenceFeatures":
        if self.arcface.ndim != 4:
            raise ValueError("arcface must have shape [B,2,R,512]")
        batch_size, people, reference_count, feature_dim = self.arcface.shape
        if people != 2 or feature_dim != 512:
            raise ValueError("arcface must have shape [B,2,R,512]")
        if batch_size == 0:
            raise ValueError("face reference batch must be non-empty")
        if not 1 <= reference_count <= 3:
            raise ValueError("face reference count must be between 1 and 3")
        expected_dino = (batch_size, 2, reference_count, 256, 1536)
        if tuple(self.dino_patches.shape) != expected_dino:
            raise ValueError(f"dino_patches must have shape {expected_dino}")
        expected_valid = (batch_size, 2, reference_count)
        if (
            tuple(self.reference_valid.shape) != expected_valid
            or self.reference_valid.dtype != torch.bool
        ):
            raise ValueError(
                f"reference_valid must have shape {expected_valid} and dtype bool"
            )
        _, device = _validate_same_float_dtype_device(
            {"arcface": self.arcface, "dino_patches": self.dino_patches}
        )
        if self.reference_valid.device != device:
            raise ValueError("all tensors must be on the same device")
        return self

    def to(self, *args, **kwargs) -> "FaceReferenceFeatures":
        return _moved(self, *args, **kwargs)

    def index_select(self, indices: torch.Tensor) -> "FaceReferenceFeatures":
        return _index_selected(self, indices)

    def expand_to_batch(self, batch_size: int) -> "FaceReferenceFeatures":
        return _expanded(self, batch_size)


@dataclass
class PreparedFaceConditioning:
    """Validated ROI-local face tensors ready for a Face Control Adapter."""

    spatial_features: torch.Tensor
    identity_tokens: torch.Tensor
    texture_tokens: torch.Tensor
    texture_delta: torch.Tensor
    geometry_tokens: torch.Tensor
    face_masks: torch.Tensor
    target_boxes: torch.Tensor
    face_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.spatial_features.shape[0])

    @property
    def device(self) -> torch.device:
        return self.spatial_features.device

    def validate(self) -> "PreparedFaceConditioning":
        batch_size = self.batch_size
        if batch_size == 0:
            raise ValueError("prepared face batch must be non-empty")
        expected = {
            "spatial_features": (batch_size, 2, 512, 16, 16),
            "identity_tokens": (batch_size, 2, 4, 512),
            "texture_tokens": (batch_size, 2, 8, 512),
            "texture_delta": (batch_size, 2, 4, 512),
            "geometry_tokens": (batch_size, 2, 4, 512),
            "target_boxes": (batch_size, 2, 4),
        }
        floating = {}
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            floating[name] = value
        if self.face_masks.ndim != 4 or self.face_masks.shape[:2] != (
            batch_size,
            2,
        ):
            raise ValueError("face_masks must have shape [B,2,H,W]")
        if min(self.face_masks.shape[-2:]) <= 0:
            raise ValueError("face mask spatial dimensions must be positive")
        floating["face_masks"] = self.face_masks
        _, device = _validate_same_float_dtype_device(floating)
        if tuple(self.face_valid.shape) != (batch_size, 2) or self.face_valid.dtype != torch.bool:
            raise ValueError("face_valid must have shape [B,2] and dtype bool")
        if self.face_valid.device != device:
            raise ValueError("all tensors must be on the same device")
        if torch.any((self.face_masks < 0) | (self.face_masks > 1)):
            raise ValueError("face_masks must be bounded in [0,1]")
        _validate_normalized_boxes(
            self.target_boxes, self.face_valid, name="target_boxes"
        )
        invalid = ~self.face_valid
        for name, value in floating.items():
            if torch.count_nonzero(value[invalid]):
                raise ValueError(f"invalid people must have exactly zero {name}")
        return self

    def to(self, *args, **kwargs) -> "PreparedFaceConditioning":
        return _moved(self, *args, **kwargs)

    def index_select(self, indices: torch.Tensor) -> "PreparedFaceConditioning":
        return _index_selected(self, indices)

    def expand_to_batch(self, batch_size: int) -> "PreparedFaceConditioning":
        return _expanded(self, batch_size)
