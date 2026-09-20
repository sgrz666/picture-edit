from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch

from ..conditions import _to_preserving_discrete_dtype


class DetailRegion(IntEnum):
    FACE = 0
    LEFT_HAND = 1
    RIGHT_HAND = 2


def _selected(instance, indices: torch.Tensor):
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


@dataclass
class FaceHandDetailCondition:
    """Normalized two-person face/hand geometry and region bindings."""

    face_keypoints: torch.Tensor
    hand_keypoints: torch.Tensor
    smplx_detail: torch.Tensor
    source_boxes: torch.Tensor
    target_boxes: torch.Tensor
    region_valid: torch.Tensor
    source_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.face_keypoints.shape[0])

    @property
    def person_valid(self) -> torch.Tensor:
        return self.region_valid.any(dim=-1)

    @property
    def device(self) -> torch.device:
        return self.face_keypoints.device

    def validate(self) -> "FaceHandDetailCondition":
        batch_size = self.batch_size
        expected_shapes = {
            "face_keypoints": (batch_size, 2, 68, 3),
            "hand_keypoints": (batch_size, 2, 2, 21, 3),
            "smplx_detail": (batch_size, 2, 103),
            "source_boxes": (batch_size, 2, 3, 4),
            "target_boxes": (batch_size, 2, 3, 4),
        }
        floating = []
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if not value.is_floating_point():
                raise ValueError(f"{name} must use a floating dtype")
            floating.append(value)
        if any(value.dtype != floating[0].dtype for value in floating[1:]):
            raise ValueError("all floating tensors must use the same dtype")
        if any(value.device != floating[0].device for value in floating[1:]):
            raise ValueError("all condition tensors must be on the same device")
        if tuple(self.region_valid.shape) != (batch_size, 2, 3) or self.region_valid.dtype != torch.bool:
            raise ValueError("region_valid must have shape [B,2,3] and dtype bool")
        if tuple(self.source_indices.shape) != (batch_size, 2) or self.source_indices.dtype != torch.long:
            raise ValueError("source_indices must have shape [B,2] and dtype int64")
        if self.region_valid.device != self.device or self.source_indices.device != self.device:
            raise ValueError("all condition tensors must be on the same device")
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("detail condition tensors must contain only finite values")

        for name in ("face_keypoints", "hand_keypoints"):
            keypoints = getattr(self, name)
            if torch.any((keypoints < 0) | (keypoints > 1)):
                raise ValueError(f"{name} must contain normalized xy/confidence values in [0,1]")

        for name in ("source_boxes", "target_boxes"):
            boxes = getattr(self, name)
            if torch.any((boxes < 0) | (boxes > 1)):
                raise ValueError(f"{name} must contain normalized boxes in [0,1]")
            ordered = (boxes[..., 0] <= boxes[..., 2]) & (
                boxes[..., 1] <= boxes[..., 3]
            )
            positive = (boxes[..., 0] < boxes[..., 2]) & (
                boxes[..., 1] < boxes[..., 3]
            )
            if not ordered.all() or not positive[self.region_valid].all():
                raise ValueError(f"{name} must use valid normalized xyxy boxes")

        valid_indices = self.source_indices[self.person_valid]
        if valid_indices.numel() and torch.any(
            (valid_indices < 0) | (valid_indices > 15)
        ):
            raise ValueError("valid source_indices must be in [0, 15]")
        both_valid = self.person_valid.all(dim=1)
        if torch.any(
            both_valid & (self.source_indices[:, 0] == self.source_indices[:, 1])
        ):
            raise ValueError("valid persons must use unique source_indices within each sample")
        return self

    def index_select(self, indices: torch.Tensor) -> "FaceHandDetailCondition":
        return _selected(self, indices)

    def to(self, *args, **kwargs) -> "FaceHandDetailCondition":
        return _moved(self, *args, **kwargs)


@dataclass
class DetailReferenceBatch:
    """Up to three local appearance references per person and detail region."""

    images: torch.Tensor
    reference_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def reference_count(self) -> int:
        return int(self.images.shape[3])

    @property
    def device(self) -> torch.device:
        return self.images.device

    def validate(self) -> "DetailReferenceBatch":
        if self.images.ndim != 7:
            raise ValueError("images must have shape [B,2,3,R,3,224,224]")
        batch_size, people, regions, reference_count, channels, height, width = self.images.shape
        if (people, regions, channels, height, width) != (2, 3, 3, 224, 224):
            raise ValueError("images must have shape [B,2,3,R,3,224,224]")
        if not 1 <= reference_count <= 3:
            raise ValueError("detail references must contain at most 3 images and at least 1")
        expected = (batch_size, 2, 3, reference_count)
        if tuple(self.reference_valid.shape) != expected or self.reference_valid.dtype != torch.bool:
            raise ValueError(f"reference_valid must have shape {expected} and dtype bool")
        if not self.images.is_floating_point():
            raise ValueError("reference images must use a floating dtype")
        if self.reference_valid.device != self.images.device:
            raise ValueError("reference images and reference_valid must be on the same device")
        if not torch.isfinite(self.images).all():
            raise ValueError("reference images must contain only finite values")
        return self

    def index_select(self, indices: torch.Tensor) -> "DetailReferenceBatch":
        return _selected(self, indices)

    def to(self, *args, **kwargs) -> "DetailReferenceBatch":
        return _moved(self, *args, **kwargs)


@dataclass
class PreparedFaceHandDetailConditioning:
    """Static face/hand detail tensors ready for the recurrent control branch."""

    detail_condition: torch.Tensor
    detail_tokens: torch.Tensor
    detail_token_mask: torch.Tensor
    region_masks: torch.Tensor
    region_valid: torch.Tensor
    detail_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.detail_condition.shape[0])

    @property
    def device(self) -> torch.device:
        return self.detail_condition.device

    def validate(self) -> "PreparedFaceHandDetailConditioning":
        batch_size = self.batch_size
        if self.detail_condition.ndim != 4 or self.detail_condition.shape[1] != 144:
            raise ValueError("detail_condition must have shape [B,144,Hlat,Wlat]")
        if self.detail_tokens.ndim != 3 or self.detail_tokens.shape[:2] != (batch_size, 48):
            raise ValueError("detail_tokens must have shape [B,48,D]")
        if tuple(self.detail_token_mask.shape) != (batch_size, 48) or self.detail_token_mask.dtype != torch.bool:
            raise ValueError("detail_token_mask must have shape [B,48] and dtype bool")
        if self.region_masks.ndim != 5 or self.region_masks.shape[:3] != (batch_size, 2, 3):
            raise ValueError("region_masks must have shape [B,2,3,Htoken,Wtoken]")
        if tuple(self.region_valid.shape) != (batch_size, 2, 3) or self.region_valid.dtype != torch.bool:
            raise ValueError("region_valid must have shape [B,2,3] and dtype bool")
        if tuple(self.detail_valid.shape) != (batch_size,) or self.detail_valid.dtype != torch.bool:
            raise ValueError("detail_valid must have shape [B] and dtype bool")
        floating = (self.detail_condition, self.detail_tokens, self.region_masks)
        if any(not value.is_floating_point() for value in floating):
            raise ValueError("prepared detail floating tensors require a floating dtype")
        if any(value.dtype != floating[0].dtype for value in floating[1:]):
            raise ValueError("prepared detail floating tensors must share a dtype")
        tensors = (*floating, self.detail_token_mask, self.region_valid, self.detail_valid)
        if any(value.device != self.device for value in tensors[1:]):
            raise ValueError("prepared detail tensors must be on the same device")
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("prepared detail tensors must contain only finite values")
        if torch.any((self.region_masks < 0) | (self.region_masks > 1)):
            raise ValueError("region_masks must be bounded in [0,1]")
        expected_mask = self.region_valid[..., None].expand(
            -1, -1, -1, 8
        ).reshape(batch_size, 48)
        if not torch.equal(self.detail_token_mask, expected_mask):
            raise ValueError("detail_token_mask must repeat each region validity eight times")
        if torch.count_nonzero(self.detail_tokens[~self.detail_token_mask]):
            raise ValueError("invalid detail tokens must be exactly zero")
        if torch.count_nonzero(self.region_masks[~self.region_valid]):
            raise ValueError("invalid region masks must be exactly zero")
        expected_valid = self.region_valid.flatten(1).any(dim=1)
        if not torch.equal(self.detail_valid, expected_valid):
            raise ValueError("detail_valid must equal region_valid.any per sample")
        if torch.count_nonzero(self.detail_condition[~self.detail_valid]):
            raise ValueError("invalid sample detail_condition must be exactly zero")
        return self

    def index_select(self, indices: torch.Tensor) -> "PreparedFaceHandDetailConditioning":
        return _selected(self, indices)

    def to(self, *args, **kwargs) -> "PreparedFaceHandDetailConditioning":
        return _moved(self, *args, **kwargs)

    def expand_to_batch(self, batch_size: int) -> "PreparedFaceHandDetailConditioning":
        if batch_size <= 0 or batch_size % self.batch_size:
            raise ValueError("expanded batch size must be a positive multiple of the current batch")
        repeats = batch_size // self.batch_size
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
        return type(self)(**values).validate()
