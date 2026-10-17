from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..conditions import _to_preserving_discrete_dtype
from ..face import FaceFineCondition, dwpose68_to_face72
from .conditions import DetailReferenceBatch, FaceHandDetailCondition


def _map_fields(instance, transform):
    return type(instance)(
        **{
            field.name: transform(getattr(instance, field.name))
            for field in fields(instance)
        }
    )


@dataclass
class HandDetailCondition:
    """Hand-only geometry and source/target bindings for two people."""

    hand_keypoints: torch.Tensor
    hand_pose: torch.Tensor
    source_boxes: torch.Tensor
    target_boxes: torch.Tensor
    region_valid: torch.Tensor
    source_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.hand_keypoints.shape[0])

    @property
    def person_valid(self) -> torch.Tensor:
        return self.region_valid.any(dim=-1)

    @property
    def device(self) -> torch.device:
        return self.hand_keypoints.device

    def validate(self) -> "HandDetailCondition":
        batch_size = self.batch_size
        if batch_size == 0:
            raise ValueError("hand condition batch must be non-empty")
        shapes = {
            "hand_keypoints": (batch_size, 2, 2, 21, 3),
            "hand_pose": (batch_size, 2, 2, 45),
            "source_boxes": (batch_size, 2, 2, 4),
            "target_boxes": (batch_size, 2, 2, 4),
        }
        floating: list[torch.Tensor] = []
        for name, shape in shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.is_floating_point():
                raise ValueError(f"{name} must use a floating dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            floating.append(value)
        reference = floating[0]
        if any(value.dtype != reference.dtype for value in floating[1:]):
            raise ValueError("all floating hand tensors must use the same dtype")
        if any(value.device != reference.device for value in floating[1:]):
            raise ValueError("all hand tensors must use the same device")
        if (
            tuple(self.region_valid.shape) != (batch_size, 2, 2)
            or self.region_valid.dtype != torch.bool
        ):
            raise ValueError("region_valid must have shape [B,2,2] and dtype bool")
        if (
            tuple(self.source_indices.shape) != (batch_size, 2)
            or self.source_indices.dtype != torch.long
        ):
            raise ValueError("source_indices must have shape [B,2] and dtype int64")
        if (
            self.region_valid.device != reference.device
            or self.source_indices.device != reference.device
        ):
            raise ValueError("all hand tensors must use the same device")
        if torch.any((self.hand_keypoints < 0) | (self.hand_keypoints > 1)):
            raise ValueError("hand_keypoints xy/confidence values must lie in [0,1]")
        for name in ("source_boxes", "target_boxes"):
            boxes = getattr(self, name)
            if torch.any((boxes < 0) | (boxes > 1)):
                raise ValueError(f"{name} must contain normalized xyxy boxes")
            positive = (boxes[..., 0] < boxes[..., 2]) & (
                boxes[..., 1] < boxes[..., 3]
            )
            if self.region_valid.any() and not positive[self.region_valid].all():
                raise ValueError(f"valid {name} must contain positive-area boxes")
        valid_indices = self.source_indices[self.person_valid]
        if valid_indices.numel() and torch.any(
            (valid_indices < 0) | (valid_indices > 15)
        ):
            raise ValueError("valid source_indices must be in [0,15]")
        both_valid = self.person_valid.all(dim=1)
        if torch.any(
            both_valid & (self.source_indices[:, 0] == self.source_indices[:, 1])
        ):
            raise ValueError("valid people must use unique source_indices")
        return self

    def to(self, *args, **kwargs) -> "HandDetailCondition":
        return _map_fields(
            self,
            lambda value: _to_preserving_discrete_dtype(value, *args, **kwargs),
        )

    def index_select(self, indices: torch.Tensor) -> "HandDetailCondition":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a one-dimensional int64 tensor")
        return _map_fields(self, lambda value: value.index_select(0, indices))

    def expand_to_batch(self, batch_size: int) -> "HandDetailCondition":
        if batch_size <= 0 or batch_size % self.batch_size:
            raise ValueError("expanded batch must be a positive multiple")
        repeats = batch_size // self.batch_size
        return _map_fields(
            self,
            lambda value: value.repeat((repeats,) + (1,) * (value.ndim - 1)),
        ).validate()


def legacy_to_face_and_hand(
    condition: FaceHandDetailCondition,
) -> tuple[FaceFineCondition, HandDetailCondition]:
    """Split the V6.4 shared condition without sharing either branch input."""

    condition.validate()
    face = FaceFineCondition(
        landmarks=dwpose68_to_face72(condition.face_keypoints),
        expression=condition.smplx_detail[..., :10],
        jaw_pose=condition.smplx_detail[..., 10:13],
        source_boxes=condition.source_boxes[:, :, 0],
        target_boxes=condition.target_boxes[:, :, 0],
        face_valid=condition.region_valid[:, :, 0],
        source_indices=condition.source_indices,
    ).validate()
    hand_pose = torch.stack(
        (
            condition.smplx_detail[..., 13:58],
            condition.smplx_detail[..., 58:103],
        ),
        dim=2,
    )
    hand = HandDetailCondition(
        hand_keypoints=condition.hand_keypoints,
        hand_pose=hand_pose,
        source_boxes=condition.source_boxes[:, :, 1:],
        target_boxes=condition.target_boxes[:, :, 1:],
        region_valid=condition.region_valid[:, :, 1:],
        source_indices=condition.source_indices,
    ).validate()
    return face, hand


def hand_to_legacy_detail(condition: HandDetailCondition) -> FaceHandDetailCondition:
    """Build a V6.4-shaped hand-only payload for the retained detail branch."""

    condition.validate()
    batch_size = condition.batch_size
    hand_keypoints = condition.hand_keypoints
    false_face = torch.zeros(
        batch_size, 2, 1, device=condition.device, dtype=torch.bool
    )
    face_boxes = hand_keypoints.new_zeros(batch_size, 2, 1, 4)
    smplx_detail = hand_keypoints.new_zeros(batch_size, 2, 103)
    smplx_detail[..., 13:58] = condition.hand_pose[:, :, 0]
    smplx_detail[..., 58:103] = condition.hand_pose[:, :, 1]
    return FaceHandDetailCondition(
        face_keypoints=hand_keypoints.new_zeros(batch_size, 2, 68, 3),
        hand_keypoints=hand_keypoints,
        smplx_detail=smplx_detail,
        source_boxes=torch.cat((face_boxes, condition.source_boxes), dim=2),
        target_boxes=torch.cat((face_boxes, condition.target_boxes), dim=2),
        region_valid=torch.cat((false_face, condition.region_valid), dim=2),
        source_indices=condition.source_indices,
    )


def references_to_hand_only(
    references: DetailReferenceBatch,
) -> DetailReferenceBatch:
    """Copy legacy references while erasing face pixels and validity."""

    references.validate()
    images = references.images.clone()
    reference_valid = references.reference_valid.clone()
    images[:, :, 0].zero_()
    reference_valid[:, :, 0] = False
    return DetailReferenceBatch(images=images, reference_valid=reference_valid)


def legacy_to_hand_only(
    condition: FaceHandDetailCondition,
    references: DetailReferenceBatch | None = None,
) -> tuple[FaceHandDetailCondition, DetailReferenceBatch | None]:
    """Deprecated V6.4 compatibility path with face leakage removed."""

    _, hand = legacy_to_face_and_hand(condition)
    return hand_to_legacy_detail(hand), (
        None if references is None else references_to_hand_only(references)
    )


__all__ = [
    "HandDetailCondition",
    "hand_to_legacy_detail",
    "legacy_to_face_and_hand",
    "legacy_to_hand_only",
    "references_to_hand_only",
]
