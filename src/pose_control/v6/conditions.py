from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Optional

import torch


def _to_preserving_discrete_dtype(
    value: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Apply Tensor.to while ignoring dtype requests for masks and indices."""

    if value.is_floating_point():
        return value.to(*args, **kwargs)
    device = kwargs.get("device")
    if args:
        target = args[0]
        if isinstance(target, torch.Tensor):
            device = target.device
        elif isinstance(target, (torch.device, str, int)):
            device = target
    return value.to(
        device=value.device if device is None else device,
        non_blocking=bool(kwargs.get("non_blocking", False)),
        copy=bool(kwargs.get("copy", False)),
    )


PART_NAMES = (
    "head",
    "torso",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
)


class TaskType(IntEnum):
    SINGLE = 0
    DUAL = 1
    HUG = 2
    HANDSHAKE = 3
    SHOULDER_HUG = 4
    HOLD_HAND = 5
    UNKNOWN = 6


@dataclass
class ContactRelationBatch:
    """Typed, padded contact relations for at most eight person-part pairs."""

    src_person: torch.Tensor
    src_part: torch.Tensor
    dst_person: torch.Tensor
    dst_part: torch.Tensor
    contact_type: torch.Tensor
    distance: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.valid_mask.shape[0])

    def validate(
        self,
        *,
        batch_size: int,
        max_relations: int = 8,
        num_parts: int = len(PART_NAMES),
        contact_type_count: int = 16,
    ) -> "ContactRelationBatch":
        expected = (batch_size, max_relations)
        for name in ("src_person", "src_part", "dst_person", "dst_part", "contact_type"):
            value = getattr(self, name)
            if tuple(value.shape) != expected or value.dtype != torch.long:
                raise ValueError(f"{name} must have shape {expected} and dtype int64")
        if tuple(self.distance.shape) != (*expected, 1) or not self.distance.is_floating_point():
            raise ValueError(f"distance must have shape {(*expected, 1)} and floating dtype")
        if tuple(self.valid_mask.shape) != expected or self.valid_mask.dtype != torch.bool:
            raise ValueError(f"valid_mask must have shape {expected} and dtype bool")
        if not torch.isfinite(self.distance).all():
            raise ValueError("contact distance must contain only finite values")
        valid = self.valid_mask
        for name in ("src_person", "dst_person"):
            value = getattr(self, name)[valid]
            if value.numel() and torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must contain person indices 0 or 1")
        for name in ("src_part", "dst_part"):
            value = getattr(self, name)[valid]
            if value.numel() and torch.any((value < 0) | (value >= num_parts)):
                raise ValueError(f"{name} must contain part ids in [0, {num_parts - 1}]")
        contact_types = self.contact_type[valid]
        if contact_types.numel() and torch.any(
            (contact_types < 0) | (contact_types >= contact_type_count)
        ):
            raise ValueError(f"contact_type must be in [0, {contact_type_count - 1}]")
        return self

    def to(self, *args, **kwargs) -> "ContactRelationBatch":
        values = {
            item.name: _to_preserving_discrete_dtype(
                getattr(self, item.name), *args, **kwargs
            )
            for item in fields(self)
        }
        return type(self)(**values)


@dataclass
class ConditionBundle:
    """Normalized SMPL-X conditions consumed by the V6 reasoning layer."""

    person_a_spatial: torch.Tensor
    person_b_spatial: Optional[torch.Tensor]
    person_a_mask: torch.Tensor
    person_b_mask: Optional[torch.Tensor]
    person_a_global_tokens: torch.Tensor
    person_b_global_tokens: Optional[torch.Tensor]
    relative_tokens: Optional[torch.Tensor]
    contact_spatial: Optional[torch.Tensor]
    contact_tokens: Optional[torch.Tensor]
    contact_mask: Optional[torch.Tensor]
    task_token: torch.Tensor
    person_valid: torch.Tensor
    person_count: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.person_valid.shape[0])

    @property
    def device(self) -> torch.device:
        return self.person_a_spatial.device

    def validate(self) -> "ConditionBundle":
        batch_size = self.batch_size
        if tuple(self.person_valid.shape) != (batch_size, 2) or self.person_valid.dtype != torch.bool:
            raise ValueError("person_valid must have shape [B,2] and dtype bool")
        if not torch.all(self.person_valid[:, 0]):
            raise ValueError("person A must be valid in every sample")
        expected_count = self.person_valid.sum(dim=1).long()
        if tuple(self.person_count.shape) != (batch_size,) or not torch.equal(
            self.person_count.long(), expected_count
        ):
            raise ValueError("person_count must equal person_valid.sum(dim=1)")
        height, width = self.person_a_spatial.shape[-2:]
        shapes = {
            "person_a_spatial": ((batch_size, 256, height, width), self.person_a_spatial),
            "person_a_mask": ((batch_size, 1, height, width), self.person_a_mask),
            "person_a_global_tokens": ((batch_size, 4, 256), self.person_a_global_tokens),
            "task_token": ((batch_size, 1, 256), self.task_token),
        }
        for name, (expected, value) in shapes.items():
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        b_values = (self.person_b_spatial, self.person_b_mask, self.person_b_global_tokens)
        if any(value is not None for value in b_values) != all(value is not None for value in b_values):
            raise ValueError("person-B bundle fields must be all present or all absent")
        if self.person_b_spatial is None and torch.any(self.person_valid[:, 1]):
            raise ValueError("valid person B requires person-B bundle fields")
        if self.person_b_spatial is not None:
            expected_b = (
                (batch_size, 256, height, width),
                (batch_size, 1, height, width),
                (batch_size, 4, 256),
            )
            for value, expected in zip(b_values, expected_b):
                if tuple(value.shape) != expected:
                    raise ValueError(f"invalid person-B bundle shape; expected {expected}")
        if self.relative_tokens is not None and tuple(self.relative_tokens.shape) != (
            batch_size,
            2,
            256,
        ):
            raise ValueError("relative_tokens must have shape [B,2,256]")
        if self.contact_spatial is not None and tuple(self.contact_spatial.shape) != (
            batch_size,
            128,
            height,
            width,
        ):
            raise ValueError("contact_spatial must have shape [B,128,h,w]")
        if (self.contact_tokens is None) != (self.contact_mask is None):
            raise ValueError("contact_tokens and contact_mask must be provided together")
        if self.contact_tokens is not None:
            if tuple(self.contact_tokens.shape) != (batch_size, 8, 256):
                raise ValueError("contact_tokens must have shape [B,8,256]")
            if tuple(self.contact_mask.shape) != (batch_size, 8) or self.contact_mask.dtype != torch.bool:
                raise ValueError("contact_mask must have shape [B,8] and dtype bool")
        return self

    def index_select(self, indices: torch.Tensor) -> "ConditionBundle":
        return type(self)(
            **{
                item.name: (
                    None
                    if getattr(self, item.name) is None
                    else getattr(self, item.name).index_select(0, indices)
                )
                for item in fields(self)
            }
        )

    def to(self, *args, **kwargs) -> "ConditionBundle":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                values[item.name] = None
                continue
            values[item.name] = _to_preserving_discrete_dtype(value, *args, **kwargs)
        return type(self)(**values)


@dataclass
class AdapterIdentityCondition:
    """Source-appearance inputs intentionally kept outside ConditionBundle."""

    source_person_latents: torch.Tensor
    source_indices: Optional[torch.Tensor] = None

    def effective_source_indices(self) -> torch.Tensor:
        if self.source_indices is not None:
            return self.source_indices.long()
        return torch.arange(2, device=self.source_person_latents.device).expand(
            self.source_person_latents.shape[0], -1
        )

    def validate(self, bundle: ConditionBundle) -> "AdapterIdentityCondition":
        if self.source_person_latents.ndim != 5 or self.source_person_latents.shape[:3] != (
            bundle.batch_size,
            2,
            16,
        ):
            raise ValueError("source_person_latents must have shape [B,2,16,h,w]")
        if self.source_indices is not None and tuple(self.source_indices.shape) != (
            bundle.batch_size,
            2,
        ):
            raise ValueError("source_indices must have shape [B,2]")
        if self.source_indices is not None and self.source_indices.device != bundle.device:
            raise ValueError("source_indices and ConditionBundle must be on the same device")
        if self.source_person_latents.device != bundle.device:
            raise ValueError("source_person_latents and ConditionBundle must be on the same device")
        return self

    def index_select(self, indices: torch.Tensor) -> "AdapterIdentityCondition":
        return type(self)(
            source_person_latents=self.source_person_latents.index_select(0, indices),
            source_indices=(
                None
                if self.source_indices is None
                else self.source_indices.index_select(0, indices)
            ),
        )

    def to(self, *args, **kwargs) -> "AdapterIdentityCondition":
        return type(self)(
            source_person_latents=_to_preserving_discrete_dtype(
                self.source_person_latents, *args, **kwargs
            ),
            source_indices=(
                None
                if self.source_indices is None
                else _to_preserving_discrete_dtype(self.source_indices, *args, **kwargs)
            ),
        )
