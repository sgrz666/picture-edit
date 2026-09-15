from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import Optional, Sequence

import torch


MODALITY_ORDER = ("normal", "depth", "skeleton", "silhouette", "part")

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
        value = self.contact_type[valid]
        if value.numel() and torch.any((value < 0) | (value >= contact_type_count)):
            raise ValueError(f"contact_type must be in [0, {contact_type_count - 1}]")
        return self

    def to(self, *args, **kwargs) -> "ContactRelationBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            tensor_kwargs = dict(kwargs)
            if not value.is_floating_point():
                tensor_kwargs.pop("dtype", None)
            values[item.name] = value.to(*args, **tensor_kwargs)
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
            tensor_kwargs = dict(kwargs)
            if not value.is_floating_point():
                tensor_kwargs.pop("dtype", None)
            values[item.name] = value.to(*args, **tensor_kwargs)
        return type(self)(**values)


@dataclass
class TaskSpec:
    """Batch-level task metadata kept outside the frozen tokenizer."""

    interaction_type: torch.Tensor
    declared_num_people: Optional[torch.Tensor] = None
    prompts: Optional[Sequence[str]] = None

    def index_select(self, indices: torch.Tensor) -> "TaskSpec":
        prompt_subset = None
        if self.prompts is not None:
            prompt_subset = tuple(self.prompts[index] for index in indices.tolist())
        return TaskSpec(
            interaction_type=self.interaction_type.index_select(0, indices),
            declared_num_people=(
                None
                if self.declared_num_people is None
                else self.declared_num_people.index_select(0, indices)
            ),
            prompts=prompt_subset,
        )


@dataclass
class UnifiedAdapterCondition:
    """The complete V6 condition contract.

    The person axis is always exactly two. A single-person sample masks slot B
    with ``person_valid``; it never fabricates a second person or a white
    semantic map. ``source_betas`` is intentionally the only body-shape field.
    """

    normal: torch.Tensor
    depth: torch.Tensor
    skeleton: torch.Tensor
    silhouette: torch.Tensor
    person_valid: torch.Tensor
    source_person_latents: torch.Tensor
    smpl_pose6d: torch.Tensor
    source_betas: torch.Tensor
    camera: torch.Tensor
    task_spec: TaskSpec
    part_ids: Optional[torch.Tensor] = None
    modality_valid: Optional[torch.Tensor] = None
    source_indices: Optional[torch.Tensor] = None
    contact_maps: Optional[torch.Tensor] = None
    contact_pairs: Optional[torch.Tensor] = None
    contact_valid: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.person_valid.shape[0])

    @property
    def device(self) -> torch.device:
        return self.person_valid.device

    def effective_modality_valid(self) -> torch.Tensor:
        if self.modality_valid is not None:
            return self.modality_valid.bool() & self.person_valid[..., None].bool()
        valid = self.person_valid[..., None].expand(-1, -1, len(MODALITY_ORDER)).clone()
        if self.part_ids is None:
            valid[..., MODALITY_ORDER.index("part")] = False
        return valid

    def effective_source_indices(self) -> torch.Tensor:
        if self.source_indices is not None:
            return self.source_indices.long()
        return torch.arange(2, device=self.device).expand(self.batch_size, -1)

    def validate(self) -> "UnifiedAdapterCondition":
        if self.person_valid.ndim != 2 or self.person_valid.shape[1] != 2:
            raise ValueError("person_valid must have shape [B,2]")
        counts = self.person_valid.bool().sum(dim=1)
        if torch.any((counts < 1) | (counts > 2)):
            raise ValueError("each sample must contain one or two valid people")

        batch_size = self.batch_size
        height, width = self.normal.shape[-2:]
        image_specs = {
            "normal": (3, self.normal),
            "depth": (1, self.depth),
            "skeleton": (3, self.skeleton),
            "silhouette": (1, self.silhouette),
        }
        for name, (channels, tensor) in image_specs.items():
            expected = (batch_size, 2, channels, height, width)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")

        if self.part_ids is not None and tuple(self.part_ids.shape) != (
            batch_size,
            2,
            height,
            width,
        ):
            raise ValueError("part_ids must have shape [B,2,H,W]")
        if self.modality_valid is not None and tuple(self.modality_valid.shape) != (
            batch_size,
            2,
            len(MODALITY_ORDER),
        ):
            raise ValueError("modality_valid must have shape [B,2,5]")
        if self.source_indices is not None and tuple(self.source_indices.shape) != (batch_size, 2):
            raise ValueError("source_indices must have shape [B,2]")
        if self.source_person_latents.ndim != 5 or self.source_person_latents.shape[:3] != (
            batch_size,
            2,
            16,
        ):
            raise ValueError("source_person_latents must have shape [B,2,16,h,w]")
        if tuple(self.smpl_pose6d.shape) != (batch_size, 2, 52, 6):
            raise ValueError("smpl_pose6d must have shape [B,2,52,6]")
        if tuple(self.source_betas.shape) != (batch_size, 2, 10):
            raise ValueError("source_betas must have shape [B,2,10]")
        if tuple(self.camera.shape) != (batch_size, 2, 7):
            raise ValueError("camera must have shape [B,2,7]")
        if self.contact_maps is not None and tuple(self.contact_maps.shape) != (
            batch_size,
            2,
            height,
            width,
        ):
            raise ValueError("contact_maps must have shape [B,2,H,W]")
        if self.contact_pairs is not None:
            if self.contact_pairs.ndim != 3 or self.contact_pairs.shape[0] != batch_size:
                raise ValueError("contact_pairs must have shape [B,K,D]")
            if self.contact_valid is None or tuple(self.contact_valid.shape) != tuple(
                self.contact_pairs.shape[:2]
            ):
                raise ValueError("contact_valid must match contact_pairs [B,K]")
        if self.task_spec.interaction_type.shape != (batch_size,):
            raise ValueError("task_spec.interaction_type must have shape [B]")
        if self.task_spec.declared_num_people is not None and self.task_spec.declared_num_people.shape != (
            batch_size,
        ):
            raise ValueError("declared_num_people must have shape [B]")
        if self.task_spec.prompts is not None and len(self.task_spec.prompts) != batch_size:
            raise ValueError("task prompts must match batch size")
        return self

    def index_select(self, indices: torch.Tensor) -> "UnifiedAdapterCondition":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "task_spec":
                values[field.name] = value.index_select(indices)
            elif torch.is_tensor(value):
                values[field.name] = value.index_select(0, indices)
            else:
                values[field.name] = value
        return type(self)(**values)

    def to(self, *args, **kwargs) -> "UnifiedAdapterCondition":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "task_spec":
                task = value
                task_kwargs = dict(kwargs)
                task_kwargs.pop("dtype", None)
                values[field.name] = replace(
                    task,
                    interaction_type=task.interaction_type.to(*args, **task_kwargs),
                    declared_num_people=(
                        None
                        if task.declared_num_people is None
                        else task.declared_num_people.to(*args, **task_kwargs)
                    ),
                )
            elif torch.is_tensor(value):
                # Integer IDs/masks keep their semantic dtype.
                dtype = kwargs.get("dtype")
                if dtype is not None and not value.is_floating_point():
                    local_kwargs = dict(kwargs)
                    local_kwargs.pop("dtype")
                    values[field.name] = value.to(*args, **local_kwargs)
                else:
                    values[field.name] = value.to(*args, **kwargs)
            else:
                values[field.name] = value
        return type(self)(**values)
