"""从原生 Qwen 元数据选择 Pose token，不假设图像是连续前缀。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


REFERENCE_ROLE_LAYOUTS = (
    ("source", "target_pose"),
    ("source", "combined_pose", "person_0_pose", "person_1_pose"),
    ("source_a", "source_b", "person_0_pose", "person_1_pose", "combined_pose"),
)
POSE_ROLES = frozenset(("target_pose", "combined_pose", "person_0_pose", "person_1_pose"))


@dataclass(frozen=True)
class PoseReference:
    role: str
    image_id: int
    indices: torch.Tensor
    token_hw: tuple[int, int]


@dataclass(frozen=True)
class PoseLayout:
    image_ids: torch.Tensor
    target_indices: torch.Tensor
    references: tuple[PoseReference, ...]
    target_token_hw: tuple[int, int]
    reference_roles: tuple[str, ...]
    image_token_hw: tuple[tuple[int, int], ...] = ()

    @property
    def joint_length(self) -> int:
        return self.image_ids.numel()

    @property
    def target_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.image_ids, dtype=torch.bool)
        mask[self.target_indices] = True
        return mask

    def validate(self) -> "PoseLayout":
        if self.reference_roles not in REFERENCE_ROLE_LAYOUTS:
            raise ValueError("unsupported reference_roles")
        if (not isinstance(self.image_ids, torch.Tensor) or self.image_ids.ndim != 1
                or self.image_ids.dtype != torch.long):
            raise ValueError("image_ids require 1D int64 native metadata")
        count = len(self.reference_roles)
        if (len(self.image_token_hw) != count + 1
                or any(not isinstance(hw, tuple) or len(hw) != 2
                       or any(type(i) is not int or i <= 0 for i in hw)
                       for hw in self.image_token_hw)
                or self.target_token_hw != self.image_token_hw[-1]):
            raise ValueError("layout must preserve the actual grid of every image")
        if bool(((self.image_ids < -1) | (self.image_ids > count)).any()):
            raise ValueError("unknown native image IDs")
        for image_id, hw in enumerate(self.image_token_hw):
            if int((self.image_ids == image_id).sum()) != math.prod(hw):
                raise ValueError("native image IDs and preserved image grids differ")
        target = (self.image_ids == count).nonzero(as_tuple=True)[0]
        if (not isinstance(self.target_indices, torch.Tensor) or self.target_indices.dtype != torch.long
                or self.target_indices.device != self.image_ids.device
                or not torch.equal(self.target_indices, target)):
            raise ValueError("Target indices must select exactly the native Target image ID")
        pose_ids = [i for i, role in enumerate(self.reference_roles) if role in POSE_ROLES]
        if (not isinstance(self.references, tuple) or len(self.references) != len(pose_ids)
                or any(not isinstance(ref, PoseReference) for ref in self.references)):
            raise ValueError("references must select exactly the declared Pose images")
        for ref, image_id in zip(self.references, pose_ids):
            indices = (self.image_ids == image_id).nonzero(as_tuple=True)[0]
            if (ref.role != self.reference_roles[image_id] or ref.image_id != image_id
                    or ref.token_hw != self.image_token_hw[image_id]
                    or not isinstance(ref.indices, torch.Tensor) or ref.indices.dtype != torch.long
                    or ref.indices.device != self.image_ids.device or not torch.equal(ref.indices, indices)):
                raise ValueError("Pose indices/grid must match the native metadata; Source/text keys are forbidden")
        return self


def build_pose_layout(model, img_mask: torch.Tensor, img_shapes, reference_roles) -> PoseLayout:
    """展开每个四-token图像槽，再使用 model.build_token_metadata 的 ID。

    batch 内只能共享一份序列布局；支持不同 reference 的矩形网格。
    img_shapes 与原生接口一致，为 [batch][image](1,H,W)，最后一图是 Target。
    """
    roles = tuple(reference_roles)
    if roles not in REFERENCE_ROLE_LAYOUTS:
        raise ValueError("reference_roles must declare the exact supported 2Ref/4Ref/5Ref roles")
    if (not isinstance(img_mask, torch.Tensor) or img_mask.dtype != torch.bool
            or img_mask.ndim != 2 or img_mask.shape[0] < 1 or img_mask.shape[1] < 1):
        raise ValueError("img_mask requires bool [B,S]")
    if not bool((img_mask == img_mask[:1]).all()):
        raise ValueError("batch sequences must share the same image-slot layout")
    if not isinstance(img_shapes, (tuple, list)) or len(img_shapes) != img_mask.shape[0]:
        raise ValueError("img_shapes must contain one image-grid list per sample")
    normalized = []
    for shapes in img_shapes:
        if not isinstance(shapes, (tuple, list)) or len(shapes) != len(roles) + 1:
            raise ValueError("image grids must match reference_roles plus Target")
        for shape in shapes:
            if (not isinstance(shape, (tuple, list)) or len(shape) != 3 or shape[0] != 1
                    or any(type(i) is not int or i <= 0 for i in shape) or math.prod(shape) % 4):
                raise ValueError("image grids require positive (1,H,W) in four-token slots")
        normalized.append(tuple(tuple(shape) for shape in shapes))
    if any(shapes != normalized[0] for shapes in normalized[1:]):
        raise ValueError("batch sequences must share actual image grids")
    shapes = normalized[0]
    repeats = torch.where(img_mask[0], 4, 1)
    expanded = torch.repeat_interleave(img_mask[0], repeats)
    if int(expanded.sum()) != sum(math.prod(shape) for shape in shapes):
        raise ValueError("img_mask slots and actual image grids disagree")
    image_ids, target_mask = model.build_token_metadata(expanded, list(shapes))
    if (image_ids.shape != expanded.shape or image_ids.dtype != torch.long
            or target_mask.shape != expanded.shape or target_mask.dtype != torch.bool
            or image_ids.device != expanded.device or target_mask.device != expanded.device):
        raise ValueError("native token metadata has an invalid shape, dtype or device")
    if (bool((image_ids[~expanded] != -1).any())
            or not torch.equal(target_mask, image_ids == len(roles))):
        raise ValueError("native token metadata disagrees with image/text/Target roles")
    references = []
    for image_id, shape in enumerate(shapes):
        indices = (image_ids == image_id).nonzero(as_tuple=True)[0]
        if indices.numel() != math.prod(shape):
            raise ValueError("native image IDs do not preserve actual grid lengths")
        if image_id < len(roles) and roles[image_id] in POSE_ROLES:
            references.append(PoseReference(roles[image_id], image_id, indices, tuple(shape[1:])))
    if bool(((image_ids[expanded] < 0) | (image_ids[expanded] > len(roles))).any()):
        raise ValueError("native image IDs include an unknown image")
    return PoseLayout(image_ids, target_mask.nonzero(as_tuple=True)[0], tuple(references),
                      tuple(shapes[-1][1:]), roles,
                      tuple(tuple(shape[1:]) for shape in shapes)).validate()
