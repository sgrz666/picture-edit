from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import MODALITY_ORDER, UnifiedAdapterCondition


def _conv(in_channels: int, out_channels: int, *, downsample: bool = False) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4 if downsample else 3,
            stride=2 if downsample else 1,
            padding=1,
        ),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(),
    )


class _PyramidBranch(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.stem = _conv(in_channels, base_channels)
        self.down_256 = _conv(base_channels, base_channels, downsample=True)
        self.down_128 = _conv(base_channels, base_channels * 2, downsample=True)
        self.down_64 = _conv(base_channels * 2, base_channels * 4, downsample=True)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.stem(value)
        level_256 = self.down_256(value)
        level_128 = self.down_128(level_256)
        level_64 = self.down_64(level_128)
        return level_256, level_128, level_64


@dataclass
class GeometryPyramid:
    level_256: torch.Tensor
    level_128: torch.Tensor
    level_64: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "GeometryPyramid":
        return GeometryPyramid(
            self.level_256.index_select(0, indices),
            self.level_128.index_select(0, indices),
            self.level_64.index_select(0, indices),
        )


class SharedGeometryEncoder(nn.Module):
    """T2I-Adapter-style hierarchy with independent CHAMP geometry stems.

    A MimicMotion-style sparse skeleton branch and four dense branches share
    weights across person slots. Input 512x512 produces 256/128/64 pyramids.
    Missing modalities are masked before their stems, and invalid people are
    masked again after fusion so convolution biases cannot leak from slot B.
    """

    def __init__(
        self,
        base_channels: int = 24,
        output_channels: int = 128,
        num_parts: int = 24,
        part_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if output_channels % 4:
            raise ValueError("output_channels must be divisible by four")
        self.output_channels = output_channels
        self.num_parts = num_parts
        self.part_embedding = nn.Embedding(num_parts, part_embedding_dim)
        self.branches = nn.ModuleDict(
            {
                "normal": _PyramidBranch(3, base_channels),
                "depth": _PyramidBranch(1, base_channels),
                "skeleton": _PyramidBranch(3, base_channels),
                "silhouette": _PyramidBranch(1, base_channels),
                "part": _PyramidBranch(part_embedding_dim, base_channels),
            }
        )
        branch_count = len(MODALITY_ORDER)
        self.fuse_256 = nn.Sequential(
            nn.Conv2d(base_channels * branch_count, output_channels // 4, 1, bias=False),
            nn.SiLU(),
        )
        self.fuse_128 = nn.Sequential(
            nn.Conv2d(base_channels * 2 * branch_count, output_channels // 2, 1, bias=False),
            nn.SiLU(),
        )
        self.fuse_64 = nn.Sequential(
            nn.Conv2d(base_channels * 4 * branch_count, output_channels, 1, bias=False),
            nn.SiLU(),
        )
        self._initialize_sparse_pose_branch()

    def _initialize_sparse_pose_branch(self) -> None:
        # MimicMotion PoseNet uses He-normal conv initialization for sparse maps.
        for module in self.branches["skeleton"].modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _flatten_people(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    @staticmethod
    def _unflatten_people(value: torch.Tensor, batch_size: int) -> torch.Tensor:
        return value.reshape(batch_size, 2, *value.shape[1:])

    def forward(self, condition: UnifiedAdapterCondition) -> GeometryPyramid:
        condition.validate()
        batch_size = condition.batch_size
        modality_valid = condition.effective_modality_valid()
        inputs = {
            "normal": condition.normal,
            "depth": condition.depth,
            "skeleton": condition.skeleton,
            "silhouette": condition.silhouette,
        }
        if condition.part_ids is None:
            height, width = condition.normal.shape[-2:]
            part = condition.normal.new_zeros(batch_size, 2, self.part_embedding.embedding_dim, height, width)
        else:
            part_ids = condition.part_ids.long().clamp(0, self.num_parts - 1)
            part = self.part_embedding(part_ids).permute(0, 1, 4, 2, 3)
        inputs["part"] = part

        levels: list[list[torch.Tensor]] = [[], [], []]
        for modality_index, name in enumerate(MODALITY_ORDER):
            value = inputs[name]
            valid = modality_valid[..., modality_index].to(value.dtype)[..., None, None, None]
            value = self._flatten_people(value * valid)
            branch_levels = self.branches[name](value)
            for level_index, branch_level in enumerate(branch_levels):
                flat_valid = valid.reshape(batch_size * 2, 1, 1, 1)
                levels[level_index].append(branch_level * flat_valid)

        person_mask = condition.person_valid.to(condition.normal.dtype)[..., None, None, None]
        fused = (
            self.fuse_256(torch.cat(levels[0], dim=1)),
            self.fuse_128(torch.cat(levels[1], dim=1)),
            self.fuse_64(torch.cat(levels[2], dim=1)),
        )
        fused = tuple(self._unflatten_people(value, batch_size) * person_mask for value in fused)
        return GeometryPyramid(*fused)


def _group_count(channels: int) -> int:
    return next(group for group in (32, 16, 8, 4, 2, 1) if channels % group == 0)


class ConditionResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class NativeConditionStem(nn.Module):
    """Three-level GroupNorm/SiLU stem for one explicit condition modality."""

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int, int],
        *,
        project_to: int | None = None,
        sparse_init: bool = False,
    ) -> None:
        super().__init__()
        first_channels = project_to or in_channels
        self.input_projection = (
            nn.Identity()
            if project_to is None
            else nn.Conv2d(in_channels, project_to, kernel_size=1)
        )
        stages = []
        current = first_channels
        for output in channels:
            stages.extend(
                (
                    nn.Conv2d(current, output, 3, stride=2, padding=1),
                    nn.GroupNorm(_group_count(output), output),
                    nn.SiLU(),
                    ConditionResBlock(output),
                )
            )
            current = output
        self.stages = nn.Sequential(*stages)
        if sparse_init:
            self._initialize_sparse_convolutions()

    def _initialize_sparse_convolutions(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.stages(self.input_projection(value))


class RoleFiLM(nn.Module):
    def __init__(self, channels: int = 256, embedding_dim: int = 128) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(2, embedding_dim)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, channels * 2),
        )

    def forward(self, value: torch.Tensor, role_id: int) -> torch.Tensor:
        roles = torch.full(
            (value.shape[0],), role_id, dtype=torch.long, device=value.device
        )
        scale, shift = self.to_scale_shift(self.role_embedding(roles)).chunk(2, dim=-1)
        return value * (1 + scale[..., None, None]) + shift[..., None, None]


class SpatialConditionEncoder(nn.Module):
    """Independent stems followed by mid-level fusion and role binding."""

    def __init__(self, *, use_depth: bool = False, output_channels: int = 256) -> None:
        super().__init__()
        self.use_depth = use_depth
        self.normal_stem = NativeConditionStem(3, (32, 64, 128))
        self.pose_stem = NativeConditionStem(
            25, (64, 96, 128), project_to=32, sparse_init=True
        )
        self.part_stem = NativeConditionStem(14, (64, 96, 128), project_to=32)
        self.depth_stem = NativeConditionStem(1, (32, 64, 128)) if use_depth else None
        fusion_input = 128 * (4 if use_depth else 3)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_input, output_channels, 1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            ConditionResBlock(output_channels),
            ConditionResBlock(output_channels),
        )
        self.role_film = RoleFiLM(output_channels)

    def forward(
        self,
        *,
        normal: torch.Tensor,
        pose_heatmap: torch.Tensor,
        part_onehot: torch.Tensor,
        human_mask: torch.Tensor,
        role_id: int,
        depth: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = human_mask.to(dtype=normal.dtype)
        features = (
            self.normal_stem(normal * mask),
            self.pose_stem(pose_heatmap * mask),
            self.part_stem(part_onehot * mask),
        )
        if self.use_depth:
            if depth is None:
                depth = normal.new_zeros(normal.shape[0], 1, *normal.shape[-2:])
            features = (*features, self.depth_stem(depth * mask))
        spatial = self.role_film(self.fusion(torch.cat(features, dim=1)), role_id)
        output_mask = F.interpolate(mask, size=spatial.shape[-2:], mode="nearest")
        if valid is not None:
            valid_map = valid.to(spatial.dtype)[:, None, None, None]
            spatial = spatial * valid_map
            output_mask = output_mask * valid_map
        return spatial, output_mask


def normalize_scene_depth(
    depth_a: torch.Tensor | None,
    depth_b: torch.Tensor | None,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor | None,
    person_b_valid: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Normalize A/B camera-space depth with one range per scene."""

    if depth_a is None and depth_b is None:
        return None, None
    template = depth_a if depth_a is not None else depth_b
    depth_a = torch.zeros_like(template) if depth_a is None else depth_a
    depth_b = torch.zeros_like(template) if depth_b is None else depth_b
    if mask_b is None:
        mask_b = torch.zeros_like(mask_a)
    valid_b = person_b_valid[:, None, None, None]
    mask_b = mask_b * valid_b.to(mask_b.dtype)
    result_a = torch.zeros_like(depth_a)
    result_b = torch.zeros_like(depth_b)
    for index in range(depth_a.shape[0]):
        values = []
        if mask_a[index].bool().any():
            values.append(depth_a[index][mask_a[index].bool()])
        if mask_b[index].bool().any():
            values.append(depth_b[index][mask_b[index].bool()])
        if not values:
            continue
        joined = torch.cat(values)
        minimum, maximum = joined.amin(), joined.amax()
        span = maximum - minimum
        if span > eps:
            result_a[index] = ((depth_a[index] - minimum) / span * 2 - 1) * mask_a[index]
            result_b[index] = ((depth_b[index] - minimum) / span * 2 - 1) * mask_b[index]
    result_b = result_b * valid_b.to(result_b.dtype)
    return result_a, result_b
