from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

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
