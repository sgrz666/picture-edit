from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        roles = torch.full((value.shape[0],), role_id, dtype=torch.long, device=value.device)
        scale, shift = self.to_scale_shift(self.role_embedding(roles)).chunk(2, dim=-1)
        return value * (1 + scale[..., None, None]) + shift[..., None, None]


class ChampConditionStemAdapter(nn.Module):
    """Adapt Normal or scalar Depth to the vendored single-frame CHAMP stem."""

    def __init__(self, in_channels: int, output_channels: int = 128) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("CHAMP condition stem supports 1-channel depth or 3-channel normal")
        from third_party.champ_guidance import ChampGuidanceEncoder

        self.in_channels = in_channels
        self.encoder = ChampGuidanceEncoder(
            guidance_embedding_channels=output_channels,
            guidance_input_channels=3,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 1:
            value = value.expand(-1, 3, -1, -1)
        return self.encoder(value)


class SpatialConditionEncoder(nn.Module):
    """Independent stems followed by mid-level fusion and role binding."""

    def __init__(
        self,
        *,
        use_depth: bool = False,
        output_channels: int = 256,
        normal_backend: str = "native",
        depth_backend: str = "native",
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        self.normal_stem = (
            NativeConditionStem(3, (32, 64, 128))
            if normal_backend == "native"
            else ChampConditionStemAdapter(3)
        )
        self.pose_stem = NativeConditionStem(
            25, (64, 96, 128), project_to=32, sparse_init=True
        )
        self.part_stem = NativeConditionStem(14, (64, 96, 128), project_to=32)
        self.depth_stem = None
        if use_depth:
            self.depth_stem = (
                NativeConditionStem(1, (32, 64, 128))
                if depth_backend == "native"
                else ChampConditionStemAdapter(1)
            )
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
