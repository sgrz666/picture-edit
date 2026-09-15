from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ReasoningResBlock, group_count
from .config import AdapterReasoningConfig


@dataclass
class DualFusionOutput:
    geometry_feature: torch.Tensor
    geometry_highres: torch.Tensor
    interaction_feature: torch.Tensor
    interaction_highres: torch.Tensor


class DualFeatureFusion(nn.Module):
    """Produce independent geometry and interaction controls at two scales."""

    def __init__(self, config: AdapterReasoningConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.geometry_projection = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.geometry_block = ReasoningResBlock(hidden_dim)
        self.interaction_projection = nn.Conv2d(hidden_dim * 5, hidden_dim + hidden_dim // 2, 1)
        interaction_mid = hidden_dim + hidden_dim // 2
        self.interaction_norm = nn.GroupNorm(group_count(interaction_mid), interaction_mid)
        self.interaction_output = nn.Conv2d(interaction_mid, hidden_dim, 3, padding=1)
        self.interaction_block = ReasoningResBlock(hidden_dim)
        self.geometry_high_projection = nn.Conv2d(hidden_dim * 3, hidden_dim, 1)
        self.geometry_high_block = ReasoningResBlock(hidden_dim)
        self.interaction_high_projection = nn.Conv2d(hidden_dim * 4, hidden_dim, 1)
        self.interaction_high_block = ReasoningResBlock(hidden_dim)

    @staticmethod
    def _as_map(tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], *size)

    def forward(
        self,
        person_a: torch.Tensor,
        person_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        local_a: torch.Tensor,
        local_b: torch.Tensor,
        contact_low: torch.Tensor,
        contact_high: torch.Tensor,
    ) -> DualFusionOutput:
        low_size = contact_low.shape[-2:]
        high_size = local_a.shape[-2:]
        map_a = self._as_map(person_a, low_size)
        map_b = self._as_map(person_b, low_size)
        low_mask_a = mask_a.reshape(mask_a.shape[0], 1, *low_size)
        low_mask_b = mask_b.reshape(mask_b.shape[0], 1, *low_size)
        geometry_mask = low_mask_a | low_mask_b
        contact_mask = contact_low.abs().sum(dim=1, keepdim=True) > 0
        interaction_mask = geometry_mask | contact_mask

        geometry = self.geometry_projection(torch.cat((map_a, map_b), dim=1))
        geometry = self.geometry_block(geometry, geometry_mask)
        interaction_input = torch.cat(
            (map_a, map_b, map_a - map_b, map_a * map_b, contact_low), dim=1
        )
        interaction = self.interaction_projection(interaction_input)
        interaction = self.interaction_output(F.silu(self.interaction_norm(interaction)))
        interaction = self.interaction_block(interaction, interaction_mask)

        high_geometry_mask = F.interpolate(
            geometry_mask.to(geometry.dtype), size=high_size, mode="nearest"
        ) > 0
        geometry_up = F.interpolate(
            geometry, size=high_size, mode="bilinear", align_corners=False
        )
        geometry_high = self.geometry_high_projection(
            torch.cat((geometry_up, local_a, local_b), dim=1)
        )
        geometry_high = self.geometry_high_block(geometry_high, high_geometry_mask)

        interaction_up = F.interpolate(
            interaction, size=high_size, mode="bilinear", align_corners=False
        )
        high_contact_mask = contact_high.abs().sum(dim=1, keepdim=True) > 0
        high_interaction_mask = high_geometry_mask | high_contact_mask
        interaction_high = self.interaction_high_projection(
            torch.cat(
                (
                    interaction_up,
                    local_a - local_b,
                    local_a * local_b,
                    contact_high,
                ),
                dim=1,
            )
        )
        interaction_high = self.interaction_high_block(
            interaction_high, high_interaction_mask
        )
        return DualFusionOutput(
            geometry,
            geometry_high,
            interaction,
            interaction_high,
        )
