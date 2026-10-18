from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .reasoning import InternalControlState


@dataclass
class ReasonerBridgedCondition:
    geometry_scene: torch.Tensor
    interaction_scene: torch.Tensor


class ReasonerControlBridge(nn.Module):
    """Keep geometry and interaction paths independent for DeepGen control."""

    def __init__(
        self,
        *,
        reasoning_dim: int = 512,
        geometry_channels: int = 128,
        token_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.reasoning_dim = reasoning_dim
        self.geometry_channels = geometry_channels
        self.token_dim = token_dim
        self.geometry_projection = nn.Conv2d(reasoning_dim, geometry_channels, 1)
        self.interaction_projection = nn.Conv2d(
            reasoning_dim, geometry_channels, 1, bias=False
        )
        self.geometry_high_downsample = nn.Conv2d(
            reasoning_dim, reasoning_dim, 3, stride=2, padding=1, bias=False
        )
        self.interaction_high_downsample = nn.Conv2d(
            reasoning_dim, reasoning_dim, 3, stride=2, padding=1, bias=False
        )
    def forward(self, state: InternalControlState) -> ReasonerBridgedCondition:
        state.validate()
        interaction_valid = state.interaction_valid[:, None, None, None].to(
            state.interaction_feature.dtype
        )
        geometry = state.geometry_feature + self.geometry_high_downsample(
            state.geometry_highres
        )
        interaction = state.interaction_feature + self.interaction_high_downsample(
            state.interaction_highres
        )
        interaction = interaction * interaction_valid
        geometry_scene = self.geometry_projection(geometry)
        interaction_scene = self.interaction_projection(interaction)
        interaction_scene = interaction_scene * interaction_valid
        return ReasonerBridgedCondition(geometry_scene, interaction_scene)
