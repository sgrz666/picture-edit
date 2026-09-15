from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reasoning import InternalControlState


@dataclass
class ReasonerBridgedCondition:
    scene_feature: torch.Tensor
    geometry_tokens: torch.Tensor
    geometry_token_mask: torch.Tensor


class ReasonerControlBridge(nn.Module):
    """Project InternalControlState to the unchanged DeepGen control widths."""

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
        self.token_projection = nn.Linear(reasoning_dim, token_dim)

    @staticmethod
    def _pool_person_tokens(
        state: InternalControlState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, people, token_count, channels = state.person_tokens.shape
        height, width = state.geometry_feature.shape[-2:]
        if token_count != height * width:
            raise ValueError("person token count must match reasoning feature size")
        features = state.person_tokens.reshape(
            batch_size * people, height, width, channels
        ).permute(0, 3, 1, 2)
        mask = state.person_token_mask.reshape(
            batch_size * people, 1, height, width
        )
        weights = mask.to(features.dtype)
        numerator = F.adaptive_avg_pool2d(features * weights, (2, 2))
        denominator = F.adaptive_avg_pool2d(weights, (2, 2))
        pooled = numerator / denominator.clamp_min(1e-6)
        pooled_mask = denominator > 0
        pooled = pooled * pooled_mask.to(pooled.dtype)
        pooled = pooled.flatten(2).transpose(1, 2).reshape(
            batch_size, people, 4, channels
        )
        pooled_mask = pooled_mask.flatten(1).reshape(batch_size, people, 4)
        return pooled, pooled_mask

    def forward(self, state: InternalControlState) -> ReasonerBridgedCondition:
        state.validate()
        interaction_valid = state.interaction_valid[:, None, None, None].to(
            state.interaction_feature.dtype
        )
        interaction = state.interaction_feature * interaction_valid
        scene = self.geometry_projection(state.geometry_feature)
        scene = scene + self.interaction_projection(interaction)
        pooled, pooled_mask = self._pool_person_tokens(state)
        geometry_tokens = self.token_projection(pooled)
        geometry_tokens = geometry_tokens * pooled_mask[..., None].to(
            geometry_tokens.dtype
        )
        return ReasonerBridgedCondition(scene, geometry_tokens, pooled_mask)
