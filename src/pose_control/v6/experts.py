from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingDecision:
    num_people: torch.Tensor
    single_mask: torch.Tensor
    dual_mask: torch.Tensor


class TaskRouter:
    """Route exclusively from the validated number of SMPL-X instances."""

    def resolve(self, person_count: torch.Tensor) -> RoutingDecision:
        if person_count.ndim != 1:
            raise ValueError("person_count must have shape [B]")
        num_people = person_count.long()
        if torch.any((num_people < 1) | (num_people > 2)):
            raise ValueError("router supports one or two people only")
        return RoutingDecision(
            num_people=num_people,
            single_mask=num_people == 1,
            dual_mask=num_people == 2,
        )


@dataclass
class ExpertOutput:
    scene_feature: torch.Tensor
    person_features: torch.Tensor
    person_tokens: torch.Tensor
    contact_tokens: torch.Tensor
    contact_token_mask: torch.Tensor


class SinglePersonExpert(nn.Module):
    def __init__(self, contact_token_count: int = 8) -> None:
        super().__init__()
        self.contact_token_count = contact_token_count

    def forward(
        self,
        person_features: torch.Tensor,
        person_tokens: torch.Tensor,
    ) -> ExpertOutput:
        scene = person_features[:, 0]
        contact_tokens = person_tokens.new_zeros(
            person_tokens.shape[0], self.contact_token_count, person_tokens.shape[-1]
        )
        contact_mask = torch.zeros(
            person_tokens.shape[0],
            self.contact_token_count,
            dtype=torch.bool,
            device=person_tokens.device,
        )
        return ExpertOutput(scene, person_features, person_tokens, contact_tokens, contact_mask)


class DualPersonExpert(nn.Module):
    """Cross-person reasoning over already encoded ConditionBundle features."""

    def __init__(
        self,
        feature_channels: int = 128,
        cross_attention_dim: int = 512,
        token_dim: int = 1536,
        num_heads: int = 8,
        contact_token_count: int = 8,
    ) -> None:
        super().__init__()
        if cross_attention_dim % num_heads:
            raise ValueError("cross attention dimension must be divisible by num_heads")
        if token_dim % num_heads:
            raise ValueError("token dimension must be divisible by num_heads")
        self.feature_channels = feature_channels
        self.contact_token_count = contact_token_count
        self.spatial_down = nn.Conv2d(
            feature_channels, cross_attention_dim, 3, stride=2, padding=1
        )
        self.spatial_attention = nn.MultiheadAttention(
            cross_attention_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.spatial_up = nn.Conv2d(cross_attention_dim, feature_channels, 1)
        self.token_attention = nn.MultiheadAttention(
            token_dim, num_heads, dropout=0.0, batch_first=True
        )

    def _bidirectional_spatial(self, person_features: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width = person_features.shape
        down = self.spatial_down(person_features.flatten(0, 1))
        down_height, down_width = down.shape[-2:]
        tokens = down.flatten(2).transpose(1, 2).reshape(
            batch_size, 2, -1, down.shape[1]
        )
        a_from_b = self.spatial_attention(
            tokens[:, 0], tokens[:, 1], tokens[:, 1], need_weights=False
        )[0]
        b_from_a = self.spatial_attention(
            tokens[:, 1], tokens[:, 0], tokens[:, 0], need_weights=False
        )[0]
        interaction = torch.stack((a_from_b, b_from_a), dim=1)
        interaction = interaction.reshape(
            batch_size * 2, down_height, down_width, -1
        ).permute(0, 3, 1, 2)
        interaction = F.interpolate(
            interaction, size=(height, width), mode="bilinear", align_corners=False
        )
        return self.spatial_up(interaction).reshape(
            batch_size, 2, self.feature_channels, height, width
        )

    def _bidirectional_tokens(self, person_tokens: torch.Tensor) -> torch.Tensor:
        a_from_b = self.token_attention(
            person_tokens[:, 0], person_tokens[:, 1], person_tokens[:, 1], need_weights=False
        )[0]
        b_from_a = self.token_attention(
            person_tokens[:, 1], person_tokens[:, 0], person_tokens[:, 0], need_weights=False
        )[0]
        return person_tokens + torch.stack((a_from_b, b_from_a), dim=1)

    @staticmethod
    def _mask_fuse(person_features: torch.Tensor, person_masks: torch.Tensor) -> torch.Tensor:
        masks = person_masks.to(person_features.dtype)
        weights = masks / masks.sum(dim=1, keepdim=True).clamp_min(1)
        return (person_features * weights).sum(dim=1)

    def forward(
        self,
        person_features: torch.Tensor,
        person_tokens: torch.Tensor,
        person_masks: torch.Tensor,
        contact_spatial: torch.Tensor,
        contact_tokens: torch.Tensor,
        contact_token_mask: torch.Tensor,
    ) -> ExpertOutput:
        if person_features.ndim != 5 or person_features.shape[1] != 2:
            raise ValueError("DualPersonExpert requires two encoded person features")
        person_features = person_features + self._bidirectional_spatial(person_features)
        person_tokens = self._bidirectional_tokens(person_tokens)
        scene = self._mask_fuse(person_features, person_masks) + contact_spatial
        contact_tokens = contact_tokens * contact_token_mask[..., None].to(contact_tokens.dtype)
        return ExpertOutput(
            scene,
            person_features,
            person_tokens,
            contact_tokens,
            contact_token_mask,
        )
