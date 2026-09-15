from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import TaskSpec, UnifiedAdapterCondition
from .token_encoders import ContactRelationEncoder


@dataclass
class RoutingDecision:
    num_people: torch.Tensor
    single_mask: torch.Tensor
    dual_mask: torch.Tensor
    interaction_ids: torch.Tensor


class TaskRouter:
    """Deterministic routing: person masks decide count; prompt only labels interaction."""

    prompt_labels = {
        "handshake": 1,
        "shake hands": 1,
        "握手": 1,
        "hug": 2,
        "embrace": 2,
        "拥抱": 2,
        "shoulder": 3,
        "搭肩": 3,
    }

    def resolve(self, person_valid: torch.Tensor, task_spec: TaskSpec) -> RoutingDecision:
        num_people = person_valid.bool().sum(dim=1).long()
        if torch.any((num_people < 1) | (num_people > 2)):
            raise ValueError("router supports one or two people only")
        interaction_ids = task_spec.interaction_type.to(device=person_valid.device, dtype=torch.long).clone()
        unresolved = interaction_ids < 0
        if unresolved.any() and task_spec.prompts is not None:
            for index in unresolved.nonzero(as_tuple=False).flatten().tolist():
                prompt = task_spec.prompts[index].lower()
                interaction_ids[index] = next(
                    (label for phrase, label in self.prompt_labels.items() if phrase in prompt),
                    0,
                )
        interaction_ids.clamp_min_(0)
        return RoutingDecision(
            num_people=num_people,
            single_mask=num_people == 1,
            dual_mask=num_people == 2,
            interaction_ids=interaction_ids,
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
        condition: UnifiedAdapterCondition,
    ) -> ExpertOutput:
        scene = person_features[:, 0]
        contact_tokens = person_tokens.new_zeros(
            person_tokens.shape[0], self.contact_token_count, person_tokens.shape[-1]
        )
        contact_mask = torch.zeros(
            person_tokens.shape[0], self.contact_token_count, dtype=torch.bool, device=person_tokens.device
        )
        return ExpertOutput(scene, person_features, person_tokens, contact_tokens, contact_mask)


class DualPersonExpert(nn.Module):
    """Symmetric two-person fusion with bidirectional spatial/token attention."""

    def __init__(
        self,
        feature_channels: int = 128,
        cross_attention_dim: int = 512,
        token_dim: int = 1536,
        num_heads: int = 8,
        contact_pair_dim: int = 4,
        contact_token_count: int = 8,
    ) -> None:
        super().__init__()
        if cross_attention_dim % num_heads:
            raise ValueError("cross attention dimension must be divisible by num_heads")
        if token_dim % num_heads:
            raise ValueError("token dimension must be divisible by num_heads")
        self.feature_channels = feature_channels
        self.spatial_down = nn.Conv2d(feature_channels, cross_attention_dim, 3, stride=2, padding=1)
        self.spatial_attention = nn.MultiheadAttention(
            cross_attention_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.spatial_up = nn.Conv2d(cross_attention_dim, feature_channels, 1)
        self.token_attention = nn.MultiheadAttention(token_dim, num_heads, dropout=0.0, batch_first=True)
        self.contact_encoder = ContactRelationEncoder(
            feature_channels=feature_channels,
            token_dim=token_dim,
            pair_dim=contact_pair_dim,
            token_count=contact_token_count,
        )

    def _bidirectional_spatial(self, person_features: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width = person_features.shape
        down = self.spatial_down(person_features.flatten(0, 1))
        down_height, down_width = down.shape[-2:]
        tokens = down.flatten(2).transpose(1, 2).reshape(batch_size, 2, -1, down.shape[1])
        a_from_b = self.spatial_attention(tokens[:, 0], tokens[:, 1], tokens[:, 1], need_weights=False)[0]
        b_from_a = self.spatial_attention(tokens[:, 1], tokens[:, 0], tokens[:, 0], need_weights=False)[0]
        interaction = torch.stack((a_from_b, b_from_a), dim=1)
        interaction = interaction.reshape(batch_size * 2, down_height, down_width, -1).permute(0, 3, 1, 2)
        interaction = F.interpolate(interaction, size=(height, width), mode="bilinear", align_corners=False)
        return self.spatial_up(interaction).reshape(batch_size, 2, self.feature_channels, height, width)

    def _bidirectional_tokens(self, person_tokens: torch.Tensor) -> torch.Tensor:
        a_from_b = self.token_attention(
            person_tokens[:, 0], person_tokens[:, 1], person_tokens[:, 1], need_weights=False
        )[0]
        b_from_a = self.token_attention(
            person_tokens[:, 1], person_tokens[:, 0], person_tokens[:, 0], need_weights=False
        )[0]
        return person_tokens + torch.stack((a_from_b, b_from_a), dim=1)

    @staticmethod
    def _depth_fuse(person_features: torch.Tensor, condition: UnifiedAdapterCondition) -> torch.Tensor:
        size = person_features.shape[-2:]
        depth = F.interpolate(
            condition.depth.flatten(0, 1), size=size, mode="bilinear", align_corners=False
        ).reshape(condition.batch_size, 2, 1, *size)
        mask = F.interpolate(
            condition.silhouette.flatten(0, 1), size=size, mode="nearest"
        ).reshape(condition.batch_size, 2, 1, *size)
        mask = mask * condition.person_valid[..., None, None, None].to(mask.dtype)
        scores = -4.0 * depth + torch.log(mask.clamp_min(1e-6))
        weights = scores.softmax(dim=1) * (mask.sum(dim=1, keepdim=True) > 0).to(scores.dtype)
        return (person_features * weights).sum(dim=1)

    def forward(
        self,
        person_features: torch.Tensor,
        person_tokens: torch.Tensor,
        condition: UnifiedAdapterCondition,
    ) -> ExpertOutput:
        if not torch.all(condition.person_valid.bool().sum(dim=1) == 2):
            raise ValueError("DualPersonExpert requires two valid people")
        person_features = person_features + self._bidirectional_spatial(person_features)
        person_tokens = self._bidirectional_tokens(person_tokens)
        scene = self._depth_fuse(person_features, condition)
        contact_spatial, contact_tokens, contact_mask = self.contact_encoder(
            condition.contact_maps,
            condition.contact_pairs,
            condition.contact_valid,
            scene.shape[-2:],
            batch_size=condition.batch_size,
            device=scene.device,
            dtype=scene.dtype,
        )
        map_present = contact_mask[:, :4].any(dim=1).to(scene.dtype)[:, None, None, None]
        scene = scene + contact_spatial * map_present
        contact_tokens = contact_tokens * contact_mask[..., None].to(contact_tokens.dtype)
        return ExpertOutput(scene, person_features, person_tokens, contact_tokens, contact_mask)
