from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import ContactRelationBatch, PART_NAMES, TaskType, UnifiedAdapterCondition


class GlobalGeometryTokenEncoder(nn.Module):
    """Encode SMPL-X pose, source shape and camera into eight tokens/person."""

    token_count = 8

    def __init__(self, token_dim: int = 1536) -> None:
        super().__init__()
        # global orientation + 21 body joints -> four body tokens
        self.body = nn.Sequential(nn.Linear(22 * 6, token_dim * 4), nn.SiLU())
        self.left_hand = nn.Sequential(nn.Linear(15 * 6, token_dim), nn.SiLU())
        self.right_hand = nn.Sequential(nn.Linear(15 * 6, token_dim), nn.SiLU())
        self.source_shape = nn.Sequential(nn.Linear(10, token_dim), nn.SiLU())
        self.camera = nn.Sequential(nn.Linear(7, token_dim), nn.SiLU())
        self.token_dim = token_dim

    def forward(self, condition: UnifiedAdapterCondition) -> torch.Tensor:
        pose = condition.smpl_pose6d
        batch_size = pose.shape[0]
        body = self.body(pose[:, :, :22].flatten(2)).reshape(batch_size, 2, 4, self.token_dim)
        left = self.left_hand(pose[:, :, 22:37].flatten(2)).unsqueeze(2)
        right = self.right_hand(pose[:, :, 37:52].flatten(2)).unsqueeze(2)
        shape = self.source_shape(condition.source_betas).unsqueeze(2)
        camera = self.camera(condition.camera).unsqueeze(2)
        tokens = torch.cat((body, left, right, shape, camera), dim=2)
        return tokens * condition.person_valid[..., None, None].to(tokens.dtype)


class AppearanceTokenEncoder(nn.Module):
    """IP-Adapter-style decoupled identity tokens from masked source latents."""

    def __init__(self, in_channels: int = 16, token_dim: int = 1536, token_count: int = 8) -> None:
        super().__init__()
        if token_count != 8:
            raise ValueError("V6 currently defines exactly eight appearance tokens/person")
        self.token_count = token_count
        self.projection = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, source_latents: torch.Tensor, person_valid: torch.Tensor) -> torch.Tensor:
        if source_latents.ndim != 5 or source_latents.shape[1:3] != (2, 16):
            raise ValueError("source latents must have shape [B,2,16,h,w]")
        batch_size = source_latents.shape[0]
        flat = source_latents.reshape(batch_size * 2, *source_latents.shape[2:])
        pooled = F.adaptive_avg_pool2d(flat, (2, 4)).flatten(2).transpose(1, 2)
        tokens = self.projection(pooled).reshape(batch_size, 2, self.token_count, -1)
        return tokens * person_valid[..., None, None].to(tokens.dtype)


class TaskTokenEncoder(nn.Module):
    def __init__(self, token_dim: int, interaction_vocab_size: int = 16) -> None:
        super().__init__()
        self.route_embedding = nn.Embedding(3, token_dim)
        self.interaction_embedding = nn.Embedding(interaction_vocab_size, token_dim)

    def forward(self, num_people: torch.Tensor, interaction_ids: torch.Tensor) -> torch.Tensor:
        interaction_ids = interaction_ids.clamp(0, self.interaction_embedding.num_embeddings - 1)
        return (self.route_embedding(num_people) + self.interaction_embedding(interaction_ids)).unsqueeze(1)


class PersonTokenBinder(nn.Module):
    """Bind geometry/appearance tokens to the same slot and source identity."""

    def __init__(self, token_dim: int, max_sources: int = 16) -> None:
        super().__init__()
        self.slot_embedding = nn.Embedding(2, token_dim)
        self.source_embedding = nn.Embedding(max_sources, token_dim)

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        appearance_tokens: torch.Tensor,
        task_token: torch.Tensor,
        person_valid: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = geometry_tokens.shape[0]
        slots = torch.arange(2, device=geometry_tokens.device).expand(batch_size, -1)
        binding = self.slot_embedding(slots) + self.source_embedding(
            source_indices.clamp(0, self.source_embedding.num_embeddings - 1)
        )
        binding = binding[:, :, None] + task_token[:, None]
        tokens = torch.cat((geometry_tokens + binding, appearance_tokens + binding), dim=2)
        valid = person_valid[..., None].expand(-1, -1, tokens.shape[2])
        tokens = tokens * valid[..., None].to(tokens.dtype)
        return tokens, valid


class ContactRelationEncoder(nn.Module):
    """Encode directional contact maps and SMPL-X part-pair relations."""

    def __init__(
        self,
        feature_channels: int,
        token_dim: int,
        pair_dim: int = 4,
        token_count: int = 8,
    ) -> None:
        super().__init__()
        if token_count != 8:
            raise ValueError("V6 contact encoder uses four map and four pair tokens")
        self.token_count = token_count
        self.map_stem = nn.Sequential(
            nn.Conv2d(1, feature_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.map_token_projection = nn.Linear(feature_channels, token_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.pair_queries = nn.Parameter(torch.randn(4, token_dim) * 0.02)

    def forward(
        self,
        contact_maps: torch.Tensor | None,
        contact_pairs: torch.Tensor | None,
        contact_valid: torch.Tensor | None,
        spatial_size: tuple[int, int],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if contact_maps is None:
            maps = torch.zeros(batch_size, 2, *spatial_size, device=device, dtype=dtype)
        else:
            maps = F.interpolate(
                contact_maps.to(device=device, dtype=dtype),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        directional = self.map_stem(maps.reshape(batch_size * 2, 1, *spatial_size)).reshape(
            batch_size, 2, -1, *spatial_size
        )
        # Spatial contact is permutation-invariant; ordered map tokens retain
        # A->B/B->A direction for the relation branch.
        spatial = directional.sum(dim=1)
        pooled = F.adaptive_avg_pool2d(
            directional.flatten(0, 1), (1, 2)
        ).flatten(2).transpose(1, 2)
        pooled = pooled.reshape(batch_size, 4, -1)
        map_tokens = self.map_token_projection(pooled)

        if contact_pairs is None or contact_pairs.shape[1] == 0:
            pair_summary = torch.zeros(batch_size, 1, map_tokens.shape[-1], device=device, dtype=dtype)
            has_pair = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            pair_tokens = self.pair_mlp(contact_pairs.to(device=device, dtype=dtype))
            pair_mask = (
                torch.ones(pair_tokens.shape[:2], dtype=torch.bool, device=device)
                if contact_valid is None
                else contact_valid.to(device=device).bool()
            )
            weights = pair_mask.to(dtype)[..., None]
            pair_summary = (pair_tokens * weights).sum(dim=1, keepdim=True) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            has_pair = pair_mask.any(dim=1)
        pair_tokens = pair_summary + self.pair_queries.to(dtype)[None]
        pair_tokens = pair_tokens * has_pair[:, None, None].to(dtype)
        tokens = torch.cat((map_tokens, pair_tokens), dim=1)
        map_present = maps.abs().flatten(1).sum(1) > 0
        token_mask = torch.cat(
            (
                map_present[:, None].expand(-1, 4),
                has_pair[:, None].expand(-1, 4),
            ),
            dim=1,
        )
        return spatial, tokens, token_mask


class GlobalConditionTokenEncoder(nn.Module):
    """Tokenize beta, root rotation, translation and normalized camera only."""

    def __init__(self, token_dim: int = 256, token_count: int = 4) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.network = nn.Sequential(
            nn.LayerNorm(26),
            nn.Linear(26, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, token_count * token_dim),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.network(value).reshape(value.shape[0], self.token_count, self.token_dim)
        if valid is not None:
            tokens = tokens * valid.to(tokens.dtype)[:, None, None]
        return tokens


class RelativeGeometryTokenEncoder(nn.Module):
    def __init__(self, token_dim: int = 256, token_count: int = 2) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.network = nn.Sequential(
            nn.LayerNorm(11),
            nn.Linear(11, 512),
            nn.SiLU(),
            nn.Linear(512, token_count * token_dim),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = self.network(value).reshape(value.shape[0], self.token_count, self.token_dim)
        return tokens * valid.to(tokens.dtype)[:, None, None]


class ContactRasterEncoder(nn.Module):
    def __init__(self, output_channels: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return output * valid.to(output.dtype)[:, None, None, None]


class StructuredContactRelationEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 256,
        max_relations: int = 8,
        contact_type_count: int = 16,
    ) -> None:
        super().__init__()
        self.max_relations = max_relations
        self.contact_type_count = contact_type_count
        self.person_embedding = nn.Embedding(2, token_dim)
        self.part_embedding = nn.Embedding(len(PART_NAMES), token_dim)
        self.type_embedding = nn.Embedding(contact_type_count, token_dim)
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(
        self,
        relations: ContactRelationBatch,
        person_b_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relations.validate(
            batch_size=person_b_valid.shape[0],
            max_relations=self.max_relations,
            contact_type_count=self.contact_type_count,
        )
        token = (
            self.person_embedding(relations.src_person)
            + self.part_embedding(relations.src_part)
            + self.person_embedding(relations.dst_person)
            + self.part_embedding(relations.dst_part)
            + self.type_embedding(relations.contact_type)
            + self.distance_mlp(relations.distance)
        )
        mask = relations.valid_mask & person_b_valid[:, None]
        token = self.output(token) * mask[..., None].to(token.dtype)
        return token, mask


class ExplicitTaskTokenEncoder(nn.Module):
    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(TaskType), token_dim)

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_id).unsqueeze(1)
