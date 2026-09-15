from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import ContactRelationBatch, PART_NAMES, TaskType


class AppearanceTokenEncoder(nn.Module):
    """IP-Adapter-style decoupled identity tokens from masked source latents."""

    def __init__(self, in_channels: int = 16, token_dim: int = 1536, token_count: int = 8) -> None:
        super().__init__()
        if token_count != 8:
            raise ValueError("V6 defines exactly eight appearance tokens per person")
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


class PersonTokenBinder(nn.Module):
    """Bind geometry and appearance tokens to the same source/role slot."""

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
        # Padding is outside the semantic contract. Replace sentinel values
        # before embedding lookup, then zero the projected tokens with mask.
        padding_mask = relations.valid_mask
        safe = lambda value: torch.where(padding_mask, value, torch.zeros_like(value))
        token = (
            self.person_embedding(safe(relations.src_person))
            + self.part_embedding(safe(relations.src_part))
            + self.person_embedding(safe(relations.dst_person))
            + self.part_embedding(safe(relations.dst_part))
            + self.type_embedding(safe(relations.contact_type))
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
