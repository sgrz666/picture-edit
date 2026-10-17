from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CrossAttentionFFN
from .config import AdapterReasoningConfig


class BidirectionalCrossPersonReasoner(nn.Module):
    """Simultaneous A↔B updates with shared weights in each direction."""

    def __init__(self, config: AdapterReasoningConfig) -> None:
        super().__init__()
        self.config = config
        self.relative_projection = nn.Linear(config.condition_dim, config.hidden_dim)
        self.direction_embedding = nn.Embedding(2, config.hidden_dim)
        self.layers = nn.ModuleList(
            CrossAttentionFFN(
                config.hidden_dim,
                config.cross_person_heads,
                ffn_ratio=config.ffn_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.cross_person_layers)
        )

    def forward(
        self,
        person_a: torch.Tensor,
        person_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        relative_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if person_a.shape != person_b.shape:
            raise ValueError("A/B person token shapes must match")
        if mask_a.shape != mask_b.shape or mask_a.shape != person_a.shape[:-1]:
            raise ValueError("A/B token masks must match person token sequences")
        batch_size = person_a.shape[0]
        if relative_tokens is None:
            relative = person_a.new_zeros(batch_size, 0, person_a.shape[-1])
            relative_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=person_a.device)
        else:
            expected = (batch_size, 2, self.config.condition_dim)
            if tuple(relative_tokens.shape) != expected:
                raise ValueError(f"relative tokens must have shape {expected}")
            relative = self.relative_projection(relative_tokens)
            relative_mask = torch.ones(
                batch_size, relative.shape[1], dtype=torch.bool, device=person_a.device
            )

        direction = self.direction_embedding.weight.to(dtype=person_a.dtype)
        for layer in self.layers:
            previous_a, previous_b = person_a, person_b
            context_for_a = torch.cat(
                (
                    previous_b + direction[0][None, None],
                    relative + direction[0][None, None],
                ),
                dim=1,
            )
            context_for_b = torch.cat(
                (
                    previous_a + direction[1][None, None],
                    relative + direction[1][None, None],
                ),
                dim=1,
            )
            context_mask_a = torch.cat((mask_b, relative_mask), dim=1)
            context_mask_b = torch.cat((mask_a, relative_mask), dim=1)
            person_a = layer(
                previous_a,
                context_for_a,
                query_mask=mask_a,
                context_mask=context_mask_a,
            )
            person_b = layer(
                previous_b,
                context_for_b,
                query_mask=mask_b,
                context_mask=context_mask_b,
            )
        return person_a, person_b
