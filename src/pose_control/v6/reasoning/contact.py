from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import CrossAttentionFFN
from .config import AdapterReasoningConfig


@dataclass
class ContactReasoningOutput:
    person_a: torch.Tensor
    person_b: torch.Tensor
    spatial_low: torch.Tensor
    spatial_high: torch.Tensor


class ContactConditionReasoner(nn.Module):
    """Gate contact raster and typed relation tokens into dual-person states."""

    def __init__(self, config: AdapterReasoningConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.spatial_projection = nn.Conv2d(128, hidden_dim, 1, bias=False)
        self.spatial_downsample = nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False
        )
        self.token_projection = nn.Linear(config.condition_dim, hidden_dim, bias=False)
        self.contact_gate = nn.Parameter(
            torch.full((2,), float(config.contact_gate_init))
        )
        self.relation_attention = nn.ModuleList(
            CrossAttentionFFN(
                hidden_dim,
                config.contact_attention_heads,
                ffn_ratio=config.ffn_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.contact_attention_layers)
        )

    def _apply_relation_attention(
        self,
        person_a: torch.Tensor,
        person_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        contact_tokens: torch.Tensor | None,
        contact_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contact_tokens is None:
            return person_a, person_b
        if contact_mask is None:
            raise ValueError("contact_mask is required when contact_tokens are present")
        expected = (person_a.shape[0], 8, self.config.condition_dim)
        if tuple(contact_tokens.shape) != expected:
            raise ValueError(f"contact tokens must have shape {expected}")
        if tuple(contact_mask.shape) != expected[:-1] or contact_mask.dtype != torch.bool:
            raise ValueError("contact mask must have shape [B,8] and dtype bool")

        active = contact_mask.any(dim=1)
        if not active.any():
            return person_a, person_b
        indices = active.nonzero(as_tuple=False).flatten()
        relation = self.token_projection(contact_tokens.index_select(0, indices))
        relation_mask = contact_mask.index_select(0, indices)
        relation = relation * relation_mask[..., None].to(relation.dtype)
        selected_a = person_a.index_select(0, indices)
        selected_b = person_b.index_select(0, indices)
        selected_mask_a = mask_a.index_select(0, indices)
        selected_mask_b = mask_b.index_select(0, indices)
        for layer in self.relation_attention:
            previous_a, previous_b = selected_a, selected_b
            selected_a = layer(
                previous_a,
                relation,
                query_mask=selected_mask_a,
                context_mask=relation_mask,
            )
            selected_b = layer(
                previous_b,
                relation,
                query_mask=selected_mask_b,
                context_mask=relation_mask,
            )
        return (
            person_a.index_copy(0, indices, selected_a),
            person_b.index_copy(0, indices, selected_b),
        )

    def forward(
        self,
        person_a: torch.Tensor,
        person_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        contact_spatial: torch.Tensor | None,
        contact_tokens: torch.Tensor | None,
        contact_mask: torch.Tensor | None,
        *,
        high_size: tuple[int, int],
    ) -> ContactReasoningOutput:
        batch_size, token_count, hidden_dim = person_a.shape
        low_height, low_width = high_size[0] // 2, high_size[1] // 2
        if token_count != low_height * low_width:
            raise ValueError("person token count does not match requested spatial size")
        if contact_spatial is None:
            contact_high = person_a.new_zeros(batch_size, hidden_dim, *high_size)
        else:
            expected = (batch_size, 128, *high_size)
            if tuple(contact_spatial.shape) != expected:
                raise ValueError(f"contact spatial feature must have shape {expected}")
            present = contact_spatial.abs().flatten(1).sum(dim=1) > 0
            contact_high = self.spatial_projection(contact_spatial)
            contact_high = contact_high * present[:, None, None, None].to(
                contact_high.dtype
            )
        contact_low = self.spatial_downsample(contact_high)
        contact_low_tokens = contact_low.flatten(2).transpose(1, 2)
        gates = self.contact_gate.sigmoid().to(person_a.dtype)
        person_a = (
            person_a + gates[0] * contact_low_tokens
        ) * mask_a[..., None].to(person_a.dtype)
        person_b = (
            person_b + gates[1] * contact_low_tokens
        ) * mask_b[..., None].to(person_b.dtype)
        person_a, person_b = self._apply_relation_attention(
            person_a,
            person_b,
            mask_a,
            mask_b,
            contact_tokens,
            contact_mask,
        )
        return ContactReasoningOutput(person_a, person_b, contact_low, contact_high)
