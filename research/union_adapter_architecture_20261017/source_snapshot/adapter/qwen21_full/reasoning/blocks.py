from __future__ import annotations

import torch
import torch.nn as nn


def group_count(channels: int) -> int:
    return next(group for group in (32, 16, 8, 4, 2, 1) if channels % group == 0)


class ReasoningResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = group_count(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            value = value * mask.to(value.dtype)
        value = value + self.block(value)
        if mask is not None:
            value = value * mask.to(value.dtype)
        return value


class CrossAttentionFFN(nn.Module):
    """Pre-norm cross-attention followed by a residual feed-forward block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        ffn_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_mask is not None and torch.any(~context_mask.any(dim=1)):
            raise ValueError("cross-attention context cannot be fully masked")
        update = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=None if context_mask is None else ~context_mask,
            need_weights=False,
        )[0]
        query = (query + update) * query_mask[..., None].to(query.dtype)
        query = (query + self.ffn(query)) * query_mask[..., None].to(query.dtype)
        return query
