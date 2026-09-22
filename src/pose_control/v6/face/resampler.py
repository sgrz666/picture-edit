from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Pre-normalized MLP used by the face Perceiver blocks."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        if dim <= 0 or mult <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class PerceiverAttention(nn.Module):
    """Cross-attention from compact latents to context plus latent keys."""

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if min(dim, dim_head, heads) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.norm_context = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        context: torch.Tensor,
        latents: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim < 3 or latents.ndim != context.ndim:
            raise ValueError("context and latents must have shape [...,N,D]")
        if context.shape[:-2] != latents.shape[:-2]:
            raise ValueError("context and latents must share leading dimensions")
        if context.shape[-1] != latents.shape[-1]:
            raise ValueError("context and latents must share their feature dimension")
        if context_mask is not None:
            if (
                tuple(context_mask.shape) != tuple(context.shape[:-1])
                or context_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "context_mask must match context token dimensions and use bool"
                )
            if context_mask.device != context.device:
                raise ValueError("context_mask must be on the context device")

        normalized_context = self.norm_context(context)
        normalized_latents = self.norm_latents(latents)
        key_values = torch.cat((normalized_context, normalized_latents), dim=-2)
        query = self.to_q(normalized_latents)
        key, value = self.to_kv(key_values).chunk(2, dim=-1)

        leading = query.shape[:-2]
        query_count = query.shape[-2]
        key_count = key.shape[-2]
        query = query.reshape(*leading, query_count, self.heads, self.dim_head).transpose(-3, -2)
        key = key.reshape(*leading, key_count, self.heads, self.dim_head).transpose(-3, -2)
        value = value.reshape(*leading, key_count, self.heads, self.dim_head).transpose(-3, -2)
        similarity = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if context_mask is not None:
            latent_mask = torch.ones(
                *latents.shape[:-1], dtype=torch.bool, device=latents.device
            )
            key_mask = torch.cat((context_mask, latent_mask), dim=-1)
            similarity = similarity.masked_fill(
                ~key_mask[..., None, None, :],
                -torch.finfo(similarity.dtype).max,
            )
        attention = similarity.float().softmax(dim=-1).to(similarity.dtype)
        output = torch.matmul(attention, value)
        output = output.transpose(-3, -2).reshape(
            *leading, query_count, self.heads * self.dim_head
        )
        return self.to_out(output)


class Resampler(nn.Module):
    """Resample 256 DINOv2 patch tokens into eight 512-D appearance tokens."""

    def __init__(
        self,
        *,
        dim: int = 512,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        num_queries: int = 8,
        embedding_dim: int = 1536,
        output_dim: int = 512,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if min(dim, depth, num_queries, embedding_dim, output_dim) <= 0:
            raise ValueError("resampler dimensions and depth must be positive")
        self.dim = dim
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.latents = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.latents, std=dim**-0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim < 3 or context.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"context must have shape [...,N,{self.embedding_dim}]"
            )
        if not context.is_floating_point() or not torch.isfinite(context).all():
            raise ValueError("context must contain finite floating values")
        if context_mask is not None:
            if (
                tuple(context_mask.shape) != tuple(context.shape[:-1])
                or context_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "context_mask must match context token dimensions and use bool"
                )
            if context_mask.device != context.device:
                raise ValueError("context_mask must be on the context device")
        hidden_states = self.proj_in(context)
        latents = self.latents.to(dtype=hidden_states.dtype).expand(
            *hidden_states.shape[:-2], -1, -1
        )
        for attention, feed_forward in self.layers:
            latents = latents + attention(
                hidden_states, latents, context_mask=context_mask
            )
            latents = latents + feed_forward(latents)
        return self.norm_out(self.proj_out(latents))
