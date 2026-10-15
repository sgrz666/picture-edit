"""Direct adaptations of Visual Persona's face-token attention primitives.

Upstream file:
``model/visual_persona/resampler.py`` and
``model/visual_persona/attention_processor.py`` at commit
``d393d09284cac20570466ae3c7ad137b10648bdc``.

Interface changes are intentionally small: arbitrary leading batch dimensions
and boolean context masks were added to the Resampler; the IP processor is
exposed as an ordinary module for ROI-local DiT states and names its projections
``*_face`` instead of ``*_ip``.  The attention equations and residual ordering
remain the upstream PyTorch 2.0 implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    """Visual Persona's pre-normalized two-layer feed-forward block."""

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _reshape_tensor(value: torch.Tensor, heads: int) -> torch.Tensor:
    leading = value.shape[:-2]
    length, width = value.shape[-2:]
    return value.reshape(*leading, length, heads, width // heads).transpose(-3, -2)


class PerceiverAttention(nn.Module):
    """Visual Persona Perceiver attention with an optional context mask."""

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
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
            if context_mask.dtype != torch.bool or context_mask.shape != context.shape[:-1]:
                raise ValueError("context_mask must match context tokens and use bool")
            if context_mask.device != context.device:
                raise ValueError("context_mask must be on the context device")

        context = self.norm1(context)
        latents = self.norm2(latents)
        query = _reshape_tensor(self.to_q(latents), self.heads)
        key, value = self.to_kv(torch.cat((context, latents), dim=-2)).chunk(2, dim=-1)
        key = _reshape_tensor(key, self.heads)
        value = _reshape_tensor(value, self.heads)

        # This split scale is retained from the pinned upstream source because
        # it is more stable than a post-matmul division in fp16.
        stable_scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weights = (query * stable_scale) @ (key * stable_scale).transpose(-2, -1)
        if context_mask is not None:
            latent_mask = torch.ones(
                latents.shape[:-1], dtype=torch.bool, device=latents.device
            )
            key_mask = torch.cat((context_mask, latent_mask), dim=-1)
            weights = weights.masked_fill(
                ~key_mask[..., None, None, :],
                -torch.finfo(weights.dtype).max,
            )
        weights = torch.softmax(weights.float(), dim=-1).to(weights.dtype)
        output = weights @ value
        output = output.transpose(-3, -2).reshape(
            *latents.shape[:-2], latents.shape[-2], self.heads * self.dim_head
        )
        return self.to_out(output)


class Resampler(nn.Module):
    """Pinned Visual Persona Resampler configured for DINOv2-G face patches."""

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
        if min(dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim) <= 0:
            raise ValueError("resampler dimensions and depth must be positive")
        self.dim = dim
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim < 3 or context.shape[-1] != self.embedding_dim:
            raise ValueError(f"context must have shape [...,N,{self.embedding_dim}]")
        if not context.is_floating_point() or not torch.isfinite(context).all():
            raise ValueError("context must contain finite floating values")
        if context_mask is not None:
            if context_mask.dtype != torch.bool or context_mask.shape != context.shape[:-1]:
                raise ValueError("context_mask must match context tokens and use bool")
            if context_mask.device != context.device:
                raise ValueError("context_mask must be on the context device")

        hidden_states = self.proj_in(context)
        leading = hidden_states.shape[:-2]
        latents = self.latents.to(dtype=hidden_states.dtype).reshape(
            *((1,) * len(leading)), self.num_queries, self.dim
        ).expand(*leading, -1, -1)
        for attention, feed_forward in self.layers:
            latents = attention(
                hidden_states, latents, context_mask=context_mask
            ) + latents
            latents = feed_forward(latents) + latents
        return self.norm_out(self.proj_out(latents))


class IPAttnProcessor2_0(nn.Module):
    """ROI-local form of Visual Persona's independent IP K/V processor."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        *,
        scale: float = 1.0,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("IPAttnProcessor2_0 requires PyTorch 2.0 or newer")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = float(scale)
        self.heads = heads
        self.head_dim = hidden_size // heads
        context_dim = cross_attention_dim or hidden_size
        self.to_q_face = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_face = nn.Linear(context_dim, hidden_size, bias=False)
        self.to_v_face = nn.Linear(context_dim, hidden_size, bias=False)
        self.to_out_face = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def face_attention(
        self,
        hidden_states: torch.Tensor,
        face_hidden_states: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.to_q_face(hidden_states))
        key = self._split_heads(self.to_k_face(face_hidden_states))
        value = self._split_heads(self.to_v_face(face_hidden_states))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape_as(hidden_states)
        multiplier = self.scale if scale is None else float(scale)
        return self.to_out_face(attended) * multiplier


__all__ = [
    "FeedForward",
    "IPAttnProcessor2_0",
    "PerceiverAttention",
    "Resampler",
]
