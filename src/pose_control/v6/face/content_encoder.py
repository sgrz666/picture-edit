from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import FaceReferenceFeatures
from .resampler import FeedForward, PerceiverAttention, Resampler


def pool_arcface_references(
    arcface: torch.Tensor,
    reference_valid: torch.Tensor,
) -> torch.Tensor:
    """FP32 normalize, valid-mean, then normalize ArcFace references again."""

    if arcface.ndim < 3 or arcface.shape[-1] != 512:
        raise ValueError("arcface must have shape [...,R,512]")
    if (
        tuple(reference_valid.shape) != tuple(arcface.shape[:-1])
        or reference_valid.dtype != torch.bool
    ):
        raise ValueError("reference_valid must match ArcFace reference dimensions")
    if reference_valid.device != arcface.device:
        raise ValueError("ArcFace features and reference_valid must be colocated")
    normalized = F.normalize(arcface.float(), dim=-1, eps=1e-12)
    weights = reference_valid[..., None].float()
    pooled = (normalized * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1)
    pooled = F.normalize(pooled, dim=-1, eps=1e-12)
    return pooled * reference_valid.any(dim=-1, keepdim=True).float()


class FacePerceiver(nn.Module):
    """Identity-seeded Perceiver that predicts a zero-initialized texture delta."""

    def __init__(
        self,
        *,
        dim: int = 512,
        depth: int = 4,
        query_count: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if min(dim, depth, query_count) <= 0:
            raise ValueError("FacePerceiver dimensions and depth must be positive")
        self.dim = dim
        self.query_count = query_count
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
        self.proj_out = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        identity_queries: torch.Tensor,
        appearance_tokens: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if appearance_tokens.ndim < 3 or appearance_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"appearance_tokens must have shape [...,N,{self.dim}]"
            )
        expected = (*appearance_tokens.shape[:-2], self.query_count, self.dim)
        if tuple(identity_queries.shape) != expected:
            raise ValueError(f"identity_queries must have shape {expected}")
        if (
            identity_queries.dtype != appearance_tokens.dtype
            or identity_queries.device != appearance_tokens.device
        ):
            raise ValueError(
                "identity_queries must match appearance token dtype and device"
            )
        latents = identity_queries
        for attention, feed_forward in self.layers:
            latents = latents + attention(
                appearance_tokens, latents, context_mask=context_mask
            )
            latents = latents + feed_forward(latents)
        return self.norm_out(self.proj_out(latents))


class FaceContentEncoder(nn.Module):
    """Encode multi-reference identity and DINO appearance for each face."""

    def __init__(
        self,
        *,
        dim: int = 512,
        identity_queries: int = 4,
        appearance_queries: int = 8,
        resampler_depth: int = 4,
        perceiver_depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        if dim != 512:
            raise ValueError("face content token dimension is fixed at 512")
        if identity_queries != 4 or appearance_queries != 8:
            raise ValueError("face content uses four identity and eight appearance tokens")
        self.dim = dim
        self.identity_queries = identity_queries
        self.identity_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, identity_queries * dim),
        )
        self.identity_norm = nn.LayerNorm(dim)
        self.resampler = Resampler(
            dim=dim,
            depth=resampler_depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=appearance_queries,
            embedding_dim=1536,
            output_dim=dim,
        )
        self.face_perceiver = FacePerceiver(
            dim=dim,
            depth=perceiver_depth,
            query_count=identity_queries,
            dim_head=dim_head,
            heads=heads,
        )

    @staticmethod
    def pool_arcface(
        arcface: torch.Tensor,
        reference_valid: torch.Tensor,
    ) -> torch.Tensor:
        return pool_arcface_references(arcface, reference_valid)

    def forward(
        self,
        references: FaceReferenceFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        references.validate()
        parameter = next(self.parameters())
        if references.device != parameter.device:
            raise ValueError("reference features and encoder must be on the same device")
        model_dtype = parameter.dtype
        batch_size = references.batch_size
        reference_count = references.reference_count
        has_reference = references.reference_valid.any(dim=-1)

        pooled_arcface = pool_arcface_references(
            references.arcface, references.reference_valid
        ).to(dtype=model_dtype)
        identity_tokens = self.identity_projection(pooled_arcface).reshape(
            batch_size, 2, self.identity_queries, self.dim
        )
        identity_tokens = self.identity_norm(identity_tokens)
        identity_tokens = identity_tokens * has_reference[..., None, None].to(
            identity_tokens.dtype
        )

        dino = references.dino_patches.to(dtype=model_dtype).reshape(
            batch_size * 2 * reference_count, 256, 1536
        )
        per_reference = self.resampler(dino).reshape(
            batch_size, 2, reference_count, 8, self.dim
        )
        weights = references.reference_valid[..., None, None].to(
            per_reference.dtype
        )
        texture_tokens = (per_reference * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1)
        texture_tokens = texture_tokens * has_reference[..., None, None].to(
            texture_tokens.dtype
        )
        texture_delta = self.face_perceiver(identity_tokens, texture_tokens)
        texture_delta = texture_delta * has_reference[..., None, None].to(
            texture_delta.dtype
        )
        return identity_tokens, texture_tokens, texture_delta
