"""Direct adaptations of StableAnimator identity-fusion components.

Upstream files ``animation/modules/id_encoder.py`` and
``animation/modules/face_model.py`` at commit
``020e7f23a768410e0183ce943397d96a609f4ec1`` (MIT).

The token dimension and query count are configurable, multi-leading-dimension
inputs are accepted, and InsightFace extraction is an offline-only wrapper.
No upstream checkpoint or model weight is bundled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_persona import FeedForward, PerceiverAttention


class FacePerceiver(nn.Module):
    """StableAnimator FacePerceiver with optional context masking."""

    def __init__(
        self,
        *,
        dim: int = 512,
        depth: int = 4,
        query_count: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        embedding_dim: int | None = None,
        output_dim: int | None = None,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if min(dim, depth, query_count, dim_head, heads) <= 0:
            raise ValueError("FacePerceiver dimensions and depth must be positive")
        embedding_dim = dim if embedding_dim is None else embedding_dim
        output_dim = dim if output_dim is None else output_dim
        self.dim = dim
        self.query_count = query_count
        self.proj_in = (
            nn.Identity() if embedding_dim == dim else nn.Linear(embedding_dim, dim)
        )
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
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        identity_queries: torch.Tensor,
        appearance_tokens: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if appearance_tokens.ndim < 3:
            raise ValueError("appearance_tokens must have shape [...,N,D]")
        appearance_tokens = self.proj_in(appearance_tokens)
        expected = (*appearance_tokens.shape[:-2], self.query_count, self.dim)
        if identity_queries.shape != expected:
            raise ValueError(f"identity_queries must have shape {expected}")
        if (
            identity_queries.dtype != appearance_tokens.dtype
            or identity_queries.device != appearance_tokens.device
        ):
            raise ValueError("identity queries and appearance tokens must share dtype/device")
        latents = identity_queries
        for attention, feed_forward in self.layers:
            latents = attention(
                appearance_tokens, latents, context_mask=context_mask
            ) + latents
            latents = feed_forward(latents) + latents
        return self.norm_out(self.proj_out(latents))


class FusionFaceId(nn.Module):
    """StableAnimator's ArcFace projection plus shortcut texture fusion."""

    def __init__(
        self,
        *,
        cross_attention_dim: int = 512,
        id_embeddings_dim: int = 512,
        clip_embeddings_dim: int = 512,
        num_tokens: int = 4,
        heads: int = 8,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        self.fusion_model = FacePerceiver(
            dim=cross_attention_dim,
            depth=depth,
            query_count=num_tokens,
            dim_head=64,
            heads=heads,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def project_identity(self, id_embeds: torch.Tensor) -> torch.Tensor:
        projected = self.proj(id_embeds)
        projected = projected.reshape(
            *id_embeds.shape[:-1], self.num_tokens, self.cross_attention_dim
        )
        return self.norm(projected)

    def forward(
        self,
        id_embeds: torch.Tensor,
        clip_embeds: torch.Tensor,
        *,
        shortcut: bool = False,
        scale: float = 1.0,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        identity = self.project_identity(id_embeds)
        output = self.fusion_model(identity, clip_embeds, context_mask=context_mask)
        if shortcut:
            output = identity + float(scale) * output
        return output


class InsightFaceArcFaceExtractor:
    """Offline ArcFace wrapper adapted from StableAnimator ``FaceModel``."""

    def __init__(
        self,
        *,
        name: str,
        root: str,
        providers: Sequence[str],
        ctx_id: int,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - optional offline dependency
            raise RuntimeError(
                "ArcFace extraction requires insightface and an ONNX runtime provider"
            ) from exc
        self.app: Any = FaceAnalysis(
            name=name,
            root=root,
            providers=list(providers),
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @torch.inference_mode()
    def extract_bgr(
        self,
        image_bgr: np.ndarray,
        full_image_bgr: np.ndarray | None = None,
    ) -> torch.Tensor:
        if full_image_bgr is not None:
            faces = self.app.get(np.ascontiguousarray(full_image_bgr))
            if faces:
                face = max(
                    faces,
                    key=lambda item: float(
                        (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                    ),
                )
                embedding = torch.as_tensor(face.embedding, dtype=torch.float32)
                if embedding.shape == (512,) and torch.isfinite(embedding).all():
                    return F.normalize(embedding, dim=0)

        faces = self.app.get(np.ascontiguousarray(image_bgr))
        if faces:
            face = max(
                faces,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                ),
            )
            embedding = torch.as_tensor(face.embedding, dtype=torch.float32)
            if embedding.shape == (512,) and torch.isfinite(embedding).all():
                return F.normalize(embedding, dim=0)

        rec_model = getattr(self.app, "models", {}).get("recognition")
        if rec_model is not None:
            import cv2
            face_112 = cv2.resize(np.ascontiguousarray(image_bgr), (112, 112))
            feat = rec_model.get_feat(face_112)
            if feat is not None:
                embedding = torch.as_tensor(feat.flatten(), dtype=torch.float32)
                if embedding.shape == (512,) and torch.isfinite(embedding).all():
                    return F.normalize(embedding, dim=0)

        raise RuntimeError("InsightFace found no face in the crop or image")


__all__ = ["FacePerceiver", "FusionFaceId", "InsightFaceArcFaceExtractor"]
