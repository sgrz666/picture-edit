from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.v65_face.stableanimator import FacePerceiver, FusionFaceId

from .conditions import FaceReferenceFeatures
from .resampler import Resampler


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
        self.fusion_face_id = FusionFaceId(
            cross_attention_dim=dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=dim,
            num_tokens=identity_queries,
            heads=heads,
            depth=perceiver_depth,
        )
        self.resampler = Resampler(
            dim=dim,
            depth=resampler_depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=appearance_queries,
            embedding_dim=1536,
            output_dim=dim,
        )

    @property
    def identity_projection(self) -> nn.Sequential:
        """Compatibility view of StableAnimator's identity projection."""

        return self.fusion_face_id.proj

    @property
    def identity_norm(self) -> nn.LayerNorm:
        """Compatibility view of StableAnimator's identity token norm."""

        return self.fusion_face_id.norm

    @property
    def face_perceiver(self) -> FacePerceiver:
        """Compatibility view of StableAnimator's texture fusion Perceiver."""

        return self.fusion_face_id.fusion_model

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
        identity_tokens = self.fusion_face_id.project_identity(pooled_arcface)
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
