from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from .conditions import DetailReferenceBatch, FaceHandDetailCondition


class ReferenceFeatureProvider(Protocol):
    """Provider boundary for future external, precomputed reference features."""

    def __call__(self, references: DetailReferenceBatch) -> torch.Tensor:
        """Return [B,2,3,R,F] floating features."""


def canonicalize_left_hand_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """Mirror normalized left-hand x coordinates into right-hand canonical space."""

    if keypoints.shape[-2:] != (21, 3) or not keypoints.is_floating_point():
        raise ValueError("left-hand keypoints must end with shape [21,3] and be floating")
    canonical = keypoints.clone()
    canonical[..., 0] = 1.0 - canonical[..., 0]
    return canonical


def restore_left_hand_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """Map canonical hand keypoints back to the anatomical left-hand frame."""

    return canonicalize_left_hand_keypoints(keypoints)

def canonicalize_left_hand_pose(pose: torch.Tensor) -> torch.Tensor:
    """Reflect SMPL-X/MANO local axis-angle triplets into right-hand space.

    SMPL-X stores the 15 MANO joint rotations as consecutive local axis-angle
    triplets [rx, ry, rz]. Under horizontal reflection an axial vector
    transforms as [rx, -ry, -rz].
    """

    if pose.shape[-1] != 45 or not pose.is_floating_point():
        raise ValueError("left-hand pose must end with 45 floating axis-angle values")
    canonical = pose.reshape(*pose.shape[:-1], 15, 3).clone()
    canonical[..., 1:] = -canonical[..., 1:]
    return canonical.reshape_as(pose)


def restore_left_hand_pose(pose: torch.Tensor) -> torch.Tensor:
    """Invert left-hand SMPL-X/MANO axis-angle canonicalization."""

    return canonicalize_left_hand_pose(pose)


class DetailSpatialEncoder(nn.Module):
    """Trainable local encoder for landmarks plus SMPL-X detail parameters."""

    output_channels = 128

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.face_encoder = nn.Sequential(
            nn.LayerNorm(68 * 3 + 13),
            nn.Linear(68 * 3 + 13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_channels),
        )
        self.hand_encoder = nn.Sequential(
            nn.LayerNorm(21 * 3 + 45),
            nn.Linear(21 * 3 + 45, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_channels),
        )

    def region_features(self, condition: FaceHandDetailCondition) -> torch.Tensor:
        batch_size = condition.batch_size
        face_input = torch.cat(
            (
                condition.face_keypoints.reshape(batch_size, 2, -1),
                condition.smplx_detail[..., :13],
            ),
            dim=-1,
        )
        left_keypoints = canonicalize_left_hand_keypoints(
            condition.hand_keypoints[:, :, 0]
        )
        left_pose = canonicalize_left_hand_pose(
            condition.smplx_detail[..., 13:58]
        )
        left_input = torch.cat(
            (
                left_keypoints.reshape(batch_size, 2, -1),
                left_pose,
            ),
            dim=-1,
        )
        right_input = torch.cat(
            (
                condition.hand_keypoints[:, :, 1].reshape(batch_size, 2, -1),
                condition.smplx_detail[..., 58:103],
            ),
            dim=-1,
        )
        return torch.stack(
            (
                self.face_encoder(face_input),
                self.hand_encoder(left_input),
                self.hand_encoder(right_input),
            ),
            dim=2,
        )

    def forward(
        self,
        condition: FaceHandDetailCondition,
        region_masks: torch.Tensor,
    ) -> torch.Tensor:
        if region_masks.ndim != 5 or region_masks.shape[:3] != (
            condition.batch_size,
            2,
            3,
        ):
            raise ValueError("region_masks must have shape [B,2,3,H,W]")
        features = self.region_features(condition)
        weights = region_masks[:, :, :, None]
        numerator = (features[..., None, None] * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return numerator / denominator


class DetailAppearanceEncoder(nn.Module):
    """Eight local appearance tokens from reference pixels or provider features."""

    token_count = 8

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 6,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_count * token_dim),
        )
        self.null_appearance = nn.Parameter(
            torch.zeros(3, self.token_count, token_dim)
        )
        nn.init.normal_(self.null_appearance, std=0.02)

    @staticmethod
    def local_image_features(images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(-1, -2))
        variance = images.var(dim=(-1, -2), unbiased=False)
        return torch.cat((mean, torch.sqrt(variance + 1e-6)), dim=-1)

    def forward(
        self,
        references: DetailReferenceBatch,
        reference_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference_features is None:
            reference_features = self.local_image_features(references.images)
        expected = (
            references.batch_size,
            2,
            3,
            references.reference_count,
            self.feature_dim,
        )
        if tuple(reference_features.shape) != expected:
            raise ValueError(f"reference_features must have shape {expected}")
        if (
            not reference_features.is_floating_point()
            or reference_features.device != references.device
        ):
            raise ValueError("reference_features must be floating and colocated with references")
        feature_valid = references.reference_valid[..., None]
        reference_features = torch.where(
            feature_valid, reference_features, torch.zeros_like(reference_features)
        )
        if not torch.isfinite(reference_features).all():
            raise ValueError("valid reference_features must contain only finite values")
        projected = self.projection(reference_features).reshape(
            references.batch_size,
            2,
            3,
            references.reference_count,
            self.token_count,
            self.token_dim,
        )
        valid = references.reference_valid[..., None, None].to(projected.dtype)
        count = valid.sum(dim=3)
        pooled = (projected * valid).sum(dim=3) / count.clamp_min(1.0)
        missing = count == 0
        null = self.null_appearance[None, None].expand(
            references.batch_size, 2, -1, -1, -1
        )
        return torch.where(missing, null, pooled)
