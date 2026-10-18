from __future__ import annotations

import torch
import torch.nn as nn

from .conditions import (
    FaceFineCondition,
    FaceReferenceFeatures,
    PreparedFaceConditioning,
)
from .content_encoder import FaceContentEncoder
from .geometry import FaceGeometryTokenizer, normalize_face_landmarks_to_roi
from .spatial import FaceSpatialEncoder, render_face_landmark_heatmaps


class FaceConditioningPreparer(nn.Module):
    """Prepare static face geometry, identity, texture, masks, and ROI bindings."""

    def __init__(
        self,
        *,
        resampler_depth: int = 4,
        perceiver_depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        heatmap_sigma: float = 2.0,
        mask_feather_sigma: float = 8.0,
    ) -> None:
        super().__init__()
        self.geometry_tokenizer = FaceGeometryTokenizer(heads=heads)
        self.content_encoder = FaceContentEncoder(
            resampler_depth=resampler_depth,
            perceiver_depth=perceiver_depth,
            heads=heads,
            dim_head=dim_head,
        )
        self.spatial_encoder = FaceSpatialEncoder()
        self.source_index_embedding = nn.Embedding(16, 512)
        self.person_slot_embedding = nn.Embedding(2, 512)
        nn.init.normal_(self.source_index_embedding.weight, std=0.02)
        nn.init.normal_(self.person_slot_embedding.weight, std=0.02)
        self.heatmap_sigma = heatmap_sigma
        self.mask_feather_sigma = mask_feather_sigma

    def _binding_tokens(self, condition: FaceFineCondition) -> torch.Tensor:
        """Return valid-only source identity plus fixed person-slot bindings."""

        safe_source_indices = torch.where(
            condition.face_valid,
            condition.source_indices,
            torch.zeros_like(condition.source_indices),
        )
        source_binding = self.source_index_embedding(safe_source_indices)
        person_slots = torch.arange(2, device=condition.device)
        slot_binding = self.person_slot_embedding(person_slots)[None].expand(
            condition.batch_size, -1, -1
        )
        binding = source_binding + slot_binding
        return binding * condition.face_valid[..., None].to(binding.dtype)

    def forward(
        self,
        condition: FaceFineCondition,
        references: FaceReferenceFeatures,
    ) -> PreparedFaceConditioning:
        condition.validate()
        references.validate()
        if references.batch_size != condition.batch_size:
            raise ValueError("condition and references must have the same batch size")
        if references.device != condition.device:
            raise ValueError("condition and references must be on the same device")

        parameter = next(self.parameters())
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        references = references.to(device=parameter.device, dtype=parameter.dtype)
        valid_scale = condition.face_valid[..., None, None]

        # Invalid rows may intentionally carry a zero sentinel box. Substitute a
        # harmless unit box only for the local-coordinate calculation, then mask.
        safe_boxes = torch.where(
            condition.face_valid[..., None],
            condition.target_boxes,
            torch.tensor(
                [0.0, 0.0, 1.0, 1.0],
                dtype=condition.target_boxes.dtype,
                device=condition.target_boxes.device,
            ),
        )
        local_landmarks = normalize_face_landmarks_to_roi(
            condition.landmarks, safe_boxes
        )
        local_landmarks = local_landmarks * valid_scale.to(local_landmarks.dtype)

        geometry_tokens = self.geometry_tokenizer(
            local_landmarks,
            condition.jaw_pose,
            condition.expression,
            condition.face_valid,
        )
        heatmaps, face_masks = render_face_landmark_heatmaps(
            local_landmarks,
            condition.face_valid,
            sigma=self.heatmap_sigma,
            feather_sigma=self.mask_feather_sigma,
        )
        spatial_features = self.spatial_encoder(
            heatmaps, face_masks, condition.face_valid
        )
        identity_tokens, texture_tokens, texture_delta = self.content_encoder(
            references
        )

        binding_tokens = self._binding_tokens(condition)
        identity_tokens = identity_tokens + binding_tokens[..., None, :]
        texture_tokens = texture_tokens + binding_tokens[..., None, :]
        geometry_tokens = geometry_tokens + binding_tokens[..., None, :]
        token_scale = condition.face_valid[..., None, None].to(
            identity_tokens.dtype
        )
        identity_tokens = identity_tokens * token_scale
        texture_tokens = texture_tokens * token_scale
        texture_delta = texture_delta * token_scale
        target_boxes = condition.target_boxes * condition.face_valid[..., None].to(
            condition.target_boxes.dtype
        )
        return PreparedFaceConditioning(
            spatial_features=spatial_features,
            identity_tokens=identity_tokens,
            texture_tokens=texture_tokens,
            texture_delta=texture_delta,
            geometry_tokens=geometry_tokens,
            face_masks=face_masks,
            target_boxes=target_boxes,
            face_valid=condition.face_valid,
        ).validate()


# Stable public aliases while downstream integration settles on one name.
FaceConditionPreparer = FaceConditioningPreparer
FacePreparer = FaceConditioningPreparer
