from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import (
    DetailReferenceBatch,
    FaceHandDetailCondition,
    PreparedFaceHandDetailConditioning,
)
from .binder import DetailTokenBinder
from .encoders import (
    DetailAppearanceEncoder,
    DetailSpatialEncoder,
)


def soft_box_masks(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    spatial_size: tuple[int, int],
    *,
    sharpness: float = 40.0,
) -> torch.Tensor:
    """Rasterize bounded soft boxes; invalid regions are exactly zero."""

    if len(spatial_size) != 2 or min(spatial_size) <= 0:
        raise ValueError("spatial_size must contain two positive dimensions")
    height, width = spatial_size
    y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
    x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
    y = y.reshape((1,) * (boxes.ndim - 1) + (height, 1))
    x = x.reshape((1,) * (boxes.ndim - 1) + (1, width))
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1, x2 = x1[..., None, None], x2[..., None, None]
    y1, y2 = y1[..., None, None], y2[..., None, None]
    mask = (
        torch.sigmoid((x - x1) * sharpness)
        * torch.sigmoid((x2 - x) * sharpness)
        * torch.sigmoid((y - y1) * sharpness)
        * torch.sigmoid((y2 - y) * sharpness)
    )
    return mask * valid[..., None, None].to(mask.dtype)


class FaceHandDetailPreparer(nn.Module):
    """Validate and statically prepare local detail control inputs."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.spatial_encoder = DetailSpatialEncoder(hidden_dim=hidden_dim)
        self.appearance_encoder = DetailAppearanceEncoder(
            token_dim=token_dim, hidden_dim=hidden_dim
        )
        self.token_binder = DetailTokenBinder(token_dim=token_dim)

    @staticmethod
    def _validate_source_latents(
        source_latents: torch.Tensor,
        condition: FaceHandDetailCondition,
    ) -> None:
        valid_shape = (
            source_latents.ndim == 4
            and source_latents.shape[:2] == (condition.batch_size, 16)
        ) or (
            source_latents.ndim == 5
            and source_latents.shape[:3] == (condition.batch_size, 2, 16)
        )
        if not valid_shape:
            raise ValueError("source_latents must have shape [B,16,H,W] or [B,2,16,H,W]")
        if not source_latents.is_floating_point():
            raise ValueError("source_latents must use a floating dtype")
        if source_latents.device != condition.device:
            raise ValueError("source_latents and condition must be on the same device")
        if not torch.isfinite(source_latents).all():
            raise ValueError("source_latents must contain only finite values")

    @staticmethod
    def _source_canvas(
        source_latents: torch.Tensor,
        condition: FaceHandDetailCondition,
        latent_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = latent_masks.shape[0], 16, latent_masks.shape[-2], latent_masks.shape[-1]
        y = (torch.arange(height, device=source_latents.device, dtype=source_latents.dtype) + 0.5) / height
        x = (torch.arange(width, device=source_latents.device, dtype=source_latents.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        numerator = source_latents.new_zeros(batch_size, 16, height, width)
        denominator = source_latents.new_zeros(batch_size, 1, height, width)
        for person in range(2):
            for region in range(3):
                source = condition.source_boxes[:, person, region]
                target = condition.target_boxes[:, person, region]
                target_width = (target[:, 2] - target[:, 0]).clamp_min(1e-6)
                target_height = (target[:, 3] - target[:, 1]).clamp_min(1e-6)
                relative_x = (
                    xx[None] - target[:, 0, None, None]
                ) / target_width[:, None, None]
                relative_y = (
                    yy[None] - target[:, 1, None, None]
                ) / target_height[:, None, None]
                source_x = source[:, 0, None, None] + relative_x * (
                    source[:, 2] - source[:, 0]
                )[:, None, None]
                source_y = source[:, 1, None, None] + relative_y * (
                    source[:, 3] - source[:, 1]
                )[:, None, None]
                grid = torch.stack((source_x * 2 - 1, source_y * 2 - 1), dim=-1)
                person_source = (
                    source_latents
                    if source_latents.ndim == 4
                    else source_latents[:, person]
                )
                warped = F.grid_sample(
                    person_source,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                weight = latent_masks[:, person, region, None]
                numerator = numerator + warped * weight
                denominator = denominator + weight
        return numerator / denominator.clamp_min(1.0)

    def forward(
        self,
        condition: FaceHandDetailCondition,
        references: DetailReferenceBatch,
        latent_spatial_size: tuple[int, int],
        token_spatial_size: tuple[int, int],
        source_latents: torch.Tensor | None = None,
        *,
        reference_features: torch.Tensor | None = None,
        person_binding: torch.Tensor,
    ) -> PreparedFaceHandDetailConditioning:
        condition.validate()
        references.validate()
        if references.batch_size != condition.batch_size:
            raise ValueError("condition and references must have the same batch size")
        if references.device != condition.device:
            raise ValueError("condition and references must be on the same device")
        if references.images.dtype != condition.face_keypoints.dtype:
            raise ValueError("condition and reference images must use the same dtype")
        if person_binding.ndim != 3 or person_binding.shape[:2] != (
            condition.batch_size,
            2,
        ):
            raise ValueError("person_binding must have shape [B,2,D]")
        if not person_binding.is_floating_point() or person_binding.device != condition.device:
            raise ValueError("person_binding must be floating and colocated with condition")

        if source_latents is not None:
            self._validate_source_latents(source_latents, condition)

        parameter = next(self.parameters())
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        references = references.to(device=parameter.device, dtype=parameter.dtype)
        person_binding = person_binding.to(
            device=parameter.device, dtype=parameter.dtype
        )
        if reference_features is not None:
            reference_features = reference_features.to(
                device=parameter.device, dtype=parameter.dtype
            )
        if source_latents is not None:
            source_latents = source_latents.to(
                device=parameter.device, dtype=parameter.dtype
            )

        latent_masks = soft_box_masks(
            condition.target_boxes, condition.region_valid, latent_spatial_size
        )
        region_masks = soft_box_masks(
            condition.target_boxes, condition.region_valid, token_spatial_size
        )
        geometry = self.spatial_encoder(condition, latent_masks)
        if source_latents is None:
            source_canvas = geometry.new_zeros(
                condition.batch_size, 16, *latent_spatial_size
            )
        else:
            source_canvas = self._source_canvas(
                source_latents, condition, latent_masks
            )
        detail_valid = condition.region_valid.flatten(1).any(dim=1)
        detail_condition = torch.cat((source_canvas, geometry), dim=1)
        detail_condition = detail_condition * detail_valid[:, None, None, None].to(
            detail_condition.dtype
        )

        region_tokens = self.appearance_encoder(
            references, reference_features=reference_features
        )
        detail_tokens, detail_token_mask = self.token_binder(
            region_tokens, condition.region_valid, person_binding
        )
        return PreparedFaceHandDetailConditioning(
            detail_condition=detail_condition,
            detail_tokens=detail_tokens,
            detail_token_mask=detail_token_mask,
            region_masks=region_masks,
            region_valid=condition.region_valid,
            detail_valid=detail_valid,
        ).validate()
