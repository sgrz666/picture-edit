from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import HandFineCondition, HandReferenceFeatures, PreparedHandConditioning
from .geometry import axis_angle45_to_6d96, hand_heatmaps, local_hand_points
from .vton_layers import HandResampler, HandStructureProjector, OneLinearLN


class HandConditioningPreparer(nn.Module):
    """Independent geometry/appearance encoding; no target-frame RGB leakage."""

    def __init__(self, *, dim: int = 512, resampler_depth: int = 2, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.structure = HandStructureProjector(dim)
        self.dino_projection = OneLinearLN(1536, dim)
        self.resampler = HandResampler(dim=dim, depth=resampler_depth, heads=heads)
        self.spatial = nn.Sequential(
            nn.Conv2d(23, dim // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1), nn.SiLU(),
        )
        self.source_embedding = nn.Embedding(16, dim)
        self.person_embedding = nn.Embedding(2, dim)
        self.side_embedding = nn.Embedding(2, dim)
        self.region_embedding = nn.Parameter(torch.randn(dim) * .02)
        self.null_appearance = nn.Parameter(torch.zeros(8, dim))

    def forward(self, condition: HandFineCondition, references: HandReferenceFeatures) -> PreparedHandConditioning:
        condition.validate()
        references.validate()
        if condition.batch_size != references.batch_size or condition.device != references.dino_patches.device:
            raise ValueError("condition/references batch and device must match")
        parameter = next(self.parameters())
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        references = references.to(device=parameter.device, dtype=parameter.dtype)
        b = condition.batch_size
        valid = condition.region_valid
        safe_boxes = torch.where(valid[..., None], condition.target_boxes, condition.target_boxes.new_tensor([0, 0, 1, 1]))
        local_points = local_hand_points(condition.hand_keypoints, safe_boxes)
        local_points = local_points * valid[..., None, None].to(local_points.dtype)
        heatmaps = hand_heatmaps(local_points)
        mask = heatmaps.amax(dim=-3).clamp(0, 1)
        mask = F.avg_pool2d(F.max_pool2d(mask.reshape(-1, 1, 16, 16), 5, stride=1, padding=2), 3, stride=1, padding=1).reshape(b, 2, 2, 16, 16)
        mask = mask * valid[..., None, None]
        side = torch.arange(2, device=condition.device, dtype=parameter.dtype).view(1, 1, 2, 1).expand(b, 2, 2, 1)
        pose6d = axis_angle45_to_6d96(condition.hand_pose)
        three_d, two_d = self.structure(
            condition.mesh_bps, pose6d, local_points[..., :2].reshape(b, 2, 2, 42),
            side, condition.mesh_valid & valid,
        )
        binding = (
            self.source_embedding(torch.where(valid, condition.source_indices[..., None].expand(-1, -1, 2), 0))
            + self.person_embedding(torch.arange(2, device=condition.device))[None, :, None]
            + self.side_embedding(torch.arange(2, device=condition.device))[None, None]
            + self.region_embedding
        )
        geometry = torch.stack((three_d, two_d, (three_d + two_d) * .5, three_d - two_d), -2)
        geometry = (geometry + binding[..., None, :]) * valid[..., None, None]
        spatial_input = torch.cat((heatmaps, mask[..., None, :, :], condition.depth), dim=-3)
        spatial = self.spatial(spatial_input.reshape(b * 4, 23, 16, 16)).reshape(b, 2, 2, self.dim, 256).transpose(-1, -2)
        spatial = (spatial + (three_d + two_d + binding)[..., None, :]) * valid[..., None, None]
        patches = references.dino_patches
        r, n = patches.shape[3:5]
        active_refs = references.reference_valid & valid[..., None]
        # Mask before projection so invalid references cannot contaminate gradients.
        flat = (patches * active_refs[..., None, None]).reshape(b * 4 * r, n, 1536)
        encoded = self.resampler(self.dino_projection(flat)).reshape(b, 2, 2, r, 8, self.dim)
        weights = active_refs.to(encoded.dtype)
        pooled = (encoded * weights[..., None, None]).sum(3) / weights.sum(3).clamp_min(1)[..., None, None]
        has_appearance = active_refs.any(-1)
        appearance = torch.where(has_appearance[..., None, None], pooled, self.null_appearance)
        appearance = (appearance + binding[..., None, :]) * valid[..., None, None]
        return PreparedHandConditioning(
            spatial_features=spatial,
            geometry_tokens=geometry,
            appearance_tokens=appearance,
            hand_masks=mask,
            target_boxes=condition.target_boxes * valid[..., None],
            hand_valid=valid,
            visibility=condition.visibility * valid,
        ).validate()
