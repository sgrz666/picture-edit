"""Reused body spatial encoder with an independent depth/visibility extension."""
from __future__ import annotations

import torch
from torch import nn

from ..geometry_encoder import SpatialConditionEncoder, ConditionResBlock
from .conditions import GeometryCondition


class DualGeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = SpatialConditionEncoder(use_depth=False, output_channels=256)
        self.depth_visibility = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.GroupNorm(8,32), nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GroupNorm(8,64), nn.SiLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.GroupNorm(8,128), nn.SiLU(),
            nn.Conv2d(128,256,1), nn.GroupNorm(16,256), nn.SiLU(), ConditionResBlock(256),
        )

    def forward(self, geometry: GeometryCondition):
        batch, _, _, height, width = geometry.normal.shape
        rows = geometry.person_valid.flatten().nonzero(as_tuple=False).flatten()
        select = lambda tensor: tensor.flatten(0,1).index_select(0,rows)
        # Both slots use the reused A RoleFiLM path; B identity is never learned.
        spatial, mask = self.spatial_encoder(
            normal=select(geometry.normal), pose_heatmap=select(geometry.pose),
            part_onehot=select(geometry.part), human_mask=select(geometry.amodal_mask), role_id=0,
        )
        depth = select(geometry.depth)
        visibility = torch.cat((torch.log1p(depth), select(geometry.visible_owner), select(geometry.amodal_mask)),dim=1)
        spatial = (spatial + self.depth_visibility(visibility)) * mask
        output = spatial.new_zeros(batch*2,256,height//8,width//8).index_copy(0,rows,spatial)
        masks = mask.new_zeros(batch*2,1,height//8,width//8).index_copy(0,rows,mask)
        return output.reshape(batch,2,256,height//8,width//8), masks.reshape(batch,2,1,height//8,width//8)
