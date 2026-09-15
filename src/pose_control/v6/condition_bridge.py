from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .conditions import ConditionBundle


@dataclass
class BridgedCondition:
    person_features: torch.Tensor
    person_masks: torch.Tensor
    global_tokens: torch.Tensor
    global_token_mask: torch.Tensor
    relative_tokens: torch.Tensor
    relative_token_mask: torch.Tensor
    contact_spatial: torch.Tensor
    contact_tokens: torch.Tensor
    contact_mask: torch.Tensor
    task_token: torch.Tensor


class ConditionBundleBridge(nn.Module):
    """Project the public 256-dim bundle into the current V6 inner widths."""

    def __init__(self, *, geometry_channels: int = 128, token_dim: int = 1536) -> None:
        super().__init__()
        self.geometry_channels = geometry_channels
        self.token_dim = token_dim
        self.spatial_projection = nn.Conv2d(256, geometry_channels, 1)
        self.contact_projection = nn.Conv2d(128, geometry_channels, 1)
        self.token_projection = nn.Linear(256, token_dim)

    def forward(self, bundle: ConditionBundle) -> BridgedCondition:
        bundle.validate()
        valid = bundle.person_valid
        spatial_a = self.spatial_projection(bundle.person_a_spatial)
        if bundle.person_b_spatial is None:
            spatial_b = torch.zeros_like(spatial_a)
            mask_b = torch.zeros_like(bundle.person_a_mask)
            global_b = torch.zeros_like(bundle.person_a_global_tokens)
        else:
            spatial_b = self.spatial_projection(bundle.person_b_spatial)
            mask_b = bundle.person_b_mask
            global_b = bundle.person_b_global_tokens
        person_features = torch.stack((spatial_a, spatial_b), dim=1)
        person_masks = torch.stack((bundle.person_a_mask, mask_b), dim=1)
        valid_map = valid[..., None, None, None].to(person_features.dtype)
        person_features = person_features * valid_map
        person_masks = person_masks * valid_map

        global_public = torch.stack((bundle.person_a_global_tokens, global_b), dim=1)
        global_tokens = self.token_projection(global_public)
        global_mask = valid[..., None].expand(-1, -1, global_public.shape[2])
        global_tokens = global_tokens * global_mask[..., None].to(global_tokens.dtype)

        batch_size = bundle.batch_size
        if bundle.relative_tokens is None:
            relative_tokens = bundle.task_token.new_zeros(batch_size, 2, 256)
            relative_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=bundle.device)
        else:
            relative_tokens = bundle.relative_tokens
            relative_mask = valid[:, 1, None].expand(-1, 2)
        relative_tokens = self.token_projection(relative_tokens)
        relative_tokens = relative_tokens * relative_mask[..., None].to(relative_tokens.dtype)

        if bundle.contact_spatial is None:
            contact_spatial = spatial_a.new_zeros(
                batch_size, self.geometry_channels, *spatial_a.shape[-2:]
            )
        else:
            contact_spatial = self.contact_projection(bundle.contact_spatial)
            contact_present = bundle.contact_spatial.abs().flatten(1).sum(dim=1) > 0
            contact_active = valid[:, 1] & contact_present
            contact_spatial = contact_spatial * contact_active[:, None, None, None].to(
                contact_spatial.dtype
            )

        if bundle.contact_tokens is None:
            contact_public = bundle.task_token.new_zeros(batch_size, 8, 256)
            contact_mask = torch.zeros(batch_size, 8, dtype=torch.bool, device=bundle.device)
        else:
            contact_public = bundle.contact_tokens
            contact_mask = bundle.contact_mask & valid[:, 1, None]
        contact_tokens = self.token_projection(contact_public)
        contact_tokens = contact_tokens * contact_mask[..., None].to(contact_tokens.dtype)

        return BridgedCondition(
            person_features=person_features,
            person_masks=person_masks,
            global_tokens=global_tokens,
            global_token_mask=global_mask,
            relative_tokens=relative_tokens,
            relative_token_mask=relative_mask,
            contact_spatial=contact_spatial,
            contact_tokens=contact_tokens,
            contact_mask=contact_mask,
            task_token=self.token_projection(bundle.task_token),
        )
