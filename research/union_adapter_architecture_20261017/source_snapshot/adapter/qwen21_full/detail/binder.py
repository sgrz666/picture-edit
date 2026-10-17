from __future__ import annotations

import torch
import torch.nn as nn

class DetailTokenBinder(nn.Module):
    """Add precomputed person/source, region-type, and anatomical-side bindings."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.region_embedding = nn.Embedding(3, token_dim)
        self.side_embedding = nn.Embedding(3, token_dim)

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_valid: torch.Tensor,
        person_binding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if region_tokens.ndim != 5 or region_tokens.shape[1:4] != (2, 3, 8):
            raise ValueError("region_tokens must have shape [B,2,3,8,D]")
        batch_size, _, _, _, token_dim = region_tokens.shape
        if tuple(region_valid.shape) != (batch_size, 2, 3) or region_valid.dtype != torch.bool:
            raise ValueError("region_valid must have shape [B,2,3] and dtype bool")
        if tuple(person_binding.shape) != (batch_size, 2, token_dim):
            raise ValueError("person_binding must have shape [B,2,D]")
        if (
            not person_binding.is_floating_point()
            or person_binding.device != region_tokens.device
            or person_binding.dtype != region_tokens.dtype
        ):
            raise ValueError(
                "person_binding must match region token dtype and device"
            )
        region_ids = torch.arange(3, device=region_tokens.device)
        region_binding = self.region_embedding(region_ids)
        side_ids = torch.tensor([0, 1, 2], device=region_tokens.device)
        side_binding = self.side_embedding(side_ids)
        binding = (
            person_binding[:, :, None, None]
            + region_binding[None, None, :, None]
            + side_binding[None, None, :, None]
        )
        tokens = region_tokens + binding
        mask = region_valid[..., None].expand(-1, -1, -1, 8)
        tokens = tokens * mask[..., None].to(tokens.dtype)
        return tokens.reshape(batch_size, 48, token_dim), mask.reshape(batch_size, 48)
