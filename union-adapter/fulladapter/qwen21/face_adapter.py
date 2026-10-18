"""Source-identity and target-ROI face residual branch for Qwen-Image-2.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


FACE_LAYERS = (3, 8, 13, 18, 23, 28)


@dataclass
class Qwen21FaceState:
    code: torch.Tensor             # [B, 512]
    mask: torch.Tensor             # [B, H*W, 1], exact zero outside target face ROI
    position: torch.Tensor         # [B, H*W, 512]
    valid: torch.Tensor            # [B]


class Qwen21FaceAdapter(nn.Module):
    """Add source-face identity/texture only inside the target head ROI.

    The source crop and ArcFace embedding belong to A. The target box is
    projected geometry; no target RGB or target identity enters this module.
    """

    def __init__(self, *, width: int = 512, hidden_size: int = 4096) -> None:
        super().__init__()
        self.identity_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width))
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256 * 4 * 4, width),
        )
        self.position_projection = nn.Linear(4, width)
        self.hidden_projection = nn.Linear(hidden_size, width, bias=False)
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.layer_embedding = nn.Embedding(len(FACE_LAYERS), width)
        self.fusion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU())
        self.zero_heads = nn.ModuleDict({str(index): nn.Linear(width, hidden_size) for index in FACE_LAYERS})
        for head in self.zero_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.hidden_size = hidden_size

    def prepare(
        self, embedding: torch.Tensor, crop: torch.Tensor, target_box: torch.Tensor,
        valid: torch.Tensor, *, token_hw: tuple[int, int] = (32, 32),
    ) -> Qwen21FaceState:
        if embedding.ndim != 2 or embedding.shape[-1] != 512:
            raise ValueError('embedding must have shape [B,512]')
        batch = embedding.shape[0]
        if crop.shape != (batch, 3, 128, 128) or target_box.shape != (batch, 4) or valid.shape != (batch,):
            raise ValueError('invalid face crop, box, or validity shape')
        if valid.dtype != torch.bool:
            raise ValueError('valid must be bool')
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        embedding = embedding.to(device=device, dtype=dtype)
        crop = crop.to(device=device, dtype=dtype) / 127.5 - 1
        box = target_box.to(device=device, dtype=dtype)
        valid = valid.to(device=device)
        code = self.identity_projection(embedding) + self.texture_encoder(crop)
        code = code * valid[:, None]
        height, width = token_hw
        y, x = torch.meshgrid(
            (torch.arange(height, device=device, dtype=dtype) + .5) / height,
            (torch.arange(width, device=device, dtype=dtype) + .5) / width,
            indexing='ij',
        )
        x, y = x.flatten()[None], y.flatten()[None]
        x0, y0, x1, y1 = [box[:, index:index + 1] for index in range(4)]
        shape_valid = (x1 > x0) & (y1 > y0)
        mask = ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & shape_valid & valid[:, None])
        box_width = (x1 - x0).clamp_min(1e-5)
        box_height = (y1 - y0).clamp_min(1e-5)
        local_x = ((x - x0) / box_width).clamp(-1, 2)
        local_y = ((y - y0) / box_height).clamp(-1, 2)
        features = torch.stack((local_x, local_y, x.expand(batch, -1), y.expand(batch, -1)), dim=-1)
        position = self.position_projection(features) * mask[..., None]
        return Qwen21FaceState(code, mask[..., None].to(dtype), position, valid)

    def residual(
        self, layer_index: int, target_hidden: torch.Tensor,
        state: Qwen21FaceState, sigma: torch.Tensor,
    ) -> torch.Tensor:
        if layer_index not in FACE_LAYERS or target_hidden.shape[-1] != self.hidden_size:
            raise ValueError('unexpected layer or hidden width')
        if state.mask.shape[:2] != target_hidden.shape[:2]:
            raise ValueError('face ROI grid does not match target tokens')
        dtype = self.hidden_projection.weight.dtype
        stage = FACE_LAYERS.index(layer_index)
        x = (self.hidden_projection(target_hidden.to(dtype)) + state.code[:, None] + state.position
             + self.time_projection(sigma.to(dtype).reshape(-1, 1))[:, None]
             + self.layer_embedding.weight[stage][None, None])
        return (self.zero_heads[str(layer_index)](self.fusion(x)) * state.mask).to(target_hidden.dtype)
