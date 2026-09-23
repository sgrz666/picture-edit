from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import PreparedHandConditioning


def scatter_hand_roi_residuals(local, masks, boxes, valid, visibility, target_hw, depth=None):
    """Inverse-warp four hand ROIs; depth/visibility weighted overlap never doubles."""
    b, people, sides, h, w, dim = local.shape
    if (people, sides) != (2, 2) or masks.shape != (b, 2, 2, h, w):
        raise ValueError("hand residual/mask shape mismatch")
    th, tw = target_hw
    y = (torch.arange(th, device=local.device, dtype=local.dtype) + .5) / th
    x = (torch.arange(tw, device=local.device, dtype=local.dtype) + .5) / tw
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None, None, None]
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
    sample_grid = ((grid - boxes[..., None, None, :2]) / sizes[..., None, None, :] * 2 - 1).reshape(b * 4, th, tw, 2)
    sample = F.grid_sample(local.permute(0, 1, 2, 5, 3, 4).reshape(b * 4, dim, h, w), sample_grid, align_corners=False)
    sample = sample.reshape(b, 4, dim, th, tw).permute(0, 1, 3, 4, 2)
    weight = F.grid_sample(masks.reshape(b * 4, 1, h, w), sample_grid, align_corners=False).reshape(b, 4, th, tw)
    weight = weight * (valid * visibility).reshape(b, 4, 1, 1)
    maximum = weight.amax(1)
    if depth is not None:
        sampled_depth = F.grid_sample(depth.reshape(b * 4, 1, h, w), sample_grid, align_corners=False).reshape(b, 4, th, tw)
        # Depth changes relative ownership only; a single hand retains its mask strength.
        blend_weight = weight * torch.exp(-4 * sampled_depth.clamp(-1, 1))
    else:
        blend_weight = weight
    blended = (sample * blend_weight[..., None]).sum(1) / blend_weight.sum(1).clamp_min(1e-6)[..., None]
    return (blended * maximum[..., None]).reshape(b, th * tw, dim)


class IndependentKVAttention(nn.Module):
    """Dynamic SDPA equivalent of VTON's separate IP K/V path."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k_hand = nn.Linear(dim, dim, bias=False)
        self.to_v_hand = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, query, appearance):
        b, n, dim = query.shape
        q = self.to_q(query).reshape(b, n, self.heads, -1).transpose(1, 2)
        k = self.to_k_hand(appearance).reshape(b, -1, self.heads, dim // self.heads).transpose(1, 2)
        v = self.to_v_hand(appearance).reshape(b, -1, self.heads, dim // self.heads).transpose(1, 2)
        update = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(update.transpose(1, 2).reshape(b, n, dim))


class HandFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = IndependentKVAttention(dim, heads)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, geometry, appearance, mask, texture_scale):
        update = self.attention(self.query_norm(geometry), self.context_norm(appearance))
        geometry = geometry + update * mask * texture_scale
        return geometry + self.ffn(geometry)


class _StageAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.silu(self.down(self.norm(x))))


class HandControlAdapter(nn.Module):
    """Two geometry-Q/appearance-KV fusions, then six DeepGen residual groups."""

    def __init__(self, *, dim: int = 512, deepgen_dim: int = 1536, rank: int = 64, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.deepgen_dim = deepgen_dim
        self.fusion = nn.ModuleList(HandFusionBlock(dim, heads) for _ in range(2))
        self.self_norm = nn.LayerNorm(dim)
        self.shared_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.shared_ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.stage_embedding = nn.Parameter(torch.randn(6, dim) * .02)
        self.stage_adapters = nn.ModuleList(_StageAdapter(dim, rank) for _ in range(6))
        self.timestep_films = nn.ModuleList(nn.Sequential(nn.SiLU(), nn.Linear(1, dim * 2)) for _ in range(6))
        self.film_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(6))
        self.zero_heads = nn.ModuleList(nn.Linear(dim, deepgen_dim) for _ in range(6))
        for head in self.zero_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, prepared: PreparedHandConditioning, *, target_token_hw: tuple[int, int], timestep: torch.Tensor, texture_scale=1.0, hand_active=None):
        prepared.validate()
        b = prepared.batch_size
        if timestep.ndim == 0:
            timestep = timestep.expand(b)
        if timestep.shape != (b,):
            raise ValueError("timestep must have shape [B]")
        valid = prepared.hand_valid
        if hand_active is not None:
            if hand_active.shape != (b,) or hand_active.dtype != torch.bool:
                raise ValueError("hand_active must be bool [B]")
            valid = valid & hand_active[:, None, None]
        if not torch.is_tensor(texture_scale):
            texture_scale = prepared.spatial_features.new_full((b,), float(texture_scale))
        elif texture_scale.ndim == 0:
            texture_scale = texture_scale.expand(b)
        if texture_scale.shape != (b,):
            raise ValueError("texture_scale must have shape [B]")
        hidden = prepared.spatial_features.reshape(b * 4, 256, self.dim)
        geometry = prepared.geometry_tokens.reshape(b * 4, 4, self.dim)
        appearance = prepared.appearance_tokens.reshape(b * 4, 8, self.dim)
        hidden = hidden + geometry.mean(1, keepdim=True)
        mask = prepared.hand_masks.reshape(b * 4, 256, 1) * valid.reshape(b * 4, 1, 1)
        texture = texture_scale.repeat_interleave(4).reshape(b * 4, 1, 1)
        for layer in self.fusion:
            hidden = layer(hidden, appearance, mask, texture)
            hidden = hidden * valid.reshape(b * 4, 1, 1)
        hand_t = timestep.to(hidden.dtype).repeat_interleave(4).reshape(b * 4, 1) / 1000
        result = []
        for stage in range(6):
            hidden = hidden + self.stage_embedding[stage]
            scale, shift = self.timestep_films[stage](hand_t).chunk(2, -1)
            hidden = hidden + self.film_norms[stage](hidden) * (1 + scale[:, None]) + shift[:, None]
            normalized = self.self_norm(hidden)
            update, _ = self.shared_attention(normalized, normalized, normalized, need_weights=False)
            hidden = hidden + update
            hidden = hidden + self.shared_ffn(hidden) + self.stage_adapters[stage](hidden)
            hidden = hidden * valid.reshape(b * 4, 1, 1)
            local = self.zero_heads[stage](hidden).reshape(b, 2, 2, 16, 16, self.deepgen_dim)
            local = local * valid[..., None, None, None]
            result.append(scatter_hand_roi_residuals(local, prepared.hand_masks, prepared.target_boxes, valid, prepared.visibility, target_token_hw, prepared.depth))
        return tuple(result)
