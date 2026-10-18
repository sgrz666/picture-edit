"""Shape-adapted VTON-HandFit Handprocessor and Perceiver Resampler primitives.

Upstream: VTON-HandFit/VTON-HandFit e69aaacc8285c2bd248ff9b1b6d8e9b92b1f8b5a
HandFit/pose_guider.py and HandFit/ip_adapter/resampler.py. The projection
topology and latent-query attention are retained; dimensions are exposed for
DeepGen's 512-wide hand stream. No UNet or CUDA-specific operation is imported.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedderLN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.linear(x)


class OneLinearLN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x):
        return self.linear(x)


class HandStructureProjector(nn.Module):
    """VTON's separate BPS/6D/side/2D projectors with an explicit null BPS."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.mano_bps = FeatureEmbedderLN(1024, max(dim, 64), dim)
        self.mano_6d = OneLinearLN(96, dim)
        self.handtype = OneLinearLN(1, dim)
        self.mano_2d = FeatureEmbedderLN(42, max(dim, 64), dim)
        self.combined = OneLinearLN(dim * 3, dim)
        self.null_bps = nn.Parameter(torch.zeros(dim))

    def forward(self, bps, pose6d, points42, side, mesh_valid):
        bps_features = self.mano_bps(bps)
        bps_features = torch.where(mesh_valid[..., None], bps_features, self.null_bps)
        volume = self.combined(torch.cat((bps_features, self.mano_6d(pose6d), self.handtype(side)), -1))
        return volume, self.mano_2d(points42)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.heads = heads
        self.dim_head = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, latents):
        b, n, d = latents.shape
        q = self.to_q(self.norm2(latents)).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        kv = self.to_kv(torch.cat((self.norm1(x), self.norm2(latents)), dim=1))
        k, v = (tensor.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2) for tensor in kv.chunk(2, -1))
        scale = self.dim_head ** -.25
        weights = torch.softmax(((q * scale) @ (k * scale).transpose(-1, -2)).float(), dim=-1).to(q.dtype)
        return self.to_out((weights @ v).transpose(1, 2).reshape(b, n, d))


class HandResampler(nn.Module):
    def __init__(self, *, dim: int = 512, depth: int = 2, heads: int = 8, num_queries: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** .5)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim, heads),
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        latents = self.latents.expand(x.shape[0], -1, -1)
        x = self.proj_in(x)
        for attention, feed_forward in self.layers:
            latents = latents + attention(x, latents)
            latents = latents + feed_forward(latents)
        return self.norm_out(self.proj_out(latents))
