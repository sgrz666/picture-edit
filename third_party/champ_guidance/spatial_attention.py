"""CHAMP's active spatial Transformer, adapted to a single frame and PyTorch SDPA.

MIT source: fudan-generative-vision/champ @ 4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140,
models/transformer_3d.py and TemporalBasicTransformerBlock in models/attention.py.
The license is in this directory. Temporal/cross attention and unused blocks are
omitted. State names match the active upstream Transformer. An optional depth
support mask excludes invalid keys; no mask reproduces the upstream spatial math.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.Sequential(nn.Linear(width, width), nn.Dropout(0.))

    def forward(self, value, key_valid=None):
        b, n, c = value.shape
        q, k, v = (layer(value).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
                   for layer in (self.to_q, self.to_k, self.to_v))
        mask = None if key_valid is None else key_valid[:, None, None, :]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.to_out(output.transpose(1, 2).reshape(b, n, c))


class _GEGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width * 8)

    def forward(self, value):
        hidden, gate = self.proj(value).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(_GEGLU(width), nn.Dropout(0.), nn.Linear(width * 4, width))

    def forward(self, value):
        return self.net(value)


class _SpatialBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn1 = _Attention(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.ff = _FeedForward(width)
        self.norm3 = nn.LayerNorm(width)

    def forward(self, value, key_valid=None):
        value = value + self.attn1(self.norm1(value), key_valid)
        return value + self.ff(self.norm3(value))


class ChampSpatialAttention(nn.Module):
    def __init__(self, channels=256, heads=8, norm_num_groups=32):
        super().__init__()
        if channels % heads or channels % norm_num_groups:
            raise ValueError("CHAMP channels must divide heads and GroupNorm groups")
        self.norm = nn.GroupNorm(norm_num_groups, channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([_SpatialBlock(channels, heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, value, key_valid=None):
        b, c, h, w = value.shape
        if key_valid is not None:
            if key_valid.dtype != torch.bool or key_valid.shape != (b, h*w):
                raise ValueError("Spatial depth support must be bool [B,H*W]")
            if not bool(key_valid.any(1).all()):
                raise ValueError("Skip all-missing scenes before CHAMP attention")
        tokens = self.proj_in(self.norm(value)).flatten(2).transpose(1, 2)
        for block in self.transformer_blocks:
            tokens = block(tokens, key_valid)
        return value + self.proj_out(tokens.transpose(1, 2).reshape(b, c, h, w))
