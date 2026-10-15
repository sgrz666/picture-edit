"""A dependency-free single-frame adaptation of CHAMP's GuidanceEncoder.

Upstream: https://github.com/fudan-generative-vision/champ
Commit: 4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140
Original file: models/guidance_encoder.py

Changes from upstream are intentionally narrow and documented: inflated 3D
convolutions become 2D convolutions for F=1, the video Transformer dependency
is replaced by a residual spatial mixer, and the final projection is not
zero-initialized because this project does not load CHAMP checkpoints.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return next(group for group in (32, 16, 8, 4, 2, 1) if channels % group == 0)


class _SpatialMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class ChampGuidanceEncoder(nn.Module):
    """CHAMP channel/downsample schedule adapted to a single image tensor."""

    def __init__(
        self,
        guidance_embedding_channels: int = 128,
        guidance_input_channels: int = 3,
        block_out_channels: Sequence[int] = (16, 32, 96, 256),
    ) -> None:
        super().__init__()
        if len(block_out_channels) != 4:
            raise ValueError("CHAMP guidance adaptation expects four channel levels")
        self.conv_in = nn.Conv2d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        blocks = []
        for channel_in, channel_out in zip(block_out_channels[:-1], block_out_channels[1:]):
            blocks.extend(
                (
                    nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
                    nn.Conv2d(
                        channel_in,
                        channel_out,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                    ),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.spatial_mixer = _SpatialMixer(block_out_channels[-1])
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], guidance_embedding_channels, kernel_size=3, padding=1
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        embedding = F.silu(self.conv_in(condition))
        for block in self.blocks:
            embedding = F.silu(block(embedding))
        return self.conv_out(self.spatial_mixer(embedding))
