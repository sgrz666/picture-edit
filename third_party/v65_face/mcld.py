"""2D adaptations of the pinned MCLD face preprocessing and PoseGuider code.

Upstream files ``src/models/pose_guider.py`` and
``preprocess_data/face_interface.py`` at commit
``6279ff7c601c11dff5e075410ed34df2973da088``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class FacePoseGuider2D(nn.Module):
    """Single-frame 2D form of MCLD's inflated-convolution PoseGuider."""

    output_channels: int

    def __init__(
        self,
        conditioning_embedding_channels: int = 512,
        conditioning_channels: int = 73,
        block_out_channels: Sequence[int] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()
        if len(block_out_channels) < 2 or min(block_out_channels) <= 0:
            raise ValueError("block_out_channels must contain positive channel sizes")
        self.output_channels = conditioning_embedding_channels
        self.stem = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList()
        for channel_in, channel_out in zip(
            block_out_channels[:-1], block_out_channels[1:]
        ):
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in,
                    channel_out,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
        # V6.5 deliberately does not carry over MCLD's zero_module here; the
        # downstream residual head is the sole zero-initialized boundary.
        self.output = nn.Conv2d(
            block_out_channels[-1],
            conditioning_embedding_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.SiLU()

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = self.activation(self.stem(conditioning))
        for block in self.blocks:
            embedding = self.activation(block(embedding))
        return self.output(embedding)


def expand_image_region(
    box: Sequence[float] | np.ndarray,
    scale_factor: float = 1.5,
) -> np.ndarray:
    """MCLD square-region expansion, retaining subpixel coordinates."""

    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    height = (y2 - y1) * scale_factor
    width = (x2 - x1) * scale_factor
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(height, width)
    return np.asarray(
        [
            center_x - side / 2,
            center_y - side / 2,
            center_x + side / 2,
            center_y + side / 2,
        ],
        dtype=np.float64,
    )


def safe_crop_image(
    image: np.ndarray,
    box: Sequence[int] | np.ndarray,
    *,
    pad_value: int = 0,
) -> np.ndarray:
    """MCLD safe crop with explicit padding for out-of-bounds regions."""

    array = np.asarray(image)
    x1, y1, x2, y2 = np.asarray(box, dtype=np.int64)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area")
    height, width = array.shape[:2]
    crop = np.full((y2 - y1, x2 - x1, 3), pad_value, dtype=array.dtype)
    src_x1, src_y1 = max(x1, 0), max(y1, 0)
    src_x2, src_y2 = min(x2, width), min(y2, height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
        crop[
            dst_y1 : dst_y1 + (src_y2 - src_y1),
            dst_x1 : dst_x1 + (src_x2 - src_x1),
        ] = array[src_y1:src_y2, src_x1:src_x2]
    return crop


def safe_square_crop_normalized(
    image: np.ndarray,
    box: torch.Tensor | np.ndarray | Sequence[float],
    *,
    scale: float = 1.0,
    pad_value: int = 0,
) -> np.ndarray:
    """Adapt MCLD expansion/safe padding to normalized xyxy coordinates."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape [H,W,3]")
    if scale <= 0:
        raise ValueError("scale must be positive")
    coords = np.asarray(torch.as_tensor(box, dtype=torch.float32).cpu(), dtype=np.float64)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("box must be a finite xyxy vector")
    height, width = array.shape[:2]
    pixel_box = coords * np.asarray([width, height, width, height], dtype=np.float64)
    expanded = expand_image_region(pixel_box, scale)
    integer_box = np.asarray(
        [
            np.floor(expanded[0]),
            np.floor(expanded[1]),
            np.ceil(expanded[2]),
            np.ceil(expanded[3]),
        ],
        dtype=np.int64,
    )
    # Floating-point rounding can differ by one pixel. Match the production
    # cache contract by forcing a square before padding.
    side = max(integer_box[2] - integer_box[0], integer_box[3] - integer_box[1], 1)
    integer_box[2] = integer_box[0] + side
    integer_box[3] = integer_box[1] + side
    return safe_crop_image(array, integer_box, pad_value=pad_value)


__all__ = [
    "FacePoseGuider2D",
    "expand_image_region",
    "safe_crop_image",
    "safe_square_crop_normalized",
]
