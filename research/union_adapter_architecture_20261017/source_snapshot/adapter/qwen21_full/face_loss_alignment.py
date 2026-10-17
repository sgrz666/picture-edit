"""Differentiable ArcFace alignment for offline face losses.

Landmarks are fixed supervision metadata, never predictions or generation
conditions. Build one grid from the Target's five points and reuse it for both
the decoded prediction and supervision image. The prediction image keeps its
gradient; estimating the similarity transform deliberately does not.

Inputs and outputs are RGB in [0, 1]. ArcFace's pinned TorchScript evaluator
expects the caller to convert the aligned crop to [-1, 1]. This module does not
change the independent detector/ONNX evaluation path.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _image_hw(image_hw: Sequence[int]) -> tuple[int, int]:
    if len(image_hw) != 2:
        raise ValueError("image_hw must contain (height, width)")
    height, width = image_hw
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           or value <= 1 for value in (height, width)):
        raise ValueError("image height and width must be integers greater than one")
    return int(height), int(width)


def build_arcface_alignment_grid(
    landmarks5: torch.Tensor,
    image_hw: Sequence[int],
    *,
    output_size: int = 112,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a detached [B,S,S,2] sampling grid for Target five-point alignment.

    ``landmarks5`` is [B,5,2] in pixels of the image being sampled, ordered as
    InsightFace's two eyes, nose, and two mouth corners. Scale original-image
    metadata to the actual decoded image size before calling. Coordinates must
    be finite, inside the image, distinct, and non-collinear. Invalid metadata
    raises ValueError; callers should exclude that face from auxiliary losses.

    This uses InsightFace's existing ArcFace template/SimilarityTransform and
    the inverse map required by grid_sample. OpenCV's interpolation and uint8
    rounding differ from floating-point PyTorch sampling, so results have the
    same alignment geometry without claiming bitwise OpenCV equivalence.
    """
    height, width = _image_hw(image_hw)
    if not torch.is_tensor(landmarks5) or landmarks5.ndim != 3 or landmarks5.shape[1:] != (5, 2):
        raise ValueError("landmarks5 must be a tensor with shape [B,5,2]")
    if landmarks5.shape[0] < 1 or not landmarks5.is_floating_point():
        raise ValueError("landmarks5 must contain at least one floating-point face")
    if (isinstance(output_size, bool) or not isinstance(output_size, int)
            or output_size <= 0 or (output_size % 112 and output_size % 128)):
        raise ValueError("output_size must be a positive multiple of 112 or 128")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("alignment grid dtype must be float32 or float64")

    from insightface.utils.face_align import estimate_norm

    points = landmarks5.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(points).all():
        raise ValueError("landmarks5 contains non-finite coordinates")
    if (points[..., 0].min() < 0 or points[..., 0].max() > width - 1
            or points[..., 1].min() < 0 or points[..., 1].max() > height - 1):
        raise ValueError("landmarks5 coordinates lie outside the sampled image")

    inverse_maps = []
    for index, face in enumerate(points):
        centered = face - face.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        pairwise = np.linalg.norm(face[:, None] - face[None], axis=-1)
        np.fill_diagonal(pairwise, np.inf)
        if (pairwise.min() <= 1e-4 or singular_values[0] <= 1e-4
                or singular_values[1] <= singular_values[0] * 1e-5):
            raise ValueError(f"landmarks5 face {index} is degenerate")
        # estimate_norm maps original pixels to the canonical ArcFace image.
        forward = np.asarray(estimate_norm(face, output_size), dtype=np.float64)
        affine = np.eye(3, dtype=np.float64)
        affine[:2] = forward
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:2, :2])) <= 1e-12:
            raise ValueError(f"landmarks5 face {index} has no invertible alignment")
        inverse_maps.append(np.linalg.inv(affine)[:2])

    target_device = landmarks5.device if device is None else torch.device(device)
    inverse = torch.as_tensor(np.stack(inverse_maps), device=target_device, dtype=dtype)
    axis = torch.arange(output_size, device=target_device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    output_pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    source_pixels = torch.einsum("hwc,bkc->bhwk", output_pixels, inverse)
    scale = source_pixels.new_tensor([width, height])
    # Pixel centers with align_corners=False. The +.5 is essential: using x/W
    # shifts every aligned sample by half an original-image pixel.
    return ((source_pixels + .5) / scale * 2 - 1).detach()


def align_face_rgb(
    images: torch.Tensor,
    landmarks5: torch.Tensor | None = None,
    *,
    grid: torch.Tensor | None = None,
    output_size: int = 112,
) -> torch.Tensor:
    """Align [B,3,H,W] [0,1] RGB without detaching image gradients.

    Supply either fixed ``landmarks5`` or an already built ``grid``. Reusing a
    grid for prediction and Target guarantees identical geometry. Half/BF16
    inputs are sampled in FP32 for stable auxiliary losses and CPU support.
    No face detection, clipping, channel swapping, or identity extraction is
    performed here.
    """
    if (not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3
            or images.shape[0] < 1 or not images.is_floating_point()):
        raise ValueError("images must be a floating-point tensor with shape [B,3,H,W]")
    height, width = _image_hw(images.shape[-2:])
    detached = images.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("images contain non-finite RGB values")
    if bool((detached < 0).any()) or bool((detached > 1).any()):
        raise ValueError("images must be RGB in [0,1]")
    if (landmarks5 is None) == (grid is None):
        raise ValueError("supply exactly one of landmarks5 or a fixed alignment grid")

    sample_dtype = torch.float64 if images.dtype == torch.float64 else torch.float32
    if grid is None:
        if not torch.is_tensor(landmarks5) or landmarks5.ndim != 3 or landmarks5.shape[1:] != (5, 2):
            raise ValueError("landmarks5 must be a tensor with shape [B,5,2]")
        if landmarks5.shape[0] != images.shape[0]:
            raise ValueError("landmarks5 and images batch sizes differ")
        grid = build_arcface_alignment_grid(
            landmarks5, (height, width), output_size=output_size,
            device=images.device, dtype=sample_dtype,
        )
    elif (not torch.is_tensor(grid) or not grid.is_floating_point()
          or grid.shape != (images.shape[0], output_size, output_size, 2)):
        raise ValueError("alignment grid must have shape [B,output_size,output_size,2]")
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("alignment grid contains non-finite coordinates")
    return F.grid_sample(
        images.to(dtype=sample_dtype), grid.detach().to(device=images.device, dtype=sample_dtype),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )


__all__ = ["build_arcface_alignment_grid", "align_face_rgb"]
