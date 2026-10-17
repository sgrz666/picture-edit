"""Offline-only SMPL-X hand quality gate and conservative refinement merge."""
from __future__ import annotations

import torch


def hand_quality_gate(keypoints: torch.Tensor, boxes: torch.Tensor, *, min_side: float = .035, min_confidence: float = .35) -> torch.Tensor:
    """True means request 2D refinement; does not imply target RGB at inference."""
    if keypoints.shape[-2:] != (21, 3) or boxes.shape[-1] != 4:
        raise ValueError("expected hand keypoints [...,21,3] and boxes [...,4]")
    size = boxes[..., 2:] - boxes[..., :2]
    confidence = keypoints[..., 2].mean(-1)
    outside = ((keypoints[..., :2] < 0) | (keypoints[..., :2] > 1)).any(-1).float().mean(-1)
    return (size.amin(-1) < min_side) | (confidence < min_confidence) | (outside > .2)


def merge_vitpose_keypoints(smplx: torch.Tensor, vitpose: torch.Tensor, refine: torch.Tensor, *, min_confidence: float = .25) -> torch.Tensor:
    """Use a confident ViTPose result for flagged samples; preserve SMPL-X otherwise."""
    if vitpose.shape != smplx.shape or smplx.shape[-2:] != (21, 3) or refine.shape != smplx.shape[:-2]:
        raise ValueError("ViTPose refinement shapes must match 21-point SMPL-X hands")
    accept = refine & (vitpose[..., 2].mean(-1) >= min_confidence)
    return torch.where(accept[..., None, None], vitpose, smplx)
