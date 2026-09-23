"""Optional HandRefiner interchange only; never executed in default generation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .conditions import HandFineCondition


def hand_refiner_payload(image: torch.Tensor, condition: HandFineCondition) -> dict[str, torch.Tensor]:
    """Return original RGB, per-hand mask/depth/ROI for an offline repair tool."""
    condition.validate()
    if image.ndim != 4 or image.shape[:2] != (condition.batch_size, 3):
        raise ValueError("image must be [B,3,H,W]")
    b, _, h, w = image.shape
    y = (torch.arange(h, device=image.device, dtype=image.dtype) + .5) / h
    x = (torch.arange(w, device=image.device, dtype=image.dtype) + .5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    boxes = condition.target_boxes.to(image.device, image.dtype)
    mask = ((xx >= boxes[..., 0, None, None]) & (xx <= boxes[..., 2, None, None]) &
            (yy >= boxes[..., 1, None, None]) & (yy <= boxes[..., 3, None, None]))
    mask = mask & condition.region_valid.to(image.device)[..., None, None]
    depth = F.interpolate(condition.depth.reshape(b * 4, 1, 16, 16).to(image.device, image.dtype), size=(h, w), mode="bilinear", align_corners=False).reshape(b, 2, 2, h, w)
    return {"image": image, "hand_masks": mask.to(image.dtype), "depth": depth * mask, "boxes": boxes, "valid": condition.region_valid}
