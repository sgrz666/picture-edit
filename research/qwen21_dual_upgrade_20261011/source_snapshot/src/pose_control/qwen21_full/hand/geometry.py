from __future__ import annotations

import torch


def split_iper_134_hand_points(points: torch.Tensor) -> torch.Tensor:
    """iPER/SMPL-X whole-body layout: left 92:113, right 113:134."""
    if points.shape[-2:] != (134, 3):
        raise ValueError("iPER whole-body keypoints must end in [134,3]")
    return torch.stack((points[..., 92:113, :], points[..., 113:134, :]), dim=-3)


def axis_angle45_to_6d96(pose: torch.Tensor) -> torch.Tensor:
    """VTON's 16 rotations: canonical wrist followed by 15 SMPL-X finger joints."""
    if pose.shape[-1] != 45:
        raise ValueError("hand pose must have 45 axis-angle values")
    angle = pose.float().reshape(*pose.shape[:-1], 15, 3)
    theta = torch.linalg.vector_norm(angle, dim=-1, keepdim=True)
    axis = angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), -1).reshape(*x.shape, 3, 3)
    eye = torch.eye(3, device=pose.device, dtype=torch.float32)
    rotation = eye + torch.sin(theta)[..., None] * skew + (1 - torch.cos(theta))[..., None] * (skew @ skew)
    # First two rotation columns, contiguous x/y vectors.
    six = rotation[..., :, :2].transpose(-1, -2).reshape(*pose.shape[:-1], 15, 6)
    wrist = pose.new_tensor([1, 0, 0, 0, 1, 0]).expand(*pose.shape[:-1], 1, 6)
    return torch.cat((wrist, six.to(pose.dtype)), dim=-2).flatten(-2)


def local_hand_points(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if points.shape[-2:] != (21, 3) or boxes.shape[-1] != 4 or points.shape[:-2] != boxes.shape[:-1]:
        raise ValueError("points/boxes must have matching hand leading dimensions")
    span = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
    local = (points[..., :2] - boxes[..., None, :2]) / span[..., None, :]
    return torch.cat((local, points[..., 2:]), -1)


def hand_heatmaps(points: torch.Tensor, *, size: int = 16, sigma: float = 1.0) -> torch.Tensor:
    """FoundHand Gaussian keypoint raster, confidence gated; invalid points are zero."""
    if points.shape[-2:] != (21, 3) or size <= 0 or sigma <= 0:
        raise ValueError("expected 21 xy/confidence points, positive size/sigma")
    coord = (torch.arange(size, device=points.device, dtype=points.dtype) + .5) / size
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    delta = (xx - points[..., 0, None, None]).square() + (yy - points[..., 1, None, None]).square()
    gauss = torch.exp(-delta / (2 * (sigma / size) ** 2))
    in_bounds = ((points[..., :2] >= 0) & (points[..., :2] <= 1)).all(-1)
    return gauss * points[..., 2, None, None].clamp(0, 1) * in_bounds[..., None, None]
