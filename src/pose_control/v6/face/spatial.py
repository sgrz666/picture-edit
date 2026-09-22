from __future__ import annotations

import torch
import torch.nn as nn


def render_face_landmark_heatmaps(
    landmarks: torch.Tensor,
    face_valid: torch.Tensor | None = None,
    spatial_size: tuple[int, int] = (128, 128),
    *,
    sigma: float = 2.0,
    feather_sigma: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render confidence-weighted landmark Gaussians and a soft union mask."""

    if (
        landmarks.ndim < 3
        or landmarks.shape[-2] != 72
        or landmarks.shape[-1] not in (2, 3)
    ):
        raise ValueError("landmarks must have shape [...,72,2|3]")
    if not landmarks.is_floating_point() or not torch.isfinite(landmarks).all():
        raise ValueError("landmarks must contain finite floating values")
    if len(spatial_size) != 2 or min(spatial_size) <= 0:
        raise ValueError("spatial_size must contain two positive dimensions")
    if sigma <= 0 or feather_sigma <= 0:
        raise ValueError("Gaussian sigmas must be positive")
    leading = landmarks.shape[:-2]
    if face_valid is None:
        face_valid = torch.ones(*leading, dtype=torch.bool, device=landmarks.device)
    elif tuple(face_valid.shape) != tuple(leading) or face_valid.dtype != torch.bool:
        raise ValueError(f"face_valid must have shape {tuple(leading)} and dtype bool")
    if face_valid.device != landmarks.device:
        raise ValueError("face_valid must be on the landmark device")

    height, width = spatial_size
    y = torch.arange(height, device=landmarks.device, dtype=landmarks.dtype)
    x = torch.arange(width, device=landmarks.device, dtype=landmarks.dtype)
    y = y.reshape((1,) * len(leading) + (1, height, 1))
    x = x.reshape((1,) * len(leading) + (1, 1, width))
    points = landmarks[..., :2].clamp(0, 1)
    point_x = (points[..., 0] * (width - 1))[..., None, None]
    point_y = (points[..., 1] * (height - 1))[..., None, None]
    squared_distance = (x - point_x).square() + (y - point_y).square()
    if landmarks.shape[-1] == 3:
        confidence = landmarks[..., 2].clamp(0, 1)
    else:
        confidence = torch.ones_like(landmarks[..., 0])
    confidence = confidence[..., None, None]
    valid_scale = face_valid[..., None, None, None].to(landmarks.dtype)
    heatmaps = (
        torch.exp(-squared_distance / (2 * sigma * sigma))
        * confidence
        * valid_scale
    )
    feather_points = (
        torch.exp(-squared_distance / (2 * feather_sigma * feather_sigma))
        * confidence
    )
    feather_mask = feather_points.amax(dim=-3)
    feather_mask = feather_mask * face_valid[..., None, None].to(
        feather_mask.dtype
    )
    return heatmaps, feather_mask.clamp(0, 1)


# Descriptive compatibility aliases for callers that prefer transformation names.
face_landmarks_to_heatmaps = render_face_landmark_heatmaps
build_face_heatmaps_and_mask = render_face_landmark_heatmaps


class FaceSpatialEncoder(nn.Module):
    """MCLD PoseGuider-style 2D stem for 72 heatmaps plus a feather mask."""

    output_channels = 512

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(73, 16, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.output = nn.Conv2d(128, self.output_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(
        self,
        heatmaps: torch.Tensor,
        feather_mask: torch.Tensor,
        face_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if heatmaps.ndim < 4 or heatmaps.shape[-3] != 72:
            raise ValueError("heatmaps must have shape [...,72,H,W]")
        leading = heatmaps.shape[:-3]
        expected_mask = (*leading, *heatmaps.shape[-2:])
        if tuple(feather_mask.shape) != expected_mask:
            raise ValueError(f"feather_mask must have shape {expected_mask}")
        if (
            not heatmaps.is_floating_point()
            or feather_mask.dtype != heatmaps.dtype
            or feather_mask.device != heatmaps.device
        ):
            raise ValueError("heatmaps and feather_mask must share float dtype/device")
        if face_valid is None:
            face_valid = torch.ones(*leading, dtype=torch.bool, device=heatmaps.device)
        elif tuple(face_valid.shape) != tuple(leading) or face_valid.dtype != torch.bool:
            raise ValueError(f"face_valid must have shape {tuple(leading)} and dtype bool")
        if face_valid.device != heatmaps.device:
            raise ValueError("face_valid must be on the heatmap device")
        height, width = heatmaps.shape[-2:]
        hidden_states = torch.cat((heatmaps, feather_mask[..., None, :, :]), dim=-3)
        hidden_states = hidden_states.reshape(-1, 73, height, width)
        hidden_states = self.activation(self.stem(hidden_states))
        for block in self.blocks:
            hidden_states = self.activation(block(hidden_states))
        hidden_states = self.output(hidden_states)
        hidden_states = hidden_states.reshape(
            *leading,
            self.output_channels,
            hidden_states.shape[-2],
            hidden_states.shape[-1],
        )
        return hidden_states * face_valid[..., None, None, None].to(
            hidden_states.dtype
        )
