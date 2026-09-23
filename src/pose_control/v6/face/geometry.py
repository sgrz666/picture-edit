from __future__ import annotations

import torch
import torch.nn as nn


def _validate_keypoint_tail(
    keypoints: torch.Tensor,
    *,
    count: int,
    name: str,
) -> None:
    if not keypoints.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if keypoints.ndim < 2 or keypoints.shape[-2] != count:
        raise ValueError(f"{name} must end with {count} keypoints")
    if keypoints.shape[-1] not in (2, 3):
        raise ValueError(f"{name} coordinates must have size 2 or 3")
    if not torch.isfinite(keypoints).all():
        raise ValueError(f"{name} must contain only finite values")


def smplx137_to_face72(keypoints: torch.Tensor) -> torch.Tensor:
    """Select the canonical 72 SMPL-X face landmarks (indices 65..136)."""

    _validate_keypoint_tail(keypoints, count=137, name="SMPL-X keypoints")
    return keypoints[..., 65:137, :]


def dwpose68_to_face72(keypoints: torch.Tensor) -> torch.Tensor:
    """Map DWPose's 68 face landmarks into the SMPL-X-aligned 72 slots."""

    _validate_keypoint_tail(keypoints, count=68, name="DWPose face keypoints")
    result = keypoints.new_zeros(*keypoints.shape[:-2], 72, keypoints.shape[-1])
    result[..., 4:, :] = keypoints
    return result


def normalize_face_landmarks_to_roi(
    landmarks: torch.Tensor,
    boxes: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize only landmark ``xy`` into its bounding-box coordinate frame."""

    _validate_keypoint_tail(landmarks, count=72, name="face landmarks")
    expected_boxes = (*landmarks.shape[:-2], 4)
    if tuple(boxes.shape) != expected_boxes:
        raise ValueError(f"boxes must have shape {expected_boxes}")
    if not boxes.is_floating_point() or boxes.dtype != landmarks.dtype:
        raise ValueError("boxes and landmarks must use the same floating dtype")
    if boxes.device != landmarks.device:
        raise ValueError("boxes and landmarks must be on the same device")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain only finite values")
    size = boxes[..., 2:] - boxes[..., :2]
    if torch.any(size <= 0):
        raise ValueError("boxes must contain positive-area xyxy coordinates")
    normalized = landmarks.clone()
    normalized[..., :2] = (
        (landmarks[..., :2] - boxes[..., None, :2])
        / size.clamp_min(eps)[..., None, :]
    ).clamp(0, 1)
    return normalized


class FaceGeometryTokenizer(nn.Module):
    """Compress 72 indexed face landmarks plus SMPL-X face parameters."""

    def __init__(
        self,
        *,
        dim: int = 512,
        token_count: int = 4,
        index_dim: int = 32,
        heads: int = 8,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if dim <= 0 or token_count <= 0 or index_dim <= 0:
            raise ValueError("tokenizer dimensions must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.token_count = token_count
        self.landmark_index = nn.Embedding(72, index_dim)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(3 + index_dim),
            nn.Linear(3 + index_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.parameter_encoder = nn.Sequential(
            nn.LayerNorm(13),
            nn.Linear(13, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.queries = nn.Parameter(torch.empty(token_count, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden_dim = dim * ff_mult
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        landmarks: torch.Tensor,
        jaw_pose: torch.Tensor,
        expression: torch.Tensor,
        face_valid: torch.Tensor,
    ) -> torch.Tensor:
        _validate_keypoint_tail(landmarks, count=72, name="face landmarks")
        leading = landmarks.shape[:-2]
        if tuple(jaw_pose.shape) != (*leading, 3):
            raise ValueError(f"jaw_pose must have shape {(*leading, 3)}")
        if tuple(expression.shape) != (*leading, 10):
            raise ValueError(f"expression must have shape {(*leading, 10)}")
        if tuple(face_valid.shape) != tuple(leading) or face_valid.dtype != torch.bool:
            raise ValueError(f"face_valid must have shape {tuple(leading)} and dtype bool")
        for name, value in (("jaw_pose", jaw_pose), ("expression", expression)):
            if (
                not value.is_floating_point()
                or value.dtype != landmarks.dtype
                or value.device != landmarks.device
            ):
                raise ValueError(
                    f"{name} must match face landmark dtype and device"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if face_valid.device != landmarks.device:
            raise ValueError("face_valid must be on the landmark device")

        if landmarks.shape[-1] == 2:
            confidence = torch.ones_like(landmarks[..., :1])
            landmarks = torch.cat((landmarks, confidence), dim=-1)
        landmark_confidence = landmarks[..., 2].clamp(0, 1)
        indices = torch.arange(72, device=landmarks.device)
        index_features = self.landmark_index(indices).to(dtype=landmarks.dtype)
        index_features = index_features.expand(*leading, -1, -1)
        points = self.point_encoder(torch.cat((landmarks, index_features), dim=-1))
        points = points * landmark_confidence[..., None]
        parameters = self.parameter_encoder(
            torch.cat((jaw_pose, expression), dim=-1)
        )
        context = torch.cat((points, parameters[..., None, :]), dim=-2)
        flat_context = context.reshape(-1, 73, self.dim)
        padding_mask = torch.cat(
            (
                landmark_confidence <= 0,
                torch.zeros(
                    *leading,
                    1,
                    dtype=torch.bool,
                    device=landmarks.device,
                ),
            ),
            dim=-1,
        ).reshape(-1, 73)
        queries = self.queries.to(dtype=landmarks.dtype).expand(
            flat_context.shape[0], -1, -1
        )
        attended, _ = self.attention(
            queries,
            flat_context,
            flat_context,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = queries + attended
        tokens = self.output_norm(tokens + self.feed_forward(tokens))
        tokens = tokens.reshape(*leading, self.token_count, self.dim)
        return tokens * face_valid[..., None, None].to(tokens.dtype)
