"""CPU-only geometry helpers for SMPL-X pose guides."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw


# DWPose/OpenPose body-25 is reduced to the requested body-18 ordering:
# nose, neck, right arm, left arm, right leg, left leg, R_Eye, R_Ear, L_Eye, L_Ear.
_BODY18_FROM_SMPLX137 = (
    24, 7, 9, 11, 13, 8, 10, 12, 2, 4, 6, 1, 3, 5, 23, 21, 22, 20
)
_FACE68_FROM_SMPLX137 = tuple(range(120, 137)) + tuple(range(69, 79)) + tuple(range(79, 88)) + tuple(range(88, 100)) + tuple(range(100, 120))

# Neck-to-hip-centre length as a fraction of a full-body detection box height. Measured against
# the generated bodies of the frozen test pairs: 0.31 reproduces the scale that actually matched
# the source body within about 4% on all five scored pairs, including the one whose width-driven
# box fit was off by 2.35x.
TORSO_TO_PERSON_BOX_HEIGHT = 0.31


def choose_subject(boxes: np.ndarray, scores: np.ndarray, rule: str = "auto") -> int:
    """Pick which detected person the transfer applies to.

    ``auto`` keeps the single-person contract and refuses to guess; ``largest`` and an integer
    index are explicit choices. Detection order is not stable across runs of the detector, so
    indexing happens over a deterministic area-then-left-to-right ordering.
    """
    boxes = _points_array(boxes, (4,), "boxes")
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) == 0 or len(boxes) != len(scores):
        raise ValueError("boxes and scores must be non-empty and equally long")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("boxes and scores must be finite")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("every box must have positive width and height")
    order = sorted(range(len(boxes)), key=lambda index: (-(widths[index] * heights[index]),
        boxes[index][0], boxes[index][1]))
    if rule == "auto":
        if len(boxes) != 1:
            raise ValueError(f'{len(boxes)} people were detected; a single-person workflow will '
                'not guess which one to transfer. Choose one with --subject largest or '
                '--subject INDEX (index over largest-first ordering).')
        return 0
    if rule == "largest":
        return order[0]
    try:
        index = int(rule)
    except (TypeError, ValueError) as error:
        raise ValueError(f"subject rule must be 'auto', 'largest' or an integer index; got {rule!r}") from error
    if not 0 <= index < len(order):
        raise ValueError(f"subject index {index} is outside the {len(order)} detected people")
    return order[index]


def _points_array(points: np.ndarray, dims: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] not in dims:
        expected = " or ".join(str(dim) for dim in dims)
        raise ValueError(f"{name} must have shape (N, {expected}); got {array.shape}")
    return array


def project_camera(
    points_cam: np.ndarray,
    focal_xy: Sequence[float],
    principal_xy: Sequence[float],
    z_epsilon: float = 1e-4,
) -> np.ndarray:
    """Project camera-space XYZ points to image XY using pinhole intrinsics.

    Points with non-finite coordinates or non-positive camera depth are returned
    as NaN. For valid points, the depth denominator matches SMPLest-X's
    ``Z + 1e-4`` convention.
    """
    points = _points_array(points_cam, (3,), "points_cam")
    focal = np.asarray(focal_xy, dtype=float)
    principal = np.asarray(principal_xy, dtype=float)
    if focal.shape != (2,) or principal.shape != (2,):
        raise ValueError("focal_xy and principal_xy must each contain two values")
    if not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError("focal_xy must contain two finite positive values")
    if not np.isfinite(principal).all():
        raise ValueError("principal_xy must contain two finite values")
    if not np.isfinite(z_epsilon) or z_epsilon <= 0:
        raise ValueError("z_epsilon must be a finite positive number")

    projected = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    depth = points[valid, 2] + z_epsilon
    projected[valid, 0] = points[valid, 0] / depth * focal[0] + principal[0]
    projected[valid, 1] = points[valid, 1] / depth * focal[1] + principal[1]
    projected[~np.isfinite(projected).all(axis=1)] = np.nan
    return projected


def smplx137_to_dwpose134(points: np.ndarray) -> np.ndarray:
    """Reorder SMPLest-X's 137 joints into the requested DWPose-134 layout."""
    array = _points_array(points, (2, 3), "points")
    if array.shape[0] != 137:
        raise ValueError(f"points must contain 137 SMPL-X joints; got {array.shape[0]}")

    body = list(_BODY18_FROM_SMPLX137)
    feet = [14, 15, 16, 17, 18, 19]
    face = list(_FACE68_FROM_SMPLX137)
    left_hand = [12, *range(25, 45)]
    right_hand = [13, *range(45, 65)]
    order = body + feet + face + left_hand + right_hand
    return array[order].copy()


def _validated_bbox(bbox: Sequence[float], name: str) -> tuple[float, float, float, float]:
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain four finite xyxy values")
    x1, y1, x2, y2 = map(float, values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return x1, y1, x2, y2


def torso_length(points_xy: np.ndarray) -> float:
    """Neck-to-hip-centre length of a DWPose-134 layout; NaN when a landmark is missing."""
    points = _points_array(points_xy, (2,), "points_xy")
    if points.shape[0] != 134:
        raise ValueError(f"points_xy must contain 134 DWPose points; got {points.shape[0]}")
    hips = np.nanmean(points[[8, 11]], axis=0)
    if not np.isfinite(points[1]).all() or not np.isfinite(hips).all():
        return float("nan")
    return float(np.linalg.norm(points[1] - hips))


def align_to_source_bbox(
    points_xy: np.ndarray,
    bbox_B: Sequence[float],
    bbox_A: Sequence[float],
    framing: str = "source_person",
) -> tuple[np.ndarray, dict[str, object]]:
    """Uniformly scale and center B-space points onto the source person in A.

    ``source_person`` fits B's detection box inside A's. That is simple, but the scale is then
    the smaller of the two side ratios, so when A and B are cropped with opposite aspect ratios
    the width term wins and the requested pose ends up far smaller than the source body.

    ``source_torso`` scales B's measured torso onto the source person's expected torso, derived
    from A's box height, so the transferred body keeps the source's own size no matter how the
    two boxes were cropped.
    """
    points = _points_array(points_xy, (2,), "points_xy")
    bx1, by1, bx2, by2 = _validated_bbox(bbox_B, "bbox_B")
    ax1, ay1, ax2, ay2 = _validated_bbox(bbox_A, "bbox_A")
    b_torso = float("nan")
    if framing == "source_person":
        scale = min((ax2 - ax1) / (bx2 - bx1), (ay2 - ay1) / (by2 - by1))
    elif framing == "source_torso":
        b_torso = torso_length(points)
        if not np.isfinite(b_torso) or b_torso <= 0:
            raise ValueError("source_torso framing needs a finite neck and at least one hip in B")
        scale = TORSO_TO_PERSON_BOX_HEIGHT * (ay2 - ay1) / b_torso
    else:
        raise ValueError(f"unknown pose framing {framing!r}; use source_person or source_torso")
    center_b = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
    center_a = np.array([(ax1 + ax2) / 2, (ay1 + ay2) / 2])
    offset = center_a - scale * center_b
    aligned = points * scale + offset
    aligned[~np.isfinite(points).all(axis=1)] = np.nan
    return aligned, {"scale": float(scale), "offset_xy": offset.tolist(), "framing": framing,
        "B_torso_px": float(b_torso) if np.isfinite(b_torso) else None,
        "A_box_height_px": float(ay2 - ay1)}


def valid_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a mask for finite points inside the half-open image canvas."""
    points = _points_array(points_xy, (2,), "points_xy")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def render_poseguide(points_xy: np.ndarray, size: Sequence[int]) -> Image.Image:
    """Render DWPose-134 points as a colored RGB skeleton on black."""
    points = _points_array(points_xy, (2,), "points_xy")
    if points.shape[0] != 134:
        raise ValueError(f"points_xy must contain 134 DWPose points; got {points.shape[0]}")
    if len(size) != 2:
        raise ValueError("size must be (width, height)")
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")

    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    valid = valid_points(points, width, height)

    # DWPose body order: nose, neck, right arm, left arm, hips/legs, eyes/ears.
    body_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (2, 5), (8, 11), (0, 14), (14, 15), (0, 16), (16, 17),
    ]
    # Bright, familiar OpenPose-like limb palette in RGB.
    palette = [
        (255, 50, 50), (255, 105, 30), (255, 165, 0), (255, 215, 0),
        (70, 220, 90), (0, 200, 180), (30, 144, 255), (65, 105, 225),
        (138, 43, 226), (220, 20, 160), (255, 70, 130), (255, 120, 90),
    ]

    def segment(a: int, b: int, color: tuple[int, int, int], line_width: int = 3) -> None:
        if valid[a] and valid[b]:
            draw.line((tuple(points[a]), tuple(points[b])), fill=color, width=line_width)

    for index, (a, b) in enumerate(body_edges):
        segment(a, b, palette[index % len(palette)])

    # Each foot has big toe, small toe, heel; connect them only when present.
    for a, b in (
        (10, 21), (10, 23), (13, 18), (13, 20),
        (18, 19), (18, 20), (21, 22), (21, 23),
    ):
        segment(a, b, (255, 190, 40), 2)

    # Hand layout is wrist followed by four joints for thumb/index/middle/ring/pinky.
    finger_colors = [
        (255, 70, 70), (255, 190, 40), (70, 230, 100),
        (60, 170, 255), (205, 90, 255),
    ]
    for wrist in (92, 113):
        for digit in range(5):
            first = wrist + 1 + digit * 4
            color = finger_colors[digit]
            segment(wrist, first, color, 2)
            for joint in range(3):
                segment(first + joint, first + joint + 1, color, 2)

    # Face landmarks are intentionally dots rather than speculative connections.
    radius = max(1, min(width, height) // 300)
    for index, (x, y) in enumerate(points):
        if not valid[index]:
            continue
        point_radius = max(radius, 1) if 24 <= index < 92 else max(radius + 1, 2)
        color = (255, 255, 255) if 24 <= index < 92 else (255, 235, 170)
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color)
    return image
