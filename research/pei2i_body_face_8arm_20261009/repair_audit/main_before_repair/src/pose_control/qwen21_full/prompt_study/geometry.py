from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw

BODY_EDGES = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
HAND_EDGES = tuple(
    (0 if n == 1 else 4 * f + n - 1, 4 * f + n)
    for f in range(5)
    for n in range(1, 5)
)
FOOT_EDGES = (
    (13, 18), (18, 19), (13, 20), (10, 21), (21, 22), (10, 23)
)


def visible_points(xy: Any, scores: Any) -> np.ndarray:
    xy_arr = np.asarray(xy, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    if xy_arr.shape != (134, 2) or scores_arr.shape != (134,):
        raise ValueError("Expected DWPose134 xy/scores")
    return (
        np.isfinite(xy_arr).all(axis=1)
        & np.isfinite(scores_arr)
        & (scores_arr >= 0.5)
        & (xy_arr >= 0).all(axis=1)
        & (xy_arr <= 1).all(axis=1)
    )


def hand_boxes(xy: Any, scores: Any) -> dict[str, list[float] | None]:
    xy_arr = np.asarray(xy, dtype=float)
    good = visible_points(xy_arr, scores)
    result: dict[str, list[float] | None] = {}
    for name, start in (("left", 92), ("right", 113)):
        ids = np.flatnonzero(good[start : start + 21]) + start
        if len(ids) < 4:
            result[name] = None
            continue
        lo = xy_arr[ids].min(axis=0)
        hi = xy_arr[ids].max(axis=0)
        side = max(float((hi - lo).max()) * 1.5, 0.02)
        center = (lo + hi) / 2.0
        box = np.concatenate((
            np.maximum(center - side / 2.0, 0.0),
            np.minimum(center + side / 2.0, 1.0),
        )).tolist()
        result[name] = box
    return result


def render_dwpose(record: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    xy = np.asarray(record["keypoints_norm"], dtype=float)
    scores = np.asarray(record["scores"], dtype=float)
    good = visible_points(xy, scores)
    width = int(record["image_width"])
    height = int(record["image_height"])
    if min(width, height) <= 0:
        raise ValueError("Invalid DWPose canvas")

    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)

    def point(i: int) -> tuple[int, int]:
        return (round(xy[i, 0] * (width - 1)), round(xy[i, 1] * (height - 1)))

    palette = (
        (255, 80, 80),
        (80, 255, 80),
        (80, 160, 255),
        (255, 220, 80),
        (220, 80, 255),
    )
    for n, (a, b) in enumerate(BODY_EDGES + FOOT_EDGES):
        if good[a] and good[b]:
            draw.line((point(a), point(b)), fill=palette[n % 5], width=3)

    for start in (92, 113):
        for a, b in HAND_EDGES:
            if good[start + a] and good[start + b]:
                draw.line(
                    (point(start + a), point(start + b)),
                    fill=palette[(b - 1) // 4],
                    width=2,
                )

    for i in np.flatnonzero(good):
        x, y = point(i)
        radius = 1 if 24 <= i < 92 else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")

    diagnostic = {
        "threshold": 0.5,
        "original_size": [width, height],
        "render_version": "dwpose134_filtered_v1",
        "filtered_indices": np.flatnonzero(~good).tolist(),
        "visible_body": int(good[:18].sum()),
        "visible_face": int(good[24:92].sum()),
        "visible_left_hand": int(good[92:113].sum()),
        "visible_right_hand": int(good[113:134].sum()),
    }
    return image, diagnostic
