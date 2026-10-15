from __future__ import annotations

import numpy as np
import pytest
from src.pose_control.qwen21_full.prompt_study.geometry import (
    visible_points,
    hand_boxes,
    render_dwpose,
)


def test_visibility_does_not_clip_out_of_frame_points():
    xy = np.full((134, 2), 0.5)
    scores = np.ones(134)
    xy[1] = (-0.1, 0.3)
    scores[2] = 0.49
    visible = visible_points(xy, scores)
    assert visible[0] and not visible[1] and not visible[2]
    assert tuple(hand_boxes(xy, scores)) == ("left", "right")


def test_visible_points_bounds_and_finite():
    xy = np.zeros((134, 2))
    scores = np.full(134, 0.6)
    # Inf / NaN handling
    xy[10, 0] = np.nan
    xy[11, 1] = np.inf
    # Out of [0, 1]
    xy[12, 0] = 1.05
    xy[13, 1] = -0.01
    vis = visible_points(xy, scores)
    assert vis[0]
    assert not vis[10]
    assert not vis[11]
    assert not vis[12]
    assert not vis[13]


def test_hand_boxes_insufficient_points_returns_none():
    xy = np.full((134, 2), 0.5)
    scores = np.zeros(134)
    # Enable only 3 points on left hand (92..113)
    scores[92:95] = 0.9
    boxes = hand_boxes(xy, scores)
    assert boxes["left"] is None
    assert boxes["right"] is None

    # Enable 4 points on left hand
    scores[95] = 0.9
    boxes = hand_boxes(xy, scores)
    assert boxes["left"] is not None
    assert len(boxes["left"]) == 4
    # All coords must be in [0, 1]
    assert all(0.0 <= c <= 1.0 for c in boxes["left"])


def test_render_dwpose_valid_and_all_invalid():
    rec = {
        "image_width": 604,
        "image_height": 1080,
        "keypoints_norm": np.full((134, 2), 0.5).tolist(),
        "scores": np.full(134, 0.8).tolist(),
    }
    img, diag = render_dwpose(rec)
    assert img.size == (604, 1080)
    assert diag["render_version"] == "dwpose134_filtered_v1"
    assert diag["visible_body"] == 18
    assert diag["visible_face"] == 68

    # All invalid
    rec_empty = {
        "image_width": 604,
        "image_height": 1080,
        "keypoints_norm": np.zeros((134, 2)).tolist(),
        "scores": np.zeros(134).tolist(),
    }
    img_empty, diag_empty = render_dwpose(rec_empty)
    assert img_empty.size == (604, 1080)
    assert diag_empty["visible_body"] == 0
    # Image must be all black
    extrema = img_empty.getextrema()
    assert all(low == 0 and high == 0 for low, high in extrema)
