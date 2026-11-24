"""Contract tests for the shared SDPose 134-slot pose-alignment evaluation module."""
from __future__ import annotations

import numpy as np

from src.evaluation.sdpose import convert_coco133_person, swap_eye_ear_slots


def test_convert_coco133_person_geometry_vs_openpose_order():
    pts133 = np.arange(133 * 2, dtype=float).reshape(133, 2)
    sc133 = np.linspace(0.1, 0.9, 133)
    sc133[5] = 0.42
    sc133[6] = 0.78

    pts_geo, sc_geo = convert_coco133_person(pts133, sc133, order="geometry")
    assert pts_geo.shape == (134, 2)
    assert sc_geo.shape == (134,)
    # Neck slot 1 is midpoint of COCO shoulders (5, 6) and weakest shoulder score
    assert np.allclose(pts_geo[1], (pts133[5] + pts133[6]) / 2.0)
    assert np.isclose(sc_geo[1], 0.42)
    # geometry.py eye/ear order: 14=R_eye(2), 15=R_ear(4), 16=L_eye(1), 17=L_ear(3)
    assert np.allclose(pts_geo[14], pts133[2])
    assert np.allclose(pts_geo[15], pts133[4])
    assert np.allclose(pts_geo[16], pts133[1])
    assert np.allclose(pts_geo[17], pts133[3])

    pts_op, sc_op = convert_coco133_person(pts133, sc133, order="openpose")
    # OpenPose eye/ear order: 14=R_eye(2), 15=L_eye(1), 16=R_ear(4), 17=L_ear(3)
    assert np.allclose(pts_op[14], pts133[2])
    assert np.allclose(pts_op[15], pts133[1])
    assert np.allclose(pts_op[16], pts133[4])
    assert np.allclose(pts_op[17], pts133[3])
    assert np.allclose(sc_op[[15, 16]], sc_geo[[16, 15]])
    assert np.allclose(swap_eye_ear_slots(pts_geo), pts_op)


def test_convert_coco133_person_missing_shoulder_sets_zero_neck_score():
    pts133 = np.zeros((133, 2), dtype=float)
    pts133[5] = [np.nan, np.nan]
    sc133 = np.ones(133, dtype=float)

    pts_geo, sc_geo = convert_coco133_person(pts133, sc133)
    assert np.isnan(pts_geo[1]).all()
    assert sc_geo[1] == 0.0
