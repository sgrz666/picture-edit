"""Bind the shared SDPose adapter used by both skeletons and generated rewards."""
import numpy as np


def test_sdpose_geometry_layout_and_neck_confidence():
    from src.evaluation.sdpose import convert_coco133_person,DEFAULT_SCORE_THRESHOLD
    p=np.arange(266,dtype=float).reshape(133,2);s=np.linspace(.1,1,133)
    points,scores=convert_coco133_person(p,s,order='geometry')
    assert points.shape==(134,2) and scores.shape==(134,)
    np.testing.assert_allclose(points[1],(p[5]+p[6])/2)
    assert scores[1]==min(s[5],s[6]) and DEFAULT_SCORE_THRESHOLD==.30
