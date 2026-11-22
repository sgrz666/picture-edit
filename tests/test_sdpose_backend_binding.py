"""Bind the shared SDPose adapter used by both skeletons and generated rewards."""
import numpy as np


def test_sdpose_geometry_layout_and_neck_confidence():
    from src.evaluation.sdpose import convert_coco133_person,DEFAULT_SCORE_THRESHOLD
    p=np.arange(266,dtype=float).reshape(133,2);s=np.linspace(.1,1,133)
    points,scores=convert_coco133_person(p,s,order='geometry')
    assert points.shape==(134,2) and scores.shape==(134,)
    np.testing.assert_allclose(points[1],(p[5]+p[6])/2)
    assert scores[1]==min(s[5],s[6]) and DEFAULT_SCORE_THRESHOLD==.30


def test_shared_pose_metric_scores_sdpose_people_in_coco17_order(monkeypatch):
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from src.evaluation import sdpose
    from src.evaluation.joint_pilot_metrics import PoseMetric

    xy = np.column_stack((np.arange(133), np.arange(133) + 1))
    points, scores = sdpose.convert_coco133_person(xy, np.ones(133))
    runtime = SimpleNamespace(device=torch.device('cpu'))
    monkeypatch.setattr(sdpose, 'load_runtime', lambda **kwargs: runtime)

    def detect(image, loaded, *, order, **kwargs):
        assert loaded is runtime and order == 'geometry'
        torch.rand(1)
        return [{'points': points, 'scores': scores, 'box_xyxy': [0, 0, 30, 30]}]

    monkeypatch.setattr(sdpose, 'detect_people', detect)
    metric = PoseMetric()
    before = torch.get_rng_state()
    predicted, confidence = metric.detect(Image.new('RGB', (32, 32)))
    np.testing.assert_array_equal(predicted, xy[:17])
    np.testing.assert_array_equal(confidence, np.ones(17))
    assert torch.equal(before, torch.get_rng_state())


def test_shared_pose_metric_penalizes_outside_canvas_and_keeps_threshold_boundary():
    from src.evaluation.joint_pilot_metrics import COCO_TO_BODY25, PoseMetric

    target = np.zeros((25, 3))
    target[np.array(COCO_TO_BODY25[:2])] = [.5, .5, 1.]
    points = np.full((17, 2), 16.)
    points[1, 0] = 32
    result = PoseMetric.compare((points, np.full(17, .3)), target, image_size=(32, 32))
    assert result['pose_matched_joints'] == 1 and result['pose_pck_002'] == .5
