import numpy as np
import pytest


def test_strict_pck_counts_missing_and_outside_requested_points():
    from src.pose_control.qwen21_union.evaluation import strict_pose_metrics
    requested = np.array([[10, 10], [20, 20], [600, 10]], float)
    generated = np.array([[10, 10], [np.nan, np.nan], [600, 10]], float)
    result = strict_pose_metrics(requested, np.ones(3), generated, np.ones(3), torso=100)
    assert result["requested"] == 3
    assert result["pck_02_strict"] == pytest.approx(1 / 3)
    assert result["requested_outside"] == 1
    assert result["detector_coverage"] == pytest.approx(1 / 3)


def test_dual_assignment_uses_pose_and_never_reuses_person():
    from src.pose_control.qwen21_union.evaluation import assign_people
    left = np.tile([100, 100], (134, 1)).astype(float)
    right = np.tile([400, 100], (134, 1)).astype(float)
    requested = [(left, np.ones(134)), (right, np.ones(134))]
    detected = [{"points": right, "scores": np.ones(134)}, {"points": left, "scores": np.ones(134)}]
    assert assign_people(requested, detected)["mapping"] == [1, 0]


def test_macro_summary_does_not_weight_large_groups_more():
    from src.pose_control.qwen21_union.evaluation import aggregate_metrics
    rows = [dict(dataset="hi4d", arm="native", group_id="p1", sample_id=f"a{i}", seed=42, pck_02_strict=1.0) for i in range(8)]
    rows.append(dict(dataset="hi4d", arm="native", group_id="p2", sample_id="b", seed=42, pck_02_strict=0.0))
    result = aggregate_metrics(rows)
    assert result[0]["pck_02_group_macro"] == pytest.approx(0.5)
