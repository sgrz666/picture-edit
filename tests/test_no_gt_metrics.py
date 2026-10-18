"""Contract tests for the ground-truth-free workflow metrics (CPU only, no models loaded).

These guard the three ways a no-GT metric can quietly start lying: a keypoint layout
remap that desynchronises from the renderer that draws the guides, an out-of-frame
accounting that silently drops misses, and a preservation region that stops excluding the
body. Everything here runs without ONNX, InsightFace or torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import no_gt_metrics as M
from src.pose_edit.geometry import cocowholebody133_to_dwpose134


def skeleton(scale=1.0, shift=(0.0, 0.0)):
    """An upright 134-slot figure: neck at (0,1), hips at y=2, shoulders at y=1.4."""
    points = np.full((M.DWSLOT_COUNT, 2), np.nan)
    points[0] = (0, 0.6)
    points[1] = (0, 1.0)
    points[2], points[5] = (-0.2, 1.4), (0.2, 1.4)
    points[3], points[6] = (-0.35, 1.0), (0.35, 1.0)
    points[4], points[7] = (-0.5, 0.6), (0.5, 0.6)
    points[8], points[11] = (-0.15, 2.0), (0.15, 2.0)
    points[9], points[12] = (-0.16, 2.8), (0.16, 2.8)
    points[10], points[13] = (-0.17, 3.6), (0.17, 3.6)
    points[14], points[15] = (-0.05, 0.55), (0.05, 0.55)
    points[16], points[17] = (-0.1, 0.5), (0.1, 0.5)
    for foot in (18, 19, 20, 21, 22, 23):
        points[foot] = (-0.2 if foot < 21 else 0.2, 3.7)
    points[24:92] = np.stack([np.linspace(-0.1, 0.1, 68), np.linspace(0.5, 0.45, 68)], axis=1)
    for wrist, base in ((92, -0.5), (113, 0.5)):
        points[wrist] = (base, 0.6)
        offsets = np.stack([np.linspace(base, base + 0.1, 20), np.linspace(0.6, 0.2, 20)], axis=1)
        points[wrist + 1:wrist + 21] = offsets
    return points * scale + np.asarray(shift)


def test_num_never_returns_nan():
    assert M.num(float('nan')) is None
    assert M.num(float('inf')) is None
    assert M.num(None) is None
    assert M.num(2.5) == 2.5


def test_slot_remap_matches_the_renderer_mapping():
    """The score/flag remap must be the same permutation geometry uses to draw guides."""
    rng = np.random.default_rng(20261013)
    raw = rng.normal(size=(M.COCO_WHOLEBODY_133, 2))
    assert np.allclose(M.dwpose134_from_coco133(raw), cocowholebody133_to_dwpose134(raw))
    scores = rng.uniform(size=M.COCO_WHOLEBODY_133)
    converted = M.dwpose134_from_coco133(scores)
    assert converted.shape == (M.DWSLOT_COUNT,)
    assert converted[1] == pytest.approx((scores[5] + scores[6]) / 2.0)
    assert converted[0] == pytest.approx(scores[0])
    with pytest.raises(ValueError):
        M.dwpose134_from_coco133(np.zeros(132))


def test_torso_normalisation_is_neck_to_hip_centre():
    points = skeleton(scale=100.0)
    hips = np.mean(points[[8, 11]], axis=0)
    assert M.torso_px(points) == pytest.approx(np.linalg.norm(points[1] - hips))
    broken = points.copy()
    broken[1] = np.nan
    assert M.torso_px(broken) is None


def test_pose_error_stats_accounts_for_dropped_joints():
    guide = skeleton(scale=100.0, shift=(500.0, 500.0))
    size = (1000, 1000)
    valid = M.validity(guide, *size)
    estimate = guide + 5.0
    block = M.pose_error_stats(guide, valid, estimate, M.validity(estimate, *size), size=size)
    assert block['joints_requested'] == int(valid.sum())
    assert block['joints_excluded_ratio'] == pytest.approx(0.0)
    assert block['groups']['body']['joints_requested'] == 18
    assert block['summary']['body_pck@0.2'] == pytest.approx(1.0)
    assert block['summary']['body_pck@0.2_strict'] == pytest.approx(1.0)

    # Push both ankles out of frame: the lenient reading is untouched because the dropped
    # joints are excluded, the strict reading charges them as misses.
    dropped_points = guide.copy()
    dropped_points[[10, 13]] = (-10.0, -10.0)
    dropped = M.pose_error_stats(guide, valid, dropped_points,
        M.validity(dropped_points, *size), size=size)
    assert dropped['groups']['body']['joints'] == 16
    assert dropped['groups']['body']['joints_dropped'] == 2
    assert dropped['summary']['body_pck@0.2'] == pytest.approx(1.0)
    assert dropped['summary']['body_pck@0.2_strict'] == pytest.approx(16.0 / 18.0)
    assert dropped['joints_excluded_ratio'] > 0.0


def test_pose_error_stats_reports_none_when_nothing_requested():
    guide = np.full((M.DWSLOT_COUNT, 2), np.nan)
    block = M.pose_error_stats(guide, np.zeros(M.DWSLOT_COUNT, dtype=bool), guide,
        np.zeros(M.DWSLOT_COUNT, dtype=bool), size=(100, 100))
    assert block['joints_requested'] == 0
    assert block['summary'] == {}


def test_limb_ratios_split_invariant_from_view_sensitive():
    base = skeleton(scale=120.0, shift=(500.0, 500.0))
    zoomed = skeleton(scale=360.0, shift=(200.0, 100.0))
    ratios_base = M.limb_ratios(base)
    ratios_zoomed = M.limb_ratios(zoomed)
    assert set(ratios_base['invariant']) == set(ratios_zoomed['invariant'])
    assert set(ratios_base['view_sensitive']) == set(ratios_zoomed['view_sensitive'])
    assert set(ratios_base['invariant']) == {
        'right_forearm_over_upper_arm', 'left_forearm_over_upper_arm',
        'right_shank_over_thigh', 'left_shank_over_thigh'}
    assert 'arm_over_leg' in ratios_base['view_sensitive']
    for family in ('invariant', 'view_sensitive'):
        for key in ratios_base[family]:
            assert ratios_base[family][key] == pytest.approx(ratios_zoomed[family][key],
                rel=1e-9)


def test_drift_headline_reads_only_the_invariant_family():
    """A torso turn must move the view family, not the headline number.

    Built from ratio dicts rather than a puppet skeleton: the point under test is the
    classification and aggregation, and no synthetic joint set can honestly model how a
    real torso turn foreshortens each limb differently.
    """
    source = {'invariant': {'right_forearm_over_upper_arm': 0.78, 'left_shank_over_thigh': 0.81},
        'view_sensitive': {'shoulder_over_hip_width': 1.9, 'arm_over_leg': 0.62}}
    generated = {'invariant': {'right_forearm_over_upper_arm': 0.781,
            'left_shank_over_thigh': 0.812},
        'view_sensitive': {'shoulder_over_hip_width': 0.4, 'arm_over_leg': 0.55}}
    drift = M.ratio_drift(source, generated)
    assert drift['max_rel_change'] < 0.005
    assert drift['view_max_rel_change'] > 0.5
    assert drift['worst_ratio'] == 'left_shank_over_thigh'
    assert drift['view_worst_ratio'] == 'shoulder_over_hip_width'


def test_stretched_shank_is_caught_by_the_invariant_family():
    base = skeleton(scale=120.0, shift=(500.0, 500.0))
    stretched = base.copy()
    stretched[10] = stretched[10] + np.array((0.0, 0.6)) * 120.0
    drift = M.ratio_drift(M.limb_ratios(base), M.limb_ratios(stretched))
    assert drift['max_rel_change'] > 0.2
    assert drift['worst_ratio'] == 'right_shank_over_thigh'


def test_ratio_drift_without_common_pairs_is_explicitly_empty():
    empty = M.ratio_drift({'invariant': {}, 'view_sensitive': {}},
        {'invariant': {}, 'view_sensitive': {}})
    assert empty['ratios_compared'] == 0
    assert empty['max_rel_change'] is None
    assert empty['view_ratios_compared'] == 0
    assert empty['view_max_rel_change'] is None


def test_estimator_agreement_verdict_thresholds():
    guide = skeleton(scale=100.0, shift=(500.0, 500.0))
    size = (1000, 1000)
    close = M.estimator_agreement(guide, guide + 3.0, size)
    assert close['verdict'] == 'consistent'
    assert close['body_agree@0.2'] == pytest.approx(1.0)
    far = guide.copy()
    far[2:14] += np.array((60.0, 60.0))
    conflict = M.estimator_agreement(guide, far, size)
    assert conflict['verdict'] in ('review', 'conflict')
    assert conflict['body_agree@0.2'] < close['body_agree@0.2']


def test_region_metrics_split_background_and_person():
    source = np.zeros((20, 20, 3), dtype=np.float64)
    generated = np.full_like(source, 10.0)
    mask_source = np.zeros((20, 20), dtype=bool)
    mask_source[5:15, 5:15] = True
    mask_generated = np.zeros((20, 20), dtype=bool)
    mask_generated[6:16, 6:16] = True
    out = M.region_metrics(source, generated, mask_source, mask_generated)
    assert out['background_mae'] == pytest.approx(10.0)
    assert out['person_mae'] == pytest.approx(10.0)
    assert out['background_pixels'] == 400 - int((mask_source | mask_generated).sum())
    assert 20 * np.log10(255.0) - 10 * np.log10(out['person_mse']) == pytest.approx(
        out['person_psnr'])
    with pytest.raises(ValueError):
        M.region_metrics(source, generated[:10, :10], mask_source, mask_generated[:10, :10])


def test_masks_are_boolean_and_dilation_only_grows():
    points = skeleton(scale=100.0, shift=(500.0, 500.0))
    mask = M.mask_from_limbs(points, (1000, 1000), torso=100.0)
    assert mask.dtype == np.bool_
    assert mask.any()
    grown = M.dilate(mask, 8)
    assert grown.sum() >= mask.sum()


def test_nonsubject_regions_track_other_people():
    boxes_a = np.array([[0, 0, 100, 200], [300, 0, 400, 200]], dtype=float)
    boxes_b = np.array([[2, 4, 102, 204], [500, 0, 600, 200]], dtype=float)
    out = M.nonsubject_regions(boxes_a, boxes_b, 0)
    assert out['nonsubject_count'] == 1
    assert out['matched_count'] == 0
    assert out['regions'][0]['matched'] is False
    aligned = M.nonsubject_regions(boxes_a, np.array([[0, 0, 100, 200], [310, 10, 410, 210]]), 0)
    assert aligned['matched_count'] == 1
    assert aligned['regions'][0]['displacement_px'] == pytest.approx(np.hypot(10, 10))


def test_beta_distance_with_standardisation():
    source = np.zeros(10)
    generated = np.zeros(10)
    generated[2] = 0.5
    block = M.beta_distance(source, generated, standard_deviation=np.full(10, 0.25))
    assert block['beta_l2'] == pytest.approx(0.5)
    assert block['beta_max_abs'] == pytest.approx(0.5)
    assert block['per_beta'][2] == pytest.approx(0.5)
    assert block['beta_mahalanobis'] == pytest.approx(2.0)
    assert block['status'] if 'status' in block else True


def test_points134_rejects_wrong_layout():
    with pytest.raises(ValueError):
        M.points134(np.zeros((133, 2)))


def test_scale_points_maps_source_frames_into_generated_pixels():
    points = np.array([[0.0, 0.0], [10.0, 20.0], [np.nan, 5.0]])
    scaled = M.scale_points(points, (0.5, 0.25))
    assert np.allclose(scaled[:2], [[0.0, 0.0], [5.0, 5.0]])
    assert np.isnan(scaled[2][0]) and scaled[2][1] == pytest.approx(1.25)
    boxes = M.scale_points(np.array([[0.0, 0.0, 100.0, 200.0]]), (0.5, 0.5))
    assert np.allclose(boxes, [[0.0, 0.0, 50.0, 100.0]])
    assert M.scale_points(np.zeros((0, 4)), (2.0, 2.0)).shape == (0, 4)
    with pytest.raises(ValueError):
        M.scale_points(points, (0.0, 1.0))
