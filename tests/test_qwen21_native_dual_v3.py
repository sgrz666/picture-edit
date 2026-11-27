"""Deferred V3 regression cases. Authored, not executed during architecture delivery."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from src.pose_control.qwen21_full.dual_hi4d_geometry import decode_contact
from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.dual_depth_geometry import build_dual_condition, ownership, relative11
from src.pose_control.qwen21_full.dual_depth_condition import NativeDualIdentityCondition, IDENTITY_SCHEMA
from src.pose_control.qwen21_full.native_dual_identity import NativeDualIdentityBinder
from src.pose_control.qwen21_full.native_dual_depth import GatedNativeResidual, DualControlOptions, NativeDualConfig


def test_downscaled_relation_raster_matches_typed_camera_z_support():
    spec = CanvasSpec((128, 128), (32, 32), (32, 32), (0, 0), policy='aspect_preserving')
    valid = np.zeros((2, 128, 128), dtype=bool)
    valid[0, :64, :64] = True
    valid[0, 96, 96] = True   # Nearest hits it, the bilinear support footprint misses it.
    valid[1, 32:96, 32:96] = True
    z = valid.astype(np.float32) * np.array([2., 3.], np.float32)[:, None, None]
    joints = np.zeros((2, 22, 3), dtype=np.float32)
    joints[..., 2] = np.array([2., 3.])[:, None]
    normal = np.zeros((2, 128, 128, 3), dtype=np.float32)
    normal[..., 2] = valid
    flags = np.zeros((14, 14), dtype=bool)
    cache = dict(z=z, valid=valid, normal=normal, part=valid.astype(np.uint8),
                 K=np.array([[128., 0., 64.], [0., 128., 64.], [0., 0., 1.]], dtype=np.float32),
                 joints_cam=joints, body_height=np.array([1.7, 1.8], dtype=np.float32),
                 part_distance=np.zeros((14, 14), np.float32), part_distance_valid=flags,
                 closest_points=np.zeros((14, 14, 2, 3), np.float32), closest_points_valid=flags,
                 part_contact=np.zeros((14, 14, 4), np.float32),
                 contact_indices=np.empty((0, 4), np.int64), contact_distance=np.empty(0, np.float32))
    metadata = dict(person_ids=['a', 'b'], record_id='scene', pair_id='example', record_sha256='0'*64,
                    camera_id='camera', scale_group_id='scale', units='model_unit', scale_status='model_scale',
                    model_family_indices=[0, 0])
    result = build_dual_condition(cache, metadata, spec, NativeDualConfig())
    assert result.projected[0, 0, 0, 24, 24]
    assert not result.person_depth_valid[0, 0, 0, 24, 24]
    assert torch.equal(result.relation_raster[:, 2:4].bool(), result.person_depth_valid.squeeze(2))
    assert result.relation_raster[0, 0, 24, 24] == 0


def test_contact_preserves_official_direction_and_one_based_indices():
    vertices = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    contact = np.array([[2., 0., 0.], [0., 1., 1.]])
    indices, distance = decode_contact(contact, vertices, 2.)
    assert indices.tolist() == [[0, 0, 1, 1], [1, 1, 0, 0], [1, 2, 0, 0]]
    assert np.allclose(distance[0], np.linalg.norm(vertices[0, 0]-vertices[1, 1])/2)
    with pytest.raises(ValueError):
        decode_contact(contact+.1, vertices, 2.)


def test_owner_excludes_invalid_depth_and_splits_ties():
    z = torch.tensor([[[[[2., 3., 0.]]], [[[3., 3., 4.]]]]])
    valid = z > 0
    scene, scene_valid, overlap, tie, front, visible, weights = ownership(z, valid, torch.tensor([[2., 2.]]), 1e-4)
    assert scene.flatten().tolist() == [2., 3., 4.]
    assert tie.flatten().tolist() == [False, True, False]
    assert weights[0, 0].flatten().tolist() == [1., .5, 0.]
    assert torch.equal(weights.sum(1), scene_valid.float())
    assert not (front[:, 0] & front[:, 1]).any()
    assert not (visible & ~valid).any()


def test_relative_orientation_uses_composed_rotations():
    joints = torch.zeros(2, 22, 3)
    joints[1, 0, 0] = 2.
    rotations = torch.stack((torch.eye(3), torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])))
    relative, valid = relative11(joints, rotations, torch.tensor([2., 2.]), torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.]]))
    assert valid
    assert torch.allclose(relative[:3], torch.tensor([1., 0., 0.]))
    assert torch.allclose(relative[8:10], torch.tensor([1., 0.]))


def test_zero_output_has_first_step_projection_gradient_without_optimizer():
    fusion = GatedNativeResidual(12)
    update = torch.randn(1, 4, 512, requires_grad=True)
    result = fusion(torch.randn_like(update), update, torch.randn(1, 4, 12), torch.ones(1, 4))
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert fusion.output.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(update.grad) == 0  # expected until the zero output head changes


def test_identity_source_reordering_with_mapping_keeps_binding():
    torch.manual_seed(9)
    mask = torch.zeros(1, 2, 1, 128, 128, dtype=torch.bool)
    mask[:, 0, :, :, :64] = True
    mask[:, 1, :, :, 64:] = True
    meta = {'schema': IDENTITY_SCHEMA, 'source_canvas_id': 'reference', 'source_rgb_sha256': '0'*64,
            'source_person_ids': ['p0', 'p1'], 'target_person_ids': ['p0', 'p1']}
    source = NativeDualIdentityCondition(torch.randn(1, 64, 8, 8), mask, torch.zeros_like(mask),
        torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool), (meta,))
    reordered = replace(source, person_mask=mask.flip(1), head_mask=source.head_mask.flip(1),
        source_indices=torch.tensor([[1, 0]]), metadata=({**meta, 'source_person_ids': ['p1', 'p0']},))
    binder = NativeDualIdentityBinder().eval()
    query = torch.randn(1, 2, 4, 512)
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    weights = torch.full((1, 2, 1, 2, 2), .5)
    with torch.no_grad():
        assert torch.equal(binder(source, query, valid, weights), binder(reordered, query, valid, weights))


def test_relation_ablation_requires_an_active_interaction_path():
    with pytest.raises(ValueError):
        DualControlOptions(interaction=False)
    with pytest.raises(ValueError):
        DualControlOptions(depth_mode='scene_only', detail=False)
