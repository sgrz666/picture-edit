"""Deferred V3 regression cases. Authored, not executed during architecture delivery."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from src.pose_control.qwen21_full.dual_hi4d_geometry import decode_contact
from src.pose_control.qwen21_full.dual_depth_geometry import ownership, relative11
from src.pose_control.qwen21_full.dual_depth_condition import NativeDualIdentityCondition, IDENTITY_SCHEMA
from src.pose_control.qwen21_full.native_dual_identity import NativeDualIdentityBinder
from src.pose_control.qwen21_full.native_dual_depth import GatedNativeResidual, DualControlOptions


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
