"""CPU-only V6.1 key support, sparse sampling and residual contracts."""
from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F

from tests.test_native_v6_poselink import layout_for, geometry_for, activate


def config(**changes):
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61
    values = dict(hidden_dim=16, width=8, heads=2, injection_layers=(0, 2),
        max_pose_tokens_per_reference=16, global_token_budget=8,
        joint_token_budget=4, local_token_budget=4, query_chunk_size=3)
    return PoseLinkConfigV61(**(values | changes))


def spatial_for(geometry):
    from src.pose_control.qwen21_full.native_v61.contracts import PoseSpatialConditionV61
    b, n = geometry.geometry.shape[:2]
    h, w = geometry.target_token_hw
    support = torch.ones(b, 2, h*16, w*16)
    uv = torch.zeros(b, 2, 21, 2)
    uv[:, 0, :2] = torch.tensor([[.2, .2], [.3, .3]])
    uv[:, 1, :2] = torch.tensor([[.7, .2], [.8, .3]])
    valid = torch.zeros(b, 2, 21, dtype=torch.bool)
    valid[:, :, :2] = True
    boxes = torch.zeros(b, 2, 5, 4)
    boxes[:, 0, :2] = torch.tensor([.1, .1, .4, .4])
    boxes[:, 1, :2] = torch.tensor([.6, .1, .9, .4])
    local_valid = torch.zeros(b, 2, 5, dtype=torch.bool)
    local_valid[:, :, :2] = True
    support.mul_(geometry.person_valid[:, :, None, None])
    valid.logical_and_(geometry.person_valid[:, :, None])
    local_valid.logical_and_(geometry.person_valid[:, :, None])
    return PoseSpatialConditionV61(support, torch.ones(b, 1, h*16, w*16),
        torch.ones(b, n), uv, valid, boxes, local_valid, geometry.canvas_ids,
        tuple({'schema': 'native_pose_spatial_v61', 'target_canvas_id': canvas, 'source_repair': 'missing'}
              for canvas in geometry.canvas_ids)).validate(geometry)


def case(count=4):
    layout = layout_for(count, [(1, 4, 4)]*(count+1))
    geometry = geometry_for((4, 4))
    joint = torch.randn(1, layout.joint_length, 16)
    return joint[:, layout.target_indices].clone(), joint, layout, geometry, spatial_for(geometry)


def test_all_features_off_are_exactly_v6_with_nonzero_heads():
    from src.pose_control.qwen21_full.native_v6.contracts import PoseLinkConfig
    from src.pose_control.qwen21_full.native_v6.poselink import PoseLinkAttention
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    cfg = config(geometry_key_mask=False, edit_region_gate=False, multiscale_tokens=False)
    old = PoseLinkAttention(PoseLinkConfig(**{key: value for key, value in asdict(cfg).items()
        if key in PoseLinkConfig.__dataclass_fields__})).eval()
    new = PoseLinkAttentionV61(cfg).eval()
    activate(old)
    new.load_state_dict(old.state_dict(), strict=False)
    target, joint, layout, geometry, _ = case()
    args = dict(layer=0, sigma=torch.tensor([.5]))
    assert torch.equal(old(target, joint, layout, geometry, **args),
                       new(target, joint, layout, geometry, spatial=None, **args))


@pytest.mark.parametrize('count', [2, 4, 5])
def test_multiscale_budget_coordinates_and_fair_person_shares(count):
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    model = PoseLinkAttentionV61(config()).eval()
    target, joint, layout, geometry, spatial = case(count)
    keys = model.build_pose_keys(joint, layout, geometry, spatial)
    assert keys.hidden.shape[1] <= len(layout.references)*16
    assert keys.xy.shape == (*keys.valid.shape, 2)
    assert ((keys.xy[keys.valid] >= 0) & (keys.xy[keys.valid] <= 1)).all()
    assert set(keys.type_ids[keys.valid].tolist()) == {0, 1, 2}
    for ref in layout.references:
        positions = keys.reference_ids == ref.image_id
        assert int(positions[0].sum()) <= 16
        if ref.role == 'combined_pose':
            sparse = positions & (keys.type_ids > 0) & keys.valid
            assert int((sparse & (keys.person_ids == 0)).sum()) == int((sparse & (keys.person_ids == 1)).sum())


def test_mask_is_applied_before_pooling_and_alignment():
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    model = PoseLinkAttentionV61(config(multiscale_tokens=False, edit_region_gate=False)).eval()
    activate(model)
    target, joint, layout, geometry, spatial = case()
    spatial.key_support[:, :, 32:] = 0
    expected = model(target, joint, layout, geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    poisoned = joint.clone()
    for ref in layout.references:
        poisoned[:, ref.indices[8:]] = 1e12
    bad_geometry = replace(geometry, geometry=geometry.geometry.clone(),
        person_tokens=geometry.person_tokens.clone(), interaction=geometry.interaction.clone())
    bad_geometry.geometry[:, 8:] = 1e12
    bad_geometry.person_tokens[:, :, 8:] = 1e12
    bad_geometry.interaction[:, 8:] = 1e12
    actual = model(target, poisoned, layout, bad_geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    assert torch.equal(expected, actual)


def test_edit_gate_applies_once_and_exterior_is_exact_zero():
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    enabled = PoseLinkAttentionV61(config()).eval()
    disabled = PoseLinkAttentionV61(config(edit_region_gate=False)).eval()
    activate(enabled)
    disabled.load_state_dict(enabled.state_dict())
    target, joint, layout, geometry, spatial = case()
    spatial.valid_pixels[:, :, :, 8:16] = 0
    coverage = F.avg_pool2d(spatial.valid_pixels, 16).flatten(1)
    geometry = replace(geometry, coverage=coverage)
    spatial.key_support.mul_(spatial.valid_pixels)
    spatial.edit_gate[:] = 0
    spatial.edit_gate[:, :2] = .25
    args = dict(spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    full = disabled(target, joint, layout, geometry, **args)
    result = enabled(target, joint, layout, geometry, **args)
    assert torch.allclose(result[:, :2], full[:, :2]*.25, atol=1e-7, rtol=1e-6)
    assert not result[:, 2:].any()


def test_all_invalid_keys_remain_zero_with_trained_head_bias():
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    model = PoseLinkAttentionV61(config()).eval()
    target, joint, layout, geometry, spatial = case()
    spatial.key_support.zero_()
    spatial.joint_valid.zero_()
    spatial.local_valid.zero_()
    with torch.no_grad():
        model.output_heads['0'].bias.fill_(2.)
    result = model(target, joint, layout, geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    assert torch.isfinite(result).all() and not result.any()


def test_zero_head_and_nonzero_head_gradients_exclude_source():
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    model = PoseLinkAttentionV61(config())
    target, joint, layout, geometry, spatial = case()
    target.requires_grad_(); joint.requires_grad_(); geometry.geometry.requires_grad_()
    result = model(target, joint, layout, geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    assert not result.any()
    result.sum().backward()
    assert model.output_heads['0'].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True); target.grad=None; joint.grad=None; geometry.geometry.grad=None
    activate(model)
    model(target, joint, layout, geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5])).square().sum().backward()
    assert geometry.geometry.grad.abs().sum() > 0
    assert model.token_type_embedding.weight.grad.abs().sum() > 0
    assert not joint.grad[:, layout.image_ids == 0].any()


def test_gate_only_preserves_the_v6_attention_before_gating():
    from src.pose_control.qwen21_full.native_v6.contracts import PoseLinkConfig
    from src.pose_control.qwen21_full.native_v6.poselink import PoseLinkAttention
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    cfg = config(geometry_key_mask=False, multiscale_tokens=False)
    old = PoseLinkAttention(PoseLinkConfig(**{k:v for k,v in asdict(cfg).items()
        if k in PoseLinkConfig.__dataclass_fields__})).eval()
    new = PoseLinkAttentionV61(cfg).eval()
    activate(old); new.load_state_dict(old.state_dict(), strict=False)
    target, joint, layout, geometry, spatial = case()
    geometry.projected.fill_(.5)
    spatial.edit_gate.fill_(.3)
    before = old(target, joint, layout, geometry, layer=0, sigma=torch.tensor([.5]))
    result = new(target, joint, layout, geometry, spatial=spatial, layer=0, sigma=torch.tensor([.5]))
    assert torch.equal(result, before*.3)


@pytest.mark.parametrize('kind', ['joint', 'local'])
def test_person_bound_combined_samples_exclude_other_person_values_and_gradients(kind):
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    target, joint, layout, geometry, spatial = case(4)
    spatial.key_support.zero_(); spatial.key_support[:,0,:,:32]=1; spatial.key_support[:,1,:,32:]=1
    spatial.joint_valid.zero_(); spatial.local_valid.zero_()
    if kind == 'joint':
        spatial.joint_valid[:,:,:1]=True
        spatial.joint_uv[:,0,0]=torch.tensor([.49,.375]); spatial.joint_uv[:,1,0]=torch.tensor([.75,.375])
    else:
        spatial.local_valid[:,:,:1]=True
        spatial.local_boxes[:,0,0]=torch.tensor([.48,.36,.50,.39])
        spatial.local_boxes[:,1,0]=torch.tensor([.7,.3,.8,.4])
    cfg = config(global_token_budget=0, joint_token_budget=4 if kind=='joint' else 0,
                 local_token_budget=4 if kind=='local' else 0)
    model = PoseLinkAttentionV61(cfg)
    joint.zero_(); joint.requires_grad_(); geometry.geometry.requires_grad_()
    first = model.build_pose_keys(joint, layout, geometry, spatial)
    ref = next(r for r in layout.references if r.role=='combined_pose')
    selected = (first.reference_ids==ref.image_id)&(first.person_ids==0)&first.valid
    (first.hidden[selected].sum()+first.geometry[selected].sum()).backward()
    assert not joint.grad[0,ref.indices[6]].any()
    assert not geometry.geometry.grad[0,6].any()
    changed = joint.detach().clone(); changed[:,ref.indices[6]]=100
    poisoned = replace(geometry, geometry=geometry.geometry.detach().clone())
    poisoned.geometry[:,6]=100
    second = model.build_pose_keys(changed, layout, poisoned, spatial)
    assert torch.equal(first.hidden[selected], second.hidden[selected])
    assert torch.equal(first.geometry[selected], second.geometry[selected])
