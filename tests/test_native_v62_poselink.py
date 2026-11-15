"""CPU-only direct routing contracts; no claim of full image identity isolation."""
from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F

from tests.test_native_v61_poselink import case, config as config61
from tests.test_native_v6_poselink import activate


def config(**changes):
    from src.pose_control.qwen21_full.native_v62.contracts import PoseLinkConfigV62
    return PoseLinkConfigV62.from_dict(asdict(config61()) | changes)


def routing_case(count=4):
    from src.pose_control.qwen21_full.native_v62.contracts import RoutingConditionV62, ROUTING_SCHEMA
    target, joint, layout, geometry, spatial = case(count)
    spatial.key_support.copy_(F.interpolate(geometry.projected.transpose(1,2).reshape(1,2,4,4), scale_factor=16, mode='nearest'))
    part = torch.zeros(1, 2, 16, 14)
    part[:,0,:8,2]=1; part[:,0,0]=0; part[:,0,0,0]=1
    part[:,1,8:,3]=1; part[:,1,8]=0; part[:,1,8,1]=1
    route = RoutingConditionV62(torch.tensor([[0]*8+[1]*8]), part,
        torch.zeros(1,14,14,dtype=torch.bool), torch.zeros(1,14,14,dtype=torch.bool),
        geometry.person_valid, geometry.target_token_hw, geometry.canvas_ids, (('a','b'),),
        ({'schema':ROUTING_SCHEMA,'target_canvas_id':geometry.canvas_ids[0]},)).validate(geometry)
    return target, joint, layout, geometry, spatial, route


@pytest.mark.parametrize('count', [2,4,5])
def test_all_global_keys_have_person_ownership_within_original_budget(count):
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    model = PoseLinkAttentionV62(config())
    _, joint, layout, geometry, spatial, routing = routing_case(count)
    keys = model.build_pose_keys(joint, layout, geometry, spatial, routing)
    assert (keys.person_ids[keys.valid]>=0).all()
    assert keys.parts.shape == (*keys.valid.shape,14)
    for ref in layout.references:
        selected = keys.reference_ids==ref.image_id
        assert int(selected.sum()) <= 16
        global_keys=selected & (keys.type_ids==0) & keys.valid
        if ref.role in ('combined_pose','target_pose'):
            assert (global_keys & (keys.person_ids==0)).any()
            assert (global_keys & (keys.person_ids==1)).any()


def test_owned_query_has_zero_direct_dependency_on_other_person_keys(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    model = PoseLinkAttentionV62(config()).eval(); activate(model)
    target, joint, layout, geometry, spatial, routing = routing_case()
    keys=model.build_pose_keys(joint,layout,geometry,spatial,routing)
    keys=replace(keys,hidden=keys.hidden.detach().clone().requires_grad_())
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:keys)
    args=dict(spatial=spatial,routing=routing,layer=0,sigma=torch.tensor([.5]))
    before=model(target,joint,layout,geometry,**args)
    before[:,:8].sum().backward()
    bkeys=(keys.person_ids==1)&keys.valid
    assert not keys.hidden.grad[bkeys].any()
    poisoned=keys.hidden.detach().clone();poisoned[bkeys]=torch.randn_like(poisoned[bkeys])*100
    modified=replace(keys,hidden=poisoned)
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:modified)
    after=model(target,joint,layout,geometry,**args)
    assert torch.equal(before[:,:8],after[:,:8])


def test_known_matching_contact_opens_only_its_query_and_key_parts():
    from src.pose_control.qwen21_full.native_v62.conditions import routing_permissions
    _,_,_,geometry,_,routing=routing_case()
    ids=torch.tensor([[0,1,1,1]])
    parts=torch.zeros(1,4,14);parts[0,0,0]=1;parts[0,1,1]=1;parts[0,2,3]=1
    valid=torch.ones(1,4,dtype=torch.bool)
    routing.contact_positive[0,0,1]=True
    allowed=routing_permissions(routing.query_person_ids,routing.query_parts,ids,parts,
        routing.contact_positive,routing.contact_forbidden,valid)
    assert allowed[0,0,1] and not allowed[0,0,2] and not allowed[0,0,3]
    assert not allowed[0,1:8,1:].any()
    routing.contact_forbidden[0,0,1]=True
    blocked=routing_permissions(routing.query_person_ids,routing.query_parts,ids,parts,
        routing.contact_positive,routing.contact_forbidden,valid)
    assert not blocked[0,0,1]


def test_other_person_pose_and_geometry_pixels_cannot_enter_owned_global_keys():
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    model=PoseLinkAttentionV62(config(global_token_budget=16,joint_token_budget=0,local_token_budget=0))
    _,joint,layout,geometry,spatial,routing=routing_case()
    joint.requires_grad_();geometry.geometry.requires_grad_()
    keys=model.build_pose_keys(joint,layout,geometry,spatial,routing)
    ref=next(ref for ref in layout.references if ref.role=='combined_pose')
    a=(keys.person_ids==0)&(keys.reference_ids==ref.image_id)&keys.valid
    (keys.hidden[a].sum()+keys.geometry[a].sum()).backward()
    assert not joint.grad[:,ref.indices[8:]].any()
    assert not geometry.geometry.grad[:,8:].any()


def test_no_legal_keys_zeroes_even_a_trained_output_bias(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    model=PoseLinkAttentionV62(config()).eval()
    target,joint,layout,geometry,spatial,routing=routing_case()
    keys=model.build_pose_keys(joint,layout,geometry,spatial,routing)
    keys=replace(keys,valid=keys.valid & (keys.person_ids==1))
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:keys)
    with torch.no_grad():model.output_heads['0'].bias.fill_(3)
    result=model(target,joint,layout,geometry,spatial=spatial,routing=routing,layer=0,sigma=torch.tensor([.5]))
    assert torch.isfinite(result).all() and not result[:,:8].any()


def test_routing_off_preserves_v61_exactly_with_trained_heads():
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    old=PoseLinkAttentionV61(config61()).eval();activate(old)
    new=PoseLinkAttentionV62(config(person_routing=False,body_spatial_gate=False,interaction_spatial_gate=False)).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    target,joint,layout,geometry,spatial=case()
    args=dict(spatial=spatial,layer=0,sigma=torch.tensor([.5]))
    assert torch.equal(old(target,joint,layout,geometry,**args),new(target,joint,layout,geometry,routing=None,**args))


def test_contact_ablation_does_not_open_positive_cross_person_edges(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    model=PoseLinkAttentionV62(config(use_contact=False)).eval();activate(model)
    target,joint,layout,geometry,spatial,routing=routing_case()
    routing.contact_positive[0,0,1]=True
    keys=model.build_pose_keys(joint,layout,geometry,spatial,routing)
    keys=replace(keys,hidden=keys.hidden.detach().clone().requires_grad_())
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:keys)
    model(target,joint,layout,geometry,spatial=spatial,routing=routing,layer=0,sigma=torch.tensor([.5]))[:,0].sum().backward()
    assert not keys.hidden.grad[(keys.person_ids==1)&keys.valid].any()


def test_diagnostics_are_scalar_only_and_do_not_change_outputs_or_rng():
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    from src.pose_control.qwen21_full.native_v62.observation import DiagnosticsV62
    model=PoseLinkAttentionV62(config()).eval();activate(model)
    target,joint,layout,geometry,spatial,routing=routing_case()
    kwargs=dict(spatial=spatial,routing=routing,layer=0,sigma=torch.tensor([.5]))
    before=torch.get_rng_state().clone()
    expected=model(target,joint,layout,geometry,**kwargs)
    observer=DiagnosticsV62()
    with observer.capture():actual=model(target,joint,layout,geometry,**kwargs)
    assert torch.equal(expected,actual) and torch.equal(before,torch.get_rng_state())
    report=observer.report();assert report['records'] and report['tensor_retention'] is False
    def check(value):
        assert not torch.is_tensor(value)
        if isinstance(value,dict):
            for x in value.values():check(x)
        elif isinstance(value,(tuple,list)):
            for x in value:check(x)
    check(report)
