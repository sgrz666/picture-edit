"""CPU contracts for V7 side attention and source-bound identity modulation."""
from dataclasses import replace

import pytest
import torch

from tests.test_native_v62_poselink import config as config62, routing_case
from tests.test_native_v6_poselink import activate


@pytest.fixture(scope='module',autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**changes):
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    return PoseLinkConfigV7.from_dict(config62().to_dict() | changes)


def attention(**changes):
    from src.pose_control.qwen21_full.native_v7.poselink import PoseLinkAttentionV7
    return PoseLinkAttentionV7(config(**changes)).eval()


def kwargs(spatial, routing):
    return dict(spatial=spatial, routing=routing, layer=0, sigma=torch.tensor([.5]))


@pytest.mark.parametrize('flag', ['mask_pose_fusion', 'spatial_rope', 'geometry_contact', 'identity_modulation'])
def test_new_flags_are_default_true_strict_bools_and_roundtrip(flag):
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    default=PoseLinkConfigV7()
    assert getattr(default,flag) is True
    assert (default.width,default.heads,len(default.injection_layers))==(512,8,6)
    assert PoseLinkConfigV7.from_dict(default.to_dict())==default
    for invalid in (0,1,None,'true'):
        with pytest.raises(ValueError,match=flag):config(**{flag:invalid})


def test_rotary_dimensions_require_two_even_axis_subspaces():
    with pytest.raises(ValueError,match='rotary|RoPE|axis'):config(width=12,heads=2)
    assert config(width=12,heads=2,spatial_rope=False).width==12


def test_geometry_contact_cannot_silently_bypass_person_routing():
    with pytest.raises(ValueError,match='geometry_contact.*person_routing'):
        config(person_routing=False)
    assert not config(person_routing=False,mask_pose_fusion=False,spatial_rope=False,geometry_contact=False).person_routing


@pytest.mark.parametrize('feature',['mask_pose_fusion','spatial_rope','geometry_contact'])
def test_routing_off_rejects_every_pose_feature_that_would_be_bypassed(feature):
    switches=dict(person_routing=False,mask_pose_fusion=False,spatial_rope=False,geometry_contact=False)
    with pytest.raises(ValueError,match=feature+'.*person_routing'):
        config(**(switches|{feature:True}))


def test_routing_off_accepts_identity_modulation_when_all_routed_pose_features_are_off():
    cfg=config(person_routing=False,mask_pose_fusion=False,spatial_rope=False,geometry_contact=False,
        identity_modulation=True)
    assert not cfg.person_routing and cfg.identity_modulation


def test_xy_rope_is_fp32_stable_relative_and_separates_canvas_axes():
    from src.pose_control.qwen21_full.native_v7.poselink import apply_xy_rope
    q=torch.tensor([[[[1.,0.,1.,0.]]]])
    origin=torch.zeros(1,1,2);x=torch.tensor([[[.5,0.]]]);y=torch.tensor([[[0.,.5]]])
    rq=apply_xy_rope(q,origin)
    rx=apply_xy_rope(q,x);ry=apply_xy_rope(q,y)
    assert torch.equal(rq,q)
    assert not torch.equal(rx[...,:2],q[...,:2]) and torch.equal(rx[...,2:],q[...,2:])
    assert torch.equal(ry[...,:2],q[...,:2]) and not torch.equal(ry[...,2:],q[...,2:])
    # Equal global translation leaves relative dot products invariant, also with unequal Q/K lengths.
    key=q.expand(1,1,3,4).clone();kxy=torch.tensor([[[.1,.2],[.7,.2],[.4,.8]]])
    offset=torch.tensor([[[.12,.08]]])
    first=apply_xy_rope(q,x)@apply_xy_rope(key,kxy).transpose(-1,-2)
    second=apply_xy_rope(q,x+offset)@apply_xy_rope(key,kxy+offset).transpose(-1,-2)
    assert torch.allclose(first,second,atol=1e-6)
    large=torch.full((1,1,3,8),20000.,dtype=torch.bfloat16)
    assert torch.isfinite(apply_xy_rope(large,kxy)).all()
    assert apply_xy_rope(large,kxy).dtype==large.dtype


def test_all_v7_flags_off_is_bit_exact_v62_with_trained_outputs():
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    old=PoseLinkAttentionV62(config62()).eval();activate(old)
    new=attention(mask_pose_fusion=False,spatial_rope=False,geometry_contact=False,identity_modulation=False)
    new.load_state_dict(old.state_dict(),strict=False)
    target,joint,layout,geometry,spatial,routing=routing_case()
    assert torch.equal(old(target,joint,layout,geometry,**kwargs(spatial,routing)),
        new(target,joint,layout,geometry,**kwargs(spatial,routing)))


@pytest.mark.parametrize('count',[2,4,5])
def test_zero_heads_preserve_zero_and_sample_budget(count):
    model=attention();target,joint,layout,g,s,r=routing_case(count)
    keys=model.build_pose_keys(joint,layout,g,s,r)
    assert keys.hidden.shape[1]<=len(layout.references)*config().max_pose_tokens_per_reference
    assert not model(target,joint,layout,g,**kwargs(s,r)).any()
    assert all(not h.weight.any() and not h.bias.any() for h in model.output_heads.values())


def test_mask_fusion_is_zero_initialized_trainable_and_person_scoped():
    model=attention();_,joint,layout,g,s,r=routing_case()
    samples=model.build_pose_keys(joint,layout,g,s,r)
    features=torch.randn(1,samples.hidden.shape[1],config().width,requires_grad=True)
    before=model._pose_key_features(features,samples)
    assert torch.equal(before,features)
    before.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.mask_pose_fusion.parameters())
    with torch.no_grad():
        for p in model.mask_pose_fusion.parameters():p.normal_(0,.05)
    model.zero_grad();features.grad=None
    after=model._pose_key_features(features,samples)
    own=(samples.person_ids==0)&samples.valid
    assert not torch.equal(after[own],features[own])
    after[own].sum().backward()
    assert not features.grad[(samples.person_ids==1)&samples.valid].any()
    invalid=replace(samples,valid=torch.zeros_like(samples.valid))
    assert torch.equal(model._pose_key_features(features,invalid),features)


def test_mask_fusion_is_equivariant_to_exchanging_person_slots():
    model=attention();_,joint,layout,g,s,r=routing_case()
    samples=model.build_pose_keys(joint,layout,g,s,r)
    features=torch.randn(1,samples.hidden.shape[1],config().width)
    with torch.no_grad():
        for parameter in model.mask_pose_fusion.parameters():parameter.normal_(0,.05)
    swapped=replace(samples,person_ids=torch.where(samples.person_ids>=0,1-samples.person_ids,samples.person_ids),
        owner=samples.owner.flip(-1),person_weights=samples.person_weights.flip(-1))
    assert torch.equal(model._pose_key_features(features,samples),model._pose_key_features(features,swapped))


def test_new_pose_and_contact_modules_receive_real_optimizer_updates_after_head_warmup():
    model=attention().train();target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    activate(model)
    selected=('mask_pose_fusion.mask_encoder.0.weight','mask_pose_fusion.output.weight',
        'contact_geometry_projection.1.weight','contact_value_projection.weight','output_heads.contact_0.weight')
    parameters=dict(model.named_parameters());before={name:parameters[name].detach().clone() for name in selected}
    optimizer=torch.optim.SGD(model.parameters(),lr=.2)
    desired=torch.randn_like(target)
    for _ in range(3):
        optimizer.zero_grad();value=model(target,joint,layout,g,**kwargs(s,r))
        (value-desired).square().mean().backward();optimizer.step()
    assert all(not torch.equal(before[name],parameters[name]) for name in selected)


def test_contact_does_not_relax_pose_hidden_person_routing(monkeypatch):
    model=attention();activate(model)
    target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    samples=model.build_pose_keys(joint,layout,g,s,r)
    samples=replace(samples,hidden=samples.hidden.detach().clone().requires_grad_())
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:samples)
    before=model(target,joint,layout,g,**kwargs(s,r))
    before[:,0].sum().backward()
    other=(samples.person_ids==1)&samples.valid
    assert not samples.hidden.grad[other].any()
    poisoned=samples.hidden.detach().clone();poisoned[other]=torch.randn_like(poisoned[other])*100
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:replace(samples,hidden=poisoned))
    assert torch.equal(before[:,:8],model(target,joint,layout,g,**kwargs(s,r))[:,:8])


@pytest.mark.parametrize('geometry_contact',[False,True])
def test_use_contact_false_disables_both_hidden_relaxation_and_geometry_bridge(monkeypatch,geometry_contact):
    model=attention(geometry_contact=geometry_contact,use_contact=False);activate(model)
    target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    samples=model.build_pose_keys(joint,layout,g,s,r)
    hidden=samples.hidden.detach().clone().requires_grad_();samples=replace(samples,hidden=hidden)
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:samples)
    model(target,joint,layout,g,**kwargs(s,r))[:,0].sum().backward()
    assert not hidden.grad[(samples.person_ids==1)&samples.valid].any()


def test_geometry_contact_is_part_matched_and_masks_unknown_forbidden_and_invalid_bias():
    model=attention();target,joint,layout,g,s,r=routing_case()
    with torch.no_grad():
        model.output_heads['contact_0'].weight.normal_(0,.1)
        model.output_heads['contact_0'].bias.fill_(3.)
    baseline=model(target,joint,layout,g,**kwargs(s,r));assert not baseline.any()
    r.contact_positive[0,0,1]=True
    positive=model(target,joint,layout,g,**kwargs(s,r))
    assert positive[:,0].abs().sum()>0 and positive[:,8].abs().sum()>0
    assert not positive[:,1:8].any() and not positive[:,9:].any()
    r.contact_forbidden[0,0,1]=True
    assert not model(target,joint,layout,g,**kwargs(s,r)).any()
    r.contact_forbidden.zero_();unknown=replace(r,query_person_ids=torch.full_like(r.query_person_ids,-1))
    assert not model(target,joint,layout,g,**kwargs(s,unknown)).any()
    model.config=replace(model.config,use_contact=False)
    assert not model(target,joint,layout,g,**kwargs(s,r)).any()


def test_contact_keys_use_geometry_only_and_all_invalid_masks_output_bias(monkeypatch):
    model=attention();target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    with torch.no_grad():
        model.output_heads['contact_0'].weight.normal_(0,.1)
        model.output_heads['contact_0'].bias.fill_(3.)
    samples=model.build_pose_keys(joint,layout,g,s,r)
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:samples)
    expected=model(target,joint,layout,g,**kwargs(s,r))
    contaminated=replace(samples,hidden=torch.randn_like(samples.hidden)*1000,
        interaction=torch.randn_like(samples.interaction)*1000 if samples.interaction is not None else None)
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:contaminated)
    assert torch.equal(expected,model(target,joint,layout,g,**kwargs(s,r)))
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:replace(samples,valid=torch.zeros_like(samples.valid)))
    assert not model(target,joint,layout,g,**kwargs(s,r)).any()


def test_contact_geometry_receives_gradient_only_for_matched_other_person(monkeypatch):
    model=attention();target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    with torch.no_grad():model.output_heads['contact_0'].weight.normal_(0,.1)
    samples=model.build_pose_keys(joint,layout,g,s,r)
    raw=samples.geometry.detach().clone().requires_grad_()
    own=samples.person_tokens.detach().clone().requires_grad_()
    hidden=samples.hidden.detach().clone().requires_grad_()
    samples=replace(samples,geometry=raw,person_tokens=own,hidden=hidden)
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:samples)
    model(target,joint,layout,g,**kwargs(s,r))[:,0].square().sum().backward()
    matched=(samples.parts[...,1]>config().routing_epsilon)&(samples.person_ids==1)&samples.valid
    assert raw.grad[matched].abs().sum()>0
    assert not raw.grad[~matched].any()
    assert own.grad[matched][:,1].abs().sum()>0 and not own.grad[matched][:,0].any()
    assert hidden.grad is None or not hidden.grad.any()


def test_contact_bridge_requires_query_and_key_support_and_both_people(monkeypatch):
    model=attention();target,joint,layout,g,s,r=routing_case();r.contact_positive[0,0,1]=True
    with torch.no_grad():model.output_heads['contact_0'].bias.fill_(3.)
    assert model(target,joint,layout,g,**kwargs(s,r))[:,0].abs().sum()>0
    samples=model.build_pose_keys(joint,layout,g,s,r)
    without_support=replace(samples,person_weights=torch.zeros_like(samples.person_weights))
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:without_support)
    assert not model(target,joint,layout,g,**kwargs(s,r)).any()
    monkeypatch.setattr(model,'build_pose_keys',lambda *args:samples)
    unsupported=replace(g,owner=torch.zeros_like(g.owner),projected=torch.zeros_like(g.projected))
    assert not model(target,joint,layout,unsupported,**kwargs(s,r)).any()


def local_case():
    from src.pose_control.qwen21_full.native_v5.contracts import PreparedLocalV5,NativeConfigV5
    from src.pose_control.qwen21_full.native_v7.local import NativeLocalControllerV7
    cfg=NativeConfigV5(verification_only=True,hidden_dim=16,num_layers=6,injection_layers=tuple(range(6)))
    valid=torch.ones(1,6,dtype=torch.bool);mask=valid[:,:,None,None].expand(-1,-1,4,16).clone()
    p=PreparedLocalV5(torch.zeros(1,6,512,16,16),torch.zeros_like(valid),torch.randn(1,6,4,16,512),mask,
        torch.zeros(1,6,512,16,16),torch.zeros_like(valid),torch.tensor([[[0.,0.,1.,1.]]*6]),
        torch.ones(1,6,32,32)/6,valid,({},)).validate()
    return NativeLocalControllerV7(cfg),p


@pytest.mark.parametrize('region,person',[(0,0),(1,1),(2,0),(3,0),(4,1),(5,1)])
def test_identity_query_modulation_uses_own_face_identity_and_no_other_person_gradient(region,person):
    model,p=local_case();q=torch.randn(1,3,512)
    assert torch.equal(model._modality_query(q,p,0,region,1),q)
    with torch.no_grad():model.identity_modulation.output.weight.normal_(0,.02)
    tokens=p.tokens.detach().clone().requires_grad_();p=replace(p,tokens=tokens)
    changed=model._modality_query(q,p,0,region,1)
    assert not torch.equal(changed,q)
    changed.square().sum().backward()
    assert tokens.grad[:,person,0].abs().sum()>0
    assert not tokens.grad[:,1-person].any()
    assert not tokens.grad[:,:,1:].any()
    assert torch.equal(model._modality_query(q,p,0,region,2),q)
    invalid=p.token_valid.clone();invalid[:,person,0]=False
    assert torch.equal(model._modality_query(q,replace(p,token_valid=invalid),0,region,1),q)
    model.identity_modulation_enabled=False
    assert torch.equal(model._modality_query(q,p,0,region,1),q)


def test_local_controller_identity_hook_changes_actual_update_and_can_be_optimized():
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5
    from src.pose_control.qwen21_full.native_v5.local import NativeLocalControllerV5
    from src.pose_control.qwen21_full.native_v5.strength import NativeStrengthController
    model,p=local_case();old=NativeLocalControllerV5(model.config)
    with torch.no_grad():model.output_heads[0].weight.normal_(0,.02)
    old.load_state_dict({n:v for n,v in model.state_dict().items() if not n.startswith('identity_modulation.')},strict=True)
    hidden=torch.randn(1,4,16);args=dict(layer=0,sigma=torch.tensor([.5]),target_hw=(2,2),regions=(0,),
        strength=NativeStrengthController(model.config),options=NativeControlOptionsV5(hand=False,detail=False))
    before=model(hidden,p,**args)
    assert torch.equal(before,old(hidden,p,**args))
    optimizer=torch.optim.SGD(model.identity_modulation.parameters(),lr=.2)
    optimizer.zero_grad();before.square().mean().backward();optimizer.step()
    assert not torch.equal(before,model(hidden,p,**args))
    model.identity_modulation_enabled=False
    assert torch.equal(model(hidden,p,**args),old(hidden,p,**args))
