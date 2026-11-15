"""Real small Qwen blocks exercise routing, spatial gates and checkpoint binding."""
from copy import deepcopy
from dataclasses import asdict,replace
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tests.test_native_v6_integration import case
from tests.test_native_v61_poselink import spatial_for


@pytest.fixture(scope='module')
def installed():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig,Qwen21NativeBody
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_v5.local import NativeLocalControllerV5
    from src.pose_control.qwen21_full.native_v5.strength import NativeStrengthController
    from src.pose_control.qwen21_full.native_v6.runtime import NativeInteractionResidualHead
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v61.poselink import PoseLinkAttentionV61
    from src.pose_control.qwen21_full.native_v61.integration import install_native_full_v61
    from src.pose_control.qwen21_full.native_v62.contracts import PoseLinkConfigV62
    from src.pose_control.qwen21_full.native_v62.runtime import NativeFullRuntimeV62
    from src.pose_control.qwen21_full.native_v62.poselink import PoseLinkAttentionV62
    from src.pose_control.qwen21_full.native_v62.integration import install_native_full_v62
    previous_threads=torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(6201)
    cfg=NativeConfigV5(verification_only=True,hidden_dim=32,num_layers=6,injection_layers=tuple(range(6)))
    new=NativeFullRuntimeV62.__new__(NativeFullRuntimeV62);nn.Module.__init__(new)
    new.config=cfg;new.base=nn.Module()
    new.base.body=Qwen21NativeBody(NativeBodyConfig(hidden_dim=32,num_layers=6,injection_layers=tuple(range(6)))).requires_grad_(False)
    new.face_controller=NativeLocalControllerV5(cfg).requires_grad_(False)
    new.hand_controller=NativeLocalControllerV5(cfg,hand=True).requires_grad_(False)
    new.interaction_head=NativeInteractionResidualHead(cfg).requires_grad_(False)
    new.strength=NativeStrengthController(cfg).requires_grad_(False)
    new.pose_config=PoseLinkConfigV62(hidden_dim=32,width=32,heads=4,injection_layers=tuple(range(6)),
        query_chunk_size=3,max_pose_tokens_per_reference=16,global_token_budget=8,joint_token_budget=4,local_token_budget=4)
    new.poselink=PoseLinkAttentionV62(new.pose_config)
    old=NativeFullRuntimeV61.__new__(NativeFullRuntimeV61);nn.Module.__init__(old);old.config=cfg
    for name in ('base','face_controller','hand_controller','interaction_head','strength'):setattr(old,name,deepcopy(getattr(new,name)))
    old.pose_config=PoseLinkConfigV61.from_dict({k:v for k,v in new.pose_config.to_dict().items() if k in PoseLinkConfigV61.__dataclass_fields__})
    old.poselink=PoseLinkAttentionV61(old.pose_config);old.poselink.load_state_dict(new.poselink.state_dict())
    bare=QwenImage21Transformer2DModel(num_layers=6,attention_head_dim=16,num_attention_heads=2,
        context_in_dim=32,axes_dims_rope=(2,6,8))
    legacy,model=deepcopy(bare),deepcopy(bare)
    install_native_full_v61(legacy,old);install_native_full_v62(model,new)
    yield bare.eval(),legacy.eval(),model.eval(),old,new
    torch.set_num_threads(previous_threads)


def control_case(roles,*,body_gate=1.,interaction_gate=1.,options=None,half_coverage=False):
    from src.pose_control.qwen21_full.native_v61.prepared import PreparedNativeFullControlV61
    from src.pose_control.qwen21_full.native_v62.prepared import PreparedNativeFullControlV62
    from src.pose_control.qwen21_full.native_v62.contracts import RoutingConditionV62,BranchSpatialGatesV62,ROUTING_SCHEMA,GATE_SCHEMA
    old,default,inputs=case(roles);options=options or default
    spatial=spatial_for(old.pose)
    if half_coverage:
        spatial.valid_pixels[:,:,:,torch.arange(spatial.valid_pixels.shape[-1])%16>=8]=0
        spatial.key_support.mul_(spatial.valid_pixels)
        coverage=torch.nn.functional.avg_pool2d(spatial.valid_pixels,16).flatten(1)
        old=replace(old,pose=replace(old.pose,coverage=coverage))
    g=old.pose;b,n=g.coverage.shape
    ids=g.owner.argmax(-1).long();parts=torch.zeros(b,2,n,14)
    for person in range(2):parts[:,person,:,person]=g.owner[:,:,person]*g.coverage
    route=RoutingConditionV62(ids,parts,torch.zeros(b,14,14,dtype=torch.bool),torch.zeros(b,14,14,dtype=torch.bool),
        g.person_valid,g.target_token_hw,g.canvas_ids,(('a','b' if g.person_valid[0,1] else '__inactive__'),),
        ({'schema':ROUTING_SCHEMA,'target_canvas_id':'canvas'},)).validate(g)
    gates=BranchSpatialGatesV62(torch.full((b,n),float(body_gate)),torch.full((b,n),float(interaction_gate)),
        g.target_token_hw,g.canvas_ids,({'schema':GATE_SCHEMA,'target_canvas_id':'canvas',
            'body_gate_reason':'missing_repair_legacy_coverage','interaction_gate_reason':'body_support_union'},)).validate(g)
    metadata=({'options':asdict(options)},);core=replace(old.core,metadata=metadata)
    before=PreparedNativeFullControlV61(core,old.relation,old.relation_valid,old.local,metadata,g,spatial).validate()
    return before,PreparedNativeFullControlV62(core,old.relation,old.relation_valid,old.local,metadata,g,spatial,route,gates).validate(),options,inputs


def bind(model,control,options,roles,*,strength=1.):
    from src.pose_control.qwen21_full.native_v62.integration import bind_native_full_v62
    return bind_native_full_v62(model,control,options=options,valid_mask=control.pose.coverage,
        reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles),poselink_strength=strength)


@pytest.mark.parametrize('roles',[('source','target_pose'),('source','combined_pose','person_0_pose','person_1_pose'),
    ('source_a','source_b','person_0_pose','person_1_pose','combined_pose')])
def test_zero_heads_preserve_qwen_and_prefix_for_all_reference_modes(installed,roles):
    bare,_,model,_,_=installed
    _,control,options,inputs=control_case(roles)
    with torch.no_grad():
        expected=bare(**inputs)[0]
        with bind(model,control,options,roles):actual=model(**inputs)[0]
    assert torch.equal(expected,actual)


def test_all_new_features_off_matches_v61_with_trained_heads(installed):
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    _,legacy,model,old,new=installed;roles=('source','combined_pose','person_0_pose','person_1_pose')
    previous=new.poselink.config
    try:
        new.poselink.config=replace(previous,person_routing=False,body_spatial_gate=False,interaction_spatial_gate=False)
        with torch.no_grad():
            for family in ('body','poselink'):
                for a,b in zip(getattr(old,family).output_heads.values() if family=='poselink' else old.body.output_heads,
                               getattr(new,family).output_heads.values() if family=='poselink' else new.body.output_heads):
                    b.weight.normal_(0,.005);a.weight.copy_(b.weight)
        before,control,options,inputs=control_case(roles)
        with torch.no_grad(),bind_native_full_v61(legacy,before,options=options,valid_mask=before.pose.coverage,
                reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles)):
            expected=legacy(**inputs)[0]
        with torch.no_grad(),bind(model,control,options,roles):actual=model(**inputs)[0]
        assert torch.equal(expected,actual)
    finally:
        new.poselink.config=previous
        with torch.no_grad():
            for runtime in (old,new):
                for head in runtime.body.output_heads:head.weight.zero_()
                for head in runtime.poselink.output_heads.values():head.weight.zero_()


def test_body_and_interaction_gates_apply_once_with_partial_coverage(installed):
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5
    bare,_,model,_,runtime=installed;roles=('source','combined_pose','person_0_pose','person_1_pose')
    options=NativeControlOptionsV5(face=False,hand=False,detail=False,interaction=True)
    _,full,options,inputs=control_case(roles,options=options,half_coverage=True)
    full=replace(full,relation=torch.randn_like(full.relation),relation_valid=torch.ones_like(full.relation_valid))
    gated=replace(full,gates=replace(full.gates,body=torch.full_like(full.gates.body,.25),interaction=torch.full_like(full.gates.interaction,.25)))
    traces={};handles=[]
    def capture(name):return lambda module,args,result:traces.__setitem__(name,result.detach().clone())
    try:
        with torch.no_grad():
            for family in (runtime.body,runtime.interaction_head):
                for head in family.output_heads:head.weight.normal_(0,.005)
        handles.append(bare.transformer_blocks[0].register_forward_hook(capture('bare')))
        with torch.no_grad():bare(**inputs)
        handles.append(model.transformer_blocks[0].register_forward_hook(capture('full')))
        with torch.no_grad(),bind(model,full,options,roles,strength=0.):model(**inputs)
        handles[-1].remove();handles.pop()
        handles.append(model.transformer_blocks[0].register_forward_hook(capture('gated')))
        with torch.no_grad(),bind(model,gated,options,roles,strength=0.):model(**inputs)
        target=model.build_token_metadata(torch.repeat_interleave(inputs['img_mask'][0],torch.where(inputs['img_mask'][0],4,1)),inputs['img_shapes'][0])[1]
        first=(traces['full']-traces['bare'])[:,target];second=(traces['gated']-traces['bare'])[:,target]
        assert first.abs().sum()>0 and torch.allclose(second,first*.25,atol=2e-7,rtol=2e-4)
        assert torch.equal(traces['gated'][:,~target],traces['bare'][:,~target])
    finally:
        for handle in handles:handle.remove()
        with torch.no_grad():
            for family in (runtime.body,runtime.interaction_head):
                for head in family.output_heads:head.weight.zero_()


def test_checkpoint_binds_routing_and_gates_for_two_forwards(installed):
    _,_,model,_,runtime=installed;model.gradient_checkpointing=True
    model._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)
    runtime.poselink.train();runtime.poselink.zero_grad(set_to_none=True)
    try:
        losses=[]
        for roles in (('source','target_pose'),('source','combined_pose','person_0_pose','person_1_pose')):
            _,control,options,inputs=control_case(roles)
            with bind(model,control,options,roles):losses.append(model(**inputs)[0][:,-8:].square().mean())
        sum(losses).backward()
        assert all(head.weight.grad is not None and head.weight.grad.abs().sum()>0 for head in runtime.poselink.output_heads.values())
        assert model._native_v62_execution.get() is None and model._native_v62_binding.get() is None
    finally:
        model.gradient_checkpointing=False;runtime.poselink.zero_grad(set_to_none=True);runtime.poselink.eval()


def test_local_face_hand_and_detail_weights_are_not_multiplied_twice(installed):
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5,PreparedLocalV5
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    _,legacy,model,old,new=installed;roles=('source','combined_pose','person_0_pose','person_1_pose')
    options=NativeControlOptionsV5(body_strength=0.,interaction=False,face=True,hand=True,detail=True)
    before,control,options,inputs=control_case(roles,body_gate=0.,interaction_gate=0.,options=options)
    weights=torch.zeros(1,6,32,64);weights[:,0]=.3;weights[:,2]=.2
    region_valid=torch.tensor([[True,False,True,False,False,False]])
    tokens=torch.randn(1,6,4,16,512);token_valid=region_valid[:,:,None,None].expand(-1,-1,4,16).clone()
    local=PreparedLocalV5(torch.zeros(1,6,512,16,16),torch.zeros(1,6,dtype=torch.bool),tokens,token_valid,
        torch.randn(1,6,512,16,16),region_valid.clone(),torch.tensor([[[0.,0.,1.,1.]]*6]),weights,region_valid,({},)).validate()
    before=replace(before,local=local);control=replace(control,local=local)
    try:
        with torch.no_grad():
            for family in ('face_controller','hand_controller'):
                for a,b in zip(getattr(old,family).output_heads,getattr(new,family).output_heads):
                    b.weight.normal_(0,.005);a.weight.copy_(b.weight)
        with torch.no_grad(),bind_native_full_v61(legacy,before,options=options,valid_mask=before.pose.coverage,
                reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles),poselink_strength=0.):
            expected=legacy(**inputs)[0]
        with torch.no_grad(),bind(model,control,options,roles,strength=0.):actual=model(**inputs)[0]
        assert torch.equal(expected,actual)
    finally:
        with torch.no_grad():
            for runtime in (old,new):
                for family in ('face_controller','hand_controller'):
                    for head in getattr(runtime,family).output_heads:head.weight.zero_()


def test_block_diagnostics_include_body_fallback_and_preserve_rng(installed):
    from src.pose_control.qwen21_full.native_v62.observation import DiagnosticsV62
    _,_,model,_,_=installed;roles=('source','target_pose')
    _,control,options,inputs=control_case(roles)
    state=torch.get_rng_state().clone()
    with torch.no_grad(),bind(model,control,options,roles):expected=model(**inputs)[0]
    observer=DiagnosticsV62()
    with torch.no_grad(),observer.capture(),bind(model,control,options,roles):actual=model(**inputs)[0]
    assert torch.equal(expected,actual) and torch.equal(state,torch.get_rng_state())
    rows=[r for r in observer.report()['records'] if r.get('branch')=='body']
    assert rows and rows[0]['gate_reasons']==['missing_repair_legacy_coverage']


def test_prepared_records_stack_index_and_convert_without_losing_new_discrete_contracts():
    from src.pose_control.qwen21_full.native_v62.prepared import PreparedNativeFullControlV62
    _,control,_,_=control_case(('source','combined_pose','person_0_pose','person_1_pose'))
    leaf=control.gates.body.clone().requires_grad_()
    control=replace(control,gates=replace(control.gates,body=leaf))
    batch=PreparedNativeFullControlV62.stack([control,control]).to(dtype=torch.bfloat16).index_select(torch.tensor([1,0]))
    assert batch.validate() is batch
    assert batch.routing.query_person_ids.dtype==torch.long and batch.routing.contact_positive.dtype==torch.bool
    assert batch.routing.part_tokens.dtype==torch.float32 and batch.gates.body.dtype==torch.float32
    batch.gates.body.sum().backward();assert leaf.grad.eq(2).all()


def test_static_local_effects_are_not_suppressed_by_the_body_gate(installed):
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5,PreparedLocalV5
    from src.pose_control.qwen21_full.native_v61.integration import bind_native_full_v61
    bare,legacy,model,old,new=installed;roles=('source','combined_pose','person_0_pose','person_1_pose')
    options=NativeControlOptionsV5(interaction=False,face=True,hand=False,detail=False)
    before,control,options,inputs=control_case(roles,body_gate=0.,options=options)
    valid=torch.tensor([[True,False,False,False,False,False]])
    weights=torch.zeros(1,6,32,64);weights[:,0]=.3
    local=PreparedLocalV5(torch.zeros(1,6,512,16,16),torch.zeros(1,6,dtype=torch.bool),
        torch.randn(1,6,4,16,512),valid[:,:,None,None].expand(-1,-1,4,16).clone(),
        torch.zeros(1,6,512,16,16),torch.zeros_like(valid),torch.tensor([[[0.,0.,1.,1.]]*6]),weights,valid,({},)).validate()
    before=replace(before,local=local);control=replace(control,local=local)
    previous=old.config;static=replace(previous,local_mode='static');traces={};handles=[]
    def capture(name):return lambda module,args,result:traces.__setitem__(name,result.detach().clone())
    try:
        old.config=static;new.config=static
        legacy._native_v61_config=static;model._native_v62_config=static
        old.face_controller.config=static;new.face_controller.config=static
        with torch.no_grad():
            new.face_controller.static_output.weight.normal_(0,.01)
            old.face_controller.static_output.weight.copy_(new.face_controller.static_output.weight)
            for a,b in zip(old.body.output_heads,new.body.output_heads):b.weight.normal_(0,.01);a.weight.copy_(b.weight)
        handles.append(bare.transformer_blocks[0].register_forward_hook(capture('bare')))
        with torch.no_grad():bare(**inputs)
        handles.append(legacy.transformer_blocks[0].register_forward_hook(capture('body')))
        with torch.no_grad(),bind_native_full_v61(legacy,replace(before,local=None),options=options,valid_mask=before.pose.coverage,
                reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles),poselink_strength=0.):legacy(**inputs)
        handles[-1].remove();handles.pop()
        handles.append(legacy.transformer_blocks[0].register_forward_hook(capture('full')))
        with torch.no_grad(),bind_native_full_v61(legacy,before,options=options,valid_mask=before.pose.coverage,
                reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles),poselink_strength=0.):legacy(**inputs)
        handles.append(model.transformer_blocks[0].register_forward_hook(capture('full62')))
        full_control=replace(control,gates=replace(control.gates,body=torch.ones_like(control.gates.body)))
        with torch.no_grad(),bind(model,full_control,options,roles,strength=0.):model(**inputs)
        assert torch.equal(traces['full62'],traces['full'])
        handles[-1].remove();handles.pop()
        handles.append(model.transformer_blocks[0].register_forward_hook(capture('gated')))
        with torch.no_grad(),bind(model,control,options,roles,strength=0.):model(**inputs)
        local_effect=traces['full']-traces['body']
        assert local_effect.abs().sum()>0
        assert torch.allclose(traces['gated'],traces['bare']+local_effect,atol=3e-7,rtol=3e-4)
    finally:
        for handle in handles:handle.remove()
        old.config=previous;new.config=previous
        legacy._native_v61_config=previous;model._native_v62_config=previous
        old.face_controller.config=previous;new.face_controller.config=previous
        with torch.no_grad():
            for runtime in (old,new):
                runtime.face_controller.static_output.weight.zero_()
                for head in runtime.body.output_heads:head.weight.zero_()
