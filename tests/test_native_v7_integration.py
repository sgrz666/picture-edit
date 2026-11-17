"""Actual random Qwen blocks verify V7 target-only installation and recomputation."""
from copy import deepcopy
from dataclasses import replace
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tests.test_native_v62_integration import installed as installed62,control_case


@pytest.fixture(scope='module')
def installed(installed62):
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    from src.pose_control.qwen21_full.native_v7.poselink import PoseLinkAttentionV7
    from src.pose_control.qwen21_full.native_v7.local import NativeLocalControllerV7
    from src.pose_control.qwen21_full.native_v7.integration import install_native_full_v7
    bare,_,legacy,_,old=installed62
    runtime=NativeFullRuntimeV7.__new__(NativeFullRuntimeV7);nn.Module.__init__(runtime)
    runtime.config=old.config
    for name in ('base','interaction_head','strength'):setattr(runtime,name,deepcopy(getattr(old,name)))
    for name,hand in (('face_controller',False),('hand_controller',True)):
        controller=NativeLocalControllerV7(old.config,hand=hand)
        controller.load_state_dict(getattr(old,name).state_dict(),strict=False)
        setattr(runtime,name,controller.requires_grad_(False))
    runtime.pose_config=PoseLinkConfigV7.from_dict(old.pose_config.to_dict())
    runtime.poselink=PoseLinkAttentionV7(runtime.pose_config)
    runtime.poselink.load_state_dict(old.poselink.state_dict(),strict=False)
    model=deepcopy(bare);report=install_native_full_v7(model,runtime)
    assert report['architecture']=='native_full_v7' and report['new_qwen_tokens']==0
    yield bare,legacy,model.eval(),old,runtime


def bind(model,control,options,roles,**extra):
    from src.pose_control.qwen21_full.native_v7.integration import bind_native_full_v7
    return bind_native_full_v7(model,control,options=options,valid_mask=control.pose.coverage,
        reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles),**extra)


ROLES=[('source','target_pose'),('source','combined_pose','person_0_pose','person_1_pose'),
    ('source_a','source_b','person_0_pose','person_1_pose','combined_pose')]


@pytest.mark.parametrize('roles',ROLES)
def test_v7_zero_heads_are_exact_base_qwen_all_reference_modes(installed,roles):
    bare,_,model,_,_=installed;_,control,options,inputs=control_case(roles)
    with torch.no_grad():
        expected=bare(**inputs)[0]
        with bind(model,control,options,roles):actual=model(**inputs)[0]
    assert torch.equal(expected,actual)


@pytest.mark.parametrize('roles',ROLES)
def test_trained_v7_heads_change_only_target_hidden_and_keep_native_attention_objects(installed,roles):
    bare,_,model,_,runtime=installed;_,control,options,inputs=control_case(roles)
    traces={};handles=[]
    def capture(name):return lambda module,args,result:traces.__setitem__(name,result.detach().clone())
    attention_before=[block.attn for block in model.transformer_blocks]
    try:
        with torch.no_grad():
            runtime.poselink.output_heads['0'].weight.normal_(0,.005)
        handles.append(bare.transformer_blocks[0].register_forward_hook(capture('bare')))
        handles.append(model.transformer_blocks[0].register_forward_hook(capture('v7')))
        with torch.no_grad():
            bare(**inputs)
            with bind(model,control,options,roles):model(**inputs)
        target=model.build_token_metadata(torch.repeat_interleave(inputs['img_mask'][0],
            torch.where(inputs['img_mask'][0],4,1)),inputs['img_shapes'][0])[1]
        assert torch.equal(traces['bare'][:,~target],traces['v7'][:,~target])
        assert not torch.equal(traces['bare'][:,target],traces['v7'][:,target])
        assert all(block.attn is before for block,before in zip(model.transformer_blocks,attention_before))
    finally:
        for handle in handles:handle.remove()
        with torch.no_grad():runtime.poselink.output_heads['0'].weight.zero_()


def test_four_new_flags_off_match_trained_v62_inside_real_qwen(installed):
    from src.pose_control.qwen21_full.native_v62.integration import bind_native_full_v62
    _,legacy,model,old,new=installed;roles=ROLES[1];_,control,options,inputs=control_case(roles)
    previous=new.poselink.config
    try:
        new.poselink.config=replace(previous,mask_pose_fusion=False,spatial_rope=False,geometry_contact=False,identity_modulation=False)
        with torch.no_grad():
            for layer,head in old.poselink.output_heads.items():
                head.weight.normal_(0,.005);new.poselink.output_heads[layer].weight.copy_(head.weight)
            with bind_native_full_v62(legacy,control,options=options,valid_mask=control.pose.coverage,
                    reference_roles=roles,reference_canvas_ids=('canvas',)*len(roles)):
                expected=legacy(**inputs)[0]
            with bind(model,control,options,roles):actual=model(**inputs)[0]
        assert torch.equal(expected,actual)
    finally:
        new.poselink.config=previous
        with torch.no_grad():
            for runtime in (old,new):
                for head in runtime.poselink.output_heads.values():head.weight.zero_()


def test_checkpoint_recomputes_geometry_bridge_across_two_distinct_reference_forwards(installed):
    _,_,model,_,runtime=installed
    model.gradient_checkpointing=True;model._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)
    runtime.poselink.train();runtime.poselink.zero_grad(set_to_none=True)
    try:
        losses=[]
        for roles in (ROLES[0],ROLES[1]):
            _,control,options,inputs=control_case(roles)
            control.routing.contact_positive[:,0,1]=True
            with bind(model,control,options,roles):losses.append(model(**inputs)[0][:,-8:].square().mean())
        sum(losses).backward()
        for head in runtime.poselink.output_heads.values():
            assert head.weight.grad is not None and head.weight.grad.abs().sum()>0
        assert model._native_v62_execution.get() is None and model._native_v62_binding.get() is None
    finally:
        model.gradient_checkpointing=False;runtime.poselink.zero_grad(set_to_none=True);runtime.poselink.eval()
