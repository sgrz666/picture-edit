"""Deferred V4 regression cases. Authored only; NOT executed in architecture delivery."""
import pytest
import torch
from src.pose_control.qwen21_full.native_local_condition import NativeLocalConditionV4,NativeLocalReferencesV4,LOCAL_SCHEMA,REFERENCE_SCHEMA
from src.pose_control.qwen21_full.native_local_control import (resolve_local_ownership,prepare_local,
    NativeLocalConfigV4,NativeFaceController,schedule_value)


def condition():
    f=lambda *shape:torch.zeros(1,*shape)
    m=lambda *shape:torch.zeros(1,*shape,dtype=torch.bool)
    return NativeLocalConditionV4(torch.tensor([[True,False]]),torch.tensor([[0,-1]]),torch.tensor([[0,1]]),
        f(2,72,3),m(2,72),f(2,3),m(2),f(2,10),m(2),torch.tensor([[[0.,0.,.25,.25],[0,0,0,0]]]),torch.tensor([[True,False]]),
        f(2,2,45),m(2,2),f(2,2,21,3),m(2,2,21),f(2,2,1024),m(2,2),f(2,2,4),m(2,2),f(2,2),
        f(6,64,64),f(6,64,64),f(6,64,64),m(6,64,64),m(6,64,64),f(2,64,64),f(1,64,64),m(1,64,64),torch.tensor([[1.7,0.]]),
        ({'schema':LOCAL_SCHEMA,'expression_mode':'source','target_canvas_id':'target','source_canvas_id':'source',
          'driver_record_sha256':'driver','camera_id':'camera','scale_group_id':'scale','normal_space':'camera_cv',
          'source_person_ids':['a','__inactive__'],'target_person_ids':['a','__inactive__'],
          'hand_pose_convention':'full_axis_angle_after_pose_mean'},)).validate()


def references():
    return NativeLocalReferencesV4(torch.zeros(1,2,1,512),torch.zeros(1,2,1,dtype=torch.bool),
        torch.zeros(1,2,1,256,1536),torch.zeros(1,2,1,dtype=torch.bool),
        torch.zeros(1,2,2,1,256,1536),torch.zeros(1,2,2,1,dtype=torch.bool),torch.zeros(1,2,4),torch.zeros(1,2,2,4),
        ({'schema':REFERENCE_SCHEMA,'role':'source_only'},)).validate()


def test_independent_dtype_and_person_masks():
    c=condition();moved=c.to(dtype=torch.bfloat16).validate()
    assert moved.face_landmarks.dtype==torch.float32 and moved.person_valid.dtype==torch.bool
    assert moved.source_indices.dtype==torch.long
    assert NativeLocalConditionV4.stack([c,c]).index_select(torch.tensor([1,0])).person_valid.shape==(2,2)
    c.jaw_valid[0,1]=True
    with pytest.raises(ValueError,match='Inactive'): c.validate()


def test_unknown_pose_not_neutral_and_null_appearance():
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_hand import NativeHandPreparer
    c=condition();c=replace(c,hand_boxes=torch.tensor([[[[0.,0.,.5,.5],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]]]]),
        hand_roi_valid=torch.tensor([[[True,False],[False,False]]]))
    c.hand_keypoints[0,0,0,:,0:2]=.1;c.hand_keypoints[0,0,0,:,2]=1;c.hand_point_valid[0,0,0]=True
    output=NativeHandPreparer()(c.validate(),references())
    masks=output[2]
    assert not masks[0,0,0] and masks[0,0,1] and not masks[0,0,2:4].any()
    assert masks[0,0,4:12].all() and not masks[0,1:].any()


def test_z_occlusion_and_pixel_tie_no_double_strength():
    from dataclasses import replace
    c=condition();c=replace(c,hand_boxes=torch.tensor([[[[0.,0.,.25,.25],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]]]]),
        hand_roi_valid=torch.tensor([[[True,False],[False,False]]]))
    for r in (0,2):
        c.region_masks[0,r,:16,:16]=1;c.visibility[0,r]=1;c.local_z_valid[0,r,:16,:16]=True;c.local_z[0,r,:16,:16]=2
    c.wrist_z[0,0,0]=2
    c.person_owner[0,0]=1;c.scene_z[0,0,:16,:16]=2;c.scene_z_valid[0,0,:16,:16]=True
    w=resolve_local_ownership(c.validate())
    assert torch.equal(w[0,0,:16,:16],torch.full((16,16),.5)) and w.sum(1).max()<=1
    c.local_z[0,0,:16,:16]=3
    w=resolve_local_ownership(c.validate());assert not w[0,0].any() and w[0,2,:16,:16].min()==1


def test_local_off_does_not_call_encoders():
    def forbidden(*args): raise AssertionError('Local disabled')
    p=prepare_local(forbidden,forbidden,condition(),references(),face_enabled=False,hand_enabled=False)
    assert not p.region_valid.any() and not p.pixel_weights.any()


def test_schedule_depends_on_sigma_not_call_counter():
    s=torch.tensor([1.,.75,0.]);knots=((0.,0.),(.5,1.),(1.,.5))
    expected=torch.tensor([0.,.5,.5])
    assert torch.equal(schedule_value(s,knots),expected)
    assert torch.equal(schedule_value(s,knots),schedule_value(s,knots))


def test_small_model_requires_explicit_verification_header():
    with pytest.raises(ValueError): NativeLocalConfigV4(hidden_dim=32,num_layers=6,injection_layers=(0,1,2,3,4,5))
    config=NativeLocalConfigV4(verification_only=True,hidden_dim=32,num_layers=6,injection_layers=(0,1,2,3,4,5))
    assert config.to_dict()['verification_only'] and config.width==512


def test_sparse_queries_zero_head_and_tiny_roi():
    from src.pose_control.qwen21_full.native_local_condition import PreparedNativeLocalV4
    config=NativeLocalConfigV4();controller=NativeFaceController(config)
    weights=torch.zeros(1,6,64,64);weights[:,0,:16,:16]=1
    tokens=torch.randn(1,6,16,512);mask=torch.zeros(1,6,16,dtype=torch.bool);mask[:,0,:4]=True
    geometry=torch.zeros_like(mask)
    p=PreparedNativeLocalV4(torch.randn(1,6,512,16,16),tokens,mask,geometry,torch.ones(1,6,dtype=torch.bool),
        condition().boxes,weights,torch.tensor([[True,False,False,False,False,False]]),condition().metadata).validate()
    calls=[];hook=controller.query.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[1]))
    hidden=torch.randn(1,16,4096);delta=controller(hidden,p,layer=5,sigma=torch.tensor([.5]),target_hw=(4,4),regions=(0,1))
    hook.remove();assert calls==[1] and not delta.any()
    delta.sum().backward();assert controller.output_heads[0].weight.grad is not None
    assert not controller.output_heads[0].weight.data.any()


def test_body_optional_delta_keeps_prefix_and_zero_bits():
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    model=Qwen21NativeBody();hidden=torch.randn(1,20,4096,dtype=torch.bfloat16)
    target=torch.tensor([False]*4+[True]*16);features=NativeGeometryFeatures(torch.randn(1,16,512),torch.zeros(1,16,512),(4,4),torch.tensor([[True,False]]))
    p=model.prepare_geometry(features);kwargs=dict(prepared=p,layer=5,sigma=torch.tensor([.5]),target_mask=target)
    result=model(hidden,**kwargs);delta=model(hidden,**kwargs,return_delta=True)
    assert torch.equal(result,hidden) and torch.equal(result[:,:4],hidden[:,:4])
    assert delta.shape==(1,16,4096) and delta.dtype==torch.float32 and not delta.any()


def test_source_array_reorder_preserves_logical_identity():
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_face import gather_source
    c=condition();value=torch.tensor([[[1.,2.],[3.,4.]]])
    c=replace(c,source_slot_ids=torch.tensor([[3,7]]))
    reordered=replace(c,source_indices=torch.tensor([[1,-1]]),source_slot_ids=torch.tensor([[7,3]]))
    assert torch.equal(gather_source(value,c)[:,0],gather_source(value.flip(1),reordered)[:,0])
    assert torch.equal(gather_source(c.source_slot_ids,c)[:,0],gather_source(reordered.source_slot_ids,reordered)[:,0])


def test_source_role_and_no_learned_offline_cache(tmp_path):
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_full_assets import write_record
    refs=references()
    with pytest.raises(ValueError,match='source-only'):
        replace(refs,metadata=({'schema':REFERENCE_SCHEMA,'role':'driver'},)).validate()
    with pytest.raises(ValueError,match='learnable'):
        write_record(tmp_path/'forbidden.npz',torch.ones(1,requires_grad=True),schema='raw')
    assert not (tmp_path/'forbidden.npz').exists()
