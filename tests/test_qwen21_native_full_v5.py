"""Deferred V5 regressions. Authored only; do NOT run during static delivery."""
from dataclasses import replace
from pathlib import Path
import json,os
import pytest
import torch
from src.pose_control.qwen21_full.native_v5.contracts import (NativeConfigV5,NativeControlOptionsV5,
    NativeDetailReferencesV5,NativeTrainingSupervisionV5,PreparedLocalV5,DETAIL_SCHEMA,SUPERVISION_SCHEMA)
from src.pose_control.qwen21_full.native_v5.strength import Schedule,NativeStrengthController


def condition():
    from src.pose_control.qwen21_full.native_local_condition import NativeLocalConditionV4,LOCAL_SCHEMA
    z=lambda *s:torch.zeros(1,*s);m=lambda *s:torch.zeros(1,*s,dtype=torch.bool)
    c=NativeLocalConditionV4(torch.tensor([[True,False]]),torch.tensor([[0,-1]]),torch.tensor([[0,1]]),
        z(2,72,3),m(2,72),z(2,3),m(2),z(2,10),m(2),torch.tensor([[[0.,0.,1.,1.],[0,0,0,0]]]),torch.tensor([[True,False]]),
        z(2,2,45),m(2,2),z(2,2,21,3),m(2,2,21),z(2,2,1024),m(2,2),z(2,2,4),m(2,2),z(2,2),
        z(6,64,64),z(6,64,64),z(6,64,64),m(6,64,64),m(6,64,64),z(2,64,64),z(1,64,64),m(1,64,64),torch.tensor([[1.7,0.]]),
        ({'schema':LOCAL_SCHEMA,'expression_mode':'source','target_canvas_id':'target','source_canvas_id':'source','source_rgb_sha256':'rgb',
            'driver_record_sha256':'driver','camera_id':'camera','scale_group_id':'scale','normal_space':'camera_cv',
            'source_person_ids':['a','__inactive__'],'target_person_ids':['a','__inactive__'],'hand_pose_convention':'unavailable'},))
    c.region_masks[0,0]=1;c.visibility[0,0]=1;c.part_visible[0,0]=True;c.person_owner[0,0]=1
    return c.validate()


def references():
    from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4,REFERENCE_SCHEMA
    return NativeLocalReferencesV4(torch.randn(1,2,1,512),torch.tensor([[[True],[False]]]),
        torch.randn(1,2,1,256,1536),torch.tensor([[[True],[False]]]),torch.zeros(1,2,2,1,256,1536),
        torch.zeros(1,2,2,1,dtype=torch.bool),torch.zeros(1,2,4),torch.zeros(1,2,2,4),
        ({'schema':REFERENCE_SCHEMA,'role':'source_only'},)).validate()


def small_config(**kwargs):
    return NativeConfigV5(verification_only=True,hidden_dim=32,num_layers=6,injection_layers=(0,1,2,3,4,5),**kwargs)


def test_schedules_are_sigma_functions_and_no_call_counter():
    sigma=torch.tensor([1.,.75,0.]);s=Schedule('piecewise',knots=((0.,0.),(.5,1.),(1.,.5)))
    assert torch.equal(s.value(sigma),torch.tensor([0.,.5,.5]))
    assert torch.equal(s.value(sigma.flip(0)),s.value(sigma).flip(0))
    assert not Schedule('piecewise',knots=((0.,0.),(1.,0.))).potentially_active()
    assert not Schedule('constant',first=0.).potentially_active()


def test_group_mapping_and_legacy_priors():
    controller=NativeStrengthController(NativeConfigV5(strength_preset='legacy_full'))
    values=[float(controller('face_identity',l,torch.tensor([0.]),NativeControlOptionsV5())) for l in (5,10,15,21,26,31)]
    assert values==pytest.approx([.1,.2,.5,.8,1.,1.])
    assert not controller('face_texture',31,torch.tensor([1.]),NativeControlOptionsV5()).any()
    assert controller('body',0,torch.tensor([.8]),NativeControlOptionsV5()).item()==1.


def test_detail_and_family_gains_are_independent():
    options=NativeControlOptionsV5(detail_strength=0.,face_strength=.5,face_texture_gain=2.)
    assert options.strength('detail')==0 and options.strength('face_texture')==1.
    assert options.strength('hand_geometry')==1.
    with pytest.raises(ValueError): NativeControlOptionsV5(face_texture_gain=-1)


def test_face_raw_identity_does_not_contain_dino_delta():
    from src.pose_control.qwen21_full.native_v5.local import NativeFacePreparerV5
    c=condition();r=references();model=NativeFacePreparerV5().eval()
    first=model(c,r,geometry=False);second=model(c,replace(r,face_dino=r.face_dino+torch.randn_like(r.face_dino)),geometry=False)
    assert torch.equal(first[1][:,:,0],second[1][:,:,0])
    assert not torch.equal(first[1][:,:,1],second[1][:,:,1])
    assert first[2][0,0,1,:12].all()


def test_texture_off_skips_dino_and_perceiver(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.local import NativeFacePreparerV5
    model=NativeFacePreparerV5();c=condition();r=references()
    def forbidden(*a,**k): raise AssertionError('Disabled texture module called')
    monkeypatch.setattr(model.content_encoder.resampler,'forward',forbidden)
    monkeypatch.setattr(model.content_encoder.face_perceiver,'forward',forbidden)
    out=model(c,r,texture=False,geometry=False)
    assert out[2][0,0,0,:4].all() and not out[2][:,:,1].any()


def test_face_parameters_without_points_do_not_run_empty_spatial_cnn(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.local import NativeFacePreparerV5
    c=condition();c.jaw_valid[0,0]=True
    c=replace(c,metadata=({**c.metadata[0],'expression_mode':'driver'},))
    model=NativeFacePreparerV5()
    def forbidden(*a,**k):raise AssertionError('No landmarks for spatial CNN')
    monkeypatch.setattr(model.spatial_encoder,'forward',forbidden)
    out=model(c,references(),identity=False,texture=False)
    assert out[2][0,0,2,:4].all() and not out[3].any()


def test_hand_2d_does_not_enter_missing_3d_projectors(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.local import NativeHandPreparerV5
    c=condition();c.hand_boxes[0,0,0]=torch.tensor([0.,0.,1.,1.]);c.hand_roi_valid[0,0,0]=True
    c.hand_keypoints[0,0,0,:,:2]=.5;c.hand_keypoints[0,0,0,:,2]=1;c.hand_point_valid[0,0,0]=True
    model=NativeHandPreparerV5()
    def forbidden(*a,**k): raise AssertionError('Unknown pose treated as neutral')
    for name in ('mano_6d','mano_bps','handtype','combined'): monkeypatch.setattr(getattr(model.structure,name),'forward',forbidden)
    out=model(c,references());assert out[2][0,0,2,1] and not out[2][0,0,2,0]


def test_source_detail_mapping_and_mask_normalization():
    from src.pose_control.qwen21_full.native_v5.detail import NativeDetailAppearanceBridge
    c=replace(condition(),person_valid=torch.ones(1,2,dtype=torch.bool),source_indices=torch.tensor([[1,0]]),
        face_roi_valid=torch.ones(1,2,dtype=torch.bool),face_boxes=torch.tensor([[[0.,0.,1.,1.],[0.,0.,1.,1.]]]))
    latent=torch.ones(1,64,8,8);latent[:,:,:,4:]=3
    support=torch.zeros(1,6,128,128,dtype=torch.bool);support[:,0,:,:64]=True;support[:,1,:,64:]=True
    boxes=torch.zeros(1,6,4);boxes[0,0]=torch.tensor([0.,0.,.5,1.]);boxes[0,1]=torch.tensor([.5,0.,1.,1.])
    ref=NativeDetailReferencesV5(boxes,support,torch.tensor([[True,True,False,False,False,False]]),c.source_indices,
        ({'schema':DETAIL_SCHEMA,'role':'primary_source_only','source_rgb_sha256':'rgb','source_canvas_id':'source','source_person_ids':['a','b']},)).validate()
    bridge=NativeDetailAppearanceBridge();sampled=[]
    hook=bridge.encoder.register_forward_pre_hook(lambda m,args:sampled.append(float(args[0].mean())))
    out,valid=bridge(latent,ref,c);hook.remove()
    assert sampled==pytest.approx([3.,1.]) and out.shape==(1,6,512,16,16) and valid[0,:2].all()


def test_feather_is_inside_owned_support_only():
    from src.pose_control.qwen21_full.native_v5.detail import inner_feather
    weight=torch.zeros(1,6,64,64);weight[:,0,16:48,16:48]=1;boxes=torch.zeros(1,6,4);boxes[:,0]=torch.tensor([.25,.25,.75,.75])
    out=inner_feather(weight,boxes,8)
    assert (out<=weight).all() and not out[weight==0].any() and 0<out[0,0,16,16]<1


def test_nonzero_head_still_exactly_off_for_zero_modality_gains():
    from src.pose_control.qwen21_full.native_v5.local import NativeLocalControllerV5
    cfg=small_config();model=NativeLocalControllerV5(cfg)
    with torch.no_grad(): model.output_heads[0].weight.normal_(0,.01)
    p=PreparedLocalV5(torch.randn(1,6,512,16,16),torch.ones(1,6,dtype=torch.bool),torch.randn(1,6,4,16,512),
        torch.ones(1,6,4,16,dtype=torch.bool),torch.randn(1,6,512,16,16),torch.ones(1,6,dtype=torch.bool),
        torch.tensor([[[0.,0.,1.,1.]]*6]),torch.cat((torch.ones(1,1,64,64),torch.zeros(1,5,64,64)),1),
        torch.tensor([[True,False,False,False,False,False]]),({},)).validate()
    off=NativeControlOptionsV5(face_identity_gain=0.,face_texture_gain=0.,face_geometry_gain=0.,detail=False)
    result=model(torch.randn(1,16,32),p,layer=0,sigma=torch.tensor([.5]),target_hw=(4,4),regions=(0,1),strength=NativeStrengthController(cfg),options=off)
    assert not result.any()


def test_interaction_zero_head_first_gradient_and_upstream_probe():
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.native_v5.runtime import NativeInteractionResidualHead
    model=NativeInteractionResidualHead(small_config());q=torch.randn(1,8,512,requires_grad=True)
    prepared=SimpleNamespace(relation=torch.randn(1,8,512,requires_grad=True),relation_valid=torch.ones(1,8,dtype=torch.bool))
    out=model(q,prepared,layer=0,sigma=torch.tensor([.5]));assert not out.any()
    out.sum().backward();assert model.output_heads[0].weight.grad.abs().sum()>0
    assert prepared.relation.grad is None or not prepared.relation.grad.any()
    model.zero_grad(set_to_none=True);prepared.relation.grad=None
    with torch.no_grad(): model.output_heads[0].weight.normal_(0,.01)
    model(q,prepared,layer=0,sigma=torch.tensor([.5])).sum().backward()
    assert prepared.relation.grad.abs().sum()>0


def test_reused_body_query_matches_original_body_algebra():
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody,NativeBodyConfig,PreparedNativeGeometry
    from src.pose_control.qwen21_full.native_v5.integration import body_delta
    model=Qwen21NativeBody(NativeBodyConfig(hidden_dim=32,num_layers=6,injection_layers=(0,1,2,3,4,5)))
    with torch.no_grad(): model.output_heads[0].weight.normal_(0,.01)
    hidden=torch.randn(1,8,32);geometry=PreparedNativeGeometry(torch.randn(1,8,512),(2,4));sigma=torch.tensor([.5])
    original=model(hidden,prepared=geometry,layer=0,sigma=sigma,target_mask=torch.ones(8,dtype=torch.bool),return_delta=True)
    q=model.query_projection(model.query_norm(hidden));new=body_delta(model,q,geometry,0,sigma)
    torch.testing.assert_close(new,original)


def test_tiny_fractional_loss_is_area_normalized_and_empty_is_zero():
    from src.pose_control.qwen21_full.native_v5.losses import region_velocity
    squared=torch.full((1,4,64),4.)
    assert region_velocity(squared,torch.full((1,4),.0001)).item()==pytest.approx(4.)
    assert region_velocity(squared,torch.zeros(1,4)).item()==0.


def test_typed_sidecars_keep_fp32_bool_and_int64():
    ref=NativeDetailReferencesV5(torch.tensor([[[0.,0.,1.,1.]]*6]),torch.ones(1,6,64,64,dtype=torch.bool),
        torch.ones(1,6,dtype=torch.bool),torch.tensor([[0,1]]),
        ({'schema':DETAIL_SCHEMA,'role':'primary_source_only','source_rgb_sha256':'rgb','source_canvas_id':'source','source_person_ids':['a','b']},)).validate()
    selected=type(ref).stack([ref,ref]).to(dtype=torch.bfloat16).index_select(torch.tensor([1]))
    assert selected.boxes.dtype==torch.float32 and selected.support.dtype==torch.bool and selected.source_indices.dtype==torch.long
    replace(ref,source_indices=torch.tensor([[0,-1]])).validate()
    s=NativeTrainingSupervisionV5(torch.zeros(1,6,64,64),torch.zeros(1,2,dtype=torch.bool),torch.zeros(1,2,5,2),torch.tensor([True]),
        ({'schema':SUPERVISION_SCHEMA,'role':'training_only','driver_record_sha256':'driver','driver_rgb_sha256':'target',
          'target_canvas_id':'canvas','target_person_ids':['a','__inactive__']},)).validate()
    assert type(s).stack([s,s]).weights.shape==(2,6,64,64)


def test_quality_audit_does_not_invalidate_conditions():
    from src.pose_control.qwen21_full.native_v5.quality import audit_hands
    c=condition();old=c.hand_point_valid.clone();report=audit_hands(c)
    assert report['mode']=='audit_only' and torch.equal(old,c.hand_point_valid)


def test_offline_repair_rejects_3d_and_wrong_source():
    from src.pose_control.qwen21_full.native_v5.quality import import_repair
    c=condition();payload={'schema':'native_hand_repair_result_v5','role':'offline_geometry_repair','hand_pose':torch.zeros(45)}
    with pytest.raises(ValueError,match='cannot replace 3D'):
        import_repair(c,payload,body_wrist_uv=torch.zeros(1,2,2,2),trusted_person_support=torch.ones(1,2,64,64,dtype=torch.bool),repair_sha256='repair',trusted_support_sha256='support')


def test_missing_hand_training_evidence_requires_explicit_fresh():
    from src.pose_control.qwen21_full.native_v5.migration import migrate_v4_to_v5
    with pytest.raises(ValueError,match='allow_fresh_detail'): migrate_v4_to_v5({})
    with pytest.raises(ValueError,match='allow_fresh_hand'): migrate_v4_to_v5({},allow_fresh_detail=True)


def test_multi_reference_requires_explicit_same_identity_order(monkeypatch):
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_v5 import data
    c=condition();spec=native_pad((64,64));c=replace(c,metadata=({**c.metadata[0],'source_canvas_id':spec.fingerprint(),'driver_rgb_sha256':'target'},))
    r=references();sources=[{'rgb':{'path':'primary','sha256':'rgb'},'person_ids':c.metadata[0]['source_person_ids'],
        'geometry_sha256':'source_geometry','canvas':spec.metadata(),'canvas_id':spec.fingerprint()},
        {'rgb':{'path':'extra','sha256':'extra'},'person_ids':c.metadata[0]['source_person_ids'],
        'geometry_sha256':'extra_geometry','canvas':spec.metadata(),'canvas_id':spec.fingerprint()}]
    meta={**r.metadata[0],'reference_count':2,'reference_sources':sources,'source_rgb_sha256':'rgb',
        'source_canvas_id':spec.fingerprint(),'source_person_ids':c.metadata[0]['source_person_ids'],'explicit_multi_reference':True}
    r=replace(r,arcface=r.arcface.repeat(1,1,2,1),arcface_valid=r.arcface_valid.repeat(1,1,2),
        face_dino=r.face_dino.repeat(1,1,2,1,1),face_dino_valid=r.face_dino_valid.repeat(1,1,2),
        hand_dino=r.hand_dino.repeat(1,1,1,2,1,1),hand_dino_valid=r.hand_dino_valid.repeat(1,1,1,2),metadata=(meta,))
    monkeypatch.setattr(data,'verified',lambda ref:Path(ref['path']))
    data.validate_references(r,c)
    with pytest.raises(ValueError,match='explicit'): data.validate_references(replace(r,metadata=({**meta,'explicit_multi_reference':False},)),c)
    bad=[sources[0],{**sources[1],'person_ids':list(reversed(sources[1]['person_ids']))}]
    with pytest.raises(ValueError,match='identities/order'): data.validate_references(replace(r,metadata=({**meta,'reference_sources':bad},)),c)
    target=[sources[0],{**sources[1],'rgb':{'path':'target','sha256':'target'}}]
    with pytest.raises(ValueError,match='target RGB'): data.validate_references(replace(r,metadata=({**meta,'reference_sources':target},)),c)


def test_auxiliary_prediction_keeps_gradient_target_is_detached(monkeypatch):
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    from src.pose_control.qwen21_full.native_v5 import losses
    class Evaluator(nn.Module):
        def __init__(self): super().__init__();self.scale=nn.Parameter(torch.ones(6),requires_grad=False)
        def forward(self,image): return torch.cat((image.mean((2,3)),image.std((2,3))),1)*self.scale
    class VAE:
        dtype=torch.float32
        config=SimpleNamespace(latents_mean=[0.]*64,latents_std=[1.]*64)
        def decode(self,latent,return_dict=False): return (F.interpolate(latent[:,:3,0],size=(32,32)).tanh()[:,:,None],)
    pipe=SimpleNamespace(vae=VAE(),vae_scale_factor=16,
        _unpack_latents=lambda t,h,w,f:t.transpose(1,2).reshape(t.shape[0],64,1,h//16,w//16))
    def grid(points,hw,device=None):
        yy,xx=torch.meshgrid(torch.linspace(-.9,.9,112),torch.linspace(-.9,.9,112),indexing='ij')
        return torch.stack((xx,yy),-1)[None].to(device)
    monkeypatch.setattr(losses,'build_arcface_alignment_grid',grid)
    weight=torch.zeros(1,6,32,32);weight[:,0]=1
    points=torch.zeros(1,2,5,2);points[0,0]=torch.tensor([[8.,10.],[24.,10.],[16.,16.],[10.,24.],[22.,24.]])
    supervision=NativeTrainingSupervisionV5(weight,torch.tensor([[True,False]]),points,torch.tensor([True]),
        ({'schema':SUPERVISION_SCHEMA,'role':'training_only','driver_record_sha256':'driver','driver_rgb_sha256':'target',
          'target_canvas_id':'canvas','target_person_ids':['a','__inactive__']},)).validate()
    prediction=torch.randn(1,4,64,requires_grad=True);target=torch.full((1,3,32,32),.7,requires_grad=True)
    total,metrics=losses.auxiliary_loss(pipe,torch.zeros_like(prediction),prediction,torch.tensor([.1]),target,supervision,
        losses.LocalLossConfig('local_aux'),step=4,evaluators=(Evaluator(),Evaluator()))
    total.backward();assert prediction.grad.abs().sum()>0 and target.grad is None and metrics['eligible']==1


@pytest.fixture
def acceptance():
    path=os.environ.get('QWEN_V5_ACCEPTANCE_JOB')
    if not path: pytest.skip('Real asset/checkpoint acceptance must be explicitly started with a bound job')
    return json.loads(Path(path).read_text())


def test_v4_and_v5_strict_rebuild_are_distinct(acceptance):
    from src.pose_control.qwen21_full.native_full_runtime import NativeFullRuntimeV4
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    old=torch.load(acceptance['v4_checkpoint'],map_location='cpu',weights_only=True)
    legacy=NativeFullRuntimeV4.from_checkpoint(old)
    assert legacy.get_extra_state()==old['runtime_config']
    with pytest.raises(ValueError,match='V5 exact'): NativeFullRuntimeV5.from_checkpoint(old)
    saved=torch.load(acceptance['v5_checkpoint'],map_location='cpu',weights_only=True)
    new=NativeFullRuntimeV5.from_checkpoint(saved)
    assert new.get_extra_state()==saved['runtime_config']


def test_real_manifest_inference_omits_training_sidecar_and_rgb(acceptance):
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    sample=NativeFullDatasetV5(acceptance['v5_manifest'],load_target=False)[acceptance['index']]
    assert 'tgt_image' not in sample and 'training_supervision_v5' not in sample
