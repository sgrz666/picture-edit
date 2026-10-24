"""Deferred V4.1 regressions. Authored, NOT executed in the static delivery.

CPU fixtures below are synthetic. Asset/strict-checkpoint/Qwen acceptance uses
explicit future fixtures; no dataset discovery, inference, or downloads occur.
"""
from dataclasses import replace, fields, is_dataclass
from types import SimpleNamespace
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.canvas_contract import native_pad, letterbox
from src.pose_control.qwen21_full.native_hybrid_assets import (HybridConfig,canonical_points,
    assign_people,frame_key,ASSET_SCHEMA,import_frame,body_anchors,ANCHORS,file_ref,geometry_key)
from src.pose_control.qwen21_full.native_hybrid_local import (build_local,local_observations,empty_references)


def fixture():
    h,w=64,96;spec=native_pad((h,w));owner=torch.zeros(1,2,1,h,w)
    owner[:,0,:,:,:48]=1;owner[:,1,:,:,48:]=1
    part=torch.zeros(1,14,h,w);part[:,0,:24]=1
    people=[SimpleNamespace(context=SimpleNamespace(part_onehot=part.clone())) for _ in range(2)]
    dual=SimpleNamespace(model_family=torch.ones(1,2,dtype=torch.long),owner_weights=owner,
        person_a=people[0],person_b=people[1],person_depth_valid=torch.zeros(1,2,1,h,w,dtype=torch.bool),
        person_camera_z=torch.zeros(1,2,1,h,w),body_height=torch.tensor([[1.7,1.8]]))
    dual.validate=lambda:dual
    sample={'dual_depth':dual,'target_canvas':spec.metadata(),'source_canvas':spec.metadata(),
        'source_indices':torch.tensor([0,1]),'joint_uv':torch.tensor([[20.,30.]]*21),
        'joint_uv_b':torch.tensor([[70.,30.]]*21),'projection_valid':torch.ones(21,dtype=torch.bool),
        'projection_valid_b':torch.ones(21,dtype=torch.bool)}
    meta={'record_sha256':'geometry','rgb':{'path':'not-opened.jpg','sha256':'rgb'},
        'person_ids':['pair:0','pair:1'],'camera_id':'camera','scale_group_id':'scale'}
    asset={'xy_native':np.zeros((2,134,2),np.float32),'scores':np.zeros((2,134),np.float32),
        'point_valid':np.zeros((2,134),bool),'person_assigned':np.ones(2,bool),
        'metadata':{'schema':ASSET_SCHEMA,'original_hw':[h,w],'geometry_sha256':'geometry',
            'person_ids':meta['person_ids'],'assignment':[{'status':'matched'},{'status':'missing'}]}}
    masks=owner[0,:,0].numpy().astype(bool)
    return sample,asset,masks,meta,spec


def local_fixture():
    sample,asset,masks,meta,spec=fixture()
    local,report=build_local(sample,asset,masks,source_meta=meta,target_meta=meta,body_record_sha256='body')
    return local,sample,asset,masks,meta,spec


def test_explicit_133_conversion_and_anatomical_sides():
    xy=np.arange(266,dtype=np.float32).reshape(133,2);score=np.ones(133,np.float32)
    pts,s,v=canonical_points(xy,score,'coco_wholebody133')
    np.testing.assert_array_equal(pts[1],(xy[5]+xy[6])/2)
    np.testing.assert_array_equal(pts[92:113],xy[91:112])
    np.testing.assert_array_equal(pts[113:134],xy[112:133])
    with pytest.raises(ValueError): canonical_points(xy,score,'dwpose134')


def test_face68_last68_slots_and_rectangular_canvas_once():
    sample,asset,masks,meta,spec=fixture()
    asset['xy_native'][0,24:92]=np.stack((np.linspace(-8,24,68),np.linspace(4,20,68)),-1)
    asset['scores'][0,24:92]=1;asset['point_valid'][0,24:92]=True
    obs=local_observations(asset,letterbox((64,96),128),torch.ones(2,128,128,dtype=torch.bool),
        torch.ones(2,128,128,dtype=torch.bool),np.full((2,2,2),np.nan),HybridConfig())
    assert not obs['face_valid'][:,:4].any()
    expected=letterbox((64,96),128).points(asset['xy_native'][0,24:92]/[96,64])
    np.testing.assert_allclose(obs['face'][0,4:,:2],expected,atol=1e-7)
    assert obs['face'][0,4,0]<0  # never clamp to the ROI/canvas edge


def test_nonfinite_points_and_no_score_fabrication():
    xy=np.zeros((134,2),np.float32);xy[92]=np.nan;score=np.ones(134,np.float32)
    pts,s,v=canonical_points(xy,score,'dwpose134')
    assert not v[92] and s[92]==0 and np.isfinite(pts).all()


def test_camera_is_part_of_join():
    row={'pair':'p','action':'hug','camera_id':4,'frame_id':26}
    assert frame_key(row)!=frame_key({**row,'camera_id':64})


def test_v3_intrinsics_already_original_not_scaled_twice():
    joints=np.ones((2,22,3),np.float32);joints[...,2]=2
    data={'joints_cam':joints,'K':np.array([[100.,0,48],[0,100,32],[0,0,1]])}
    uv,valid=body_anchors(data,{'asset_to_original':[[2,0,0],[0,2,0],[0,0,1]]})
    np.testing.assert_allclose(uv[0,0],[98.,82.]);assert valid.all()


def test_assignment_injective_crossing_and_reordered_candidates():
    config=HybridConfig();masks=np.zeros((2,64,96),bool);masks[0,:,48:]=True;masks[1,:,:48]=True
    anchors=np.zeros((2,22,2),np.float32);anchors[0]=[70,30];anchors[1]=[20,30]
    av=np.ones((2,22),bool)
    candidates=[(np.tile([20.,30.],(134,1)),np.ones(134),np.ones(134,bool)),
                (np.tile([70.,30.],(134,1)),np.ones(134),np.ones(134,bool))]
    a,_=assign_people(candidates,masks,anchors,av,config)
    b,_=assign_people(candidates[::-1],masks,anchors,av,config)
    assert [r['candidate'] for r in a]==[1,0] and [r['candidate'] for r in b]==[0,1]
    duplicate,_=assign_people([candidates[0],candidates[0]],masks,anchors,av,config)
    assert all(r['candidate']==-1 for r in duplicate)


def test_assignment_ambiguous_not_a_negative_contact():
    masks=np.ones((2,64,96),bool);anchors=np.full((2,22,2),30.)
    candidates=[(np.full((134,2),30.),np.ones(134),np.ones(134,bool))]
    assignment,_=assign_people(candidates,masks,anchors,np.ones((2,22),bool),HybridConfig())
    assert all(r['status']!='matched' for r in assignment)
    assert all('contact' not in r for r in assignment)


@pytest.mark.parametrize('conflict,expected', [('camera','camera_source_conflict'),('rgb','rgb_source_conflict'),
    ('dimensions','native_grid_conflict'),('missing_manifest','missing_or_ambiguous_generation_row')])
def test_legacy_source_conflicts_keep_rejected_slots(tmp_path,conflict,expected):
    rgb=tmp_path/'rgb.bin';rgb.write_bytes(b'rgb')
    camera=tmp_path/'camera.bin';camera.write_bytes(b'camera')
    meta={'pair':'pair00','action':'hug00','camera_id':'hi4d:pair00:hug00:4','frame_id':26,
        'original_hw':[64,96],'asset_to_original':np.eye(3).tolist(),'model_family_indices':[1,1],
        'person_ids':['pair00:0','pair00:1'],'scale_group_id':'scale','record_sha256':'geometry',
        'rgb':file_ref(rgb),'sources':{'camera':file_ref(camera)}}
    row={'pair':'pair00','action':'hug00','camera_id':4,'frame_id':26,'rgb':str(rgb),'camera_file':str(camera),
        'person_mask_A':'not_opened_a','person_mask_B':'not_opened_b','person_A':0,'person_B':1,
        'dwpose_native_height':64,'dwpose_native_width':96,'dwpose_pre_crop':False,'dwpose_pre_resize':False}
    if conflict=='camera': meta['sources']['camera']={**meta['sources']['camera'],'sha256':'different'}
    if conflict=='rgb': meta['rgb']={**meta['rgb'],'sha256':'different'}
    if conflict=='dimensions': row['dwpose_native_width']=512
    rows={} if conflict=='missing_manifest' else {geometry_key(meta):[row]}
    data={'joints_cam':np.ones((2,22,3),np.float32),'K':np.eye(3,dtype=np.float32)}
    dataset=SimpleNamespace(geometry_record=lambda record:(data,meta))
    asset=import_frame(dataset,{'path':'geometry','sha256':'geometry'},rows,file_ref(rgb),'dwpose134',HybridConfig())
    assert not asset['person_assigned'].any() and len(asset['metadata']['assignment'])==2
    assert all(x['status']=='rejected' and x['reason']==expected for x in asset['metadata']['assignment'])


def test_untrusted_wrist_does_not_enable_hand_roi():
    sample,asset,masks,meta,spec=fixture()
    asset['xy_native'][0,92:113]=np.stack((np.linspace(5,20,21),np.linspace(8,20,21)),-1)
    asset['point_valid'][0,92:113]=True;asset['scores'][0,92:113]=1
    wrist=np.full((2,2,2),1000.,np.float32)
    obs=local_observations(asset,spec,torch.from_numpy(masks),torch.zeros(2,64,96,dtype=torch.bool),wrist,HybridConfig())
    assert not obs['hand_roi'].any() and obs['hand_valid'][0,0].all()


def test_missing_parameters_and_local_depth_remain_unavailable():
    local,*_=local_fixture()
    assert not any(getattr(local,k).any() for k in ('jaw_valid','expression_valid','hand_pose_valid','mesh_valid','local_z_valid'))
    assert local.metadata[0]['driver_record_sha256']=='geometry' and local.metadata[0]['body_record_sha256']=='body'
    assert local.metadata[0]['hand_pose_convention']=='unavailable'
    local.hand_pose_valid[0,0,0]=True
    with pytest.raises(ValueError): local.validate()


def test_no_mask_extension_and_unknown_local_overlap_disabled():
    sample,asset,masks,meta,spec=fixture()
    # Left hand projected over A's face: both supports lose contested pixels.
    asset['xy_native'][0,92:113]=np.stack((np.linspace(10,30,21),np.linspace(8,20,21)),-1)
    asset['scores'][0,92:113]=1;asset['point_valid'][0,92:113]=True
    sample['joint_uv'][19]=torch.tensor(asset['xy_native'][0,92])
    local,report=build_local(sample,asset,masks,source_meta=meta,target_meta=meta,body_record_sha256='body')
    effective=local.region_masks*local.visibility
    assert not effective[0,(0,2),:,48:].any()
    assert not ((effective[0,0]>0)&(effective[0,2]>0)).any()
    assert report['ambiguous_overlap_pixels']>0


def test_wholly_invalid_preserved_with_coverage_report():
    sample,asset,masks,meta,spec=fixture();masks[:]=False;asset['person_assigned'][:]=False
    local,report=build_local(sample,asset,masks,source_meta=meta,target_meta=meta,body_record_sha256='body')
    assert not local.region_valid.any() and report['support_pixels']==[0.]*6


def test_empty_references_explicit_and_dtype_roundtrip():
    local,sample,asset,masks,meta,spec=local_fixture()
    refs=empty_references(meta,spec,declaration='explicit_empty')
    assert not refs.arcface_valid.any() and not refs.face_dino_valid.any() and not refs.hand_dino_valid.any()
    with pytest.raises(ValueError): empty_references(meta,spec,declaration='missing_file')
    moved=local.to(dtype=torch.bfloat16).validate()
    assert moved.face_landmarks.dtype==torch.float32 and moved.person_valid.dtype==torch.bool and moved.source_indices.dtype==torch.long
    assert type(local).stack([local,local]).index_select(torch.tensor([1])).metadata==local.metadata


def forbidden(*args,**kwargs): raise AssertionError('Unavailable modality entered an encoder')


def test_face_parameter_and_identity_projections_not_called(monkeypatch):
    from src.pose_control.qwen21_full.native_face import NativeFacePreparer
    local,sample,asset,masks,meta,spec=local_fixture();module=NativeFacePreparer()
    for name in ('parameter_encoder','jaw_partial','expression_partial'):
        monkeypatch.setattr(getattr(module.geometry_tokenizer,name),'forward',forbidden)
    monkeypatch.setattr(module.content_encoder.fusion_face_id,'project_identity',forbidden)
    module(local,empty_references(meta,spec,declaration='explicit_empty'))


@pytest.mark.parametrize('identity,texture',[(True,False),(False,True),(False,False)])
def test_source_face_modalities_have_independent_token_masks(identity,texture):
    from src.pose_control.qwen21_full.native_face import NativeFacePreparer
    local,sample,asset,masks,meta,spec=local_fixture()
    refs=empty_references(meta,spec,declaration='explicit_empty')
    refs.arcface[0,0,0]=torch.randn(512);refs.face_dino[0,0,0]=torch.randn(256,1536)
    refs.arcface_valid[0,0,0]=identity;refs.face_dino_valid[0,0,0]=texture
    output=NativeFacePreparer()(local,refs)
    assert output[2][0,0,:4].all().item()==identity
    assert output[2][0,0,4:12].all().item()==texture


def test_hand_missing_3d_paths_not_called(monkeypatch):
    from src.pose_control.qwen21_full.native_hand import NativeHandPreparer
    local,sample,asset,masks,meta,spec=local_fixture();module=NativeHandPreparer()
    for name in ('mano_6d','mano_bps','combined','handtype'):
        monkeypatch.setattr(getattr(module.structure,name),'forward',forbidden)
    module(local,empty_references(meta,spec,declaration='explicit_empty'))


def test_source_only_guard_precedes_model_loading():
    from src.pose_control.qwen21_full.native_full_references import extract_source_references
    with pytest.raises(ValueError,match='Source only'):
        extract_source_references({'role':'source_only','target_rgb':'forbidden'})


def test_empty_source_extraction_never_imports_models(tmp_path,monkeypatch):
    import builtins
    from PIL import Image
    from src.pose_control.qwen21_full.native_full_assets import sha256
    from src.pose_control.qwen21_full.native_full_references import extract_source_references
    local,sample,asset,masks,meta,spec=local_fixture()
    rgb=tmp_path/'source.png';Image.new('RGB',(96,64)).save(rgb)
    real=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('onnxruntime','transformers'): raise AssertionError('Empty reference initialized a model')
        return real(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    person={'face_box':[0,0,.2,.2],'hand_boxes':[[0,0,0,0],[0,0,0,0]],
            'identity_valid':False,'face_texture_valid':False,'hand_texture_valid':[False,False]}
    job={'role':'source_only','modalities':{'arcface':True,'face_dino':False,'hand_dino':True},
        'source_person_ids':meta['person_ids'],'references':[{'rgb':{'path':str(rgb),'sha256':sha256(rgb)},
        'canvas':spec.metadata(),'person_ids':meta['person_ids'],'geometry_sha256':'geometry','people':[person,person]}]}
    refs=extract_source_references(job)
    assert refs.metadata[0]['models_loaded']=={'arcface':False,'dino':False}
    assert not refs.arcface_valid.any() and not refs.hand_dino_valid.any()


def test_hybrid_source_reordering_preserves_logical_id():
    local,sample,asset,masks,meta,spec=local_fixture()
    reordered_meta={**meta,'person_ids':meta['person_ids'][::-1]}
    sample={**sample,'source_indices':torch.tensor([1,0])}
    changed,_=build_local(sample,asset,masks,source_meta=reordered_meta,target_meta=meta,body_record_sha256='other_body')
    assert torch.equal(local.source_slot_ids.gather(1,local.source_indices),changed.source_slot_ids.gather(1,changed.source_indices))


def assert_raw_equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b) and a.dtype==b.dtype
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b);assert a.dtype==b.dtype
    elif is_dataclass(a):
        for f in fields(a): assert_raw_equal(getattr(a,f.name),getattr(b,f.name))
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: assert_raw_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_raw_equal(x,y)
    else: assert a==b


@pytest.fixture
def acceptance():
    path=os.environ.get('QWEN_V41_ACCEPTANCE_JOB')
    if path is None: pytest.skip('Later acceptance requires explicit V3/V4/Hybrid fixture paths')
    return json.loads(Path(path).read_text())


def test_v3_body_exact_raw_roundtrip(acceptance):
    from src.pose_control.qwen21_full.dual_data_runtime import NativeDualDataset
    from src.pose_control.qwen21_full.native_full_assets import read_record
    from src.pose_control.qwen21_full.native_full_data import BODY_SCHEMA
    original=NativeDualDataset(acceptance['v3_manifest'],load_target=False)[acceptance['index']]
    row=json.loads(Path(acceptance['hybrid_manifest']).read_text().splitlines()[acceptance['index']])
    cached=read_record(row['body'],schema=BODY_SCHEMA)
    for key,value in original.items():
        if key not in ('src_image','pair_metadata'): assert_raw_equal(value,cached[key])


def test_inference_does_not_open_target_rgb_and_profile_assertion(acceptance,monkeypatch):
    from src.pose_control.qwen21_full import native_full_data as data
    real=data._rgb;opened=[]
    def rgb(record,spec): opened.append(record['sha256']);return real(record,spec)
    monkeypatch.setattr(data,'_rgb',rgb)
    dataset=data.NativeFullDatasetV4(acceptance['hybrid_manifest'],load_target=False,data_profile='hybrid_hi4d')
    sample=dataset[acceptance['index']]
    assert opened==[dataset.rows[acceptance['index']]['source_rgb']['sha256']] and 'tgt_image' not in sample
    with pytest.raises(ValueError): data.NativeFullDatasetV4(acceptance['hybrid_manifest'],data_profile='full_smplx')


def test_counterfactual_supervision_refused_before_target_rgb(acceptance,tmp_path,monkeypatch):
    from src.pose_control.qwen21_full import native_full_data as data
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record
    from src.pose_control.qwen21_full.native_local_condition import LOCAL_SCHEMA
    row=json.loads(Path(acceptance['hybrid_manifest']).read_text().splitlines()[acceptance['index']])
    local=read_record(row['local'],schema=LOCAL_SCHEMA)
    local=replace(local,metadata=({**local.metadata[0],'counterfactual_geometry':True,'paired_rgb_compatible':False},))
    row['local']=write_record(tmp_path/'counterfactual.npz',local,schema=LOCAL_SCHEMA)
    manifest=tmp_path/'pairs.jsonl';manifest.write_text(json.dumps(row)+'\n')
    opened=[]
    def rgb(record,spec):
        opened.append(record['sha256']);return torch.zeros(3,*spec.canvas_hw)
    monkeypatch.setattr(data,'_rgb',rgb)
    with pytest.raises(ValueError,match='Counterfactual'): data.NativeFullDatasetV4(manifest,load_target=True)[0]
    assert opened==[row['source_rgb']['sha256']]


def test_legacy_v4_strict_load_and_local_off_v3(acceptance):
    from src.pose_control.qwen21_full.native_full_runtime import NativeFullRuntimeV4,FullControlOptionsV4
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    from src.pose_control.qwen21_full.native_full_data import NativeFullDatasetV4,collate_native_full_v4
    saved=torch.load(acceptance['v4_checkpoint'],map_location='cpu',weights_only=True)
    runtime=NativeFullRuntimeV4.from_checkpoint(saved).eval()
    sample=NativeFullDatasetV4(acceptance['hybrid_manifest'],load_target=False)[acceptance['index']]
    batch=collate_native_full_v4([sample]);base=DualControlOptions(identity=False)
    with torch.no_grad():
        full=runtime.encode(batch,target_token_hw=batch['target_token_hw'],options=FullControlOptionsV4(face=False,hand=False),base_options=base)
        old=runtime.base.encode(batch,target_token_hw=batch['target_token_hw'],options=base)
    assert_raw_equal(full.body,old)
