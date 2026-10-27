"""CPU contracts for two independent Native V5 appearance images."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA
from src.pose_control.qwen21_full.native_v5.contracts import NativeDetailReferencesV5, DETAIL_SCHEMA
from src.pose_control.qwen21_full.native_v5.conditioning import validate_pair, reference_inputs


def dual_pair():
    grids = [(2, 4), (4, 2), (4, 4), (4, 4), (4, 4)]
    refs = [torch.full((1, h*w, 64), float(i+1)) for i, (h, w) in enumerate(grids)]
    count = sum(x.shape[1] for x in refs)
    return dict(qwen_condition_profile='dual_source_5ref',
        reference_roles=['source_a','source_b','person_0_pose','person_1_pose','combined_pose'],
        reference_token_hws=grids, reference_latents=torch.cat(refs, 1), source_latent=refs[0],
        source_token_hw=grids[0], target_token_hw=(4,4),
        prompt_embeds=torch.zeros(1,count//4+2,16), prompt_mask=torch.ones(1,count//4+2,dtype=torch.bool),
        img_mask=torch.tensor([[False,False]+[True]*(count//4+4)]))


def source(index):
    from src.pose_control.qwen21_full.native_v5.source_bank import NativeAppearanceSourceV5
    spec=native_pad((32,64) if index==0 else (64,32));h,w=spec.canvas_hw
    meta={'source_rgb_sha256':f'rgb{index}','source_canvas_id':spec.fingerprint(),
        'source_person_ids':[f'person{index}','__inactive__']}
    r=NativeLocalReferencesV4(torch.full((1,2,1,512),float(index+1)),torch.tensor([[[True],[False]]]),
        torch.full((1,2,1,256,1536),float(index+1)),torch.tensor([[[True],[False]]]),
        torch.full((1,2,2,1,256,1536),float(index+1)),torch.tensor([[[[True],[True]],[[False],[False]]]]),
        torch.tensor([[[0.,0.,1.,1.],[0,0,0,0]]]),torch.tensor([[[[0.,0.,1.,1.]]*2,[[0.,0.,0.,0.]]*2]]),
        ({**meta,'schema':REFERENCE_SCHEMA,'role':'source_only','reference_count':1,
            'reference_sources':[{'rgb':{'sha256':f'rgb{index}'},'canvas_id':spec.fingerprint(),
                'geometry_sha256':f'geo{index}','person_ids':meta['source_person_ids']}]},)).validate()
    valid=torch.tensor([[True,False,True,True,False,False]])
    detail=NativeDetailReferencesV5(torch.tensor([[[0.,0.,1.,1.]]*6]),
        valid[...,None,None].expand(1,6,h,w).clone(),valid,torch.tensor([[0,-1]]),
        ({**meta,'schema':DETAIL_SCHEMA,'role':'primary_source_only'},)).validate()
    return NativeAppearanceSourceV5(index,(f'person{index}',),index,spec.metadata(),f'rgb{index}',f'geo{index}',
        torch.ones(1,1,1,h,w,dtype=torch.bool),torch.ones(1,1,1,h,w,dtype=torch.bool),r,detail)


def bank():
    from src.pose_control.qwen21_full.native_v5.source_bank import NativeSourceBankV5
    return NativeSourceBankV5((source(0),source(1)),torch.tensor([[0,1]]),torch.tensor([[0,0]]),
        (('person0','person1'),)).validate()


def test_five_references_keep_both_sources_and_three_pose_grids():
    p=validate_pair(dual_pair())
    inputs=reference_inputs(p,torch.zeros(1,16,64),torch.tensor([.5]))
    assert len(inputs['img_shapes'][0])==6
    assert inputs['img_shapes'][0][:2]==[(1,2,4),(1,4,2)]
    p['reference_roles'][0]='source'
    with pytest.raises(ValueError,match='role'): validate_pair(p)


def test_bank_binds_independent_latents_and_offline_features_without_compositing():
    p=dual_pair();b=bank().with_pair(p)
    assert b.sources[0].latent.shape==(1,8,64) and b.sources[1].latent.shape==(1,8,64)
    assert b.sources[0].latent.data_ptr()==p['reference_latents'].data_ptr()
    assert b.sources[1].latent.storage_offset()==8*64
    assert b.sources[0].token_hw==(2,4) and b.sources[1].token_hw==(4,2)
    refs=b.merged_references()
    assert (refs.arcface[:,0]==1).all() and (refs.arcface[:,1]==2).all()
    assert (refs.hand_dino[:,0]==1).all() and (refs.hand_dino[:,1]==2).all()


def test_mapping_rejects_skeleton_source_and_stable_identity_swap():
    b=bank()
    with pytest.raises(ValueError,match='source image'):
        replace(b,target_qwen_image_indices=torch.tensor([[0,2]])).validate()
    with pytest.raises(ValueError,match='stable'):
        replace(b,target_person_ids=(('person1','person0'),)).validate()
    with pytest.raises(ValueError,match='logical'):
        replace(b,sources=(b.sources[0],replace(b.sources[1],source_slot_id=0))).validate()


def test_offline_feature_sets_keep_their_own_reference_counts_and_validity():
    b=bank();r=b.sources[1].references
    fields={}
    for name in ('arcface','arcface_valid','face_dino','face_dino_valid','hand_dino','hand_dino_valid'):
        value=getattr(r,name);dim=3 if name.startswith('hand') else 2
        fields[name]=torch.cat([value]*3,dim)
    three=replace(r,**fields,metadata=({**r.metadata[0],'reference_count':3,
        'reference_sources':r.metadata[0]['reference_sources']*3,'explicit_multi_reference':True},))
    refs=replace(b,sources=(b.sources[0],replace(b.sources[1],references=three))).merged_references()
    assert refs.arcface.shape==(1,2,3,512)
    assert refs.arcface_valid[0,0].tolist()==[True,False,False]
    assert refs.arcface_valid[0,1].all() and refs.hand_dino_valid[0,1].all()


def test_source_geometry_cannot_be_rebound_to_an_unrelated_offline_roi():
    b=bank();s=b.sources[1];r=s.references;meta={**r.metadata[0],
        'reference_sources':[{**r.metadata[0]['reference_sources'][0],'geometry_sha256':'wrong_geometry'}]}
    with pytest.raises(ValueError,match='geometry'):
        replace(s,references=replace(r,metadata=(meta,))).validate()


def test_body_identity_rois_select_correct_image_with_different_native_grids():
    from src.pose_control.qwen21_full.native_dual_identity import NativeDualIdentityBinder
    from src.pose_control.qwen21_full.native_v5.frontend import identity_update
    b=bank().with_pair(dual_pair());model=NativeDualIdentityBinder();seen=[]
    hook=model.encoder.register_forward_pre_hook(lambda m,args:seen.append(float(args[0].mean())))
    state=SimpleNamespace(person_tokens=torch.zeros(1,2,4,512),
        person_token_mask=torch.ones(1,2,4,dtype=torch.bool),person_valid=torch.ones(1,2,dtype=torch.bool))
    result=identity_update(model,b,state,torch.ones(1,2,4),'roi48');hook.remove()
    assert seen==pytest.approx([1.,1.,2.,2.]) and result.shape==(1,4,512)


def test_detail_routes_faces_and_anatomical_hands_to_their_own_source():
    from src.pose_control.qwen21_full.native_v5.detail import NativeDetailAppearanceBridge
    b=bank().with_pair(dual_pair());model=NativeDetailAppearanceBridge();seen=[]
    c=SimpleNamespace(person_valid=torch.ones(1,2,dtype=torch.bool),
        region_valid=torch.ones(1,6,dtype=torch.bool),source_indices=torch.tensor([[0,1]]),source_slot_ids=torch.tensor([[0,1]]))
    hook=model.encoder.register_forward_pre_hook(lambda m,args:seen.append(float(args[0].mean())))
    result,valid=model.forward_bank(b,c);hook.remove()
    assert seen==pytest.approx([1.,2.,1.,1.,2.,2.])
    assert result.shape==(1,6,512,16,16) and valid.all()
    changed=replace(b,sources=(b.sources[0],replace(b.sources[1],latent=b.sources[1].latent+2)))
    second,_=model.forward_bank(changed,c)
    assert torch.equal(result[:,[0,2,3]],second[:,[0,2,3]])
    assert not torch.equal(result[:,[1,4,5]],second[:,[1,4,5]])
