"""Numerical and compatibility contracts for the production regional objective."""
from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch


def payload(n=64):
    counts={'person':2,'joint':2,'hands':4,'background':1}
    return {'coverage':torch.ones(1,n),
            'supports':{k:torch.zeros(r,n) for k,r in counts.items()},
            'quality':{k:torch.zeros(r) for k,r in counts.items()}}


def test_uniform_matches_native_value_and_gradient():
    from src.pose_control.qwen21_full.region_objective import regional_micro_loss
    from src.pose_control.qwen21_full.native_depth_runtime import native_velocity_loss
    p=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
    c=torch.tensor([[1.,.5]]); r=payload(2);r['coverage']=c
    result,parts=regional_micro_loss(p,torch.zeros_like(p),c,r,'uniform',{'velocity':1})
    expected=native_velocity_loss(p,torch.zeros_like(p),c)
    assert torch.equal(result,expected)
    assert torch.equal(torch.autograd.grad(result,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0])
    assert torch.equal(parts['velocity'],expected)


def test_independent_hands_and_window_denominator():
    from src.pose_control.qwen21_full.region_objective import regional_micro_loss, regional_counts
    r=payload();r['supports']['hands'][0,0]=1;r['supports']['hands'][1,1:17]=1;r['quality']['hands'][:2]=1
    status={'region_eligible':{'person':False,'joint':False,'hands':True,'background':False}}
    route=SimpleNamespace(loss=SimpleNamespace(region_mode='region_balanced_v1'))
    entries=[{'route':'joint','status':status}]+[{'route':'joint','status':{'region_eligible':{k:False for k in status['region_eligible']}}}]*3
    counts=regional_counts(entries,{'joint':route},{'velocity':4})
    assert counts['region_hands']==1
    p=torch.ones(1,64,2,requires_grad=True)
    _,parts=regional_micro_loss(p,torch.zeros_like(p),r['coverage'],r,'region_balanced_v1',counts)
    g=torch.autograd.grad(parts['region_hands'],p)[0].abs().sum(-1)[0]
    torch.testing.assert_close(g[:1].sum(),g[1:17].sum())


def test_quality_outside_mean_and_support_detached():
    from src.pose_control.qwen21_full.region_objective import regional_micro_loss
    p=torch.ones(1,64,2,requires_grad=True);r=payload()
    r['supports']['hands'][0,0]=1;r['quality']['hands'][0]=.1
    r['supports']['hands'].requires_grad_();r['quality']['hands'].requires_grad_()
    _,parts=regional_micro_loss(p,torch.zeros_like(p),r['coverage'],r,'region_balanced_v1',
        {'velocity':1,'region_person':0,'region_joint':0,'region_hands':1,'region_background':0})
    torch.testing.assert_close(parts['region_hands'],torch.tensor(.1))
    torch.testing.assert_close(parts['region_hands_effective_coefficient'],torch.tensor(.035))
    gp,gm,gq=torch.autograd.grad(parts['region_hands'],(p,r['supports']['hands'],r['quality']['hands']),allow_unused=True)
    assert gp.abs().sum()>0 and gm is None and gq is None


def test_empty_families_do_not_reallocate_to_global():
    from src.pose_control.qwen21_full.region_objective import regional_micro_loss
    p=torch.ones(1,64,2,requires_grad=True);r=payload()
    value,parts=regional_micro_loss(p,torch.zeros_like(p),r['coverage'],r,'region_balanced_v1',
        {'velocity':1,'region_person':0,'region_joint':0,'region_hands':0,'region_background':0})
    torch.testing.assert_close(value,torch.tensor(.1));assert parts['region_hands_count']==0


@pytest.mark.parametrize('version',['native_v5','native_v6'])
def test_existing_trainer_objective_replaces_velocity(version):
    import importlib
    objective=importlib.import_module('src.pose_control.qwen21_full.'+version+'.training.objective')
    from src.pose_control.qwen21_full.native_v5.training.config import LossSpec
    from src.pose_control.qwen21_full.native_v5.contracts import NativeTrainingSupervisionV5,SUPERVISION_SCHEMA
    metadata={'schema':SUPERVISION_SCHEMA,'role':'training_only','driver_record_sha256':'a'*64,
              'driver_rgb_sha256':'b'*64,'target_canvas_id':'c'*64,'target_person_ids':['p0',None]}
    supervision=NativeTrainingSupervisionV5(torch.zeros(1,6,16,32),torch.zeros(1,2,dtype=torch.bool),
        torch.zeros(1,2,5,2),torch.tensor([True]),(metadata,))
    r=payload(2);r['coverage']=torch.ones(1,2)
    spec=LossSpec(region_mode='region_balanced_v1')
    route=SimpleNamespace(loss=spec)
    status={'face_area':0,'hand_area':0,'auxiliary_faces':0,
            'region_eligible':{k:False for k in ('person','joint','hands','background')}}
    counts=objective.counts_for_window([{'route':'body','status':status,'sigma':.5}],{'body':route},1)
    p=torch.ones(1,2,1,requires_grad=True)
    value,parts=objective.compute_micro_loss(None,p,torch.zeros_like(p),torch.ones(1,2),supervision,
        torch.zeros_like(p),torch.tensor([.5]),torch.zeros(1,3,16,32),spec,counts,1,(None,None),spatial={'regional':r})
    torch.testing.assert_close(value,torch.tensor(.1))
    torch.testing.assert_close(parts['velocity'],torch.tensor(1.))


def test_region_config_keeps_legacy_signature_and_rejects_duplicates(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.config import LossSpec,TrainingConfig,digest
    c=TrainingConfig(model_path=str(tmp_path));old=asdict(c)
    for route in old['stage']['routes']:route['loss'].pop('region_mode',None)
    for source in old['sources']:source.pop('region_supervision',None)
    old.pop('gradient_report');old.pop('body_baseline_report')
    assert c.signature()==digest(old)
    for field in ('face','hand','spatial_body','spatial_joint','spatial_hand'):
        with pytest.raises(ValueError):LossSpec(region_mode='region_balanced_v1',**{field:.1})
    assert LossSpec(region_mode='uniform').region_mode=='uniform'


@pytest.mark.parametrize('bad',['negative','padding','quality_nan','missing_count'])
def test_bad_region_payload_fails(bad):
    from src.pose_control.qwen21_full.region_objective import regional_micro_loss
    r=payload();r['supports']['hands'][0,0]=1;r['quality']['hands'][0]=1
    counts={'velocity':1,'region_person':0,'region_joint':0,'region_hands':1,'region_background':0}
    if bad=='negative':r['supports']['hands'][0,0]=-.1
    elif bad=='padding':r['coverage'][0,0]=0
    elif bad=='quality_nan':r['quality']['hands'][0]=float('nan')
    else:counts['region_hands']=0
    with pytest.raises(ValueError):regional_micro_loss(torch.ones(1,64,2),torch.zeros(1,64,2),r['coverage'],r,'region_balanced_v1',counts)
