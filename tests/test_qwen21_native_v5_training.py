"""Deferred training-management regressions. Authored only, not run at delivery.

All default optimizer tests use synthetic two-dimensional modules, not Adapter
weights. Real model acceptance requires an explicit external config.
"""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json,os,random
import numpy as np
import pytest
import torch
from torch import nn
from src.pose_control.qwen21_full.native_v5.training.config import (TrainingConfig,SourceSpec,RouteSpec,
    StageSpec,LossSpec,OptimizerSpec,stage_preset,digest)
from src.pose_control.qwen21_full.native_v5.training.parameters import ParameterCatalog,UpdateScheduler
from src.pose_control.qwen21_full.native_v5.training.sampling import DeterministicSampler


class ToyRuntime(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Module();self.base.body=nn.Module()
        self.base.body.output_heads=nn.ModuleList([nn.Linear(2,2,bias=False)])
        self.base.body.fusion=nn.Linear(2,2)
        self.face_preparer=nn.Linear(2,2);self.hand_preparer=nn.Linear(2,2)
        self.face_controller=nn.Module();self.face_controller.fusion=nn.Linear(2,2)
        self.face_controller.output_heads=nn.ModuleList([nn.Linear(2,2,bias=False)])
        self.hand_controller=nn.Module();self.hand_controller.output_heads=nn.ModuleList([nn.Linear(2,2,bias=False)])
        self.retired=nn.Parameter(torch.ones(2))
        self.inheritance_report={'inherited':{'hand_preparer.weight':'legacy'},'new_initialized':['face_preparer.weight','face_preparer.bias']}
        self.trainability={}
    def named_parameter_groups(self):
        groups={k:[] for k in ('native_body','face_encoder','hand_encoder','face_controller','face_heads','hand_heads','inactive')}
        for n,p in self.named_parameters():
            k='inactive' if n=='retired' else 'native_body' if n.startswith('base.') else 'face_encoder' if n.startswith('face_preparer.') else 'hand_encoder' if n.startswith('hand_preparer.') else 'hand_heads' if n.startswith('hand_controller.') else 'face_heads' if '.output_heads.' in n else 'face_controller'
            groups[k].append((n,p))
        return groups
    def set_trainability(self,profile,overrides=None):
        self.trainability=overrides or {}
        for k,rows in self.named_parameter_groups().items():
            for _,p in rows:p.requires_grad_(self.trainability.get(k,False))
    def forward(self,x):return self.face_controller.output_heads[0](self.face_controller.fusion(self.face_preparer(x)))
    def checkpoint_payload(self,metadata):return {'architecture':'synthetic_test_only','state_dict':self.state_dict(),'metadata':metadata}

def config(stage=None):
    return TrainingConfig(stage=stage or stage_preset('S1','face',budget=8),sources=(SourceSpec('train',weight=1.),))

def test_prepare_has_no_asset_model_or_optimizer_side_effects(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    report=NativeV5Trainer(TrainingConfig()).prepare()
    assert not report['training_ready'] and not report['model_constructed'] and not report['optimizer_created']
    assert 'initialization' in report['missing_bindings']

def test_cli_defaults_to_prepare_without_training(monkeypatch,capsys):
    from src.pose_control.qwen21_full.native_v5.training import cli
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    def forbidden(*a,**k):raise AssertionError('Implicit train')
    monkeypatch.setattr(NativeV5Trainer,'run',forbidden)
    cli.main([]);assert json.loads(capsys.readouterr().out)['action']=='prepare'

def test_unknown_controls_or_mixed_weights_are_rejected():
    with pytest.raises(ValueError):RouteSpec('bad','face',controls={'face_strenght':1})
    with pytest.raises(ValueError):StageSpec(phase='S1',routes=(RouteSpec('a','face'),RouteSpec('b','hand')))
    with pytest.raises(ValueError):TrainingConfig(sources=(SourceSpec('a'),SourceSpec('b')))
    with pytest.raises(ValueError):SourceSpec('champ_motions_example')

def test_config_roundtrip_and_gate_reference_not_circular():
    c=config();assert TrainingConfig.from_dict(c.record()).signature()==c.signature()
    changed=replace(c,gradient_report={'path':'/data/report.json','sha256':None})
    assert changed.signature()==c.signature()
    assert replace(c,stage=replace(c.stage,accumulation=2)).signature()!=c.signature()

def test_shared_heads_are_unique_and_inactive_excluded():
    catalog=ParameterCatalog(ToyRuntime());groups=catalog.group_spec(OptimizerSpec())
    names=[n for g in groups for n in g['names']]
    assert len(names)==len(set(names)) and 'retired' not in names
    active=catalog.active_names(stage_preset('S1','detail',budget=2))
    assert 'face_controller.output_heads.0.weight' in active and 'hand_controller.output_heads.0.weight' in active
    assert not any('fusion' in n for n in active)

def test_body_head_and_encoder_lineage_lr_classification():
    catalog=ParameterCatalog(ToyRuntime());rows={r.name:r for r in catalog.rows};o=OptimizerSpec()
    assert rows['base.body.output_heads.0.weight'].role=='head'
    assert catalog.base_lr(rows['base.body.output_heads.0.weight'],o)==1e-4
    assert catalog.base_lr(rows['face_preparer.weight'],o)==5e-5
    assert catalog.base_lr(rows['hand_preparer.weight'],o)==1e-5
    assert catalog.base_lr(rows['base.body.fusion.weight'],OptimizerSpec(preset='full_legacy'))==1e-5

def test_head_warmup_exact_frozen_mask():
    model=ToyRuntime();catalog=ParameterCatalog(model);active=catalog.apply(stage_preset('S1','face',budget=2))
    assert active=={'face_controller.output_heads.0.weight'}
    assert not model.face_preparer.weight.requires_grad and model.face_controller.output_heads[0].weight.requires_grad
    with pytest.raises(ValueError):catalog.apply(replace(stage_preset('S1','face'),include_groups=('inactive',)))

def test_stage_advance_retains_adam_moments_and_lazy_new_states():
    model=ToyRuntime();catalog=ParameterCatalog(model);s1=stage_preset('S1','face',budget=2);catalog.apply(s1)
    optimizer=catalog.build_optimizer(s1.optimizer)
    model(torch.ones(1,2)).square().sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    head=model.face_controller.output_heads[0].weight;before=optimizer.state[head]['exp_avg'].clone()
    s2=stage_preset('S2','face',budget=2);catalog.apply(s2);catalog.set_stage_lrs(optimizer,s2.optimizer)
    assert torch.equal(before,optimizer.state[head]['exp_avg']) and model.face_preparer.weight not in optimizer.state
    model(torch.ones(1,2)).square().sum().backward();optimizer.step()
    assert model.face_preparer.weight in optimizer.state

def test_frozen_parameters_with_existing_moments_do_not_decay():
    model=ToyRuntime();catalog=ParameterCatalog(model);stage=stage_preset('S2','face',budget=2);catalog.apply(stage)
    optimizer=catalog.build_optimizer(stage.optimizer)
    model(torch.ones(1,2)).square().sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    original=model.face_preparer.weight.detach().clone();state=deepcopy(optimizer.state[model.face_preparer.weight])
    catalog.apply(stage_preset('S1','face',budget=2));model(torch.ones(1,2)).square().sum().backward();optimizer.step()
    assert torch.equal(original,model.face_preparer.weight)
    assert torch.equal(state['exp_avg'],optimizer.state[model.face_preparer.weight]['exp_avg'])

def test_sampler_snapshot_is_independent_and_resumes_order():
    c=config();pools={('face','train'):[0,1,2]};s=DeterministicSampler(c,pools)
    s.draw();state=s.state_dict();expected=[s.draw() for _ in range(12)]
    restored=DeterministicSampler(c,pools);restored.load_state_dict(state)
    assert expected==[restored.draw() for _ in range(12)]

def test_sampler_rejects_changed_membership():
    s=DeterministicSampler(config(),{('face','train'):[0,1]})
    other=DeterministicSampler(config(),{('face','train'):[0,1,2]})
    with pytest.raises(ValueError):other.load_state_dict(s.state_dict())

def test_schedulers_advance_only_on_update_and_restore():
    model=ToyRuntime();catalog=ParameterCatalog(model);spec=OptimizerSpec(scheduler='warmup_constant',warmup_updates=4)
    optimizer=catalog.build_optimizer(spec);scheduler=UpdateScheduler(optimizer,spec,8)
    first=[g['lr'] for g in optimizer.param_groups];state=scheduler.state_dict()
    assert scheduler.completed==0 and first==[g['lr'] for g in optimizer.param_groups]
    scheduler.step();scheduler.load_state_dict(state)
    assert scheduler.completed==0 and first==[g['lr'] for g in optimizer.param_groups]

def test_accumulation_matches_effective_synthetic_batch():
    first=nn.Linear(2,2,bias=False);second=deepcopy(first);x=torch.randn(4,2);y=torch.randn(4,2)
    for i in range(4):((first(x[i:i+1])-y[i:i+1]).square().mean()/4).backward()
    (second(x)-y).square().mean().backward()
    torch.testing.assert_close(first.weight.grad,second.weight.grad)

def test_window_counts_do_not_dilute_missing_face_or_auxiliary():
    from src.pose_control.qwen21_full.native_v5.training.objective import counts_for_window
    route=RouteSpec('face','face',1.,loss=LossSpec(face=.5,identity=.1))
    entries=[{'route':'face','sigma':.2,'status':{'face_area':10.,'hand_area':0.,'auxiliary_faces':2}},
             {'route':'face','sigma':.8,'status':{'face_area':0.,'hand_area':0.,'auxiliary_faces':0}}]
    counts=counts_for_window(entries,{'face':route},4)
    assert counts=={'velocity':2,'face':1,'hand':0,'identity':2,'perceptual':0}

def eligibility_fixture():
    from importlib.util import spec_from_file_location,module_from_spec
    from types import SimpleNamespace
    spec=spec_from_file_location('deferred_v5_fixture',Path(__file__).with_name('test_qwen21_native_full_v5.py'))
    module=module_from_spec(spec);spec.loader.exec_module(module)
    c=module.condition();r=module.references()
    s=SimpleNamespace(paired=torch.tensor([True]),weights=c.region_masks.clone(),auxiliary_valid=torch.zeros(1,2,dtype=torch.bool))
    return {'local_condition_v4':c,'local_references_v4':r,'training_supervision_v5':s,'native_scene_kind':'single',
        'source_person_mask':torch.stack((torch.ones(1,64,64,dtype=torch.bool),torch.zeros(1,64,64,dtype=torch.bool))),
        'source_token_hw':(4,4),'target_token_hw':(4,4),'data_profile':'hybrid_fixture',
        'detail_references_v5':SimpleNamespace(valid=torch.zeros(1,6,dtype=torch.bool))}

def test_tiny_geometry_only_roi_is_not_head_training_eligible():
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    sample=eligibility_fixture();c=sample['local_condition_v4'];r=sample['local_references_v4']
    c.face_boxes[0,0]=torch.tensor([0.,0.,.25,.25]);c.face_landmarks[0,0,31]=torch.tensor([.1,.1,1.]);c.face_point_valid[0,0,31]=True
    r.arcface_valid[:]=False;r.face_dino_valid[:]=False
    result=sample_status(sample,RouteSpec('face','face',1.),config())
    assert result['cells'][0]==1 and not result['eligible']

def test_joint_keeps_paired_rows_without_local_modalities():
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    sample=eligibility_fixture();r=sample['local_references_v4'];r.arcface_valid[:]=False;r.face_dino_valid[:]=False
    result=sample_status(sample,RouteSpec('joint','joint',1.),config())
    assert result['eligible'] and not result['branch_usable']['face'] and not result['branch_usable']['interaction']
    assert not sample['local_condition_v4'].hand_pose_valid.any()

def test_rng_roundtrip_is_weights_only_safe(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import capture_rng,restore_rng
    state=capture_rng();path=tmp_path/'rng.pt';torch.save(state,path)
    expected=(random.random(),float(np.random.rand()),torch.rand(3))
    restore_rng(torch.load(path,weights_only=True))
    assert expected[0]==random.random() and expected[1]==float(np.random.rand()) and torch.equal(expected[2],torch.rand(3))

def test_journal_tail_is_preserved_and_new_segment_log_is_separate(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.io import Journal,verify_prefix
    journal=Journal(tmp_path/'first.jsonl');journal.append({'update':1});committed=journal.state_dict()
    journal.append({'update':2});original=journal.path.read_bytes();tail=verify_prefix(committed)
    next_journal=Journal(tmp_path/'second.jsonl');next_journal.append({'update':2,'resumed':True})
    assert journal.path.read_bytes()==original and tail['uncommitted_tail_bytes']>0

def test_corrupt_committed_prefix_is_rejected(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.io import Journal,verify_prefix
    journal=Journal(tmp_path/'metrics.jsonl');journal.append({'value':1});state=journal.state_dict()
    journal.path.write_bytes(journal.path.read_bytes().replace(b'"value":1',b'"value":2'))
    with pytest.raises(ValueError):verify_prefix(state)

def test_old_optimizer_only_or_uncommitted_file_is_not_resume(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v5.training.io import file_sha
    p=tmp_path/'old.pt';torch.save({'optimizer_state_dict':{}},p)
    with pytest.raises(ValueError,match='commit'):load_committed({'path':str(p),'sha256':file_sha(p)})

def test_snapshot_rejects_partial_accumulation_before_writing(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import save_snapshot
    with pytest.raises(ValueError,match='boundary'):
        save_snapshot(tmp_path,None,None,None,None,None,config=None,identity=None,catalog=None,active=None,
            global_update=0,stage_update=0,attempts=1,journal=None,micro_position=1,gradient_report=None,stage_origin=None)
    assert not list(tmp_path.iterdir())

def test_immutable_snapshot_and_commit_restore_named_training_state(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training import checkpointing as cp
    from src.pose_control.qwen21_full.native_v5.training.io import Journal
    monkeypatch.setattr(cp,'environment',lambda:{'synthetic_cpu':True})
    model=ToyRuntime();catalog=ParameterCatalog(model);c=config();active=catalog.apply(c.stage)
    optimizer=catalog.build_optimizer(c.stage.optimizer);scheduler=UpdateScheduler(optimizer,c.stage.optimizer,8)
    sampler=DeterministicSampler(c,{('face','train'):[0,1]});journal=Journal(tmp_path/'metrics.jsonl');journal.append({'update':0})
    ref=cp.save_snapshot(tmp_path/'checkpoints',model,optimizer,scheduler,sampler,torch.Generator().manual_seed(42),
        config=c,identity={'synthetic':True},catalog=catalog,active=active,global_update=0,stage_update=0,attempts=0,
        journal=journal,micro_position=0,gradient_report={'path':'/not_used'},stage_origin={'sha256':'synthetic'})
    original=Path(ref['checkpoint']['path']).read_bytes();journal.append({'uncommitted_attempt':1})
    saved,bound,tail=cp.load_committed(ref['checkpoint'])
    state=cp.validate_resume(saved,c,{'synthetic':True},catalog)
    assert state['active_names']==sorted(active) and tail['tail_preserved'] and tail['uncommitted_tail_bytes']>0
    assert Path(bound['path']).read_bytes()==original
    with pytest.raises(ValueError):cp.validate_resume(saved,replace(c,seed=c.seed+1),{'synthetic':True},catalog)

def test_continuous_and_reconstructed_toy_optimizer_trajectory_match():
    torch.manual_seed(789);model=ToyRuntime();c=config(stage_preset('S2','face',budget=6));catalog=ParameterCatalog(model);catalog.apply(c.stage)
    optimizer=catalog.build_optimizer(c.stage.optimizer);scheduler=UpdateScheduler(optimizer,c.stage.optimizer,6)
    sampler=DeterministicSampler(c,{('face','train'):[0,1,2]});noise=torch.Generator().manual_seed(42)
    def step(m,o,sc,sa,g):
        draw=sa.draw();x=torch.randn(1,2,generator=g);target=torch.randn(1,2,generator=g)
        o.zero_grad(set_to_none=True);(m(x)-target).square().mean().backward();o.step();sc.step();return draw,x,target
    for _ in range(3):step(model,optimizer,scheduler,sampler,noise)
    saved=deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'sampler':sampler.state_dict(),'noise':noise.get_state()})
    expected=[step(model,optimizer,scheduler,sampler,noise) for _ in range(3)]
    other=ToyRuntime();other.load_state_dict(saved['model']);other_catalog=ParameterCatalog(other);other_catalog.apply(c.stage)
    oo=other_catalog.build_optimizer(c.stage.optimizer);ss=UpdateScheduler(oo,c.stage.optimizer,6);oo.load_state_dict(saved['optimizer']);ss.load_state_dict(saved['scheduler'])
    sa=DeterministicSampler(c,{('face','train'):[0,1,2]});sa.load_state_dict(saved['sampler']);g=torch.Generator();g.set_state(saved['noise'])
    actual=[step(other,oo,ss,sa,g) for _ in range(3)]
    for a,b in zip(expected,actual):assert a[0]==b[0] and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
    for n,p in model.state_dict().items():assert torch.equal(p,other.state_dict()[n])

def test_real_gradient_gate_is_opt_in_and_has_no_optimizer(tmp_path):
    path=os.environ.get('NATIVE_V5_TRAINING_ACCEPTANCE_CONFIG')
    if not path:pytest.skip('Real asset/model acceptance is separately started with a bound config')
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    cfg=TrainingConfig.from_dict(json.loads(Path(path).read_text()))
    result=NativeV5Trainer(cfg).preflight_grad(tmp_path/'gradient.json')
    assert result['passed'] and not result['optimizer_created'] and not result['optimizer_step_executed']
