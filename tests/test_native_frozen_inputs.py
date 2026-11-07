"""Frozen encoding needs exact pixels/conditions, never full Adapter tensors."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,digest
from src.pose_control.qwen21_full.native_v5.training.io import file_sha


def bound(path):return {'path':str(path),'sha256':file_sha(path)}


def fixture(tmp_path):
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA,text_sha
    source=tmp_path/'source.png';target=tmp_path/'target.png';guide=tmp_path/'guide.png'
    Image.new('RGB',(31,45),'red').save(source);Image.new('RGB',(45,31),'blue').save(target)
    sc,tc=native_pad((45,31)),native_pad((31,45))
    Image.new('RGB',(tc.canvas_hw[1],tc.canvas_hw[0]),'gray').save(guide)
    bank=tmp_path/'bank.json';bank.write_text('{}')
    text='Use the source person and target pose.'
    spec={'schema':SCHEMA,'profile':'single_2ref','pair_id':'p','primary_source_index':0,
          'target_canvas_id':tc.fingerprint(),'images':[
          {'role':'source','file':bound(source),'canvas':sc.metadata(),'canvas_id':sc.fingerprint(),'space':'original'},
          {'role':'target_pose','file':bound(guide),'canvas':tc.metadata(),'canvas_id':tc.fingerprint(),'space':'canvas'}],
          'prompt':{'text':text,'sha256':text_sha(text),'field':'compiled_prompt_inference','bank':bound(bank)}}
    condition=tmp_path/'condition.json';condition.write_text(json.dumps(spec))
    v4=tmp_path/'v4.jsonl';v4row={'schema':'native_full_pair_v4','pair_id':'p','scene_kind':'single','data_profile':'full_smplx',
        'source_rgb':bound(source),'target_rgb':bound(target),'body':{'path':str(tmp_path/'must_not_decode.npz'),'sha256':'a'*64}}
    v4.write_text(json.dumps(v4row)+'\n')
    row={'schema':'native_full_pair_v5','pair_id':'p','v4_manifest':bound(v4),'detail':None,'supervision':None,
         'references':None,'qwen_condition':bound(condition)}
    manifest=tmp_path/'pairs.jsonl';manifest.write_text(json.dumps(row)+'\n')
    groups=tmp_path/'groups.json';groups.write_text(json.dumps({'p':'video'}))
    src=SourceSpec('train',role='train',manifest=bound(manifest),groups=bound(groups),scene_kind='single',data_profile='full_smplx')
    config=replace(TrainingConfig(),sources=(src,),qwen_condition_profile='single_2ref')
    identity={'sources':{'train':{'manifest':bound(manifest),'groups':bound(groups),'role':'train',
        'closure':[{k:row.get(k) for k in ('pair_id','v4_manifest','detail','supervision','references','qwen_condition')}] }},
        'config_sha256':'qualified_other_stage'}
    qualified=tmp_path/'qualified.json';qualified.write_text(json.dumps({'schema':'native_v5_data_preflight_v1','level':'data',
        'passed':True,'all_rows_retained':True,'identity':identity,'signature':digest(identity)}))
    return config,bound(qualified),sc,tc,v4row,spec,identity


def test_frozen_plan_reuses_qualification_without_loading_native_arrays(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    from src.pose_control.qwen21_full.native_full_data import _rgb
    c,q,sc,tc,v4,spec,identity=fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Full Native arrays must not be decoded')))
    plan=FrozenInputPlan(c,q)
    assert plan.identity['sources']==identity['sources']
    meta=plan.binding('train',0,'model','unused')
    assert meta=={'key':'train\0p','source_rgb_sha256':v4['source_rgb']['sha256'],
        'driver_rgb_sha256':v4['target_rgb']['sha256'],'source_canvas_id':sc.fingerprint(),'target_canvas_id':tc.fingerprint(),
        'source_token_hw':list(sc.token_hw),'target_token_hw':list(tc.token_hw),'model_manifest_sha256':'model',
        'condition_sha256':c.sources[0].manifest and plan.conditions[('train',0)][0]['sha256'],'prompt_sha256':spec['prompt']['sha256']}
    sample=plan.datasets['train'][0]
    assert torch.equal(sample['tgt_image'],_rgb(v4['target_rgb'],tc))
    assert sample['source_token_hw']==sc.token_hw and sample['target_token_hw']==tc.token_hw


def test_frozen_plan_rejects_failed_or_changed_qualification(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    c,q,*_=fixture(tmp_path)
    p=Path(q['path']);obj=json.loads(p.read_text());obj['passed']=False;p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='qualification'):FrozenInputPlan(c,bound(p))
    obj['passed']=True;obj['identity']['sources']['train']['role']='validation';p.write_text(json.dumps(obj))
    with pytest.raises(ValueError):FrozenInputPlan(c,bound(p))


def test_frozen_pixels_still_reject_changed_rgb(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    c,q,sc,tc,v4,*_=fixture(tmp_path);plan=FrozenInputPlan(c,q)
    with Path(v4['target_rgb']['path']).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='SHA'):plan.datasets['train'][0]


def test_lazy_plan_does_not_decode_all_conditions_before_first_batch(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    from src.pose_control.qwen21_full.native_v5 import conditioning
    c,q,*_=fixture(tmp_path);calls=[];original=conditioning.read_condition
    def counted(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(conditioning,'read_condition',counted)
    plan=FrozenInputPlan(c,q,lazy=True)
    assert not calls and not plan.metadata
    assert len(plan.condition_bindings)==1
    plan.binding('train',0,'model','prompt')
    assert calls and ('train',0) in plan.metadata


def test_body_projection_matches_original_real_status_formula():
    from test_qwen21_native_v5_training import eligibility_fixture
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    from src.pose_control.qwen21_full.native_v5.training.config import RouteSpec
    from src.pose_control.qwen21_full.native_v5.training.streaming import body_status
    sample=eligibility_fixture();c=TrainingConfig()
    joint=sample_status(sample,RouteSpec('joint','joint',1.),c)
    assert body_status(joint)==sample_status(sample,RouteSpec('body','body',1.),c)


def valid_pair_for(spec,sc,tc):
    grids=[sc.token_hw,tc.token_hw];n=sum(h*w for h,w in grids);th,tw=tc.token_hw
    ref=torch.zeros(1,n,64)
    return {'qwen_condition_profile':'single_2ref','reference_roles':['source','target_pose'],'reference_token_hws':grids,
        'reference_latents':ref,'source_latent':ref[:,:sc.token_hw[0]*sc.token_hw[1]],'target_latent':torch.ones(1,th*tw,64),
        'source_token_hw':sc.token_hw,'target_token_hw':tc.token_hw,'prompt_embeds':torch.zeros(1,n//4,8),
        'prompt_mask':torch.ones(1,n//4,dtype=torch.bool),'img_mask':torch.ones(1,(n+th*tw)//4,dtype=torch.bool),
        'prompt_sha256':spec['prompt']['sha256']}


def test_on_demand_encodes_once_preserves_rng_and_rejects_lost_used_entry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.native_v5.training.streaming import OnDemandCache
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    from src.pose_control.qwen21_full.native_v5.training import backend
    c,q,sc,tc,v4,spec,identity=fixture(tmp_path);c=replace(c,cache_root=str(tmp_path/'cache'))
    plan=FrozenInputPlan(c,q,lazy=True)
    prep={'model_manifest_sha256':'model','profile':'single_2ref','prompt_sha256':digest(c.prompt),'sources':plan.identity['sources']}
    store=FrozenCacheStore(c.cache_root,prep)
    descriptor={'model_manifest_sha256':'model','qualification':q,'entry_preparation':bound(Path(c.cache_root)/'preparation.json')}
    calls=[];moves=[]
    class Frozen(torch.nn.Linear):
        def to(self,device):moves.append(device);return self
    def load(config):
        torch.rand(3)
        return SimpleNamespace(text_encoder=Frozen(1,1),vae=Frozen(1,1))
    def encode(*a):calls.append(1);torch.rand(5);return valid_pair_for(spec,sc,tc)
    monkeypatch.setattr(backend,'load_cache_pipeline',load);monkeypatch.setattr(backend,'encode_frozen_pair',encode)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    manager=OnDemandCache(c,plan,descriptor);rng=torch.get_rng_state().clone()
    manager.ensure([('train',0),('train',0)])
    first=manager.read('train',0)
    assert len(calls)==1 and moves==['cpu','cpu'] and torch.equal(rng,torch.get_rng_state())
    state=manager.state();again=OnDemandCache(c,plan,descriptor,state)
    assert again.read('train',0)==first and not again.pipe
    # Simulate a missing commit without deleting any artifact.
    monkeypatch.setattr(again.store,'get',lambda binding:None)
    with pytest.raises(ValueError,match='not committed'):again.read('train',0)
    with pytest.raises(ValueError):OnDemandCache(c,plan,descriptor,{'schema':'bad','used':{}})


def test_checkpoint_binds_used_cache_state_and_rejects_signature_tampering(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.native_v5.training import checkpointing as cp
    from src.pose_control.qwen21_full.native_v5.training.io import Journal
    monkeypatch.setattr(cp,'environment',lambda:{'test':'cpu-only'})
    parameter=torch.nn.Parameter(torch.ones(1))
    runtime=SimpleNamespace(checkpoint_payload=lambda **k:{'state_dict':{'weight':parameter.detach().clone()}})
    optimizer=torch.optim.AdamW([parameter])  # No step is executed.
    journal=Journal(tmp_path/'metrics.jsonl');journal.append({'test':'checkpoint protocol'})
    state={'schema':'native_on_demand_used_cache_v1','used':{'train\0p':{'path':'/data/test-only.pt','sha256':'a'*64}}}
    result=cp.save_snapshot(tmp_path/'checkpoints',runtime,optimizer,SimpleNamespace(state_dict=lambda:{}),
        SimpleNamespace(state_dict=lambda:{}),torch.Generator(),config=TrainingConfig(),identity={},
        catalog=SimpleNamespace(record=lambda:[],signature=lambda:digest([])),active=set(),global_update=0,stage_update=0,
        attempts=0,journal=journal,micro_position=0,gradient_report=None,stage_origin=None,frozen_cache_state=state)
    saved,_,_=cp.load_committed(result['checkpoint'])
    assert saved['training_state']['frozen_cache_state']==state
    marker=Path(result['commit']['path']);value=json.loads(marker.read_text())
    assert value['frozen_cache_state_sha256']==digest(state)
    value['frozen_cache_state_sha256']='0'*64;marker.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='cache checkpoint state'):cp.load_committed(result['checkpoint'])
