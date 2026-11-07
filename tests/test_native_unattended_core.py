"""CPU-only contracts for unattended production training safety."""
from dataclasses import replace
from types import SimpleNamespace
from collections import Counter
import json
import pytest
import torch
from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,stage_preset


def test_config_accepts_platform_independent_absolute_bindings_only():
    c=TrainingConfig()
    assert c.gradient_case_budget is None and c.validation_loss_limit_ratio is None
    assert replace(c,gradient_case_budget=32,validation_loss_limit_ratio=2).validation_failure_patience==3
    assert TrainingConfig.from_dict(c.record()).signature()==c.signature()
    replace(c,initialization={'path':'/data/shangguanrz/model.pt'},output_root='/data/shangguanrz/run')
    with pytest.raises(ValueError,match='Absolute'):replace(c,model_path='models/qwen')
    for kw in ({'gradient_case_budget':0},{'validation_loss_limit_ratio':float('nan')},{'validation_failure_patience':0}):
        with pytest.raises(ValueError):replace(c,**kw)


def selection_data():
    rows=[];groups={};dataset_rows=[]
    for g in range(5):
        for i in range(8):
            pair=f'{g}_{i}';groups[pair]=str(g)
            dataset_rows.append({'pair_id':pair,'target_rgb_sha256':f'target_{g}_{i//2}'})
            rows.append({'source_id':'val','role':'validation','index':len(dataset_rows)-1,'pair_id':pair,'route':'body','eligible':True})
    rows.extend({'source_id':'train','role':'train','index':i,'pair_id':f'train{i}','route':'body','eligible':True} for i in range(11010))
    return SimpleNamespace(records=rows,group_indices={'val':groups},datasets={'val':SimpleNamespace(rows=dataset_rows)})


def test_selection_is_bounded_group_balanced_validation_only_and_stable():
    from src.pose_control.qwen21_full.native_v5.training.evaluation import fixed_evaluation_selection
    c=replace(TrainingConfig(),probe_samples=10);d=selection_data()
    first=fixed_evaluation_selection(c,d)
    assert len(first)==10 and {r['source_id'] for r in first}=={'val'}
    assert Counter(d.group_indices['val'][r['pair_id']] for r in first)=={str(g):2 for g in range(5)}
    assert len({d.datasets['val'].rows[r['index']]['target_rgb_sha256'] for r in first})==10
    assert first==fixed_evaluation_selection(c,d)


def test_validation_guard_preserves_baseline_and_patience_across_resume():
    from src.pose_control.qwen21_full.native_v5.training.evaluation import ValidationGuard
    c=replace(TrainingConfig(),validation_loss_limit_ratio=2)
    guard=ValidationGuard(c,selection=[{'source_id':'val','index':0}])
    def report(loss,update):return {'update':update,'records':[{'objective':loss}]}
    guard.observe(report(1.,0));guard.observe(report(3.,100))
    restored=ValidationGuard(c,state=json.loads(json.dumps(guard.state)))
    restored.observe(report(3.,200))
    assert restored.state['baseline_loss']==1. and restored.state['consecutive_failures']==2
    with pytest.raises(RuntimeError,match='validation'):restored.observe(report(3.,300))
    assert restored.state['triggered'] and not restored.state['passed']


def test_parent_randomness_continues_after_stage_reconstruction():
    import random
    import numpy as np
    from src.pose_control.qwen21_full.native_v5.training import checkpointing
    noise=torch.Generator().manual_seed(789)
    parent={'rng':checkpointing.capture_rng(),'noise_generator_state':noise.get_state()}
    expected=(random.random(),float(np.random.rand()),torch.rand(3),torch.rand(3,generator=noise))
    # Module construction and a fresh-stage generator must not restart the streams.
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    next_noise=torch.Generator().manual_seed(42)
    checkpointing.restore_training_randomness(parent,next_noise)
    actual=(random.random(),float(np.random.rand()),torch.rand(3),torch.rand(3,generator=next_noise))
    assert actual[:2]==expected[:2]
    assert torch.equal(actual[2],expected[2]) and torch.equal(actual[3],expected[3])


def test_validation_guard_fails_nonfinite_immediately_and_resets_after_recovery():
    from src.pose_control.qwen21_full.native_v5.training.evaluation import ValidationGuard
    c=replace(TrainingConfig(),validation_loss_limit_ratio=2)
    g=ValidationGuard(c)
    g.observe({'update':0,'records':[{'objective':1.}]})
    g.observe({'update':100,'records':[{'objective':3.}]})
    g.observe({'update':200,'records':[{'objective':1.5}]})
    assert g.state['consecutive_failures']==0 and g.state['passed']
    with pytest.raises(RuntimeError,match='validation'):g.observe({'update':300,'records':[{'objective':float('nan')}]})
    assert not g.state['finite']


def test_optional_frozen_components_are_absent_in_gradient_audit():
    from src.pose_control.qwen21_full.native_v5.training.validation import frozen_digest,audit_gradients
    class Runtime(torch.nn.Linear):
        def get_extra_state(self):return {'test':True}
    runtime=Runtime(1,1);runtime.requires_grad_(False)
    backend=SimpleNamespace(runtime=runtime,active=set(),pipe=SimpleNamespace(vae=None,text_encoder=None,transformer=torch.nn.Linear(1,1)),
        evaluators=[],base_state_names=set(),base_parameters={},catalog=SimpleNamespace(assert_flags=lambda x:None,parameters={},rows=[]))
    assert frozen_digest(backend)['frozen_auxiliary']
    assert audit_gradients(backend,require_any=False)=={}


def test_pipeline_loads_only_components_for_the_requested_work(monkeypatch):
    import sys
    from src.pose_control.qwen21_full.native_v5.training import backend
    calls=[]
    class Pipeline:
        @classmethod
        def from_pretrained(cls,*args,**kw):calls.append(kw);return object()
    monkeypatch.setitem(sys.modules,'diffusers',SimpleNamespace(QwenImage21Pipeline=Pipeline))
    backend.load_training_pipeline(TrainingConfig())
    backend.load_cache_pipeline(TrainingConfig())
    assert calls[0]['text_encoder'] is None and calls[0]['vae'] is None and 'transformer' not in calls[0]
    assert calls[1]['transformer'] is None and 'text_encoder' not in calls[1] and 'vae' not in calls[1]
    assert all(c['device_map']=='cuda' and c['low_cpu_mem_usage'] and c['dtype']==torch.bfloat16 for c in calls)


def test_lazy_vae_uses_exact_qwen21_class(monkeypatch):
    import sys
    from src.pose_control.qwen21_full.native_v5.training.backend import NativeBackend
    calls=[]
    class VAE(torch.nn.Linear):
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            calls.append((args,kwargs));return cls(1,1)
    monkeypatch.setitem(sys.modules,'diffusers',SimpleNamespace(AutoencoderKLQwenImage21=VAE))
    backend=object.__new__(NativeBackend)
    backend.config=SimpleNamespace(model_path='/model/qwen21')
    backend.pipe=SimpleNamespace(vae=None)
    backend.frozen_component_digests={}
    loaded=backend.ensure_vae()
    assert len(calls)==1 and calls[0][1]['subfolder']=='vae'
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert backend.ensure_vae() is loaded


def test_evaluation_only_plan_rejects_any_training_source():
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan
    with pytest.raises(ValueError,match='evaluation_only'):
        DataPlan(TrainingConfig(sources=(SourceSpec('train'),)),evaluation_only=True)


def body_fixture(tmp_path,monkeypatch,*,updates=10000,gradient_overrides=None):
    from src.pose_control.qwen21_full.native_v5.training import checkpointing as cp
    from src.pose_control.qwen21_full.native_v5.training.config import digest
    from src.pose_control.qwen21_full.native_v5.training.io import Journal,write_json
    from src.pose_control.qwen21_full.native_v5.training.evaluation import ValidationGuard
    monkeypatch.setattr(cp,'environment',lambda:{'test_cpu':True})
    auth={'source':'direct_user_message','text':'S0 10000 then S1/S2/S3 unattended'}
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=10000),validation_loss_limit_ratio=2,continuous_stage_authorization=auth)
    identity={'data':{'sources':{'train':{'manifest':'bound'}}},'model_manifest':{'sha256':'m'},'cache_identity':{'sha256':'c'}}
    frozen={'qwen':'fixed','native':'fixed','architecture':'fixed','frozen_auxiliary':'fixed'}
    rows=[{'name':'body','runtime_group':'native_body','role':'controller'}]
    catalog=SimpleNamespace(record=lambda:rows,signature=lambda:digest(rows))
    gradient={'schema':'native_v5_gradient_preflight_v1','level':'grad','passed':True,'production_passed':True,
        'verification_only':False,'config_sha256':c.signature(),'data_signature':digest(identity['data']),
        'identity_sha256':digest(identity),'catalog_sha256':catalog.signature(),'initialization_sha256':'origin',
        'frozen_before':frozen,'frozen_after':frozen,'optimizer_created':False,'optimizer_step_executed':False}
    gradient.update(gradient_overrides or {})
    gref=write_json(tmp_path/'gradient.json',gradient)
    model=SimpleNamespace(checkpoint_payload=lambda **k:{'test_model':True})
    stateful=SimpleNamespace(state_dict=lambda:{})
    journal=Journal(tmp_path/'journal.jsonl');journal.append({'update':updates})
    guard=ValidationGuard(c);guard.observe({'update':0,'records':[{'objective':1.}]});guard.observe({'update':updates,'records':[{'objective':.8}]})
    snap=cp.save_snapshot(tmp_path/'snap',model,stateful,stateful,stateful,torch.Generator(),config=c,identity=identity,
        catalog=catalog,active={'body'},global_update=updates,stage_update=updates,attempts=updates*4,journal=journal,
        micro_position=0,gradient_report=gref,stage_origin={'sha256':'origin'},frozen_state=frozen,validation_state=guard.state)
    return snap,gref,auth,c


def test_automatic_body_gate_requires_committed_completed_production_s0(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.validation import build_automatic_body_gate,check_body_baseline
    from src.pose_control.qwen21_full.native_v5.training.io import write_json
    snap,gref,auth,c=body_fixture(tmp_path,monkeypatch)
    gate=build_automatic_body_gate(snap['checkpoint'],gref,auth)
    assert gate['acceptance_kind']=='technical_training_gate_v1' and gate['body_accepted']
    assert gate.get('manual_acknowledgment') is not True
    baseline=write_json(tmp_path/'body.json',gate)
    check_body_baseline(replace(c,stage=stage_preset('S1','face',budget=2000),body_baseline_report=baseline),snap['checkpoint'])
    with pytest.raises(ValueError):check_body_baseline(replace(c,stage=stage_preset('S1','face',budget=2000),body_baseline_report=baseline,
        continuous_stage_authorization={'source':'direct_user_message','text':'different'}),snap['checkpoint'])
    with pytest.raises(ValueError):build_automatic_body_gate({**snap['checkpoint'],'sha256':'0'*64},gref,auth)


@pytest.mark.parametrize('updates,overrides',[(9999,{}),(10000,{'production_passed':False}),(10000,{'data_signature':'different'}),
    (10000,{'frozen_after':{}})])
def test_automatic_body_gate_rejects_incomplete_or_unbound_evidence(tmp_path,monkeypatch,updates,overrides):
    from src.pose_control.qwen21_full.native_v5.training.validation import build_automatic_body_gate
    snap,gref,auth,c=body_fixture(tmp_path,monkeypatch,updates=updates,gradient_overrides=overrides)
    with pytest.raises(ValueError):build_automatic_body_gate(snap['checkpoint'],gref,auth)


def test_generation_uses_only_shared_bounded_validation_selection(tmp_path,monkeypatch):
    from PIL import Image
    from src.pose_control.qwen21_full.native_v5.training.observations import generation_probe
    from src.pose_control.qwen21_full.native_v5 import generation
    d=selection_data();calls=[]
    def pair(source,index):
        calls.append((source,index));return {'native_scene_kind':'single','pair_metadata':{'pair_id':d.datasets[source].rows[index]['pair_id']}},{},{}
    backend=SimpleNamespace(runtime=torch.nn.Linear(1,1),pair=pair,pipe=object(),ensure_vae=lambda:torch.rand(1))
    monkeypatch.setattr(generation,'generate_native',lambda *a,**k:Image.new('RGB',(16,16)))
    rng=torch.get_rng_state().clone()
    report=generation_probe(backend,replace(TrainingConfig(),probe_samples=10),d,1000,tmp_path)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(calls)==len(report['records'])==10 and all(s=='val' for s,i in calls)
    before={r['path']:open(r['path'],'rb').read() for r in report['records']}
    repeated=generation_probe(backend,replace(TrainingConfig(),probe_samples=10),d,1000,tmp_path)
    assert repeated['records']==report['records']
    assert all(open(path,'rb').read()==value for path,value in before.items())
    with torch.no_grad():backend.runtime.weight.add_(1)
    with pytest.raises(ValueError,match='generation'):
        generation_probe(backend,replace(TrainingConfig(),probe_samples=10),d,1000,tmp_path)


def trainer_context(tmp_path,monkeypatch,c):
    from src.pose_control.qwen21_full.native_v5.training import trainer,checkpointing,validation,objective,probes,observations
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    runtime=torch.nn.Linear(1,1,bias=False);runtime.config=SimpleNamespace(injection_layers=(0,))
    runtime.strength=lambda *a:torch.ones(1)
    p=runtime.weight
    optimizer=torch.optim.SGD([p],lr=.01)
    status={'face_area':0,'hand_area':0,'auxiliary_faces':0,'bucket':[[1,4],[1,4],'single','full_smplx',512]}
    data=selection_data();data.status={('body','train',0):status};data.sample=lambda *a:{'native_scene_kind':'single'}
    backend=SimpleNamespace(runtime=runtime,device='cpu',ensure_evaluators=lambda r:(),
        pair=lambda *a:({'native_scene_kind':'single'},{'training_supervision_v5':torch.ones(1),'tgt_image':torch.zeros(1,3,16,16)},
            {'target_latent':torch.zeros(1,4,64)}),
        forward=lambda *a:(p,torch.zeros_like(p),torch.ones(1),torch.zeros_like(p)),pipe=None,identity={})
    events=[];writes={}
    context=dict(backend=backend,data=data,catalog=SimpleNamespace(parameters={'weight':p}),active={'weight'},optimizer=optimizer,
        scheduler=SimpleNamespace(step=lambda:None),sampler=SimpleNamespace(draw=lambda:('body','train',0)),noise_generator=torch.Generator(),
        segment=tmp_path,journal=SimpleNamespace(append=lambda report:events.append(('journal',report.get('update')))),gradient={},origin={},parent=None,
        global_update=0,stage_update=0,attempts=0,frozen={})
    monkeypatch.setattr(NativeV5Trainer,'_start',lambda *a:context)
    monkeypatch.setattr(validation,'audit_gradients',lambda b:{})
    monkeypatch.setattr(validation,'frozen_digest',lambda b:{})
    monkeypatch.setattr(objective,'compute_micro_loss',lambda *a,**k:(p.square().sum(),{}))
    def probe(b,c,d,u):events.append(('loss',u));return {'update':u,'records':[{'objective':1.}]}
    monkeypatch.setattr(probes,'loss_probe',probe)
    monkeypatch.setattr(observations,'generation_probe',lambda b,c,d,u,o:events.append(('generation',u)) or {'update':u,'records':[]})
    def save(*a,**kw):events.append(('save',kw['global_update'],deepcopy(kw.get('validation_state'))));return {'checkpoint':{'path':'synthetic','sha256':str(kw['global_update'])}}
    from copy import deepcopy
    monkeypatch.setattr(checkpointing,'save_snapshot',save)
    monkeypatch.setattr(trainer,'write_json',lambda path,value:writes.update({path.name:value}))
    optimizer.param_groups[0]['name']='test'
    return NativeV5Trainer(c),events,writes,context


def test_training_reads_each_micro_once_and_preserves_noise_sequence(tmp_path,monkeypatch):
    c=replace(TrainingConfig(),stage=replace(stage_preset('S0',budget=2),accumulation=4),
              output_root='/data/shangguanrz/test',probe_every_updates=0,save_every_updates=100)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    original_sample=ctx['data'].sample; reads=[]; observed=[]
    def sample(source,index):
        reads.append((source,index));return original_sample(source,index)
    ctx['data'].sample=sample
    original_pair=ctx['backend'].pair
    def pair(source,index):
        _,batch,latent=original_pair(source,index)
        return sample(source,index),batch,latent
    ctx['backend'].pair=pair
    forward=ctx['backend'].forward
    def capture(batch,pair,route,sigma,noise,options,base):
        observed.append((sigma.clone(),noise.clone()))
        return forward(batch,pair,route,sigma,noise,options,base)
    ctx['backend'].forward=capture
    t.run()
    assert reads==[('train',0)]*8
    expected=[];generator=torch.Generator()
    for update in range(2):
        sigmas=[torch.rand(1,generator=generator) for _ in range(4)]
        expected.extend((sigma,torch.randn((1,4,64),generator=generator)) for sigma in sigmas)
    for actual,wanted in zip(observed,expected):
        assert all(torch.equal(a,b) for a,b in zip(actual,wanted))
    assert len(observed)==8 and ctx['global_update']==ctx['stage_update']==2
    assert torch.equal(ctx['noise_generator'].get_state(),generator.get_state())


def test_trainer_validates_before_commits_and_probes_stage_end_without_duplicate_generation(tmp_path,monkeypatch):
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=3),output_root='/data/shangguanrz/test',
        probe_samples=10,probe_every_updates=1,save_every_updates=1,generation_every_updates=2,validation_loss_limit_ratio=2)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    t.run()
    assert [e[1] for e in events if e[0]=='loss']==[0,1,2,3]
    assert [e[1] for e in events if e[0]=='generation']==[2,3]
    for e in events:
        if e[0]=='save':assert e[2]['last_probe_update']==e[1]
    assert writes['summary.json']['validation_state']['baseline_loss']==1.


def test_trainer_validation_failure_retains_previous_committed_checkpoint(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training import probes
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=4),output_root='/data/shangguanrz/test',
        probe_samples=10,probe_every_updates=1,save_every_updates=1,validation_loss_limit_ratio=2)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    monkeypatch.setattr(probes,'loss_probe',lambda b,c,d,u:{'update':u,'records':[{'objective':1. if u==0 else 3.}]})
    with pytest.raises(RuntimeError,match='validation'):t.run()
    assert [e[1] for e in events if e[0]=='save']==[1,2]
    assert writes['failure.json']['last_committed_snapshot']['checkpoint']['sha256']=='2'
    assert writes['failure.json']['validation_state']['consecutive_failures']==3


def test_resumed_full_stage_only_fills_missing_generation_and_keeps_baseline(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.evaluation import ValidationGuard,fixed_evaluation_selection
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=3),output_root='/data/shangguanrz/test',
        probe_samples=10,probe_every_updates=1,save_every_updates=1,generation_every_updates=2,validation_loss_limit_ratio=2)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    guard=ValidationGuard(c,selection=fixed_evaluation_selection(c,ctx['data']))
    guard.observe({'update':0,'records':[{'objective':2.}]});guard.observe({'update':3,'records':[{'objective':1.}]})
    ctx.update(global_update=3,stage_update=3,validation_state=guard.state)
    t.run(mode='resume',checkpoint={'path':'synthetic'})
    assert [e[1] for e in events if e[0]=='generation']==[3]
    assert not [e for e in events if e[0]=='loss']
    assert writes['summary.json']['validation_state']['baseline_loss']==2.


def test_completed_generation_failure_cannot_publish_final_checkpoint(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training import observations
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=2),output_root='/data/shangguanrz/test',
        probe_samples=10,probe_every_updates=1,save_every_updates=1,generation_every_updates=1000,validation_loss_limit_ratio=2)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    def fail(*a,**kw):raise RuntimeError('generation failed')
    monkeypatch.setattr(observations,'generation_probe',fail)
    with pytest.raises(RuntimeError,match='generation'):t.run()
    assert [e[1] for e in events if e[0]=='save']==[1]
    assert writes['failure.json']['last_committed_snapshot']['checkpoint']['sha256']=='1'


def test_resume_with_enabled_guard_refuses_missing_validation_state(tmp_path,monkeypatch):
    c=replace(TrainingConfig(),stage=stage_preset('S0',budget=2),output_root='/data/shangguanrz/test',
        probe_samples=10,validation_loss_limit_ratio=2)
    t,events,writes,ctx=trainer_context(tmp_path,monkeypatch,c)
    with pytest.raises(ValueError,match='validation state'):t.run(mode='resume',checkpoint={'path':'synthetic'})
    assert events==[]


def cache_pair():
    return {'source_latent':torch.zeros(1,4,64),'target_latent':torch.ones(1,4,64),
        'source_token_hw':(2,2),'target_token_hw':(2,2),'prompt_embeds':torch.zeros(1,2,8),
        'prompt_mask':torch.ones(1,2,dtype=torch.bool),'img_mask':torch.ones(1,3,dtype=torch.bool)}


def cache_binding(key):
    return {'key':key,'source_rgb_sha256':'source','driver_rgb_sha256':'target','source_canvas_id':'sourcecanvas',
        'target_canvas_id':'targetcanvas','source_token_hw':[2,2],'target_token_hw':[2,2],
        'model_manifest_sha256':'model','prompt_sha256':'prompt','condition_sha256':None}


def test_frozen_cache_resumes_verified_pairs_and_publishes_index_once(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    from src.pose_control.qwen21_full.native_v5.training.io import file_sha
    root=tmp_path/'cache';identity={'model':'pinned','data':'pinned'}
    store=FrozenCacheStore(root,identity);entry=store.put(cache_binding('one'),cache_pair())
    resumed=FrozenCacheStore(root,identity,resume=True)
    assert resumed.get(cache_binding('one'))==entry and resumed.get(cache_binding('two')) is None
    other=resumed.put(cache_binding('two'),cache_pair())
    value={'entries':{'one':entry,'two':other}}
    ref=resumed.publish(value);before=(root/'index.json').read_bytes()
    assert resumed.publish(value)==ref and (root/'index.json').read_bytes()==before
    assert ref['sha256']==file_sha(root/'index.json')
    with pytest.raises(ValueError,match='binding'):resumed.get({**cache_binding('one'),'driver_rgb_sha256':'drift'})
    with pytest.raises(ValueError,match='identity'):FrozenCacheStore(root,{'model':'changed'},resume=True)
    with open(entry['file']['path'],'ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):resumed.get(cache_binding('one'))


def test_frozen_cache_refuses_existing_directory_without_explicit_resume(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    root=tmp_path/'cache';FrozenCacheStore(root,{'model':'pinned'})
    with pytest.raises(FileExistsError):FrozenCacheStore(root,{'model':'pinned'})


def test_frozen_cache_rejects_grid_drift_even_when_blob_sha_is_valid(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    store=FrozenCacheStore(tmp_path/'cache',{'model':'pinned'})
    pair=cache_pair();pair['target_latent']=torch.zeros(1,3,64)
    with pytest.raises(ValueError,match='grid'):store.put(cache_binding('one'),pair)


def test_frozen_cache_rejects_corrupt_entry_metadata(tmp_path):
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    store=FrozenCacheStore(tmp_path/'cache',{'model':'pinned'});store.put(cache_binding('one'),cache_pair())
    record=next((tmp_path/'cache'/'entries').glob('*.json'));value=json.loads(record.read_text())
    value['entry']['driver_rgb_sha256']='different';record.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='binding'):store.get(cache_binding('one'))


def test_single_condition_keeps_dict_prompt_bank_out_of_hi4d_reader(tmp_path):
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition,text_sha,SCHEMA
    from src.pose_control.qwen21_full.native_v5.training.io import file_sha
    canvas=native_pad((32,32));text='Use the source person and target pose.'
    bank=tmp_path/'bank.json';bank.write_text(json.dumps({'schema':'bilibili_native_prompt_v1','prompts':['first','last']},indent=2))
    def bound(p):return {'path':str(p),'sha256':file_sha(p)}
    images=[]
    for role,color in (('source','red'),('target_pose','blue')):
        path=tmp_path/(role+'.png');Image.new('RGB',(32,32),color).save(path)
        images.append({'role':role,'file':bound(path),'canvas':canvas.metadata(),'canvas_id':canvas.fingerprint(),'space':'canvas'})
    spec={'schema':SCHEMA,'profile':'single_2ref','pair_id':'single','primary_source_index':0,
        'target_canvas_id':canvas.fingerprint(),'images':images,'prompt':{'schema':'bilibili_native_prompt_v1',
            'text':text,'sha256':text_sha(text),'field':'compiled_prompt_inference','bank':bound(bank)}}
    path=tmp_path/'condition.json';path.write_text(json.dumps(spec))
    assert read_condition(bound(path))==spec
    bank.write_text(bank.read_text()+' ')
    with pytest.raises(ValueError,match='SHA'):read_condition(bound(path))


def test_hi4d_schema_still_rejects_nonobject_and_duplicate_bank_rows(tmp_path):
    from src.pose_control.qwen21_full.native_v5.hi4d_guide_binding import verify_grounded_hi4d,SCHEMA
    from src.pose_control.qwen21_full.native_v5.training.io import file_sha
    bank=tmp_path/'bank.jsonl';bank.write_text('"not a bank row"\n')
    def spec():return {'profile':'dual_4ref','pair_id':'p','prompt':{'schema':SCHEMA,'bank':{'path':str(bank),'sha256':file_sha(bank)}}}
    with pytest.raises(ValueError,match='bank'):verify_grounded_hi4d(spec())
    row=json.dumps({'schema':SCHEMA,'pair_id':'p'})+'\n';bank.write_text(row+row)
    with pytest.raises(ValueError,match='Duplicate'):verify_grounded_hi4d(spec())


def test_json_commit_is_invisible_until_complete_and_retry_preserves_orphan(tmp_path,monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training import io
    path=tmp_path/'checkpoint.commit.json';original_link=io.os.link
    def interrupt(source,destination):
        assert not path.exists()
        assert json.loads(open(source,encoding='utf8').read())=={'complete':True}
        raise RuntimeError('interrupted before publication')
    monkeypatch.setattr(io.os,'link',interrupt)
    with pytest.raises(RuntimeError,match='publication'):io.write_json(path,{'complete':True})
    assert not path.exists()
    orphans={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert orphans
    monkeypatch.setattr(io.os,'link',original_link)
    ref=io.write_json(path,{'complete':True})
    assert io.verify(ref)==ref and all((tmp_path/name).read_bytes()==value for name,value in orphans.items())
    with pytest.raises(FileExistsError):io.write_json(path,{'changed':True})
