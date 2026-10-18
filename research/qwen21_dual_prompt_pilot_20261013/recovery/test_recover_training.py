"""Operational resume proofs; no production completion is fabricated by tests."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def module():
    try:
        return importlib.import_module('research.qwen21_dual_prompt_pilot_evidence.recovery.recover_training')
    except ModuleNotFoundError as exc:
        if exc.name.endswith('recover_training'):
            pytest.fail('outside-mirror operational recovery is not implemented')
        raise


def schedule(steps=4):
    ids=('a','b','c','d')
    return tuple(SimpleNamespace(step=i//4+1,micro=i%4,sample_index=i%4,sample_id=ids[i%4],noise_seed=100+i)
                 for i in range(steps*4))


def fixture():
    torch.manual_seed(19)
    adapter=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
    return adapter,optimizer


def train_forward(adapter):
    def forward(item,scale):
        generator=torch.Generator().manual_seed(item.noise_seed)
        noise=torch.randn((1,2),generator=generator)
        sigma=torch.rand(1,generator=generator).clamp(.01,.99)
        prediction=adapter(noise+item.sample_index)
        flow=prediction.square().mean()
        background=flow*0;budget=flow*0
        loss=flow+.1*background+.01*budget
        (loss/scale).backward()
        return loss,dict(flow=float(flow.detach()),background=0.,budget=0.,sigma=float(sigma[0]))
    return forward


def validation(step):
    return .01 if step==1 else .1+step*.01,dict(selection='true_val_split_only',gradients_none=True,
        optimizer_steps=0,inference_mode=True,autograd_enabled=False)


def test_checkpoint_resume_matches_uninterrupted_adam_and_global_noise(tmp_path):
    p=module();items=schedule()
    first,opt=fixture()
    uninterrupted=p.resume_updates(first,opt,items,start_step=0,target_step=4,history=[],archived={},
        train_forward=train_forward(first),validate=validation,save_every=1,on_save=lambda *args:None)
    second,second_opt=fixture()
    prefix=p.resume_updates(second,second_opt,items,start_step=0,target_step=2,history=[],archived={},
        train_forward=train_forward(second),validate=validation,save_every=1,on_save=lambda *args:None)
    checkpoint=tmp_path/'step2.pt'
    torch.save(dict(adapter=second.state_dict(),optimizer=second_opt.state_dict(),step=2),checkpoint)
    loaded=torch.load(checkpoint,weights_only=True)
    resumed,resumed_opt=fixture();resumed.load_state_dict(loaded['adapter']);resumed_opt.load_state_dict(loaded['optimizer'])
    saved=[]
    continuation=p.resume_updates(resumed,resumed_opt,items,start_step=2,target_step=4,history=prefix['records'],
        archived={row['step']:row for row in uninterrupted['records'] if row['step']>2},
        train_forward=train_forward(resumed),validate=validation,save_every=1,
        on_save=lambda step,score,best,record:saved.append((step,best)),metrics_path=tmp_path/'segment.jsonl')
    assert p.tensor_digest(first.state_dict())==p.tensor_digest(resumed.state_dict())
    assert p.tensor_digest(opt.state_dict())==p.tensor_digest(resumed_opt.state_dict())
    assert continuation['best_step']==1 and continuation['best_validation']==.01
    assert saved==[(3,False),(4,False)]
    assert [row['step'] for row in continuation['records']]==[3,4]
    assert [m['noise_seed'] for row in continuation['records'] for m in row['microbatches']]==list(range(108,116))
    assert [row['step'] for row in continuation['validation_records']]==[1,2,3,4]
    assert all(proof['passed'] for proof in continuation['replay_audits'])
    assert all(proof['rng_unchanged'] for proof in continuation['replay_audits'])
    assert not (tmp_path/'training_complete.json').exists()


@pytest.mark.parametrize('field',['sample_id','noise_seed','sigma','flow','background','budget','total_loss','mean_loss','gradient_norm','step'])
def test_replay_divergence_fails_before_optimizer_update(field):
    p=module();adapter,opt=fixture();items=schedule()
    original=p.resume_updates(adapter,opt,items,start_step=0,target_step=2,history=[],archived={},
        train_forward=train_forward(adapter),validate=validation,save_every=1,on_save=lambda *args:None)
    boundary_adapter=copy.deepcopy(adapter.state_dict());boundary_opt=copy.deepcopy(opt.state_dict())
    reference=p.resume_updates(adapter,opt,items,start_step=2,target_step=3,history=original['records'],archived={},
        train_forward=train_forward(adapter),validate=validation,save_every=1,on_save=lambda *args:None)['records'][0]
    changed=copy.deepcopy(reference)
    if field=='mean_loss':changed['loss']+=1
    elif field=='gradient_norm':changed[field]+=1
    elif field=='step':changed['step']=4
    elif field=='sample_id':changed['microbatches'][0][field]='other'
    else:changed['microbatches'][0][field]+=1
    restored,restored_opt=fixture();restored.load_state_dict(boundary_adapter);restored_opt.load_state_dict(boundary_opt)
    before=p.tensor_digest(restored.state_dict());before_opt=p.tensor_digest(restored_opt.state_dict())
    with pytest.raises(ValueError,match='replay'):
        p.resume_updates(restored,restored_opt,items,start_step=2,target_step=3,history=original['records'],archived={3:changed},
            train_forward=train_forward(restored),validate=validation,save_every=1,on_save=lambda *args:None)
    assert p.tensor_digest(restored.state_dict())==before and p.tensor_digest(restored_opt.state_dict())==before_opt


def test_legacy_replay_totals_are_reconstructed_in_float32():
    p=module()
    old=dict(step=201,loss=1.,gradient_norm=.5,microbatches=[dict(sample_id='a',noise_seed=14862,
             sigma=.3,flow=1.,background=0.,budget=0.) for _ in range(4)])
    current=copy.deepcopy(old)
    for row in current['microbatches']:row['total_loss']=1.
    evidence=p.compare_replay(old,current)
    assert evidence['passed'] and evidence['micro_total_source']=='fp32 reconstruction from archived components'
    assert len(evidence['microbatch_comparisons'])==4


@pytest.mark.parametrize('field',['dataset_sha256','code_sha256','cache_content','schedule_sha256','optimizer_parameter_names'])
def test_contract_changes_are_rejected(field):
    expected={key:{'proof':'x'} for key in ('dataset_sha256','code_sha256','cache_content','schedule_sha256','optimizer_parameter_names')}
    current=copy.deepcopy(expected);current[field]={'proof':'wrong'}
    with pytest.raises(ValueError,match=field):module().require_equal_contract(expected,current)


@pytest.mark.parametrize('damage',['order','clock','missing_state','nonfinite'])
def test_optimizer_contract_is_checked_before_any_resume(damage):
    p=module();adapter,opt=fixture();items=schedule()
    p.resume_updates(adapter,opt,items,start_step=0,target_step=2,history=[],archived={},
        train_forward=train_forward(adapter),validate=validation,save_every=1,on_save=lambda *args:None)
    names=[name for name,param in adapter.named_parameters() if param.requires_grad]
    if damage=='order':names.reverse()
    elif damage=='clock':next(iter(opt.state.values()))['step'].fill_(1)
    elif damage=='missing_state':opt.state.pop(next(iter(opt.state)))
    else:next(iter(opt.state.values()))['exp_avg'].fill_(float('nan'))
    with pytest.raises(ValueError,match='optimizer'):
        p.optimizer_proof(adapter,opt,names,expected_step=2)


def test_ambient_rng_usage_is_rejected_before_commit():
    p=module();adapter,opt=fixture();base=train_forward(adapter)
    def stochastic(item,scale):
        torch.rand(1)
        return base(item,scale)
    with pytest.raises(ValueError,match='RNG'):
        p.resume_updates(adapter,opt,schedule(),start_step=0,target_step=1,history=[],archived={},
            train_forward=stochastic,validate=validation,save_every=1,on_save=lambda *args:None)
    assert not opt.state


@pytest.mark.parametrize('kind',['missing_step','missing_validation','wrong_noise'])
def test_incomplete_or_inconsistent_history_cannot_prove_completion(kind):
    p=module();items=schedule(300)
    records=[dict(step=step,loss=1.,gradient_norm=.1,finite_gradients=True,microbatches=[
        dict(sample_id=item.sample_id,noise_seed=item.noise_seed,flow=1.,background=0.,budget=0.,sigma=.5)
        for item in items[(step-1)*4:step*4]]) for step in range(1,301)]
    for row in records:
        if row['step']%50==0:
            row.update(validation_loss=.1,validation=dict(selection='true_val_split_only',gradients_none=True,
                optimizer_steps=0,inference_mode=True,autograd_enabled=False))
    if kind=='missing_step':records.pop()
    elif kind=='missing_validation':del records[49]['validation_loss']
    else:records[299]['microbatches'][3]['noise_seed']+=1
    with pytest.raises(ValueError):p.verify_history(records,items,expected_tail=300,save_every=50)


def test_native_integrity_keeps_original_final_hash_unknown():
    report=module().native_integrity_report('same','same','same')
    assert report['original_process']['final_memory_sha256'] is None
    assert report['recovery_process']['sha256_before']==report['recovery_process']['sha256_after']=='same'
    assert report['recovery_process']['unchanged'] is True
    with pytest.raises(ValueError,match='backbone'):
        module().native_integrity_report('original','different','different')


def test_frozen_import_routing_builds_namespace_and_rejects_root_source(tmp_path,monkeypatch):
    p=module();mirror=tmp_path/'mirror';mirror.mkdir()
    monkeypatch.setattr(p.importlib,'import_module',lambda name:SimpleNamespace(__file__=str(mirror/(name+'.py'))))
    pkg=p._imports(mirror)
    assert Path(pkg.pilot.__file__).is_relative_to(mirror)
    assert Path(pkg.trainer.__file__).is_relative_to(mirror)
    monkeypatch.setattr(p.importlib,'import_module',lambda name:SimpleNamespace(__file__=str(tmp_path/'outside.py')))
    with pytest.raises(ValueError,match='outside'):
        p._imports(mirror)


def test_actual_legacy201_archive_reconstructs_its_exact_mean_total():
    path=Path(__file__).resolve().parents[1]/'recovery_contracts.json'
    if not path.exists():pytest.skip('actual production audit snapshot is external to these two test files')
    row=json.loads(path.read_text())['structured_archived201']
    assert row['step']==201
    assert [micro['noise_seed'] for micro in row['microbatches']]==[14862,14863,14864,14865]
    assert sum(module()._micro_total(micro) for micro in row['microbatches'])/4==row['loss']


@pytest.mark.parametrize('failure',['validator','partial_publication'])
def test_failed_publication_withdraws_own_markers_and_preserves_checkpoint_evidence(tmp_path,monkeypatch,failure):
    p=module();run=tmp_path/'run';run.mkdir()
    checkpoint=run/'checkpoint_last.pt';checkpoint.write_bytes(b'preserved completed checkpoint evidence')
    markers={run/'short/training_complete.json':{'proof':'short'},
             run/'structured/training_complete.json':{'proof':'structured'},run/'training_complete.json':{'proof':'root'}}
    real=p.atomic_json
    if failure=='partial_publication':
        def faulty(path,value):
            real(path,value)
            if Path(path)==run/'structured/training_complete.json':raise OSError('publication interrupted after replace')
        monkeypatch.setattr(p,'atomic_json',faulty)
    def validator():
        assert all(path.exists() for path in markers)
        raise ValueError('final consumer validation rejected candidate')
    with pytest.raises((ValueError,OSError)):
        p.publish_completion_fail_closed(markers,validator)
    assert not any(path.exists() for path in markers)
    assert checkpoint.read_bytes()==b'preserved completed checkpoint evidence'


def test_completion_publication_refuses_existing_markers(tmp_path):
    p=module();path=tmp_path/'training_complete.json';path.write_bytes(b'existing unrelated completion')
    with pytest.raises(ValueError,match='existing'):
        p.publish_completion_fail_closed({path:{'proof':'replacement'}},lambda:None)
    assert path.read_bytes()==b'existing unrelated completion'
