"""Fresh fallback guards, with CPU doubles for the unchanged GPU trainer."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    return importlib.import_module('research.qwen21_dual_prompt_pilot_evidence.recovery.retrain_structured')


def fixture(tmp_path):
    m=module();h=m.load_helper(Path(__file__).with_name('recover_training.py'))
    root=dict(production_training=True,requested_steps=300,variants=['short','structured'],save_every=50,
              learning_rate=1e-4,seed=14062,effective_batch=4,accumulation=4,
              model_id='/model',warm_start_path='/warm/checkpoint_best.pt',test_data_read=False)
    common={k:v for k,v in root.items() if k not in ('variants','save_every')}
    manifest=dict(run_contract=dict(common),initial_adapter_sha256='init')
    cache={'train':[{'sample_id':str(i)} for i in range(4)],'val':[{'sample_id':'val'}]}
    run=tmp_path/'run';run.mkdir()
    h.atomic_json(run/'cache_manifest.json',dict(short=cache,structured=cache))
    short=run/'short';short.mkdir();(short/'checkpoint_last.pt').write_bytes(b'actual short300')
    protected={str(short/'checkpoint_last.pt'):h.file_sha(short/'checkpoint_last.pt')}
    plan=dict(root=root,manifests={'short':copy.deepcopy(manifest),'structured':copy.deepcopy(manifest)},
              caches={'structured':{'train':['a','b','c','d'],'val':['v']}},data=SimpleNamespace(assert_unchanged=lambda:None),
              schedule=['schedule1..300'],adapter=object(),optimizer=object(),protected=protected,cache_files={})
    args=SimpleNamespace(attempt_dir=tmp_path/'attempt',run_dir=run,data_dir=tmp_path/'data')
    args.attempt_dir.mkdir()
    return m,h,args,plan,cache


def fake_success(args,plan,h):
    directory=args.output/'structured';directory.mkdir(parents=True)
    initial=plan['manifests']['structured']['initial_adapter_sha256']
    manifest=copy.deepcopy(plan['manifests']['structured'])
    manifest['loading_report']={'matched':[None]*178,'source_step':19500,'source_step_verified':True,
                                'frozen_coverage':{'complete':True,'matched_tensor_count':178}}
    plan['manifests']['structured']['loading_report']=copy.deepcopy(manifest['loading_report'])
    h.atomic_json(directory/'run_manifest.json',manifest)
    gate=dict(zero_init_exact=True,gradients_none=True,optimizer_steps=0,source_step=19500,
              initial_adapter_sha256=initial,frozen_coverage={'complete':True,'matched_tensor_count':178})
    h.atomic_json(directory/'real_model_gates.json',gate)
    for name in ('checkpoint_best.pt','checkpoint_last.pt'):(directory/name).write_bytes(name.encode())
    records=[{'step':i} for i in range(1,301)]
    (directory/'train_metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    validations=[dict(step=i,loss=float(300-i)) for i in range(50,301,50)]
    result=dict(status='success',schema_version=h.TRAINING_SCHEMA,production_training=True,
                requested_steps=300,completed_steps=300,variant='structured',initial_adapter_sha256=initial,
                validation_selection='true_val_split_only',caches_unchanged=True,native_gradients_none=True,
                best_step=300,best_validation=0.,validation_records=validations)
    for kind in ('best','last'):
        path=directory/f'checkpoint_{kind}.pt'
        result[f'{kind}_checkpoint']=dict(path=str(path.resolve()),sha256=h.file_sha(path),step=300)
    return result,initial


def test_common_removes_only_two_root_fields_and_does_not_mutate(tmp_path):
    m,h,args,plan,_=fixture(tmp_path);original=copy.deepcopy(plan['root'])
    common=m.reconstruct_common(h,plan)
    assert common=={k:v for k,v in original.items() if k not in ('variants','save_every')}
    common['seed']=-1
    assert plan['root']==original


@pytest.mark.parametrize('field,value',[('requested_steps',200),('learning_rate',2e-4),('save_every',100),
                                     ('variants',['structured']),('test_data_read',True)])
def test_common_rejects_unapproved_experiment(tmp_path,field,value):
    m,h,args,plan,_=fixture(tmp_path);plan['root'][field]=value
    with pytest.raises(ValueError):m.reconstruct_common(h,plan)


def test_common_rejects_changed_original_branch_contract(tmp_path):
    m,h,args,plan,_=fixture(tmp_path);plan['manifests']['structured']['run_contract']['seed']=0
    with pytest.raises(ValueError,match='common'):m.reconstruct_common(h,plan)


def test_pins_unchanged_resume_dependency(tmp_path):
    m=module();path=tmp_path/'recover_training.py';path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='dependency'):m.load_helper(path)


def test_fresh_invocation_uses_original_loop_full300_no_resume_or_completion(tmp_path,monkeypatch):
    m,h,args,plan,cache=fixture(tmp_path);calls=[]
    def train(call_args,pipe,data,samples,evidence,variant,common,schedule,expected_initial):
        assert 'adapter' not in plan and 'optimizer' not in plan
        assert call_args.steps==300 and call_args.lr==1e-4 and call_args.save_every==50
        assert call_args.output==args.attempt_dir/'training'
        assert call_args.warm_start==Path('/warm/checkpoint_best.pt')
        assert not hasattr(call_args,'resume')
        assert variant=='structured' and evidence==cache and samples is plan['caches']['structured']
        assert schedule is plan['schedule'] and expected_initial=='init'
        assert 'recovery_provenance' not in common
        calls.append(variant)
        return fake_success(call_args,plan,h)
    pkg=SimpleNamespace(trainer=SimpleNamespace(train_one_variant=train,validate_training_settings=lambda a:None))
    monkeypatch.setattr(h,'verify_history',lambda *a,**k:None)
    monkeypatch.setattr(h,'validation_history',lambda rows:[dict(step=i,loss=float(300-i)) for i in range(50,301,50)])
    result,rows,gate=m.train_fresh_structured(args,pkg,plan,h,object())
    assert calls==['structured'] and result['completed_steps']==300 and len(rows)==300 and gate['zero_init_exact']
    assert not list(tmp_path.rglob('training_complete.json'))


def test_short_changed_refuses_before_any_new_training(tmp_path):
    m,h,args,plan,_=fixture(tmp_path)
    (args.run_dir/'short/checkpoint_last.pt').write_bytes(b'changed')
    pkg=SimpleNamespace(trainer=SimpleNamespace(train_one_variant=lambda *a:pytest.fail('must not train')))
    with pytest.raises(ValueError,match='changed'):m.train_fresh_structured(args,pkg,plan,h,object())
    assert not list(tmp_path.rglob('training_complete.json'))


@pytest.mark.parametrize('failure',['exception','partial','initial','native_gradients','wrong_path'])
def test_failed_or_partial_fresh_result_never_claims_completion(tmp_path,monkeypatch,failure):
    m,h,args,plan,_=fixture(tmp_path)
    def train(call_args,*rest):
        if failure=='exception':raise RuntimeError('interrupted fresh run')
        result,initial=fake_success(call_args,plan,h)
        if failure=='partial':result['completed_steps']=200
        if failure=='initial':initial='wrong'
        if failure=='native_gradients':result['native_gradients_none']=False
        if failure=='wrong_path':result['last_checkpoint']['path']=str(args.run_dir/'structured/checkpoint_last.pt')
        return result,initial
    pkg=SimpleNamespace(trainer=SimpleNamespace(train_one_variant=train,validate_training_settings=lambda a:None))
    monkeypatch.setattr(h,'verify_history',lambda *a,**k:None)
    monkeypatch.setattr(h,'validation_history',lambda rows:[dict(step=i,loss=float(300-i)) for i in range(50,301,50)])
    with pytest.raises((ValueError,RuntimeError)):m.train_fresh_structured(args,pkg,plan,h,object())
    assert not list(tmp_path.rglob('training_complete.json'))


def test_previous_strict_failure_is_required_and_preserved(tmp_path):
    m,h,args,plan,_=fixture(tmp_path)
    args.failed_attempt=tmp_path/'failed';args.failed_attempt.mkdir()
    args.failed_log=tmp_path/'failed.log';args.failed_log.write_text('ValueError: replay gradient_norm divergence\n')
    path=args.failed_attempt/'recovery_result.json'
    h.atomic_json(path,dict(status='failed_no_completion_claim',error='replay gradient_norm divergence',
                           recovery={'source_sha256':m.HELPER_SHA256,'committed_boundary':200,'resumed_steps':[201,300]}))
    evidence,files=m.failed_attempt_proof(args,h)
    assert evidence['exact_continuation_rejected'] and files[str(path)]==h.file_sha(path)
    h.atomic_json(path,dict(status='success'))
    with pytest.raises(ValueError):m.failed_attempt_proof(args,h)


def test_actual_root_snapshot_common_matches_both_original_contracts():
    m=module();h=m.load_helper(Path(__file__).with_name('recover_training.py'))
    path=Path(__file__).resolve().parents[1]/'recovery_contracts.json'
    if not path.exists():pytest.skip('real parent-owned snapshot is absent in this isolated test copy')
    contracts=h.read_json(path)
    plan=dict(root=contracts['root_manifest'],manifests={v:contracts[f'{v}_manifest'] for v in ('short','structured')})
    common=m.reconstruct_common(h,plan)
    assert len(common)==len(contracts['root_manifest'])-2 and common['requested_steps']==300
