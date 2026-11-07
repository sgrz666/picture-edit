"""CPU-only evidence for the unattended experiment orchestration contract."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'experiments/bilibili-single-spatial-training/scripts'
sys.path.insert(0, str(SCRIPTS))


def module(name):
    path = SCRIPTS / (name + '.py')
    assert path.is_file(), f'Missing experiment implementation: {path}'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_resource_probe_falls_back_to_cuda_when_nvml_library_mismatches(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    s=module('supervisor');calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='nvidia-smi':
            raise subprocess.CalledProcessError(255,command,stderr='Failed to initialize NVML: Driver/library version mismatch')
        return SimpleNamespace(stdout=json.dumps({'free_bytes':40*1024**3,'total_bytes':83*1024**3}),returncode=0)
    monkeypatch.setattr(s.subprocess,'run',run)
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'MemAvailable: 12582912 kB\n' if p.as_posix()=='/proc/meminfo' else original(p,*a,**k))
    result=s.resources('0')
    assert result['gpu_free_gib']==40 and result['host_available_gib']==12
    assert result['gpu_probe']=='cuda_runtime' and len(calls)==2


def test_plan_pins_budget_scope_and_never_exposes_test_to_training(tmp_path):
    p = module('run_plan')
    authorization = {'source': 'direct_user_message', 'text': '明确授权一次性无人值守完成单人S0-S3。'}
    prepared = {'sources': [{'source_id': 'bilibili_train', 'role': 'train'},
                            {'source_id': 'bilibili_validation', 'role': 'validation'}],
                'initialization': {'path': '/data/native_v5_init_v2.pt', 'sha256': 'a'*64},
                'model_path': '/data/qwen', 'model_manifest': {'path': '/data/model.json', 'sha256': 'b'*64},
                'cache_root': '/data/run/frozen_qwen', 'bindings': {'qwen_condition_profile': 'single_2ref'}}
    plan = p.make_plan('/data/run', '/repo', authorization)
    assert [s['budget_updates'] for s in plan['stages']] == [10000, 2000, 4000, 4000]
    assert plan['total_updates'] == 20000
    for stage in plan['stages']:
        config = p.stage_config(plan, prepared, stage['phase'], initialization=prepared['initialization'])
        assert len(config['sources']) == 2
        assert all(s['source_id'] != 'bilibili_test' for s in config['sources'])
        assert config['stage']['accumulation'] == 4
        assert not config['stage']['unfreeze_geometry']
        assert config['probe_samples'] == 10
        assert config['probe_every_updates'] == config['save_every_updates'] == 100
        assert config['generation_every_updates'] == 1000
        assert config['validation_loss_limit_ratio'] == 2
        assert config['validation_failure_patience'] == 3
        assert config['continuous_stage_authorization'] == authorization
        for route in config['stage']['routes']:
            assert route['controls']['interaction'] is False
            assert route['loss']['spatial_body'] == .5
            assert route['loss']['spatial_joint'] == .5
            assert route['loss']['spatial_hand'] == 1
            assert route['loss']['spatial_max'] == 2.5
            assert all(route['loss'][k] == 0 for k in ('face', 'hand', 'identity', 'perceptual'))
    assert p.stage_config(plan, prepared, 'S3', initialization=prepared['initialization'])['stage']['optimizer']['multiplier'] == .1


def test_hash_journal_detects_changed_or_truncated_record(tmp_path):
    s = module('supervisor')
    j = s.EventJournal(tmp_path)
    j.append({'event': 'started'})
    j.append({'event': 'finished'})
    assert len(j.read()) == 2
    part = sorted((tmp_path/'journal').glob('*.json'))[0]
    part.write_text('{}', encoding='utf8')
    with pytest.raises(s.IntegrityError):
        j.read()


def test_only_committed_sha_bound_checkpoints_can_be_recovered(tmp_path):
    s = module('supervisor')
    directory = tmp_path/'stages/S0/checkpoints'
    directory.mkdir(parents=True)
    orphan = directory/'update_99999999_orphan.pt'
    orphan.write_bytes(b'orphan')
    checkpoint = directory/'update_00000100_good.pt'
    checkpoint.write_bytes(b'fixture checkpoint')
    ref = s.file_ref(checkpoint)
    s.write_new_json(checkpoint.with_suffix('.commit.json'), {
        'schema':'native_v5_training_commit_v1', 'checkpoint': ref,
        'global_update': 100, 'stage_update': 100, 'training_state_sha256': 'a'*64})
    assert s.latest_committed(tmp_path, 'S0')['checkpoint'] == ref
    checkpoint.write_bytes(b'corrupted')
    with pytest.raises(s.IntegrityError):
        s.latest_committed(tmp_path, 'S0')


def test_state_replays_only_accepted_results_and_retries_survive_restart(tmp_path):
    s = module('supervisor')
    journal = s.EventJournal(tmp_path)
    journal.append({'event': 'attempt', 'task':'prepare_data', 'attempt':1})
    journal.append({'event': 'attempt_failed', 'task':'prepare_data', 'attempt':1, 'failure_class':'transient'})
    unaccepted = tmp_path/'ignored.json'
    s.write_new_json(unaccepted, {'passed':True})
    state = s.replay(journal.read())
    assert state['accepted'] == {}
    assert state['attempts']['prepare_data'] == 1
    assert s.next_task(state) == ('prepare_data', None)
    journal.append({'event':'accepted', 'task':'prepare_data', 'result':s.file_ref(unaccepted)})
    assert s.next_task(s.replay(journal.read())) == ('prepare_cache', None)


def test_all_stages_and_final_evidence_are_required_for_completion(tmp_path):
    s = module('supervisor')
    p = module('run_plan')
    plan = p.make_plan(str(tmp_path), '/repo', {'source':'direct_user_message','text':'批准完整训练'})
    accepted = {}
    for task in ('prepare_data', 'prepare_cache', 'data_preflight'):
        accepted[task] = {'passed':True}
    total = 0
    for stage in plan['stages']:
        phase, budget = stage['phase'], stage['budget_updates']
        accepted[phase+'.stage_grad'] = {'passed':True, 'production_passed':True}
        total += budget
        accepted[phase+'.stage_train'] = {'passed':True, 'budget_completed':True,
            'stage_update':budget, 'global_update':total, 'checkpoint': {'path':'unused','sha256':'a'*64},
            'validation_state': {'passed':True,'finite':True,'triggered':False}}
    accepted['eval_test'] = {'passed':True,'technical_only':True,'loss_report':{},'generation_report':{},'metrics_report':{},'validation_loss_report':{}}
    accepted['export_final'] = {'passed':True,'successful_updates':20000,
        'checkpoint':accepted['S3.stage_train']['checkpoint'],
        'weights':{'path':'fixture'},'summary':{'path':'fixture'},'readme':{'path':'fixture'}}
    assert s.completion_errors(plan, accepted) == []
    accepted['S3.stage_train']['stage_update'] = 3999
    assert any('S3' in err for err in s.completion_errors(plan, accepted))
    accepted['S3.stage_train']['stage_update'] = 4000
    del accepted['eval_test']['metrics_report']
    assert s.completion_errors(plan, accepted)


def test_failure_classes_and_two_retry_ceiling_are_explicit():
    s = module('supervisor')
    assert s.classify_failure('CUDA out of memory', 1) == 'resource'
    assert s.classify_failure('Artifact SHA changed: x', 1) == 'terminal'
    assert s.classify_failure('Frozen state changed', 1) == 'terminal'
    assert s.classify_failure('Nonfinite gradient', 1) == 'terminal'
    assert s.classify_failure('', -9) == 'transient'
    assert s.may_retry('resource', 1)
    assert s.may_retry('transient', 2)
    assert not s.may_retry('transient', 3)
    assert not s.may_retry('terminal', 1)


def test_gradient_case_budget_counts_every_representative_without_truncation():
    w = module('stage_worker')
    from types import SimpleNamespace
    statuses = {}
    for i in range(19):
        statuses[('joint','train',i)] = {'bucket':[[],[],'single','full_smplx'],
            'branch_usable':{'body':True,'face':bool(i%2)},'capabilities':{'case':i},'auxiliary_faces':0}
    data = SimpleNamespace(pools={('joint','train'):list(range(19))},status=statuses)
    assert w.required_gradient_cases(data, ['joint']) == {'joint':19}


def test_stop_requests_are_append_only_and_resume_acknowledges_old_requests(tmp_path):
    s = module('supervisor')
    first = s.request_stop(tmp_path)
    assert Path(first['path']).exists()
    assert len(s.pending_stops(tmp_path, set())) == 1
    assert s.pending_stops(tmp_path, {first['sha256']}) == []
    second = s.request_stop(tmp_path)
    assert Path(second['path']).exists() and Path(first['path']).exists()
    assert len(s.pending_stops(tmp_path, set())) == 2


def test_explicit_stop_does_not_spend_automatic_retry_budget(tmp_path):
    s = module('supervisor')
    rows = [{'event':'attempt','task':'S0.stage_train','attempt':1},
            {'event':'stopped','reason':'explicit stop request'},
            {'event':'attempt','task':'S0.stage_train','attempt':2},
            {'event':'stopped','reason':'explicit stop request'}]
    assert s.replay(rows)['failure_counts'].get('S0.stage_train', 0) == 0
    rows.append({'event':'attempt_failed','task':'S0.stage_train','failure_class':'transient'})
    assert s.replay(rows)['failure_counts']['S0.stage_train'] == 1


def test_production_config_accepts_all_group_names_and_new_core_fields(tmp_path):
    p = module('run_plan')
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,stage_preset
    prepared = {'sources':[{'source_id':'train','role':'train'},{'source_id':'val','role':'validation'}],
        'model_path':str(tmp_path),'model_manifest':None,'cache_root':str(tmp_path)}
    plan = p.make_plan(str(tmp_path), str(tmp_path), {'source':'direct_user_message','text':'批准20k'})
    for phase in ('S0','S1','S2','S3'):
        raw = p.stage_config(plan, prepared, phase, initialization=None)
        config = TrainingConfig.from_dict(raw)
        assert config.signature()


def test_final_guard_requires_generation_and_probe_at_exact_last_update():
    w = module('stage_worker')
    state = {'passed':True,'finite':True,'triggered':False,'selection':[{'pair_id':'x'}],
             'last_probe_update':20000,'last_generation_update':19000}
    assert not w.final_validation_ok(state, 20000)
    state['last_generation_update'] = 20000
    assert w.final_validation_ok(state, 20000)


def test_explicit_verify_hashes_nested_generated_image_evidence(tmp_path):
    s = module('supervisor')
    image = tmp_path/'test.png'
    image.write_bytes(b'image fixture')
    report = s.write_new_json(tmp_path/'generation.json', {'records':[{**s.file_ref(image),'pair_id':'pair'}]})
    s.audit_references({'generation_report':report})
    image.write_bytes(b'changed')
    with pytest.raises(s.IntegrityError):
        s.audit_references({'generation_report':report})


def test_pixel_metrics_are_finite_and_exact_match_is_explicit(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from types import SimpleNamespace
    w = module('stage_worker')
    target = np.full((32,32,3), 128, dtype=np.uint8)
    path = tmp_path/'test.png'
    Image.fromarray(target).save(path)
    sample = {'tgt_image':torch.from_numpy(target.copy()).permute(2,0,1).float()/127.5-1,
              'valid_image_mask':torch.ones(1,32,32)}
    data = SimpleNamespace(datasets={'test':SimpleNamespace(rows=[{'pair_id':'p'}])}, sample=lambda *_:sample)
    report = {'records':[{'source_id':'test','pair_id':'p','path':str(path),'sha256':'a'*64}]}
    result = w.generation_metrics(report,data)
    assert result['used_for_selection'] is False
    assert result['mse_mean'] < 1e-12
    assert result['ssim_mean'] > .999


def test_data_qualification_previews_all_four_stages_from_initialization():
    w, p = module('stage_worker'), module('run_plan')
    plan = p.make_plan('/data/run','/repo',{'source':'direct_user_message','text':'批准20k'})
    initialization = {'path':'/data/native_v5_init_v2.pt','sha256':'a'*64}
    prepared = {'sources':[{'source_id':'train','role':'train'},{'source_id':'val','role':'validation'}],
        'initialization':initialization,'model_path':'/data/qwen','model_manifest':None,'cache_root':'/data/run/cache'}
    configs = w.data_preflight_configs(plan,prepared,{'path':'/data/run/cache/index.json','sha256':'b'*64})
    assert [raw['stage']['phase'] for raw in configs] == ['S0','S1','S2','S3']
    assert all(raw['initialization'] == initialization and raw['body_baseline_report'] is None for raw in configs)
    assert [r['branch'] for r in configs[1]['stage']['routes']] == ['face','hand','detail']


def test_immutable_configs_compare_json_semantics_on_retry(tmp_path):
    w = module('stage_worker')
    config = {'routes':({'name':'body','source_ids':()},), 'sources':()}
    first = w.immutable_config(tmp_path/'config.json',config)
    assert w.immutable_config(tmp_path/'config.json',config) == first
    with pytest.raises(ValueError):
        w.immutable_config(tmp_path/'config.json', {'routes':[{'name':'different'}]})


def test_terminal_failure_survives_crash_before_final_failed_record():
    s = module('supervisor')
    rows = [{'event':'attempt','task':'S0.stage_grad','attempt':1},
            {'event':'attempt_failed','task':'S0.stage_grad','attempt':1,'failure_class':'terminal'}]
    assert 'S0.stage_grad' in s.replay(rows)['terminal_tasks']


def test_child_exit_from_explicit_stop_is_not_counted_as_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    s = module('supervisor')
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(poll=lambda:-15,returncode=-15))
    s.request_stop(tmp_path)
    with pytest.raises(s.StopRequested):
        s.launch_child(['fixture'],tmp_path/'log',{},tmp_path,set())


def test_committed_report_is_reused_once_and_rejects_binding_or_sha_change(tmp_path):
    w = module('stage_worker')
    calls = []
    def create():
        calls.append(1)
        return {'objective':1.}
    binding = {'checkpoint_sha256':'a'*64}
    first = w.committed_report(tmp_path/'loss.json',binding,create)
    assert w.committed_report(tmp_path/'loss.json',binding,create) == first
    assert calls == [1]
    with pytest.raises(ValueError):
        w.committed_report(tmp_path/'loss.json',{'checkpoint_sha256':'b'*64},create)
    Path(first['path']).write_text('{}',encoding='utf8')
    with pytest.raises(ValueError):
        w.committed_report(tmp_path/'loss.json',binding,create)


def test_interrupted_journal_publish_preserves_uncommitted_pending_file(tmp_path, monkeypatch):
    s = module('supervisor')
    journal = s.EventJournal(tmp_path)
    original = s.os.link
    monkeypatch.setattr(s.os, 'link', lambda *_: (_ for _ in ()).throw(OSError('simulated interruption before publication')))
    with pytest.raises(OSError):
        journal.append({'event':'started'})
    assert journal.read() == []
    monkeypatch.setattr(s.os, 'link', original)
    journal.append({'event':'started'})
    assert len(journal.read()) == 1
    assert len(list((tmp_path/'journal/.pending').glob('*.json'))) == 2


def test_runtime_snapshot_rejects_changed_or_new_executable_code(tmp_path):
    s = module('supervisor')
    code = tmp_path/'snapshot'
    code.mkdir()
    source = code/'worker.py'
    source.write_text('x = 1\n',encoding='utf8')
    manifest = s.snapshot_runtime_code([code],tmp_path/'runtime_manifest.json')
    plan = {'runtime_code_manifest':manifest}
    s.verify_runtime_code(plan)
    source.write_text('x = 2\n',encoding='utf8')
    with pytest.raises(s.IntegrityError):
        s.verify_runtime_code(plan)
    source.write_text('x = 1\n',encoding='utf8')
    (code/'new.py').write_text('x = 3\n',encoding='utf8')
    with pytest.raises(s.IntegrityError):
        s.verify_runtime_code(plan)


def test_last_commit_at_same_update_uses_publication_time(tmp_path):
    import os
    s = module('supervisor')
    directory = tmp_path/'stages/S3/checkpoints'
    directory.mkdir(parents=True)
    refs = []
    for index in (1,2):
        path = directory/f'update_00020000_snapshot{index}.pt'
        path.write_bytes(str(index).encode())
        refs.append(s.file_ref(path))
        marker = path.with_suffix('.commit.json')
        s.write_new_json(marker,{'schema':'native_v5_training_commit_v1','checkpoint':refs[-1],
            'global_update':20000,'stage_update':4000,'training_state_sha256':'a'*64})
        os.utime(marker,ns=(index*10**9,index*10**9))
    assert s.latest_committed(tmp_path,'S3')['checkpoint'] == refs[-1]


def test_completion_requires_exported_weights_and_summary():
    s, p = module('supervisor'), module('run_plan')
    plan = p.make_plan('/data/run','/repo',{'source':'direct_user_message','text':'批准20k'})
    assert any('export' in error for error in s.completion_errors(plan, {}))
    assert s.next_task({'accepted':{key:{} for key in (
        'prepare_data','prepare_cache','data_preflight','eval_test',
        *[phase+'.'+action for phase in p.PHASES for action in ('stage_grad','stage_train')])}}) == ('export_final',None)


def test_model_export_omits_optimizer_and_reuses_verified_commit(tmp_path):
    import torch
    s, w = module('supervisor'), module('stage_worker')
    source = tmp_path/'source.pt'
    payload = {'architecture':'fixture','state_dict':{'weight':torch.tensor([1.])},
               'runtime_config':{'config':{}}, 'metadata':{},
               'training_schema':'fixture','training_state':{'optimizer_state':{'large':'not for inference'}}}
    torch.save(payload,source)
    source_ref = s.file_ref(source)
    first = w.export_model_weights(payload,source_ref,tmp_path/'export')
    exported = torch.load(first['path'],weights_only=True)
    assert 'training_state' not in exported and 'training_schema' not in exported
    assert torch.equal(exported['state_dict']['weight'],payload['state_dict']['weight'])
    assert w.export_model_weights(payload,source_ref,tmp_path/'export') == first
    Path(first['path']).write_bytes(b'corrupted weights')
    with pytest.raises(ValueError):
        w.export_model_weights(payload,source_ref,tmp_path/'export')


def test_full_loss_evaluation_rejects_missing_or_duplicate_pairs(monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,stage_preset
    from src.pose_control.qwen21_full.native_v5.training import probes
    w = module('stage_worker')
    data = SimpleNamespace(datasets={'val':range(3)},source_specs={'val':SimpleNamespace(role='validation')})
    calls=[]
    def probe(backend,config,data,update):
        calls.append(config.probe_samples)
        return {'records':[{'source_id':'val','index':i,'objective':1.} for i in (0,1,2)]}
    monkeypatch.setattr(probes,'loss_probe',probe)
    config=replace(TrainingConfig(),probe_samples=10,stage=stage_preset('S3'),sources=(SourceSpec('val',role='validation'),))
    config=replace(config,stage=replace(config.stage,routes=tuple(r for r in config.stage.routes if r.branch=='joint')))
    report=w.full_evaluation_loss(None,config,data,20000)
    assert report['evaluated_pairs']==3 and report['complete_split'] and calls==[3]
    monkeypatch.setattr(probes,'loss_probe',lambda *a:{'records':[{'source_id':'val','index':i,'objective':1.} for i in (0,1,1)]})
    with pytest.raises(ValueError,match='coverage'):
        w.full_evaluation_loss(None,config,data,20000)


def test_active_plan_changes_only_through_append_only_revision(tmp_path):
    s=module('supervisor');j=s.EventJournal(tmp_path)
    first=s.write_new_json(tmp_path/'run_plan.json',{'repo_root':'old','total_updates':20000})
    j.append({'event':'plan','plan':first})
    assert s.active_plan_reference(j.read())==first
    second=s.write_new_json(tmp_path/'revision.json',{'repo_root':'new','total_updates':20000})
    j.append({'event':'code_revision','plan':second,'parent':first,'successful_updates':0})
    assert s.active_plan_reference(j.read())==second
    assert s.read_ref(first)['repo_root']=='old'
    assert len(j.read())==2


@pytest.mark.parametrize('status,attempts,checkpoint',[
    ('running',{},False),('failed',{},False),('stopped',{'S0.stage_grad':1},False),
    ('stopped',{'S0.stage_train':1},False),('stopped',{},True)])
def test_code_update_refuses_running_terminal_or_production_started(status,attempts,checkpoint):
    u=module('update_training_code')
    state={'status':status,'attempts':attempts,'accepted':{'prepare_data':{'passed':True}}}
    with pytest.raises(ValueError):u.validate_pretraining_update(state,checkpoint)


def test_pretraining_update_requires_accepted_data_and_retains_prefix():
    u=module('update_training_code')
    with pytest.raises(ValueError):u.validate_pretraining_update({'status':'stopped','attempts':{},'accepted':{}},False)
    state={'status':'stopped','attempts':{'prepare_data':1,'prepare_cache':1},
           'accepted':{'prepare_data':{'passed':True,'prepared':{'path':'fixed','sha256':'a'*64}}}}
    before=json.dumps(state,sort_keys=True)
    u.validate_pretraining_update(state,False)
    assert json.dumps(state,sort_keys=True)==before


def test_recorded_code_update_preserves_original_plan_budget_and_data(tmp_path,monkeypatch):
    from contextlib import nullcontext
    s,u,p=module('supervisor'),module('update_training_code'),module('run_plan')
    monkeypatch.setattr(u,'single_owner',lambda *a:nullcontext())
    monkeypatch.setattr(u,'idle_worker',lambda *a:nullcontext())
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    for name in ('trainer.py','validation.py','parameters.py'):
        target=new/'src/pose_control/qwen21_full/native_v5/training'/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text('x=1\n')
    auth=s.write_new_json(tmp_path/'auth.json',{'source':'direct_user_message','text':'允许本次代码更新，预算不变'})
    plan=p.make_plan(str(tmp_path),str(old),s.read_ref(auth))
    original=s.write_new_json(tmp_path/'run_plan.json',plan)
    report=s.write_new_json(tmp_path/'audit.json',{'passed':True,'optimizer_step_executed':False,
        'baseline_repo':str(old),'repo_root':str(new),'new_code':{name:s.file_ref(new/'src/pose_control/qwen21_full/native_v5/training'/name)
        for name in ('trainer.py','validation.py','parameters.py')}})
    accepted=s.write_new_json(tmp_path/'prepared_result.json',{'passed':True,'prepared':{'binding':'unchanged'}})
    journal=s.EventJournal(tmp_path);journal.append({'event':'plan','plan':original})
    journal.append({'event':'accepted','task':'prepare_data','result':accepted})
    journal.append({'event':'stopped'})
    revision=u.record_update(tmp_path,new,auth,report)
    assert s.read_ref(original)==plan
    changed=s.read_ref(revision)
    assert changed['repo_root']==str(new) and changed['stages']==plan['stages'] and changed['total_updates']==20000
    assert s.replay(journal.read())['accepted']['prepare_data']==s.read_ref(accepted)
    assert s.active_plan_reference(journal.read())==revision
    assert u.record_update(tmp_path,new,auth,report)==revision
