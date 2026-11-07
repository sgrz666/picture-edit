"""One bounded supervisor action using the formal NativeV5Trainer APIs.

Each invocation is a child process so model/CUDA memory is released at every gate.
There is no legacy fixed-300 runner, optimizer smoke, or manual quality acceptance.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import traceback
import threading
import time
import uuid
from datetime import datetime
from contextlib import contextmanager
from collections import deque

from run_plan import PHASES, stage_config
from supervisor import (IntegrityError, canonical, checked_ref, classify_failure, digest,
                        file_ref, latest_committed, pending_stops, read_ref, verify_runtime_code, write_new_json)


class ActionProcessError(RuntimeError):
    def __init__(self, message, returncode):
        super().__init__(message)
        self.returncode = returncode
        self.failure_class = classify_failure(message, returncode)


def required_gradient_cases(data, route_names):
    result = {}
    for name in route_names:
        representatives = set()
        for (route, source), pool in data.pools.items():
            if route != name:
                continue
            for index in pool:
                status = data.status[(name, source, index)]
                # Keep exactly the equivalence key used by native gradient_preflight.
                key = (source, status['bucket'][2], status['bucket'][3],
                       tuple(k for k, v in status['branch_usable'].items() if v),
                       digest(status['capabilities']), bool(status['auxiliary_faces']))
                representatives.add(key)
        if not representatives:
            raise IntegrityError('Empty production gradient representatives for '+name)
        result[name] = len(representatives)
    return result


def immutable_config(path, value):
    if Path(path).exists():
        if digest(json.loads(Path(path).read_text(encoding='utf8'))) != digest(value):
            raise IntegrityError('Immutable stage config differs: '+str(path))
        return file_ref(path)
    return write_new_json(path, value)


def committed_report(path, binding, create):
    """A test report becomes reusable only after its exclusive SHA commit exists."""
    path = Path(path)
    marker = path.with_suffix('.commit.json')
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf8'))
        if saved.get('schema') != 'bilibili_evaluation_commit_v1' or saved.get('binding') != binding:
            raise IntegrityError('Committed evaluation binding differs')
        return checked_ref(saved['report'])
    actual = path if not path.exists() else path.with_name(path.stem+'_'+uuid.uuid4().hex+path.suffix)
    ref = write_new_json(actual, create())
    write_new_json(marker, {'schema':'bilibili_evaluation_commit_v1','binding':binding,'report':ref})
    return ref


def prepared_inputs(request):
    prepared_ref = request['accepted']['prepare_data']['prepared']
    prepared = read_ref(prepared_ref)
    if prepared.get('schema') != 'bilibili_single_prepared_v1':
        raise IntegrityError('Unexpected prepared data schema')
    return prepared


def prepare_data(request, output):
    command = [sys.executable, str(Path(__file__).with_name('data_prepare.py')), 'prepare',
               '--run-root', request['run_root'], '--repo-root', request['repo_root']]
    tail = deque(maxlen=100)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in child.stdout:
        print(line, end='', flush=True)
        tail.append(line)
    if child.wait() != 0:
        raise ActionProcessError(''.join(tail), child.returncode)
    path = Path(request['run_root'])/'prepared.json'
    prepared = json.loads(path.read_text(encoding='utf8'))
    if prepared.get('schema') != 'bilibili_single_prepared_v1':
        raise IntegrityError('Data prepare did not produce the expected manifest')
    return {'passed': True, 'prepared': file_ref(path), 'counts': prepared['counts'],
            'training_started': False, 'optimizer_created': False}


def config_for_phase(request, phase, *, cache=True):
    plan = read_ref(request['plan'])
    prepared = prepared_inputs(request)
    accepted = request['accepted']
    previous = PHASES[PHASES.index(phase)-1] if phase != 'S0' else None
    initialization = accepted[previous+'.stage_train']['checkpoint'] if previous else prepared['initialization']
    checked_ref(initialization)
    identity = accepted['prepare_cache']['cache_identity'] if cache else None
    body = accepted.get('S0.stage_train', {}).get('body_gate')
    return stage_config(plan, prepared, phase, initialization=initialization,
                        cache_identity=identity, body_gate=body)


def prepare_cache_action(request, output):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    raw = config_for_phase(request, 'S0', cache=False)
    config = TrainingConfig.from_dict(raw)
    prepared=prepared_inputs(request)
    native_audit=read_ref(prepared['audits']['native_data'])
    if native_audit.get('passed') is not True or native_audit.get('pairs_read')!=12156:
        raise IntegrityError('Full Native data qualification required before frozen encoding')
    qualification=checked_ref(native_audit['train_val_route_audit'])
    identity = NativeV5Trainer(config).cache_prepare(config.cache_root,resume=Path(config.cache_root).exists(),qualification=qualification,mode='on_demand')
    checked_ref(identity)
    index = read_ref(identity)
    expected = sum(len(json.loads(Path(s['groups']['path']).read_text(encoding='utf8'))) for s in raw['sources'])
    if index.get('loading_mode')!='on_demand' or index.get('pair_count')!=expected or len(index['condition_bindings'])!=expected:
        raise IntegrityError('Frozen train/validation cache count differs')
    cached=sum(1 for _ in (Path(config.cache_root)/'entries').glob('*.json'))
    return {'passed': True, 'cache_identity': identity, 'entries': cached,'declared_pairs':expected,
            'test_included': False, 'optimizer_created': False,'loading_mode':'on_demand','all_pairs_preencoded':False}


def data_preflight_configs(plan, prepared, identity):
    """Preview route eligibility without claiming local heads or stages are train-ready."""
    return [stage_config(plan, prepared, phase, initialization=prepared['initialization'],
                         cache_identity=identity, body_gate=None) for phase in PHASES]


def data_preflight(request, output):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    plan = read_ref(request['plan'])
    prepared = prepared_inputs(request)
    identity = request['accepted']['prepare_cache']['cache_identity']
    phases = {}
    for raw in data_preflight_configs(plan, prepared, identity):
        config = TrainingConfig.from_dict(raw)
        if config.stage.phase!='S0':continue  # Local stages retain their own real data/gradient gate at entry.
        phase = config.stage.phase
        report = output/(phase+'_data_preflight.json')
        result = NativeV5Trainer(config).preflight_data(report)
        if result.get('passed') is not True or result.get('all_rows_retained') is not True:
            raise IntegrityError('Complete train/validation data preflight not passed for '+phase)
        pools = {}
        for row in result['rows']:
            if row['eligible']:
                key = row['role']+'/'+row['route']+'/'+row['source_id']
                pools[key] = pools.get(key,0)+1
        # Native DataPlan already rejects empty requested train and validation pools.
        phases[phase] = {'report':file_ref(report),'eligible_pool_counts':pools,
                         'model_constructed':False,'optimizer_created':False,'training_ready_claimed':False}
    report_ref = write_new_json(output/'all_stage_data_preflight.json', {
        'schema':'bilibili_all_stage_data_preflight_v1','passed':True,'phases':phases,
        'all_rows_retained':True,'future_stage_initialization_for_data_only':True,'deferred_phase_gates':['S1','S2','S3']})
    return {'passed':True,'report':report_ref,'all_rows_retained':True,'phases':list(phases)}


def stage_grad(request, output):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.streaming import training_data_plan
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    raw = config_for_phase(request, request['phase'])
    config = TrainingConfig.from_dict(raw)
    data = training_data_plan(config)
    required = required_gradient_cases(data, [r.name for r in config.stage.routes])
    raw['gradient_case_budget'] = max(required.values())
    config = TrainingConfig.from_dict(raw)
    path = Path(request['run_root'])/'stages'/request['phase']/'config.json'
    config_ref = immutable_config(path, config.record())
    write_new_json(output/'gradient_case_coverage.json', {'required_per_route': required,
                   'budget_per_route': raw['gradient_case_budget'], 'truncated': False})
    report_path = output/'gradient_preflight.json'
    report = NativeV5Trainer(config).preflight_grad(report_path)
    if (report.get('passed') is not True or report.get('production_passed') is not True
            or report.get('optimizer_created') is not False or report.get('optimizer_step_executed') is not False):
        raise IntegrityError('Actual production gradient preflight not passed')
    return {'passed': True, 'production_passed': True, 'config': config_ref,
            'report': file_ref(report_path), 'required_per_route': required, 'optimizer_created': False}


def final_validation_ok(state, global_update):
    return (state.get('passed') is True and state.get('finite') is True and not state.get('triggered')
            and state.get('last_probe_update') == global_update
            and state.get('last_generation_update') == global_update
            and bool(state.get('selection')))


def stage_train(request, output):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v5.training.validation import build_automatic_body_gate
    phase = request['phase']
    accepted = request['accepted']
    gate = accepted[phase+'.stage_grad']
    raw = read_ref(gate['config'])
    raw['gradient_report'] = checked_ref(gate['report'])
    config = TrainingConfig.from_dict(raw)
    immutable_config(Path(config.output_root)/'execution_config.json', config.record())
    committed = latest_committed(request['run_root'], phase)
    result = None
    if committed:
        saved, bound, _ = load_committed(committed['checkpoint'])
        state = saved['training_state']
        if state['config_sha256'] != config.signature():
            raise IntegrityError('Recovered committed stage config differs')
        if state['stage_update'] > config.stage.budget_updates:
            raise IntegrityError('Committed stage exceeds authorized budget')
        if state['stage_update'] == config.stage.budget_updates and final_validation_ok(state.get('validation_state', {}), state['global_update']):
            result = {'checkpoint': bound, 'commit': committed['commit']}
        del saved
    if result is None:
        mode = 'resume' if committed else 'train' if phase == 'S0' else 'advance-stage'
        checkpoint = committed['checkpoint'] if committed else None if phase == 'S0' else config.initialization
        result = NativeV5Trainer(config).run(mode=mode, checkpoint=checkpoint)
    saved, ref, _ = load_committed(result['checkpoint'])
    state = saved['training_state']
    expected = sum(s['budget_updates'] for s in read_ref(request['plan'])['stages'][:PHASES.index(phase)+1])
    validation = state.get('validation_state', {})
    if state['stage_update'] != config.stage.budget_updates or state['global_update'] != expected:
        raise IntegrityError('Training exited before committing the entire stage budget')
    if not final_validation_ok(validation, expected):
        raise IntegrityError('Stage final finite validation/generation not passed')
    report = write_new_json(output/'final_validation.json', {
        'schema':'bilibili_single_stage_validation_v1', 'phase':phase, 'checkpoint':ref,
        'validation_state':validation, 'technical_only':True})
    value = {'passed': True, 'budget_completed':True, 'stage_update':state['stage_update'],
             'global_update':state['global_update'], 'checkpoint':ref, 'commit':result['commit'],
             'validation_state':validation, 'validation_report':report, 'config':file_ref(Path(config.output_root)/'execution_config.json'),
             'optimizer_state_retained':phase != 'S0', 'technical_only':True}
    del saved
    if phase == 'S0':
        authorization = read_ref(request['plan'])['authorization']
        body = build_automatic_body_gate(ref, gate['report'], authorization)
        if body.get('acceptance_kind') != 'technical_training_gate_v1' or body.get('manual_acknowledgment') is True:
            raise IntegrityError('Body gate must record technical acceptance without fabricated manual acknowledgment')
        value['body_gate'] = write_new_json(output/'body_training_gate.json', body)
    return value


def generation_metrics(report, data):
    """Paired pixel metrics are descriptive only; never choose a training checkpoint."""
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    records = []
    by_id = {(source,row['pair_id']):i for source,dataset in data.datasets.items() for i,row in enumerate(dataset.rows)}
    for row in report['records']:
        sample = data.sample(row['source_id'], by_id[(row['source_id'],row['pair_id'])])
        target = ((sample['tgt_image'].float()+1)*.5).clamp(0,1).permute(1,2,0).numpy()
        predicted = np.asarray(Image.open(row['path']).convert('RGB'), dtype=np.float32)/255.
        if target.shape != predicted.shape:
            raise IntegrityError('Generated and target canvas dimensions differ')
        valid = sample['valid_image_mask'].detach().cpu().numpy().squeeze().astype(bool)
        difference = (target-predicted)**2
        mse = float(difference[valid].mean())
        # Seven-pixel uniform SSIM, using existing torch rather than installing a dependency.
        x = torch.from_numpy(target.copy()).permute(2,0,1)[None]
        y = torch.from_numpy(predicted.copy()).permute(2,0,1)[None]
        mean_x, mean_y = (F.avg_pool2d(v, 7, 1, 3) for v in (x, y))
        var_x = F.avg_pool2d(x*x, 7, 1, 3)-mean_x*mean_x
        var_y = F.avg_pool2d(y*y, 7, 1, 3)-mean_y*mean_y
        covariance = F.avg_pool2d(x*y, 7, 1, 3)-mean_x*mean_y
        ssim_map = ((2*mean_x*mean_y+.01**2)*(2*covariance+.03**2)/
                    ((mean_x.square()+mean_y.square()+.01**2)*(var_x+var_y+.03**2)))[0].mean(0).numpy()
        # Exclude windows touching padding so arbitrary padded pixels do not alter the score.
        interior = F.avg_pool2d(torch.from_numpy(valid.astype(np.float32))[None,None],7,1,3)[0,0].numpy()>.999999
        if not interior.any():
            raise IntegrityError('No complete valid 7x7 window for held-out SSIM')
        ssim = float(ssim_map[interior].mean())
        records.append({'source_id':row['source_id'],'pair_id':row['pair_id'],'mse':mse,
                        'psnr_db':None if mse == 0 else -10*math.log10(mse), 'perfect_match':mse == 0,
                        'ssim':ssim,'valid_pixels':int(valid.sum()), 'generation_sha256':row['sha256']})
    return {'kind':'held_out_test_pixel_metrics','technical_only':True,'used_for_selection':False,
            'records':records, 'mse_mean':sum(r['mse'] for r in records)/len(records),
            'ssim_mean':sum(r['ssim'] for r in records)/len(records),
            'note':'Paired MSE/PSNR on valid pixels; uniform-window7 SSIM on fully valid windows; no human perceptual quality claim.'}


def full_evaluation_loss(backend, config, data, update):
    from src.pose_control.qwen21_full.native_v5.training.probes import loss_probe
    if len(config.stage.routes) != 1:
        raise IntegrityError('Full final evaluation requires the single joint route')
    expected = {(source.source_id,index) for source in config.sources if source.role == 'validation'
                for index in range(len(data.datasets[source.source_id]))}
    if not expected:
        raise IntegrityError('Empty final evaluation split')
    report = loss_probe(backend, replace(config, probe_samples=len(expected)), data, update)
    actual = [(row.get('source_id'),row.get('index')) for row in report['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise IntegrityError('Final evaluation loss coverage differs from the complete split')
    values = [row.get('objective',float('nan')) for row in report['records']]
    if not all(math.isfinite(value) for value in values):
        raise IntegrityError('Final evaluation loss is nonfinite')
    return {**report,'complete_split':True,'evaluated_pairs':len(expected),
            'objective_mean':sum(values)/len(values),'used_for_selection':False}


def eval_test(request, output):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan
    from src.pose_control.qwen21_full.native_v5.training.backend import NativeBackend, prepare_cache
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v5.training.observations import generation_probe
    prepared = prepared_inputs(request)
    final = request['accepted']['S3.stage_train']
    evaluation = Path(request['run_root'])/'evaluation'/'test'
    evaluation.mkdir(parents=True, exist_ok=True)
    raw = read_ref(final['config'])
    raw['sources'] = [deepcopy(prepared['test'])]
    raw['sources'][0]['role'] = 'validation'
    raw['stage']['routes'] = [r for r in raw['stage']['routes'] if r['branch']=='joint']
    for route in raw['stage']['routes']:
        route['source_ids'] = []
    raw['initialization'] = checked_ref(final['checkpoint'])
    raw['gradient_report'] = None
    raw['cache_identity'] = None
    raw['cache_root'] = str(Path(request['run_root'])/'frozen_qwen_test')
    raw['output_root'] = str(evaluation)
    raw['body_baseline_report'] = None
    config = TrainingConfig.from_dict(raw)
    data = DataPlan(config, evaluation_only=True)
    identity = prepare_cache(config, data, config.cache_root, resume=Path(config.cache_root).exists())
    config = replace(config, cache_identity=identity)
    # The final test is a single logical evaluation. Preserve a completed loss/report
    # across transient process failure instead of repeating the held-out loss pass.
    immutable_config(evaluation/'test_config.json', config.record())
    saved, ref, _ = load_committed(final['checkpoint'])
    if saved['training_state']['global_update'] != 20000:
        raise IntegrityError('Held-out test requires the full 20000 committed updates')
    backend = NativeBackend(config, data, saved=saved, purpose='eval')
    del saved
    binding = {'checkpoint_sha256':ref['sha256'],'config_sha256':config.signature(),
               'data_signature':data.signature,'test_only':True,'global_update':20000}
    # Full held-out loss, while image generation remains bounded to ten examples.
    loss_ref = committed_report(evaluation/'test_loss.json',binding,lambda:full_evaluation_loss(backend,config,data,20000))
    generation_ref = committed_report(evaluation/'test_generation.json',binding,
        lambda:generation_probe(backend,config,data,20000,evaluation/'images'))
    generation = read_ref(generation_ref)
    if len(generation['records']) != 10:
        raise IntegrityError('Final held-out generation must contain ten fixed images')
    # Reused results are tied to the immutable final checkpoint/config; audit every image.
    from supervisor import audit_references
    audit_references(generation)
    metrics_ref = committed_report(evaluation/'test_metrics.json',binding,lambda:generation_metrics(generation,data))
    del backend
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    validation_config = TrainingConfig.from_dict(read_ref(final['config']))
    validation_data = DataPlan(validation_config)
    validation_binding = {'checkpoint_sha256':ref['sha256'],'config_sha256':validation_config.signature(),
                          'data_signature':validation_data.signature,'validation_only':True,'global_update':20000}
    def final_validation():
        payload, _, _ = load_committed(final['checkpoint'])
        validation_backend = NativeBackend(validation_config, validation_data, saved=payload, purpose='eval')
        return full_evaluation_loss(validation_backend,validation_config,validation_data,20000)
    validation_ref = committed_report(evaluation.parent/'final_validation_loss.json',validation_binding,final_validation)
    return {'passed':True,'technical_only':True,'checkpoint':ref,'loss_report':loss_ref,
            'generation_report':generation_ref,'metrics_report':metrics_ref,
            'validation_loss_report':validation_ref,
            'test_pairs':sum(len(d) for d in data.datasets.values()),'optimizer_created':False,
            'used_for_stage_selection':False,'cache_identity':identity,
            'evaluation_commits':[file_ref(p) for p in sorted(evaluation.parent.rglob('*.commit.json'))]}


def export_model_weights(saved, checkpoint, root):
    """Export the native inference payload only, with an immutable source binding."""
    import torch
    checkpoint = checked_ref(checkpoint)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'adapter.commit.json'
    if marker.exists():
        record = json.loads(marker.read_text(encoding='utf8'))
        if record.get('schema') != 'bilibili_adapter_export_v1' or record.get('checkpoint') != checkpoint:
            raise IntegrityError('Model export source binding differs')
        return checked_ref(record['weights'])
    payload = {key:value for key,value in saved.items() if key not in ('training_schema','training_state')}
    if 'state_dict' not in payload or 'runtime_config' not in payload:
        raise IntegrityError('Final checkpoint lacks Native inference weights/configuration')
    pending = root/'.pending'/('adapter_'+uuid.uuid4().hex+'.pt')
    pending.parent.mkdir(exist_ok=True)
    with pending.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    target = root/'native_v5_single_spatial_20000.pt'
    if target.exists():
        target = target.with_name(target.stem+'_'+uuid.uuid4().hex+'.pt')
    os.link(pending, target)
    if os.name == 'posix':
        fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    weights = file_ref(target)
    write_new_json(marker, {'schema':'bilibili_adapter_export_v1','checkpoint':checkpoint,'weights':weights,
                            'successful_updates':20000,'optimizer_included':False})
    return weights


def export_final(request, output):
    from supervisor import EventJournal, audit_references
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    plan, accepted = read_ref(request['plan']), request['accepted']
    final = accepted['S3.stage_train']
    saved, checkpoint, _ = load_committed(final['checkpoint'])
    if saved['training_state']['global_update'] != 20000 or accepted['eval_test'].get('passed') is not True:
        raise IntegrityError('Final export requires 20000 updates and independent test acceptance')
    root = Path(request['run_root'])/'export'
    weights = export_model_weights(saved, checkpoint, root)
    del saved
    records = EventJournal(request['run_root']).read()
    stages = []
    for spec in plan['stages']:
        phase = spec['phase']
        trained = accepted[phase+'.stage_train']
        if trained['stage_update'] != spec['budget_updates']:
            raise IntegrityError('Final export stage budget differs: '+phase)
        start, elapsed = None, 0.
        for row in records:
            if row['event'] == 'attempt' and row.get('task') == phase+'.stage_train':
                start = datetime.fromisoformat(row['time'])
            elif start is not None and (row['event'] == 'stopped' or
                    row['event'] in ('accepted','attempt_failed') and row.get('task') == phase+'.stage_train'):
                elapsed += (datetime.fromisoformat(row['time'])-start).total_seconds()
                start = None
        stages.append({'phase':phase,'successful_updates':trained['stage_update'],
                       'global_update':trained['global_update'],'attempt_wall_seconds':elapsed,
                       'checkpoint':trained['checkpoint'],'validation_report':trained['validation_report'],
                       'validation_state':trained['validation_state']})
    summary = {'schema':'bilibili_single_training_summary_v1','successful_updates':20000,
               'technical_only':True,'stages':stages,'weights':weights,'checkpoint':checkpoint,
               'test':accepted['eval_test'],'authorization':plan['authorization_ref'],
               'runtime_code_manifest':plan['runtime_code_manifest'],
               'elapsed_note':'Per-stage attempt wall time includes failed attempts and probes; excludes resource waits and preparation.'}
    summary_ref = immutable_config(root/'summary.json', summary)
    text = '# 单人 Native V5 空间加权训练产物\n\n累计成功 optimizer 更新：20,000。技术验收通过，不代表人工画质验收。\n\n'
    text += 'Adapter 权重（不含 optimizer）：`'+weights['path']+'`\n\n'
    text += '完整恢复断点：`'+checkpoint['path']+'`\n\n'
    text += '用 `NativeFullRuntimeV5.from_checkpoint(torch.load(权重路径, map_location="cpu", weights_only=True))` 加载 Adapter。\n\n'
    text += '|阶段|成功更新|尝试耗时（秒）|最终验证 loss|\n|---|---:|---:|---:|\n'
    for stage in stages:
        text += f"|{stage['phase']}|{stage['successful_updates']}|{stage['attempt_wall_seconds']:.1f}|{stage['validation_state']['last_loss']:.8g}|\n"
    text += '\n完整 SHA、验证和独立 test 报告见 `summary.json`。完整断点保留 optimizer/RNG；主管仅允许中断且未完成的运行使用 `resume`。推理权重不能恢复 optimizer。\n'
    readme = root/'README.md'
    if readme.exists():
        if readme.read_text(encoding='utf8') != text:
            raise IntegrityError('Final README differs on retry')
    else:
        pending = root/'.pending'/('readme_'+uuid.uuid4().hex+'.md')
        with pending.open('xb') as stream:
            stream.write(text.encode('utf8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, readme)
    audit_references(summary)
    return {'passed':True,'successful_updates':20000,'checkpoint':checkpoint,
            'weights':weights,'summary':summary_ref,'readme':file_ref(readme),'technical_only':True}


ACTIONS = {'prepare_data':prepare_data,'prepare_cache':prepare_cache_action,'data_preflight':data_preflight,
           'stage_grad':stage_grad,'stage_train':stage_train,'eval_test':eval_test,'export_final':export_final}


@contextmanager
def worker_lock(root):
    """A surviving child and its resumed owner must never train concurrently."""
    import fcntl
    with (Path(root)/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def watch_stop_requests(request):
    """A child still honors stop if its owning supervisor disappeared."""
    ignored = set(request.get('ignored_stops', []))
    while True:
        if pending_stops(request['run_root'], ignored):
            print('Explicit stop requested; retaining only committed optimizer boundaries', flush=True)
            if os.name == 'posix' and os.getsid(0) == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGTERM)
            else:
                os.kill(os.getpid(), signal.SIGTERM)
            return
        time.sleep(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    request = read_ref({'path':str(Path(args.input).resolve()),'sha256':args.input_sha256})
    verify_runtime_code(read_ref(request['plan']))
    threading.Thread(target=watch_stop_requests, args=(request,), daemon=True).start()
    sys.path.insert(0, request['repo_root'])
    output = Path(request['run_root'])/'attempts'/((request['phase'] or '')+request['action']+'_'+uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        torch.set_num_threads(4)
        with worker_lock(request['run_root']):
            result = ACTIONS[request['action']](request, output)
        write_new_json(args.result, {'schema':'bilibili_single_worker_result_v1','action':request['action'],
                                   'phase':request['phase'],'attempt_output':str(output),**result})
    except BaseException as error:
        traceback.print_exc()
        returncode = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        failure = getattr(error,'failure_class',None) or ('terminal' if isinstance(error, (ValueError, AssertionError, FloatingPointError)) else classify_failure(str(error), returncode))
        write_new_json(args.result, {'schema':'bilibili_single_worker_result_v1','passed':False,
                                   'failure_class':failure,'error_type':type(error).__name__,'error':str(error),
                                   'attempt_output':str(output)})
        raise


if __name__ == '__main__':
    main()
