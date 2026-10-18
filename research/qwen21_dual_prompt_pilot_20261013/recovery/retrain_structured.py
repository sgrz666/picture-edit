#!/usr/bin/env python3
"""Train structured afresh for300, retain short300, and publish verified results.

This operational wrapper stays outside the frozen implementation. Launch it
detached with stdout/stderr connected directly to a durable log file. Candidate
training is outside RUN; the pinned recovery helper audits and publishes only
after the unchanged trainer and evaluator prove all300 updates.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT=Path('/home/shangguanrz/project/pic-edit')
WORK=ROOT/'research/qwen21_dual_prompt_pilot_20261013'
RUN=ROOT/'experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1'
HELPER_SHA256='f682af355750b6fc4663648b5a8a658b2e9c4ce63929e2b5db539a2db127b4a8'
SCHEMA='qwen21_dual_prompt_structured_fresh_retraining_v1'


def load_helper(path):
    """Pin executable operational dependencies before importing them."""
    path=Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=HELPER_SHA256:
        raise ValueError('operational recovery dependency SHA differs from reviewed f682 source')
    name='_qwen21_prompt_recovery_f682'
    spec=importlib.util.spec_from_file_location(name,path)
    helper=importlib.util.module_from_spec(spec)
    sys.modules[name]=helper
    spec.loader.exec_module(helper)
    return helper


def reconstruct_common(helper,plan):
    root=plan['root']
    if (root.get('production_training'),root.get('requested_steps'),root.get('learning_rate'),root.get('save_every'),
        root.get('variants'),root.get('effective_batch'),root.get('accumulation'),root.get('test_data_read'))!=(
        True,300,1e-4,50,['short','structured'],4,4,False):
        raise ValueError('fresh fallback requires the original authorized matched300 experiment')
    common=copy.deepcopy({key:value for key,value in root.items() if key not in ('variants','save_every')})
    for variant in ('short','structured'):
        contract=plan['manifests'][variant]['run_contract']
        if any(key not in contract or contract[key]!=value for key,value in common.items()):
            raise ValueError('original branch common contract differs from root manifest')
    if plan['manifests']['short']['initial_adapter_sha256']!=plan['manifests']['structured']['initial_adapter_sha256']:
        raise ValueError('original initial adapter hashes differ')
    return common


def failed_attempt_proof(args,helper):
    path=args.failed_attempt/'recovery_result.json'
    result=helper.read_json(path)
    provenance=result.get('recovery',{})
    if (result.get('status'),result.get('error'),provenance.get('source_sha256'),provenance.get('committed_boundary'),
        provenance.get('resumed_steps'))!=('failed_no_completion_claim','replay gradient_norm divergence',HELPER_SHA256,200,[201,300]):
        raise ValueError('previous exact continuation failure is not authenticated')
    if 'replay gradient_norm divergence' not in args.failed_log.read_text(encoding='utf-8',errors='replace'):
        raise ValueError('previous failure log disagrees with result')
    if list(args.failed_attempt.rglob('training_complete.json')):
        raise ValueError('failed continuation contains unexpected completion markers')
    files={str(item.resolve()):helper.file_sha(item) for item in (path,args.failed_log)}
    return dict(exact_continuation_rejected=True,error=result['error'],committed_boundary=200,
                result_path=str(path.resolve()),log_path=str(args.failed_log.resolve()),files_sha256=files,
                discarded_original_uncheckpointed_steps=provenance.get('discarded_uncheckpointed_steps',[]),
                reuse_partial_optimizer_or_weights=False),files


def _audit_fresh_result(args,pkg,plan,helper,result,initial):
    expected=plan['manifests']['short']['initial_adapter_sha256']
    if initial!=expected or result.get('initial_adapter_sha256')!=expected:
        raise ValueError('fresh adapter initialization differs from matched original SHA')
    if (result.get('status'),result.get('schema_version'),result.get('production_training'),result.get('requested_steps'),
        result.get('completed_steps'),result.get('variant'),result.get('validation_selection'),
        result.get('caches_unchanged'),result.get('native_gradients_none'))!=(
        'success',helper.TRAINING_SCHEMA,True,300,300,'structured','true_val_split_only',True,True):
        raise ValueError('fresh trainer did not return a complete frozen300 production result')
    directory=args.attempt_dir/'training'/'structured'
    manifest=helper.read_json(directory/'run_manifest.json')
    helper.require_equal_contract(plan['manifests']['structured']['run_contract'],manifest['run_contract'])
    helper.require_equal_contract(plan['manifests']['structured']['loading_report'],manifest['loading_report'])
    warm=manifest['loading_report']
    if (len(warm['matched']),warm.get('source_step'),warm.get('source_step_verified'),
        warm['frozen_coverage'].get('complete'),warm['frozen_coverage'].get('matched_tensor_count'))!=(178,19500,True,True,178):
        raise ValueError('fresh warm loading did not prove all178 frozen tensors')
    gate=helper.read_json(directory/'real_model_gates.json')
    if (gate.get('zero_init_exact'),gate.get('gradients_none'),gate.get('optimizer_steps'),gate.get('source_step'),
        gate.get('initial_adapter_sha256'),gate['frozen_coverage'].get('complete'),
        gate['frozen_coverage'].get('matched_tensor_count'))!=(True,True,0,19500,expected,True,178):
        raise ValueError('actual fresh zero initialization gate is incomplete')
    rows=helper._read_metrics(directory/'train_metrics.jsonl')
    helper.verify_history(rows,plan['schedule'],expected_tail=300,save_every=50)
    validations=helper.validation_history(rows)
    chosen=min(validations,key=lambda row:(row['loss'],row['step']))
    if result['validation_records']!=validations or (result['best_step'],result['best_validation'])!=(chosen['step'],chosen['loss']):
        raise ValueError('fresh result does not match its actual six true-val records')
    for kind,step in (('last',300),('best',chosen['step'])):
        path=(directory/f'checkpoint_{kind}.pt').resolve()
        if result[f'{kind}_checkpoint']!=dict(path=str(path),sha256=helper.file_sha(path),step=step):
            raise ValueError('fresh exported checkpoint path/SHA/step differs from candidate')
    if list((args.attempt_dir/'training').rglob('training_complete.json')):
        raise ValueError('fresh trainer unexpectedly published a completion marker')
    return rows,gate


def train_fresh_structured(args,pkg,plan,helper,pipe):
    """Invoke only the original fresh loop, with the full original schedule."""
    helper.verify_files(plan['protected']);helper.verify_files(plan['cache_files'])
    plan['data'].assert_unchanged()
    common=reconstruct_common(helper,plan)
    output=args.attempt_dir/'training'
    if output.exists():raise ValueError('fresh training candidate directory already exists')
    call_args=SimpleNamespace(data_dir=args.data_dir,output=output,steps=300,variants=['short','structured'],lr=1e-4,
                              seed=common['seed'],save_every=50,model=Path(common['model_id']),warm_start=Path(common['warm_start_path']))
    pkg.trainer.validate_training_settings(call_args)
    cache=helper.read_json(args.run_dir/'cache_manifest.json')['structured']
    # Checkpoint200 was loaded exclusively for the CPU evidence audit. Never pass
    # its trained adapter or optimizer into the fresh training function.
    plan.pop('adapter',None);plan.pop('optimizer',None);gc.collect()
    result,initial=pkg.trainer.train_one_variant(call_args,pipe,plan['data'],plan['caches']['structured'],cache,
        'structured',common,plan['schedule'],plan['manifests']['short']['initial_adapter_sha256'])
    rows,gate=_audit_fresh_result(args,pkg,plan,helper,result,initial)
    helper.verify_files(plan['protected']);helper.verify_files(plan['cache_files'])
    plan['data'].assert_unchanged()
    return result,rows,gate


def assert_native_frozen(pipe):
    for name in ('transformer','vae','text_encoder'):
        component=getattr(pipe,name)
        if component.training or any(param.requires_grad or param.grad is not None for param in component.parameters()):
            raise ValueError(f'fresh native {name} is not frozen eval with no gradients')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,default=WORK)
    p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--mirror',type=Path)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--evidence-dir',type=Path)
    p.add_argument('--attempt-dir',type=Path)
    p.add_argument('--recovery-helper',type=Path,default=Path(__file__).with_name('recover_training.py'))
    p.add_argument('--failed-attempt',type=Path)
    p.add_argument('--failed-log',type=Path)
    p.add_argument('--preflight-only',action='store_true',help='CPU evidence audits; no native load, training or RUN writes')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.work_dir=args.work_dir.resolve();args.run_dir=args.run_dir.resolve()
    args.mirror=(args.mirror or args.work_dir/'implementation').resolve()
    args.data_dir=(args.data_dir or args.work_dir/'data').resolve()
    args.evidence_dir=(args.evidence_dir or args.work_dir/'recovery').resolve()
    args.attempt_dir=(args.attempt_dir or args.evidence_dir/'retrain_structured_v1_300').resolve()
    args.failed_attempt=(args.failed_attempt or args.evidence_dir/'continuation_v1_200_to_300').resolve()
    args.failed_log=(args.failed_log or args.evidence_dir/'recovery_training.log').resolve()
    args.recovery_helper=args.recovery_helper.resolve()
    source=Path(__file__).resolve()
    if any(path.is_relative_to(args.mirror) for path in (source,args.recovery_helper,args.attempt_dir)) or args.attempt_dir.is_relative_to(args.run_dir):
        raise ValueError('operational sources/output must stay outside MIRROR; candidate output must stay outside RUN')
    if args.attempt_dir.exists():raise ValueError('fresh attempt directory must be new; prior attempts stay preserved')
    helper=load_helper(args.recovery_helper)
    args.attempt_dir.mkdir(parents=True)
    recovery=dict(schema_version=SCHEMA,source_path=str(source),source_sha256=helper.file_sha(source),
                  helper_path=str(args.recovery_helper),helper_sha256=HELPER_SHA256,
                  started_utc=datetime.now(timezone.utc).isoformat(),training_execution='fresh_structured_steps1_to300',
                  retained_short_completed_steps=300,resumed_optimizer=False,frozen_implementation_edited=False,
                  original_final_native_memory_hash_observed=False,
                  bit_identical_curves_claimed=False,controlled_initialization_schedule_and_configuration=True)
    sibling=source.with_name('test_retrain_structured.py')
    if sibling.exists():recovery['cpu_regression_source_sha256']=helper.file_sha(sibling)
    published=None
    try:
        failure,failure_files=failed_attempt_proof(args,helper)
        recovery['failed_continuation']=failure
        pkg=helper._imports(args.mirror)
        torch.set_num_threads(8)
        plan=helper._preflight(args,pkg)
        reconstruct_common(helper,plan)
        plan['protected'].update(failure_files)
        plan['protected'][str(source)]=recovery['source_sha256']
        plan['protected'][str(args.recovery_helper)]=HELPER_SHA256
        recovery.update(original_live_inputs_sha256_at_preflight=copy.deepcopy(plan['protected']),
                        cache_files_sha256=plan['cache_files'],original_training_code_sha256=plan['root']['code_sha256'],
                        original_optimizer_boundary_proofs=plan['optimizer_proofs'],
                        replaced_partial_structured_steps=len(plan['structured_rows']))
        helper.atomic_json(args.attempt_dir/'retrain_preflight.json',dict(status='preflight_passed',recovery=recovery))
        # Release the audit-only partial state even for CPU preflight mode.
        plan.pop('adapter',None);plan.pop('optimizer',None);gc.collect()
        if args.preflight_only:
            helper.atomic_json(args.attempt_dir/'retrain_result.json',dict(status='preflight_passed_no_training_no_completion',recovery=recovery))
            print('FRESH_STRUCTURED_PREFLIGHT_PASSED',flush=True)
            return dict(status='preflight_passed_no_training_no_completion',recovery=recovery)
        with pkg.precision.qwen21_precision() as policy:
            helper.require_equal_contract(plan['root']['precision_policy'],policy)
            if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                raise ValueError('fresh native BF16 reduction policy is inactive')
            torch.manual_seed(plan['root']['seed']);torch.cuda.manual_seed_all(plan['root']['seed'])
            pipe=pkg.runtime.load_native(Path(plan['root']['model_id']))
            before=pkg.runtime.tensor_model_sha256(pipe.transformer)
            if before!=plan['root']['backbone_sha256']:raise ValueError('fresh native before SHA differs from original initial SHA')
            actual_code={name:helper.file_sha(inspect.getfile(type(component))) for name,component in
                         (('transformer',pipe.transformer),('pipeline',pipe),('vae',pipe.vae),('text_encoder',pipe.text_encoder))}
            helper.require_equal_contract(plan['root']['native_code_sha256'],actual_code)
            if helper.file_sha(Path(plan['root']['model_id'])/'transformer/config.json')!=plan['root']['model_config_sha256']:
                raise ValueError('fresh native model config SHA differs')
            assert_native_frozen(pipe)
            pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
            pipe.transformer.enable_gradient_checkpointing()
            result,rows,gate=train_fresh_structured(args,pkg,plan,helper,pipe)
            after=pkg.runtime.tensor_model_sha256(pipe.transformer)
            assert_native_frozen(pipe)
            scope=helper.native_integrity_report(plan['root']['backbone_sha256'],before,after)
            scope['original_process']['reason']='Original short300 completed before SSH interruption; its final native memory SHA was not observed.'
            scope['recovery_process']['scope']='Actual newly loaded native process for fresh structured steps1..300; before/after SHA observed.'
            recovery['native_integrity_scope']=scope
            candidate=args.attempt_dir/'training'/'structured'
            recovery['fresh_initialization_evidence']=dict(gate=gate,gate_path=str(candidate/'real_model_gates.json'),
                gate_sha256=helper.file_sha(candidate/'real_model_gates.json'),
                run_manifest_path=str(candidate/'run_manifest.json'),run_manifest_sha256=helper.file_sha(candidate/'run_manifest.json'),
                warm_matched_tensor_count=178,initial_adapter_sha256=result['initial_adapter_sha256'])
            for variant in plan['caches'].values():
                for samples in variant.values():
                    for sample in samples:sample.assert_unchanged()
            helper.verify_files(plan['protected']);helper.verify_files(plan['cache_files']);plan['data'].assert_unchanged()
            if helper.file_sha(plan['root']['warm_start_path'])!=plan['root']['warm_start_sha256']:raise ValueError('warm start changed')
            branches={
                'short':helper._branch_completion(args,plan,'short',plan['short_rows'],args.run_dir/'short/checkpoint_last.pt',
                    args.run_dir/'short/checkpoint_best.pt',scope,recovery,pkg),
                'structured':helper._branch_completion(args,plan,'structured',rows,candidate/'checkpoint_last.pt',
                    candidate/'checkpoint_best.pt',scope,recovery,pkg)}
            if branches['structured']['trained_adapter_sha256']!=result['trained_adapter_sha256']:
                raise ValueError('fresh final checkpoint tensors differ from actual final in-memory adapter')
            branches['short']['training_execution']='retained_original_short_steps1_to300'
            branches['structured']['training_execution']='fresh_structured_steps1_to300'
            branches['structured']['elapsed_seconds']=result['elapsed_seconds']
            branches['structured']['elapsed_seconds_scope']='Actual unchanged fresh trainer elapsed time; no interrupted prefix added.'
            root=dict(status='success',schema_version=helper.TRAINING_SCHEMA,production_training=True,requested_steps=300,completed_steps=300,
                dataset_path=str(plan['data'].dataset_path),dataset_sha256=plan['data'].dataset_sha256,split_manifest=plan['data'].split_manifest,
                model_id=plan['root']['model_id'],backbone_sha256_before=before,backbone_sha256_after=after,backbone_unchanged=True,
                initialization_sha256_equal=True,initial_adapter_sha256=plan['manifests']['short']['initial_adapter_sha256'],
                initial_adapter_sha256_by_variant={name:manifest['initial_adapter_sha256'] for name,manifest in plan['manifests'].items()},
                warm_start_sha256=plan['root']['warm_start_sha256'],train_pair_ids=plan['root']['train_pair_ids'],val_pair_ids=plan['root']['val_pair_ids'],
                validation_selection='true_val_split_only',effective_batch=4,schedule_sha256=plan['root']['schedule_sha256'],
                test_data_read=False,variants=branches,recovery_provenance=recovery,native_integrity_scope=scope)
            published=helper._stage_and_commit(args,plan,branches,root,candidate/'checkpoint_last.pt',candidate/'checkpoint_best.pt',rows,pkg)
            helper.atomic_json(args.attempt_dir/'retrain_result.json',dict(status='success',recovery=recovery,completed_steps=300,
                structured_optimizer_proof=branches['structured']['optimizer_completion_proof'],
                completion_sha256=helper.file_sha(args.run_dir/'training_complete.json')))
            print('FRESH_STRUCTURED_COMPLETE',str(args.run_dir/'training_complete.json'),flush=True)
            return root
    except BaseException as exc:
        if published is not None:helper.withdraw_own_completion_markers(published)
        helper.atomic_json(args.attempt_dir/'retrain_result.json',dict(status='failed_no_completion_claim',
            error_type=type(exc).__name__,error=str(exc),recovery=recovery))
        raise


if __name__=='__main__':main()
