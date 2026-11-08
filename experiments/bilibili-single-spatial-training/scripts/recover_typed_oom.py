"""Repair a proven OOM classification, separate frozen encoding, resume exact worker code."""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from supervisor import (EventJournal,active_plan_reference,read_ref,file_ref,write_new_json,digest,
    single_owner,snapshot_runtime_code,verify_runtime_code,resources,status,pending_stops,checked_ref)


def wait_memory(root,minimum,ignored):
    while True:
        if pending_stops(root,ignored):raise RuntimeError('Explicit stop requested during resource recovery')
        available=resources('0')
        if available['gpu_free_gib']>=minimum:return available
        print(json.dumps({'event':'recovery_waiting_resources',**available}),flush=True)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args();root=Path(args.run_root).resolve()
    import fcntl
    with (root/'control/oom_recovery.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=EventJournal(root);rows=journal.read();current=status(root)
        if current['status']!='failed' or current['owner_alive']:raise ValueError('Recovery requires an inactive failed supervisor')
        failure=next(row for row in reversed(rows) if row['event']=='attempt_failed')
        worker_result=file_ref(failure['result_path']);error=read_ref(worker_result)
        if error.get('error_type') not in ('OutOfMemoryError','CUDAOutOfMemoryError') or error.get('passed') is not False:
            raise ValueError('Only a proven typed OOM can use this recovery path')
        if current['failure_counts'][failure['task']]>=3:raise ValueError('Original resource retry budget exhausted')
        plan_ref=active_plan_reference(rows);plan=read_ref(plan_ref);verify_runtime_code(plan)
        repo=Path(plan['repo_root']);sys.path.insert(0,str(repo))
        sys.path.insert(0,str(repo/'experiments/bilibili-single-spatial-training/scripts'))
        import torch
        torch.set_num_threads(4)
        from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
        from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,digest as config_digest
        from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
        from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
        from src.pose_control.qwen21_full.native_v5.training.backend import load_cache_pipeline,encode_frozen_pair,model_identity
        checkpoint=current['stage_commits']['S0']['checkpoint']
        saved,bound,_=load_committed(checkpoint);training=saved['training_state']
        if not training['validation_state']['passed'] or training['validation_state']['triggered']:
            raise ValueError('Cannot resource-recover a failed validation guard')
        gradient=read_ref(training['gradient_report'])
        if gradient['frozen_before']!=training['frozen_state'] or gradient['frozen_after']!=training['frozen_state']:
            raise ValueError('Frozen state acceptance differs')
        config=TrainingConfig.from_dict(training['config']);cache_state=training['frozen_cache_state']
        for ref in cache_state['used'].values():checked_ref(ref)
        saved_update=training['global_update'];del saved,training
        descriptor=read_ref(config.cache_identity)
        frozen=FrozenInputPlan(config,descriptor['qualification'],lazy=True)
        manifest,_=model_identity(config)
        identity={'model_manifest_sha256':manifest['sha256'],'profile':config.qwen_condition_profile,
                  'prompt_sha256':config_digest(config.prompt),'sources':frozen.identity['sources']}
        store=FrozenCacheStore(config.cache_root,identity,resume=True)
        ignored={ref['sha256'] for ref in pending_stops(root,set())}
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output=root/'source'/('oom_recovery_'+tag);output.mkdir(exist_ok=False)
        controller_manifest=snapshot_runtime_code([Path(__file__).resolve().parent],output/'controller_manifest.json')
        write_new_json(output/'started.json',{'worker_result':worker_result,'checkpoint':bound,
            'checkpoint_update':saved_update,'training_code_unchanged':True,'plan':plan_ref,'scope':'Frozen-encoding-only process before exact optimizer resume'})
        pending=[];total=sum(len(ds) for ds in frozen.datasets.values());scanned=0
        for source,dataset in frozen.datasets.items():
            for index in range(len(dataset)):
                if pending_stops(root,ignored):raise RuntimeError('Explicit recovery stop')
                binding=frozen.binding(source,index,manifest['sha256'],config_digest(config.prompt))
                if store.get(binding) is None:pending.append((source,index,binding))
                scanned+=1
                if scanned%250==0 or scanned==total:
                    print(json.dumps({'event':'recovery_cache_audit','scanned':scanned,'total':total,'missing':len(pending)}),flush=True)
        done=total-len(pending);retry=0;pipe=None
        while pending:
            wait_memory(root,28,ignored)
            try:
                pipe=load_cache_pipeline(config)
                pipe.text_encoder.requires_grad_(False).eval();pipe.vae.requires_grad_(False).eval()
                versions=[(p,p._version) for model in (pipe.text_encoder,pipe.vae) for p in model.parameters()]
                with torch.no_grad():
                    while pending:
                        if pending_stops(root,ignored):raise RuntimeError('Explicit recovery stop')
                        source,index,binding=pending[0];spec=frozen.conditions[(source,index)][1]
                        pair=encode_frozen_pair(pipe,frozen.datasets[source][index],config,spec)
                        if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in versions):
                            raise RuntimeError('Frozen encoder parameter changed during cache-only recovery')
                        store.put(binding,pair);pending.pop(0);done+=1
                        if done%20==0 or not pending:print(json.dumps({'event':'recovery_frozen_encoded','completed':done,'total':total}),flush=True)
            except torch.OutOfMemoryError:
                retry+=1
                if retry>2:raise
                print(json.dumps({'event':'recovery_encoder_oom','retry':retry}),flush=True)
            finally:
                if pipe is not None:del pipe;pipe=None
                gc.collect();torch.cuda.empty_cache()
        complete=write_new_json(output/'cache_complete.json',{'passed':True,'cached_pairs':total,'cache_identity':config.cache_identity,
                  'no_optimizer_constructed':True,'training_model_not_resident_during_encoding':True})
        evidence=write_new_json(output/'classification_evidence.json',{'schema':'native_oom_reclassification_v1',
            'failure_record_sha256':digest(failure),'worker_result':worker_result,'checkpoint':bound,'cache_backfill':complete})
        # The old event chain and failed result remain untouched.
        with single_owner(root):
            latest=status(root)
            if latest['status']!='failed' or latest['owner_alive']:raise ValueError('Run changed during recovery')
            journal.append({'event':'resource_failure_reclassified','task':failure['task'],'attempt':failure['attempt'],
                            'evidence':evidence,'reason':'Typed torch.OutOfMemoryError was misclassified by short message'})
        verify_runtime_code({'runtime_code_manifest':controller_manifest})
        worker=repo/'experiments/bilibili-single-spatial-training/scripts/stage_worker.py'
        log=root/'logs'/('resource_resume_'+tag+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('supervisor.py')),'resume',
                '--run-root',str(root),'--repo-root',str(repo),'--python',sys.executable,'--gpu','0',
                '--worker-script',str(worker),'--controller-manifest',controller_manifest['path'],
                '--controller-manifest-sha256',controller_manifest['sha256']],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        report=write_new_json(output/'resume_launched.json',{'passed':True,'supervisor_pid':child.pid,'log':str(log),
            'resume_checkpoint':bound,'resume_from_update':saved_update,'optimizer_rng_preserved':True,'plan':plan_ref,
            'controller_manifest':controller_manifest,'original_retry_budget_preserved':True})
        print(json.dumps({'event':'resource_recovery_resumed','report':report}),flush=True)


if __name__=='__main__':main()
