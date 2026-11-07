"""One authorized pre-training code update; data/model/cache/checkpoint SHA stay strict."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from supervisor import (DEFAULT_RUN, EventJournal, IntegrityError, active_plan_reference, audit_references,
    file_ref, read_ref, replay, request_stop, runtime_manifest_path, single_owner, snapshot_runtime_code,
    status, verify_runtime_code, write_new_json)


def production_attempted(state):
    return any(task.endswith(('.stage_grad','.stage_train')) for task in state['attempts'])


def validate_pretraining_update(state,checkpoint_present):
    if state['status']!='stopped' or checkpoint_present or production_attempted(state):
        raise IntegrityError('Code update requires a stopped run before any production gradient/train attempt or checkpoint')
    if state['accepted'].get('prepare_data',{}).get('passed') is not True:
        raise IntegrityError('Code update requires accepted complete data preparation')


@contextmanager
def idle_worker(root):
    import fcntl
    with (Path(root)/'worker.lock').open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise IntegrityError('A worker is still active') from error
        yield


def record_update(root,repo,authorization_ref,audit_ref):
    root,repo=Path(root).resolve(),Path(repo).resolve()
    with single_owner(root),idle_worker(root):
        journal=EventJournal(root);rows=journal.read();state=replay(rows)
        previous=active_plan_reference(rows);plan=read_ref(previous)
        if Path(plan['repo_root']).resolve()==repo:
            if plan.get('code_update',{}).get('audit')!=audit_ref:
                raise IntegrityError('Existing code update has a different acceptance report')
            return previous
        validate_pretraining_update(state,any((root/'stages').glob('*/checkpoints/*.commit.json')))
        authorization=read_ref(authorization_ref);report=read_ref(audit_ref)
        if authorization.get('source')!='direct_user_message' or not authorization.get('text'):
            raise IntegrityError('Explicit user code-update authorization required')
        if (report.get('passed') is not True or report.get('optimizer_step_executed') is not False or
                report.get('baseline_repo')!=plan['repo_root'] or report.get('repo_root')!=str(repo)):
            raise IntegrityError('Missing matching no-optimizer throughput equivalence report')
        audit_references(report)
        for name in ('trainer.py','validation.py','parameters.py'):
            actual=file_ref(repo/'src/pose_control/qwen21_full/native_v5/training'/name)
            if report['new_code'][name]!=actual:raise IntegrityError('Audited code changed: '+name)
        updated=deepcopy(plan);updated['repo_root']=str(repo)
        if report.get('cache_mode')=='on_demand':
            updated['cache_mode']='on_demand'
            updated['gpu_free_gib']=40  # Admit encoder + frozen DiT weights with a memory margin.
        roots=[repo,Path(__file__).resolve().parent]
        if (root/'source/legacy').is_dir():roots.append(root/'source/legacy')
        updated['runtime_code_manifest']=snapshot_runtime_code(roots,runtime_manifest_path(root,roots))
        updated['code_update']={'parent':previous,'authorization':authorization_ref,'audit':audit_ref,'successful_updates':0}
        verify_runtime_code(updated)
        revision=write_new_json(root/'source'/('run_plan_code_update_'+uuid.uuid4().hex+'.json'),updated)
        journal.append({'event':'code_revision','plan':revision,'parent':previous,'successful_updates':0,
                        'authorization':authorization_ref,'audit':audit_ref})
        if report.get('cache_mode')=='on_demand':
            for task in ('prepare_cache','data_preflight'):
                if task in state['accepted']:
                    journal.append({'event':'task_superseded','task':task,'reason':'Authorized on-demand cache protocol; historical result preserved','plan':revision})
        return revision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',default=DEFAULT_RUN)
    parser.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--authorization-json',required=True)
    parser.add_argument('--audit-report',required=True)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args();root=Path(args.run_root).resolve();repo=Path(args.repo_root).resolve()
    root.joinpath('control').mkdir(exist_ok=True)
    authorization_ref,audit_ref=file_ref(args.authorization_json),file_ref(args.audit_report)
    import fcntl
    with (root/'control/code_update.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Do not stop the producer halfway through hours of data preparation.
            while True:
                current=status(root)
                if production_attempted(current):raise IntegrityError('Production preflight/train already started; no automatic code update')
                if current['status'] in ('failed','completed','stopped') or not current['owner_alive']:
                    raise IntegrityError('Existing run stopped or failed before code-update handoff')
                if current['accepted'].get('prepare_data',{}).get('passed') is True:break
                print(json.dumps({'event':'waiting_accepted_data','task':current.get('task')}),flush=True)
                time.sleep(15)
            # Validate authorization and report before asking the old owner to stop.
            authorization=read_ref(authorization_ref);report=read_ref(audit_ref)
            active=read_ref(active_plan_reference(EventJournal(root).read()))
            if (authorization.get('source')!='direct_user_message' or not authorization.get('text') or
                report.get('passed') is not True or report.get('optimizer_step_executed') is not False or
                report.get('baseline_repo')!=active['repo_root'] or report.get('repo_root')!=str(repo)):
                raise IntegrityError('Code-update authorization or acceptance does not match')
            audit_references(report)
            print(json.dumps({'event':'stopping_after_data_acceptance','request':request_stop(root)}),flush=True)
            deadline=time.monotonic()+120
            while True:
                current=status(root)
                if not current['owner_alive'] and current['status']=='stopped':
                    try:
                        with idle_worker(root):break
                    except IntegrityError:pass
                if time.monotonic()>deadline:raise IntegrityError('Timed out awaiting the old owner/worker; no forced process termination')
                time.sleep(1)
            revision=record_update(root,repo,authorization_ref,audit_ref)
            launch=subprocess.run(['bash',str(Path(__file__).with_name('launch.sh')),'resume','--run-root',str(root),
                '--repo-root',str(repo),'--python',sys.executable,'--gpu',args.gpu],check=True,capture_output=True,text=True)
            time.sleep(3)
            current=status(root)
            if not current['owner_alive'] or current['active_repo_root']!=str(repo):
                raise IntegrityError('Updated owner did not start; inspect its launch log')
            result=write_new_json(root/'control'/('code_update_completed_'+uuid.uuid4().hex+'.json'),{
                'passed':True,'plan':revision,'authorization':authorization_ref,'audit':audit_ref,
                'launch':launch.stdout,'committed_updates_at_handoff':current['committed_updates'],
                'accepted_tasks_preserved':list(current['accepted']),'task':current.get('task')})
            print(json.dumps({'event':'code_update_completed','result':result}),flush=True)
        except BaseException:
            write_new_json(root/'control'/('code_update_failed_'+uuid.uuid4().hex+'.json'),
                           {'passed':False,'error':traceback.format_exc(),'authorization':authorization_ref,'audit':audit_ref})
            raise


if __name__=='__main__':main()
