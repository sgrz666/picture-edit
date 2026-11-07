"""Linux control-flow integration only: impossible resource gate, no model/optimizer worker."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--repo-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();production=Path(args.run_root).resolve();repo=Path(args.repo_root).resolve()
    scripts=repo/'experiments/bilibili-single-spatial-training/scripts'
    sys.path.insert(0,str(scripts))
    import supervisor as s
    import update_training_code as update
    root=production/'control_checks'/('code_update_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    original=deepcopy(s.read_ref(s.active_plan_reference(s.EventJournal(production).read())))
    original['run_root']=str(root)
    # No worker can pass this synthetic resource gate on this host.
    original.update(gpu_free_gib=10**9,host_available_gib=10**9,resource_poll_seconds=1)
    original_ref=s.write_new_json(root/'run_plan.json',original)
    accepted=s.write_new_json(root/'control_only_prepared_fixture.json',{
        'passed':True,'schema':'control_flow_fixture_not_production_data_acceptance','control_test_only':True})
    journal=s.EventJournal(root)
    journal.append({'event':'plan','plan':original_ref})
    journal.append({'event':'accepted','task':'prepare_data','result':accepted,'control_test_only':True})
    journal.append({'event':'stopped','control_test_only':True})
    auth=s.file_ref(production/'source/performance_update_authorization.json')
    audit=s.file_ref(production/'source/throughput_audit_v6.json')
    revised=update.record_update(root,repo,auth,audit)
    assert s.read_ref(original_ref)==original
    assert s.replay(journal.read())['accepted']['prepare_data']==s.read_ref(accepted)
    assert update.record_update(root,repo,auth,audit)==revised
    with s.single_owner(root):
        try:
            with s.single_owner(root):raise AssertionError('Duplicate supervisor lock unexpectedly succeeded')
        except RuntimeError as error:
            assert 'already owns' in str(error)
    log=root/'resume_control.log'
    with log.open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(scripts/'supervisor.py'),'resume','--run-root',str(root),
                                '--repo-root',str(repo),'--python',sys.executable],
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+45
            while True:
                state=s.status(root)
                if state['status']=='waiting_resources':break
                if child.poll() is not None:raise AssertionError(log.read_text(encoding='utf8'))
                if time.monotonic()>deadline:raise AssertionError('Control resume did not reach impossible resource gate')
                time.sleep(.5)
            assert state['active_plan']==revised and state['active_repo_root']==str(repo)
            assert state['committed_updates']==0 and not state['attempts']
            assert not list(root.glob('requests/*.json'))
            s.request_stop(root);assert child.wait(timeout=20)==0
        finally:
            if child.poll() is None:s.terminate_owned_child(child)
    final=s.status(root)
    assert final['status']=='stopped' and not final['owner_alive']
    assert not list(root.glob('stages/*/checkpoints/*'))
    result={'schema':'native_code_update_control_integration_v1','passed':True,'control_test_only':True,
            'model_constructed':False,'optimizer_created':False,'optimizer_step_executed':False,
            'original_plan_preserved':s.read_ref(original_ref)==original,'accepted_prefix_preserved':True,
            'duplicate_owner_rejected':True,'resumed_active_plan':revised,'resource_gate_prevented_worker':True,
            'explicit_stop_passed':True,'fixture_root':str(root),'log':s.file_ref(log),'producer':s.file_ref(__file__)}
    print(json.dumps(s.write_new_json(args.output,result)),flush=True)


if __name__=='__main__':main()
