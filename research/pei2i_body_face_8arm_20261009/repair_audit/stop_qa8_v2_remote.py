import json,pathlib,os,signal,time
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
record=json.loads((run/'repair_audit/qa8_v2_launch.json').read_text())
if record.get('status')!='running':raise RuntimeError('QA2 is no longer running; inspect before stop')
wrapper=json.loads((run/'repair_audit/qa8_v2_pid.json').read_text())['pid']
children=pathlib.Path(f'/proc/{wrapper}/task/{wrapper}/children').read_text().split()
expected=str(run/'code/scripts/precompute_qwen21_pei2i_prompts.py')
for pid in children:
 command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
 if expected not in command or '--qa8' not in command:raise RuntimeError('Child is not the intended QA2 process')
 os.kill(int(pid),signal.SIGTERM)
(run/'repair_audit/qa8_v2_stop_reason.json').write_text(json.dumps({'stopped_children':children,
 'reason':'Unknown expression must be omitted; explicitly preserving A expression/head direction conflicts with target Face geometry.',
 'preserve_all_completed_evidence':True,'time':time.time()},indent=2))
print(json.dumps({'terminated_qa2_children':children,'wrapper_will_record_exit':True}))
