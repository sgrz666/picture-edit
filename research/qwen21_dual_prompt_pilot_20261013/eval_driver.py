from pathlib import Path
import os,json,subprocess,time,sys
work=Path('/home/shangguanrz/project/pic-edit/research/qwen21_dual_prompt_pilot_20261013')
root=Path('/home/shangguanrz/project/pic-edit')
mirror=work/'implementation'
run=root/'experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1'
output=run/'test_pair00_v1'
if output.exists(): raise RuntimeError('Final test output exists; inspect it before any further action')
completion=json.loads((run/'training_complete.json').read_text())
assert completion['status']=='success' and completion['completed_steps']==300
assert completion['production_training'] is True and completion['backbone_unchanged'] is True
assert completion['initialization_sha256_equal'] is True and completion['test_data_read'] is False
cmd=['/home/shangguanrz/nihaochen/envs/qwen21/bin/python','-u','scripts/eval/eval_qwen21_dual_prompt_pilot.py','--data-dir',str(work/'data'),'--training-root',str(run),'--output',str(output),'--steps','40','--seeds','42','43','44','--cfg','1','--mesh-python',str(root/'add/envs/multihmr/bin/python'),'--mesh-device','cuda']
env=dict(os.environ,PYTHONPATH=str(mirror)+':'+str(root),CUDA_VISIBLE_DEVICES='0')
started=time.time()
with (work/'evaluation.log').open('x') as log:
 proc=subprocess.Popen(cmd,cwd=mirror,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 (work/'evaluation_process.json').write_text(json.dumps(dict(pid=proc.pid,started_unix=started,command=cmd,cwd=str(mirror),output=str(output)),indent=2)+'\n')
 print('STARTED_EVAL_PID',proc.pid,flush=True)
 for line in proc.stdout:
  log.write(line);log.flush();print(line,end='',flush=True)
 result=proc.wait()
(work/'evaluation_result.json').write_text(json.dumps(dict(exit_code=result,elapsed_seconds=time.time()-started,output=str(output)),indent=2)+'\n')
print('EVAL_DRIVER_EXIT',result,flush=True)
sys.exit(result)
