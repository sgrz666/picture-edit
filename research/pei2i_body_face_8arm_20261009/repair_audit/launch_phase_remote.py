"""Launch the next named phase once, with inspectable process identity."""
import os,pathlib,json,subprocess,time
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
phase='migration'
script=run/'repair_audit/migrate_pretraining_lock_remote.py'
record=run/f'repair_audit/{phase}_launch.json'
if record.exists(): raise RuntimeError('Launch already exists; inspect before retrying')
record.write_text(json.dumps({'argv':[python,str(script)],'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/f'logs/{phase}.log').open('w') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/f'repair_audit/{phase}_pid.json').write_text(json.dumps({'pid':process.pid,'record':str(record),'log':str(run/f'logs/{phase}.log')}))
print(json.dumps({'pid':process.pid,'record':str(record),'log':str(run/f'logs/{phase}.log')}))
