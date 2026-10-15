import json,os,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
phase='measurement_smoke'
record=run/f'repair_audit/{phase}_launch.json'
if record.exists():raise RuntimeError('Smoke check already launched; inspect first')
record.write_text(json.dumps({'argv':[python,str(run/'repair_audit/measurement_smoke_remote.py')],
 'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/f'logs/{phase}.log').open('w') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/f'repair_audit/{phase}_pid.json').write_text(json.dumps({'pid':process.pid}))
print(json.dumps({'pid':process.pid,'record':str(record)}))
