import json,os,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
record=run/'repair_audit/full_pipeline_launch.json'
if record.exists():raise RuntimeError('Full pipeline already launched; inspect before retry')
record.write_text(json.dumps({'argv':[python,str(run/'repair_audit/full_pipeline_remote.py')],
 'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/'logs/full_pipeline.log').open('x') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/'repair_audit/full_pipeline_pid.json').write_text(json.dumps({'pid':process.pid}))
print(json.dumps({'pid':process.pid,'first_gate':'wait for signed review of all 24 QA v3 outputs'}))
