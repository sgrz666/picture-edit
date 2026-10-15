import json,os,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
phase='verification_remote'
tests=sorted(p.relative_to(run/'code').as_posix() for p in (run/'code/tests').glob('test_qwen21_prompt_study*.py'))
tests+=['tests/test_qwen21_joint_pilot_visuals.py','tests/test_qwen21_native_resolution.py','tests/test_qwen21_frozen_bfs_lora.py']
record=run/f'repair_audit/{phase}_launch.json'
if record.exists():raise RuntimeError('Verification already launched; inspect first')
record.write_text(json.dumps({'argv':[python,'-m','pytest',*tests,'-q'],'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/f'logs/{phase}.log').open('w') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/f'repair_audit/{phase}_pid.json').write_text(json.dumps({'pid':process.pid}))
print(json.dumps({'pid':process.pid,'tests':len(tests)}))
