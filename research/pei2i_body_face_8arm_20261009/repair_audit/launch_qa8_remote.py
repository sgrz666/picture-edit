import os, pathlib, json, subprocess, time
r = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
record = r/'repair_audit/qa8_launch.json'
if record.exists():
 raise RuntimeError('QA launch already recorded; inspect it before retrying')
argv = ['/home/shangguanrz/nihaochen/envs/qwen21/bin/python', str(r/'code/scripts/precompute_qwen21_pei2i_prompts.py'), '--run', str(r), '--qa8', '--model', '/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I']
record.write_text(json.dumps(dict(argv=argv,cwd=str(r/'code'),status='queued'),indent=2))
env = dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (r/'logs/qa8.log').open('w') as out:
 proc = subprocess.Popen(['/home/shangguanrz/nihaochen/envs/qwen21/bin/python',str(r/'repair_audit/phase_worker.py'),str(record)],stdout=out,stderr=subprocess.STDOUT,cwd=r/'code',env=env,start_new_session=True)
print(json.dumps(dict(pid=proc.pid,record=str(record),log=str(r/'logs/qa8.log'))))
