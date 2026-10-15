import json,os,pathlib,shutil,subprocess,sys
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
previous=json.loads((run/'repair_audit/qa8_v2_launch.json').read_text())
if previous.get('status')!='failed' or previous.get('exit_code')!=-15 or not (run/'repair_audit/qa8_v2_stop_reason.json').is_file():
 raise RuntimeError('Expected intentional QA2 termination must be verified')
phase='qa8_v3'
record=run/f'repair_audit/{phase}_launch.json'
if record.exists():raise RuntimeError('QA v3 already launched; inspect before retry')
sys.path.insert(0,str(run/'code'))
from scripts.precompute_qwen21_pei2i_prompts import default_template
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json
archive=run/'prompts/development_archive/instruction_v2'
if archive.exists():raise RuntimeError('QA2 archive already exists; inspect before retry')
archive.mkdir(parents=True)
shutil.copy2(run/'prompts/instruction_v1.json',archive/'instruction_v1.json')
if (run/'prompts/qa8.json').exists():(run/'prompts/qa8.json').rename(archive/'qa8_review_original.json')
counts={}
for proto in ('pg','pb','pbg'):
 path=run/f'prompts/qa8/{proto}/train.jsonl'
 counts[proto]=len(path.read_text().splitlines()) if path.exists() else 0
(run/'prompts/qa8').rename(archive/'qa8')
atomic_json(archive/'review.json',{'approved_for_formal':False,'completed_counts':counts,'intentionally_stopped':True,
 'finding':'The first PG reply explicitly preserves A head tilt/expression, which can oppose B Face geometry.',
 'decision':'Use reliable target head orientation. Omit unknown expression rather than freezing A expression. Repeat QA with global instruction v3.'})
atomic_json(run/'prompts/instruction_v1.json',default_template())
record.write_text(json.dumps({'argv':[python,str(run/'code/scripts/precompute_qwen21_pei2i_prompts.py'),
 '--run',str(run),'--qa8','--model','/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I'],
 'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/f'logs/{phase}.log').open('w') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/f'repair_audit/{phase}_pid.json').write_text(json.dumps({'pid':process.pid}))
print(json.dumps({'pid':process.pid,'archived_v2_counts':counts}))
