import json,os,pathlib,shutil,subprocess,sys
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
for phase in ('migration','verification_remote','qa8'):
 record=json.loads((run/f'repair_audit/{phase}_launch.json').read_text())
 if record.get('status')!='complete' or record.get('exit_code')!=0:
  raise RuntimeError(f'Prerequisite phase is not verified complete: {phase}')
phase='qa8_v2'
record=run/f'repair_audit/{phase}_launch.json'
if record.exists():raise RuntimeError('QA v2 already launched; inspect its state')
sys.path.insert(0,str(run/'code'))
from scripts.precompute_qwen21_pei2i_prompts import default_template
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json
archive=run/'prompts/development_archive/instruction_v1'
if archive.exists():raise RuntimeError('QA v1 archival already exists; inspect before retry')
archive.mkdir(parents=True)
for name in ('instruction_v1.json','qa8.json'):
 shutil.copy2(run/'prompts'/name,archive/name)
(run/'prompts/qa8').rename(archive/'qa8')
(run/'prompts/qa8.json').rename(archive/'qa8_review_original.json')
atomic_json(archive/'review.json',{'reviewed':True,'approved_for_formal':False,'pairs':8,'replies':24,
 'findings':['PG train_0007410 requests removal of the source golf club despite pose-only intent.',
 'PG train_0002794 and train_0000092 infer uncertain lower-leg details outside the source frame.',
 'Several PG replies infer speaking/closed-mouth expressions from fitted geometry.',
 'Compiled replies repeatedly refer to guides unavailable as RGB to Qwen.'],
 'decision':'Preserve v1 evidence; apply one global v2 instruction and repeat all 24 QA requests.'})
atomic_json(run/'prompts/instruction_v1.json',default_template())
record.write_text(json.dumps({'argv':[python,str(run/'code/scripts/precompute_qwen21_pei2i_prompts.py'),
 '--run',str(run),'--qa8','--model','/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I'],
 'cwd':str(run/'code'),'status':'queued'},indent=2))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (run/f'logs/{phase}.log').open('w') as output:
 process=subprocess.Popen([python,str(run/'repair_audit/phase_worker.py'),str(record)],cwd=run/'code',
  stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(run/f'repair_audit/{phase}_pid.json').write_text(json.dumps({'pid':process.pid}))
print(json.dumps({'pid':process.pid,'record':str(record),'archived_v1':str(archive)}))
