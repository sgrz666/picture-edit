"""Continue the approved experiment after signed QA; retain every phase/log."""
import hashlib,json,os,pathlib,subprocess,sys,time
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
state_path=run/'repair_audit/full_pipeline_state.json'
def save(state):
 temporary=state_path.with_suffix('.json.tmp')
 temporary.write_text(json.dumps(state,indent=2,ensure_ascii=False))
 temporary.replace(state_path)
state={'schema':'qwen21_repair_full_pipeline_v1','pid':os.getpid(),'status':'waiting_for_signed_qa_review',
 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'stages':[]}
save(state)
try:
 while True:
  qa=json.loads((run/'repair_audit/qa8_v3_launch.json').read_text())
  if qa.get('status')=='failed':raise RuntimeError('QA v3 process failed; preserve artifacts and investigate')
  review=run/'prompts/qa8.json'
  if qa.get('status')=='complete' and qa.get('exit_code')==0 and review.is_file() and json.loads(review.read_text()).get('reviewed') is True:
   break
  time.sleep(30)
 sys.path.insert(0,str(run/'code'))
 from scripts.precompute_qwen21_pei2i_prompts import validate_qa8_review
 from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
 validate_qa8_review(run)
 state.update(status='running',review_sha256=sha256_file(review),lock_sha256=sha256_file(run/'data/lock.json'))
 save(state)
 phases=[('rewrite',['--splits','train','val','diagnostic']),('cache',['--splits','train','val','diagnostic']),
  ('selfcheck',[]),('preflight',[]),('initial',[]),('train',['--resume']),('finaltest',[]),('report',[]),('export',[])]
 runner=run/'code/scripts/run/run_qwen21_pei2i_study.py'
 for phase,extra in phases:
  stage={'phase':phase,'status':'running','started_at':time.time(),
   'argv':[python,str(runner),'--run',str(run),'--phase',phase,*extra],
   'log':f'logs/full_pipeline_{phase}.log'}
  state['stages'].append(stage);state['phase']=phase;save(state)
  with (run/stage['log']).open('x') as output:
   process=subprocess.Popen(stage['argv'],cwd=run/'code',stdout=output,stderr=subprocess.STDOUT)
   stage['pid']=process.pid;save(state)
   stage['exit_code']=process.wait()
  stage['finished_at']=time.time();stage['status']='complete' if stage['exit_code']==0 else 'failed';save(state)
  if stage['exit_code']!=0:raise RuntimeError(f'Phase failed: {phase}, exit {stage["exit_code"]}; see retained log')
 summary=json.loads((run/'report/summary.json').read_text())
 if summary.get('is_completed') is not True:raise RuntimeError('Report does not prove the complete experiment')
 state.update(status='remote_complete_waiting_local_copy',remote_complete=True,finished_at=time.time(),
  report='report/index.html',export='export/results.tar.gz')
 save(state)
except BaseException as exc:
 state.update(status='failed',failure=f'{type(exc).__name__}: {exc}',finished_at=time.time())
 save(state)
 raise
