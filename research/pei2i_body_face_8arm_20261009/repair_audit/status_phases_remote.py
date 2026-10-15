import json,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
for phase in ('migration','verification_remote','qa8_v2','rewrite','cache','train'):
 record=run/f'repair_audit/{phase}_launch.json'
 if not record.exists():continue
 print(phase,record.read_text())
 pidfile=run/f'repair_audit/{phase}_pid.json'
 if pidfile.exists():
  pid=json.loads(pidfile.read_text())['pid']
  children=pathlib.Path(f'/proc/{pid}/task/{pid}/children')
  pids=[str(pid)]+(children.read_text().split() if children.exists() else [])
  print(subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout)
 log=run/f'logs/{phase}.log'
 if log.exists():print(log.read_text(errors='replace')[-2500:])
