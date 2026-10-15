import json,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
for phase in ('qa8_v3','measurement_smoke'):
 record=run/f'repair_audit/{phase}_launch.json'
 print(phase,record.read_text())
 pid=json.loads((run/f'repair_audit/{phase}_pid.json').read_text())['pid']
 children=pathlib.Path(f'/proc/{pid}/task/{pid}/children')
 pids=[str(pid)]+(children.read_text().split() if children.exists() else [])
 print(subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout)
 if phase == 'qa8_v3':print((run/f'logs/{phase}.log').read_text(errors='replace')[-900:])
for proto in ('pg','pb','pbg'):
 path=run/f'prompts/qa8/{proto}/train.jsonl'
 print(proto, sum(bool(line.strip()) for line in path.read_text().splitlines()) if path.exists() else 0)
