import pathlib,json,time,subprocess
r=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
print((r/'repair_audit/qa8_launch.json').read_text())
wrapper_pid=2806393
children=pathlib.Path(f'/proc/{wrapper_pid}/task/{wrapper_pid}/children')
pids=[str(wrapper_pid)]+(children.read_text().split() if children.exists() else [])
print(subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout)
attempts=list((r/'prompts/qa8').rglob('attempt_*.json'))
print('raw_replies',len(attempts))
for p in attempts[-3:]:
 rec=json.loads(p.read_text());print('attempt',rec['pair_id'],rec['protocol'],rec['status'],rec['tokens'],round(rec['seconds'],2),rec.get('error'))
for p in sorted((r/'prompts/qa8').rglob('*.jsonl')):
 print(str(p.relative_to(r)),len(p.read_text().splitlines()))
print('log_tail', (r/'logs/qa8.log').read_text(errors='replace')[-1800:])
