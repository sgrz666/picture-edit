import hashlib,json,subprocess,tarfile
from pathlib import Path
run=Path(__file__).resolve().parents[1]
audit=run/'repair_audit'
remote='/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/export/'
for name in ('progress_report_20261011.tar.gz','progress_report_20261011.files.json'):
 subprocess.run(['scp','sg:'+remote+name,str(audit/name)],check=True)
manifest=json.loads((audit/'progress_report_20261011.files.json').read_text(encoding='utf-8'))
expected={r['path']:r for r in manifest}
with tarfile.open(audit/'progress_report_20261011.tar.gz') as tar:
 for member in tar.getmembers():
  if not member.isfile() or member.name not in expected:raise ValueError('Unexpected report member')
  target=(run/member.name).resolve()
  if not target.is_relative_to(run.resolve()) or member.name.split('/')[0] not in ('report','prompts','data','evaluation','preflight','training','history'):raise ValueError('Invalid report path')
  payload=tar.extractfile(member).read()
  if hashlib.sha256(payload).hexdigest()!=expected[member.name]['sha256']:raise ValueError('Downloaded report hash mismatch')
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(payload)
print(json.dumps({'verified_report_files':len(expected),'local_report':str(run/'report/index.html')}))
