from pathlib import Path
import subprocess,time,json,sys
r=Path('/home/shangguanrz/project/pic-edit')
j=r/'research/cleanup_20261003'
started=time.time()
with (j/'full_verification.log').open('w') as log:
 result=subprocess.run([sys.executable,str(r/'scripts/audit/verify_project_organization.py'),'--full-assets'],stdout=log,stderr=subprocess.STDOUT)
record={'exit_code':result.returncode,'started_at':started,'finished_at':time.time()}
try:record['verification']=json.loads((j/'full_verification.log').read_text())
except Exception as error:record['read_error']=str(error)
(j/'full_verification_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
