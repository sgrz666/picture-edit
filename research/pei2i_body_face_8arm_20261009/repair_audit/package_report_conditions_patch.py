import hashlib,json,shutil,tarfile
from pathlib import Path
audit=Path(__file__).resolve().parent
source=Path('C:/Users/sg/.codex/worktrees/pei2i-experiment-repair/pictureedit')
main=Path('D:/codeplus/pictureedit')
previous={r['path']:r['repaired_sha256'] for r in json.loads((audit/'code_snapshot_repaired.json').read_text())}
names=['scripts/reports/build_qwen21_pei2i_study_report.py','tests/test_qwen21_prompt_study_report.py']
records=[]
backup=audit/'report_conditions_patch_before'
if backup.exists():raise RuntimeError('Report patch already applied; inspect before repeating')
for relative in names:
 if hashlib.sha256((main/relative).read_bytes()).hexdigest()!=previous[relative]:
  raise ValueError('Main file changed since scoped backsync')
 records.append({'path':relative,'original_sha256':previous[relative],
  'repaired_sha256':hashlib.sha256((source/relative).read_bytes()).hexdigest()})
backup.mkdir()
with tarfile.open(audit/'code_report_conditions_patch.tar.gz','w:gz') as archive:
 for record in records:
  relative=record['path'];old=backup/relative;old.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(main/relative,old)
  shutil.copyfile(source/relative,main/relative)
  archive.add(source/relative,arcname=relative,recursive=False)
(audit/'code_report_conditions_patch.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'files':len(records),'only_report_and_test':True}))
