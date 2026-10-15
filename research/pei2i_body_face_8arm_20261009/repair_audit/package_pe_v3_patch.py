import hashlib,json,shutil,tarfile
from pathlib import Path
audit=Path(__file__).resolve().parent
source=Path('C:/Users/sg/.codex/worktrees/pei2i-experiment-repair/pictureedit')
main=Path('D:/codeplus/pictureedit')
previous={r['path']:r['repaired_sha256'] for r in json.loads((audit/'code_snapshot_repaired.json').read_text())}
names=['scripts/precompute_qwen21_pei2i_prompts.py','src/pose_control/qwen21_full/prompt_study/prompts.py']
records=[]
backup=audit/'pe_instruction_v3_before'
if backup.exists():raise RuntimeError('PE patch already applied; inspect first')
for relative in names:
 if hashlib.sha256((main/relative).read_bytes()).hexdigest()!=previous[relative]:raise ValueError('Main PE code changed')
 records.append({'path':relative,'original_sha256':previous[relative],
  'repaired_sha256':hashlib.sha256((source/relative).read_bytes()).hexdigest()})
backup.mkdir()
with tarfile.open(audit/'code_pe_instruction_v3_patch.tar.gz','w:gz') as archive:
 for record in records:
  relative=record['path'];old=backup/relative;old.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(main/relative,old);shutil.copyfile(source/relative,main/relative)
  archive.add(source/relative,arcname=relative,recursive=False)
(audit/'code_pe_instruction_v3_patch.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'files':len(records),'formal_training_not_started':True}))
