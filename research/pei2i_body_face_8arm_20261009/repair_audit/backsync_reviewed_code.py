import hashlib,json,shutil
from pathlib import Path
audit=Path(__file__).resolve().parent
source=Path('C:/Users/sg/.codex/worktrees/pei2i-experiment-repair/pictureedit').resolve()
main=Path('D:/codeplus/pictureedit').resolve()
records=json.loads((audit/'code_snapshot_repaired.json').read_text(encoding='utf-8'))
backup=audit/'main_before_repair'
if backup.exists():raise RuntimeError('Backsync already recorded; inspect before retrying')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for record in records:
 target=(main/record['path']).resolve()
 if not target.is_relative_to(main) or not (source/record['path']).resolve().is_relative_to(source):
  raise ValueError('Path escapes scoped roots')
 if digest(source/record['path'])!=record['repaired_sha256']:raise ValueError('Snapshot source changed')
 if record['original_sha256'] is None:
  if target.exists():raise ValueError(f'New repair path already exists: {target}')
 elif not target.is_file() or digest(target)!=record['original_sha256']:
  raise ValueError(f'User edits changed since initial snapshot: {target}')
backup.mkdir()
for record in records:
 target=main/record['path']
 if target.exists():
  old=backup/record['path'];old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,old)
 target.parent.mkdir(parents=True,exist_ok=True)
 temporary=target.with_name(target.name+'.pei2i_repair.tmp')
 shutil.copyfile(source/record['path'],temporary)
 temporary.replace(target)
 if digest(target)!=record['repaired_sha256']:raise ValueError('Backsync hash mismatch')
(backup/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'backsynced':len(records),'backup':str(backup),'git_commit_created':False}))
