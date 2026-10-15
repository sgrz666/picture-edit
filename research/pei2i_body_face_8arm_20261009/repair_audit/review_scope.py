"""Inventory changes relative to the repair source snapshot, without touching main."""
import hashlib, json
from pathlib import Path
root = Path('C:/Users/sg/.codex/worktrees/pei2i-experiment-repair/pictureedit')
original = {r['path']: r['sha256'] for r in json.loads((root/'.pei2i_repair_source_snapshot.json').read_text())}
changed=[]
for prefix in ('scripts','src','tests','configs'):
 for p in sorted((root/prefix).rglob('*')):
  if not p.is_file() or p.suffix not in ('.py','.json','.yaml','.yml','.md','.sh'): continue
  rel=p.relative_to(root).as_posix(); digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if original.get(rel)!=digest:
   changed.append(dict(path=rel, original_sha256=original.get(rel), repaired_sha256=digest))
print(json.dumps(changed,indent=2))
Path('D:/codeplus/pictureedit/research/pei2i_body_face_8arm_20261009/repair_audit/scoped_changes_draft.json').write_text(json.dumps(changed,indent=2))
