from pathlib import Path
import json
r=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008')
print('children',[(p.name,p.is_dir()) for p in sorted(r.iterdir())])
for p in sorted(r.rglob('config.json')):
 if len(p.relative_to(r).parts)>5: continue
 try:
  rec=json.loads(p.read_text())
  if 'checkpoint' not in rec: continue
  samples=p.parent/'per_sample.jsonl'
  rows=[json.loads(line) for line in samples.read_text().splitlines() if line.strip()] if samples.exists() else []
  print(str(p.relative_to(r)),{k:rec.get(k) for k in ('label','variant','stage','checkpoint','lora_sha256','steps','seed','resolution','align_mode')},'rows',len(rows),'complete',sum(x.get('status')=='complete' for x in rows))
 except Exception as e: print(str(p),type(e).__name__,str(e))
