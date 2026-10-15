"""Upgrade proof bindings before any cache/checkpoint is created; preserve originals."""
import hashlib,json,pathlib,shutil,sys,time
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
sys.path.insert(0,str(run/'code'))
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json,sha256_file,load_study_lock
from src.pose_control.qwen21_full.prompt_study.source_assets import add_source_asset_hashes
from src.pose_control.qwen21_full.prompt_study.measurement import measurement_artifacts,scoring_signature
for prefix in ('training','cache','evaluation'):
 if any(p.is_file() for p in (run/prefix).rglob('*')):
  raise RuntimeError(f'Cannot change lock after formal artifacts exist: {prefix}')
audit=run/'repair_audit/pretraining_lock_migration'
if audit.exists():
 raise RuntimeError('Migration was already recorded; inspect its state instead of rerunning')
audit.mkdir(parents=True)
data=run/'data'
for name in ('lock.json','assets.jsonl'):
 shutil.copy2(data/name,audit/name)
lock=json.loads((data/'lock.json').read_text())
pairs=[]
for split,record in lock['splits'].items():
 pairs += [(split,json.loads(line)) for line in (data/record['path']).read_text().splitlines() if line.strip()]
records=[json.loads(line) for line in (data/'assets.jsonl').read_text().splitlines() if line.strip()]
started=time.time()
records=add_source_asset_hashes(records,pairs)
new_assets=data/'assets.jsonl.tmp'
new_assets.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
new_assets.replace(data/'assets.jsonl')
lock['source_assets_manifest']={'path':'assets.jsonl','sha256':sha256_file(data/'assets.jsonl'),'count':len(records)}
lock['measurement_artifacts']=measurement_artifacts(run/'code/models',code_root=run/'code')
lock['scoring_signature_sha256']=scoring_signature(lock['measurement_artifacts'])
atomic_json(data/'lock.json',lock)
load_study_lock(data/'lock.json',verify_assets=True)
atomic_json(audit/'summary.json',{'schema':'pretraining_lock_proof_upgrade_v1','old_lock_sha256':sha256_file(audit/'lock.json'),
 'new_lock_sha256':sha256_file(data/'lock.json'),'pairs':len(records),'source_files':sum(len(r['source_files']) for r in records),
 'scoring_signature_sha256':lock['scoring_signature_sha256'],'seconds':time.time()-started,'verified':True})
print((audit/'summary.json').read_text())
