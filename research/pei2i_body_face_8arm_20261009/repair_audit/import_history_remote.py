from pathlib import Path
import json,shutil,hashlib
old=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008')
run=Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for name,label,extra in [('J_common_tiktok','历史 J256',0),('Qwen_final_tiktok','历史 Qwen：J256 后续256步',256),('BFSHead_hotload_tiktok','历史 BFS Head：J256 热挂载',0),('BFSHead_final_tiktok','历史 BFS Head：J256 后续256步',256)]:
 src=old/'evaluation'/name
 if not src.is_dir():
  print('missing independent history',name);continue
 config=json.loads((src/'config.json').read_text())
 rows=[json.loads(line) for line in (src/'per_sample.jsonl').read_text().splitlines() if line.strip()]
 dst=run/'history'/name;dst.mkdir(parents=True,exist_ok=True)
 metadata={'config':config,'source_directory':str(src),'source_files_sha256':{},'independent_from_formal_study':True}
 for filename in ('config.json','per_sample.jsonl','checkpoint_metadata.json','base_provenance.json','lora_load.json','summary.json'):
  p=src/filename
  if p.is_file():
   shutil.copy2(p,dst/filename);metadata['source_files_sha256'][filename]=digest(p)
   if filename not in ('config.json','per_sample.jsonl'):metadata[filename[:-5]]=json.loads(p.read_text())
 prompts={r.get('prompt') for r in rows if r.get('prompt')}
 if config.get('prompt'):prompts.add(config['prompt'])
 if len(prompts)!=1: raise RuntimeError(f'No unambiguous actual old prompt: {name}, {prompts}')
 images=[]
 for row in rows:
  if row.get('status')!='complete':continue
  p=(src/row['image']).resolve()
  if not p.is_relative_to(src.resolve()):raise ValueError('Historical image escapes source')
  dest=dst/row['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  images.append({'pair_id':row.get('pair_id') or row.get('metadata',{}).get('pair_id') or f"legacy_index_{row['pair_index']}", 'path':dest.relative_to(run).as_posix(),'sha256':digest(dest)})
 loading={'variant':config.get('variant'),'stage':config.get('stage'),'lora_sha256':config.get('lora_sha256'),'base_provenance':metadata.get('base_provenance'),'checkpoint':config.get('checkpoint'),'checkpoint_sha256':config.get('checkpoint_sha256')}
 records.append(dict(id=name,label=label,prompt=next(iter(prompts)),budget={'initial_history_steps':256,'additional_updates':extra,'note':'旧实验自身的先验 Body/Face 权重历史另见实际 config 与 initialization provenance'},loading=loading,metadata=metadata,images=images))
 print(name,'native',config.get('resolution'),config.get('align_mode'),'images',len(images))
out=run/'data/historical_references.json'
out.write_text(json.dumps({'schema':'qwen21_study_historical_references_v1','records':records},ensure_ascii=False,indent=2))
print('saved',str(out),'sha256',digest(out))
