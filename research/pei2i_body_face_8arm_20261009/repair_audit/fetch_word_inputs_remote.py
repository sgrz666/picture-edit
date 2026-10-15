import hashlib,json,pathlib,tarfile,time

run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
old=pathlib.Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/data/val24_pairs.jsonl')
rows=[json.loads(s) for s in old.read_text().splitlines() if s.strip()]
manifest={'captured_at':time.time(),'pairs':[],'files':[]}
archive=run/'repair_audit/word_inputs_20261011.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for row in rows:
        item={k:row[k] for k in ('pair_id','sequence_id','source_video_group','source_stem','target_stem')}
        item['inputs']={}
        for role in ('source_rgb','target_rgb','target_normal','target_depth_vis','target_dwpose'):
            path=pathlib.Path(row[role]); rel=f"word_inputs/{row['pair_id']}/{role}.png"
            data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
            tar.add(path,arcname=rel,recursive=False)
            item['inputs'][role]={'path':rel,'sha256':digest,'original_path':str(path)}
            manifest['files'].append({'path':rel,'sha256':digest,'bytes':len(data)})
        manifest['pairs'].append(item)
    statuses={}
    for proto in ('pg','pb','pbg'):
        p=run/f'prompts/qa8/{proto}/train.jsonl'
        statuses[proto]=sum(bool(s.strip()) for s in p.read_text().splitlines()) if p.exists() else 0
    manifest['qa_v3_counts']=statuses
    for name in ('qa8_v3_launch.json','full_pipeline_state.json'):
        p=run/'repair_audit'/name
        if p.exists():manifest[name]=json.loads(p.read_text())
    p=run/'repair_audit/word_inputs_manifest_20261011.json'
    p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    tar.add(p,arcname='word_inputs/manifest.json',recursive=False)
print(json.dumps({'pairs':len(rows),'files':len(manifest['files']),'bytes':archive.stat().st_size,'qa_v3_counts':statuses,'archive':str(archive)}))
