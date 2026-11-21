"""Relocate a pilot's verified references; preserve every semantic model input."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def rebind_ref(reference,mappings):
    from src.pose_control.qwen21_full.native_full_assets import sha256
    result=dict(reference);original=Path(result['path'])
    candidate=original
    if not candidate.is_file():
        for before,after in mappings:
            try:relative=original.relative_to(Path(before))
            except ValueError:continue
            candidate=Path(after)/relative
            break
    if not candidate.is_file() or sha256(candidate)!=result['sha256']:
        raise ValueError('Relocation requires the original SHA256: '+str(original))
    result['path']=str(candidate)
    return result


def verified_path(reference):
    """Verify an original reference before reading or assigning it a new SHA."""
    return Path(rebind_ref(reference,[])['path'])


def condition_semantics(spec):
    return {'profile':spec['profile'],'target_canvas_id':spec['target_canvas_id'],
            'prompt':{key:spec['prompt'][key] for key in ('text','sha256')},
            'images':[{key:image[key] for key in ('role','canvas','canvas_id','space')}|
                      {'rgb_sha256':image['file']['sha256']} for image in spec['images']]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.cache_output.exists():raise FileExistsError('Use fresh relocation outputs')
    args.output.mkdir(parents=True);args.cache_output.mkdir(parents=True)
    import torch
    from src.pose_control.qwen21_full.native_full_assets import sha256,read_record,write_record
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v6.conditioning import bind_pair_provenance,read_condition
    mappings=[('/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1','/data/shangguanrz/pic-edit-datasets-test/Hi4D_pilot_v1'),
              ('/data/shangguanrz/pic-edit-datasets','/data/shangguanrz/pic-edit-datasets-test')]
    def ref(path):return {'path':str(path.resolve()),'sha256':sha256(path)}
    def write(path,value):
        with path.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
        return ref(path)
    def rewrite(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:return rebind_ref(value,mappings)
            return {key:rewrite(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(rewrite(item) for item in value)
        return value
    old_rows={role:json.loads((args.root/'data-v2'/(role+'.jsonl')).read_text()) for role in ('train','observation')}
    selected={row['pair_id'] for row in old_rows.values()}
    reference_cache={};bank_cache={};new_rows={};conditions={}
    for role,row in old_rows.items():
        old_v4=next(json.loads(line) for line in verified_path(row['v4_manifest']).read_text().splitlines() if json.loads(line)['pair_id']==row['pair_id'])
        v4=rewrite(old_v4)
        old_refs=old_v4['references']
        if old_refs['sha256'] not in reference_cache:
            references=read_record(old_refs,schema='native_local_reference_v4')
            relocated=replace(references,metadata=tuple(rewrite(meta) for meta in references.metadata)).validate()
            reference_cache[old_refs['sha256']]=write_record(args.output/('references-'+old_refs['sha256'][:12]+'.npz'),relocated,schema='native_local_reference_v4')
            assert torch.equal(references.arcface,relocated.arcface) and torch.equal(references.face_dino,relocated.face_dino)
        v4['references']=reference_cache[old_refs['sha256']]
        v4_path=args.output/(role+'-v4.jsonl')
        with v4_path.open('x',encoding='utf8') as stream:stream.write(json.dumps(v4)+'\n')
        spec=json.loads(verified_path(row['qwen_condition']).read_text())
        bank=spec['prompt']['bank']
        if bank['sha256'] not in bank_cache:
            rows=[rewrite(json.loads(line)) for line in verified_path(bank).read_text().splitlines() if line.strip() and json.loads(line).get('pair_id') in selected]
            if {item['pair_id'] for item in rows}!=selected:raise ValueError('Selected bank rows missing')
            path=args.output/('bank-'+bank['sha256'][:12]+'.jsonl')
            with path.open('x',encoding='utf8') as stream:
                for item in rows:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            bank_cache[bank['sha256']]=ref(path)
        rebound=rewrite(spec);rebound['prompt']['bank']=bank_cache[bank['sha256']]
        assert condition_semantics(spec)==condition_semantics(rebound)
        condition=write(args.output/(role+'-condition.json'),rebound)
        updated={**row,'v4_manifest':ref(v4_path),'qwen_condition':condition}
        if row.get('references') is not None:updated['references']=v4['references']
        path=args.output/(role+'.jsonl')
        with path.open('x',encoding='utf8') as stream:stream.write(json.dumps(updated)+'\n')
        sample=NativeFullDatasetV62(path,load_target=True,data_profile='hybrid_hi4d')[0]
        read_condition(condition,sample)
        new_rows[role]=ref(path);conditions[role+'\0'+row['pair_id']]=(condition,rebound)
        new_rows[role+'_groups']=ref(args.root/'data-v2'/(role+'-groups.json'))
    old_index=args.root/'cache/native-4ref/index.json';cache=json.loads(old_index.read_text())
    entries={};bindings={}
    for key,(condition,spec) in conditions.items():
        old=cache['entries'][key];pair=torch.load(verified_path(old['file']),map_location='cpu',weights_only=True,mmap=True)
        assert pair['prompt_sha256']==spec['prompt']['sha256']
        assert pair['reference_sha256s']==[item['file']['sha256'] for item in spec['images']]
        pair={**pair,'condition_sha256':condition['sha256']}
        pair=bind_pair_provenance(pair,spec,condition_sha256=condition['sha256'])
        path=args.cache_output/(key.split('\0')[0]+'.pt')
        with path.open('xb') as stream:torch.save(pair,stream)
        entries[key]={**old,'file':ref(path),'condition_sha256':condition['sha256']};bindings[key]=condition['sha256']
    index=write(args.cache_output/'index.json',{**cache,'entries':entries,'condition_bindings':bindings,'relocated_from':ref(old_index)})
    report={'manifests':new_rows,'cache_index':index,'old_prepared':ref(args.root/'data-v2/prepared.json'),
            'semantic_inputs_unchanged':True,'images_or_geometry_moved':False,'training_started':False}
    write(args.output/'prepared.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
