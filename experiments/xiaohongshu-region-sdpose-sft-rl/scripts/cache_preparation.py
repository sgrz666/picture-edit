"""Frozen Qwen inputs/cache independent of SAM, DINO and Native mesh assembly."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from prepare import read,ref,sha,write

PROMPT="Change the person's pose to match <image2>. Preserve the identity, face, clothing and background in <image1>."


def write_once(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=json.loads(json.dumps(value,ensure_ascii=False)):raise ValueError('Immutable metadata differs: '+str(path))
        return ref(path)
    return write(path,value)


def model_definition():
    base=read('/data/shangguanrz/pic-edit-experiments/one/bilibili-region-sdpose-sft-rl-20261120-v1/configs/templates/uniform/S0.json')
    manifest=base['model_manifest'];path=Path(manifest['path'])
    if not path.exists():
        old='/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/'
        if not str(path).startswith(old):raise FileNotFoundError(path)
        path=Path(str(path).replace(old,'/data/shangguanrz/pic-edit-experiments/one/qwen21_full_native_v1/',1))
    if sha(path)!=manifest['sha256']:raise ValueError('Pinned Qwen manifest changed')
    return {'model_path':base['model_path'],'model_manifest':ref(path)}


def prepare_inputs(root):
    from PIL import Image
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA,read_condition
    selection=read(root/'data/selection.json');frames={};entries={};prompt_sha=hashlib.sha256(PROMPT.encode()).hexdigest()
    for fid,frame in selection['frames'].items():
        spec=canvas_for(frame['original_hw'],selection['canvas_long_side'])
        if sha(frame['skeleton']['path'])!=frame['skeleton']['sha256']:raise ValueError('SDPose guide changed')
        with Image.open(frame['skeleton']['path']) as image:guide=spec.image(image.convert('RGB'))
        stream=io.BytesIO();guide.save(stream,format='PNG');data=stream.getvalue()
        path=root/f'conditioning/guides/{fid}.png';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Canonical guide differs')
        else:
            with path.open('xb') as f:f.write(data)
        rgb=ref(frame['rgb_path'])
        if rgb['sha256']!=frame['rgb_sha256']:raise ValueError('Locked original RGB changed: '+fid)
        frames[fid]={'canvas':spec.metadata(),'canvas_id':spec.fingerprint(),'rgb':rgb,'guide':ref(path)}
    for pair in selection['pairs']:
        source=frames[pair['source_frame_id']];target=frames[pair['target_frame_id']];pid=pair['pair_id']
        bank=write_once(root/f'conditioning/banks/{pid}.json',{'schema':'xiaohongshu_native_prompt_v1','pair_id':pid,
            'compiled_prompt_inference':PROMPT,'compiled_prompt_inference_sha256':prompt_sha,'producer':'fixed_source_identity_target_sdpose'})
        condition={'schema':SCHEMA,'profile':'single_2ref','pair_id':pid,'primary_source_index':0,
            'target_canvas_id':target['canvas_id'],'target_pose_backend':'sdpose','images':[
            {'role':'source','file':source['rgb'],'canvas':source['canvas'],'canvas_id':source['canvas_id'],'space':'original'},
            {'role':'target_pose','file':target['guide'],'canvas':target['canvas'],'canvas_id':target['canvas_id'],'space':'canvas'}],
            'prompt':{'text':PROMPT,'sha256':prompt_sha,'field':'compiled_prompt_inference','bank':bank,
                      'schema':'xiaohongshu_native_prompt_v1','pair_id':pid}}
        bound=write_once(root/f'conditions/{pid}.json',condition);read_condition(bound)
        key='xiaohongshu_'+pair['split']+'\0'+pid
        entries[key]={'pair_id':pid,'split':pair['split'],'condition':bound,'source_rgb':source['rgb'],
            'target_rgb':target['rgb'],'source_canvas':source['canvas'],'target_canvas':target['canvas'],
            'source_rgb_sha256':source['rgb']['sha256'],'driver_rgb_sha256':target['rgb']['sha256'],
            'source_canvas_id':source['canvas_id'],'target_canvas_id':target['canvas_id']}
    result={'schema':'xhs_independent_qwen_inputs_v1','selection':ref(root/'data/selection.json'),
            'model':model_definition(),'entries':entries}
    write_once(root/'data/qwen_condition_index.json',result)
    print(json.dumps({'canonical_conditions':len(entries),'frames':len(frames)}),flush=True)


def validate_native_bindings(cache,native):
    if cache['condition_bindings']!={key:row['condition_sha256'] for key,row in native.items()}:
        raise ValueError('Final Native condition set differs from independent cache')
    if set(cache['entries'])!=set(native):raise ValueError('Cache condition entries differ')
    for key,row in native.items():
        if any(cache['entries'][key].get(k)!=row[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')):
            raise ValueError('Cache/Native provenance differs: '+key)


def encode(root):
    import fcntl
    lock=(root/'cache-encoder.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from diffusers import QwenImage21Pipeline
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_full_data import _rgb
    from src.pose_control.qwen21_full.native_v62.training.backend import model_identity
    from src.pose_control.qwen21_full.native_v62.training.config import digest
    from src.pose_control.qwen21_full.native_v62.conditioning import encode_pair,CACHE_SCHEMA,read_condition
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REGION_PILOT_MEMORY_FRACTION','.35')))
    index=read(root/'data/qwen_condition_index.json');model=index['model'];model_identity(SimpleNamespace(**model))
    if ref(index['selection']['path'])!=index['selection']:raise ValueError('Locked selection changed')
    cache_root=root/'cache';cache_root.mkdir(exist_ok=True)
    pipe=QwenImage21Pipeline.from_pretrained(model['model_path'],dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.requires_grad_(False).eval().to('cuda');pipe.vae.requires_grad_(False).eval().to('cuda')
    entries={};bindings={}
    for number,(key,item) in enumerate(index['entries'].items(),1):
        condition=read_condition(item['condition']);name='pair_'+digest(key.split('\0'))[:24]
        marker=cache_root/(name+'.ready.json')
        if marker.exists():
            row=read(marker)
            if row['condition_sha256']!=item['condition']['sha256'] or ref(row['file']['path'])!=row['file']:
                raise ValueError('Saved cache entry changed')
        else:
            target=CanvasSpec.from_record(item['target_canvas']);source=CanvasSpec.from_record(item['source_canvas'])
            sample={'target_token_hw':target.token_hw,'source_token_hw':source.token_hw,'tgt_image':_rgb(item['target_rgb'],target)}
            pair=encode_pair(pipe,sample,condition);pair['condition_sha256']=item['condition']['sha256']
            path=cache_root/(name+'.pt')
            if path.exists():
                import uuid
                path=cache_root/(name+'_attempt_'+uuid.uuid4().hex[:10]+'.pt')
            with path.open('xb') as f:torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in pair.items()},f)
            row={'file':ref(path),'condition_sha256':item['condition']['sha256'],'model_manifest_sha256':model['model_manifest']['sha256'],
                 **{k:item[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')}}
            write(marker,row)
        if row.get('model_manifest_sha256')!=model['model_manifest']['sha256'] or any(
            row.get(k)!=item[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')):
            raise ValueError('Saved independent cache provenance changed')
        entries[key]=row;bindings[key]=item['condition']['sha256']
        print(json.dumps({'cache_pair':number,'total':len(index['entries']),'pair_id':item['pair_id']}),flush=True)
    write_once(cache_root/'index.json',{'schema':CACHE_SCHEMA,'model_manifest_sha256':model['model_manifest']['sha256'],
        'profile':'single_2ref','condition_bindings':bindings,'entries':entries})
    write_once(root/'report/independent-cache-complete.json',{'cache_index':ref(cache_root/'index.json'),
        'input_index':ref(root/'data/qwen_condition_index.json'),'pairs':len(entries),'peak_allocated':torch.cuda.max_memory_allocated(),
        'source_script':ref(__file__),'native_assets_required_for_encoding':False})


def finalize(root):
    from src.pose_control.qwen21_full.native_v7.training import DataPlanV7,TrainingConfigV7
    cache=read(root/'cache/index.json');config=TrainingConfigV7.from_dict(read(root/'configs/templates/uniform/S0.json'))
    data=DataPlanV7(config);native={}
    for (source,index),(bound,spec) in data.conditions.items():
        sample=data.sample(source,index);meta=sample['local_condition_v4'].metadata[0]
        key=source+'\0'+data.datasets[source].rows[index]['pair_id']
        native[key]={'condition_sha256':bound['sha256'],
                    **{k:meta[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')}}
    validate_native_bindings(cache,native);cache_ref=ref(root/'cache/index.json')
    for arm in ('uniform','regional'):
        for path in sorted((root/f'configs/templates/{arm}').glob('*.json')):
            raw=read(path);raw['cache_identity']=cache_ref
            write_once(root/f'configs/cache-ready/{arm}/{path.name}',TrainingConfigV7.from_dict(raw).record())
    write_once(root/'cache_ready.json',{'cache':cache_ref,'target_pose_backend':'sdpose',
        'canonical_inputs':ref(root/'data/qwen_condition_index.json'),'native_bindings_verified':True})
    print(json.dumps({'cache_ready':True,'verified_native_pairs':len(native)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','encode','finalize'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repo));{'prepare':prepare_inputs,'encode':encode,'finalize':finalize}[a.action](a.root)


if __name__=='__main__':main()
