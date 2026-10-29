"""Exact-canvas five-reference Qwen baseline or genuine two-source Native V5."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abc_contracts import ROLES,PROFILE,image_records
from abc_runner import sha256
from abc_variants import control_records, PHOTO_SUFFIX
from src.pose_edit.common import atomic_json

def condition_spec(request,scene):
    from src.data.canvas_contract import CanvasSpec
    from abc_native_assets import target_canvas
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA,text_sha
    target=target_canvas(scene);images=[]
    for index,role in enumerate(ROLES):
        if index<2:
            source=scene['sources'][index];path=source['image'];canvas=CanvasSpec.from_record(source['canvas']);space='original'
        else:
            path=scene['guides'][('person_0','person_1','combined')[index-2]];canvas=target;space='canvas'
        images.append(dict(role=role,file={'path':str(Path(path).resolve()),'sha256':sha256(path)},
            canvas=canvas.metadata(),canvas_id=canvas.fingerprint(),space=space))
    prompt_path=Path(request['prompt']);prompt=prompt_path.read_text(encoding='utf8').strip()
    return dict(schema=SCHEMA,profile=PROFILE,primary_source_index=0,appearance_source_indices=[0,1],
        pair_id=sha256(request['scene']) if request.get('scene') else Path(request['run_dir']).name,
        images=images,target_canvas_id=target.fingerprint(),
        prompt={'text':prompt,'sha256':text_sha(prompt),'field':'compiled_prompt_inference',
            'bank':{'path':str(prompt_path.resolve()),'sha256':sha256(prompt_path)}})

def prepare_bank(scene,index,settings,output,*,feature_cache=None):
    import numpy as np
    import torch
    from PIL import Image
    from abc_native_assets import source_reference_job
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record
    from src.pose_control.qwen21_full.native_full_references import extract_source_references
    from src.pose_control.qwen21_full.native_v5.source_bank import NativeAppearanceSourceV5,NativeSourceBankV5
    from src.pose_control.qwen21_full.native_v5.contracts import NativeDetailReferencesV5,DETAIL_SCHEMA
    from src.data.canvas_contract import CanvasSpec
    dino=Path(settings['native_dino_model'])
    models=dict(arcface_model=settings['arcface_model'],arcface_sha256=sha256(settings['arcface_model']),
        dino_model=str(dino),dino_files={str(p.relative_to(dino)):sha256(p) for p in sorted(dino.iterdir()) if p.is_file()},device='cuda')
    bank=[];reports=[]
    for i,(source,ref) in enumerate(zip(scene['sources'],index['sources'])):
        geom=read_record(ref,schema='abc_native_source_geometry_v1');local=geom['local'];canvas=CanvasSpec.from_record(source['canvas'])
        with np.load(source['keypoints_npz'],allow_pickle=False) as archive:observation={k:archive[k] for k in archive.files}
        job=source_reference_job(source,observation,local['boxes'][0],models)
        if feature_cache is None:
            features=extract_source_references(job)
        else:
            from abc_feature_cache import frozen_source_features
            binding=dict(source_rgb_sha256=sha256(source['image']),source_mask_sha256=sha256(source['mask']),
                source_canvas_id=canvas.fingerprint(),geometry=ref,models=models)
            features=frozen_source_features(feature_cache,binding,lambda:extract_source_references(job))
        with Image.open(source['mask']) as im:mask=torch.from_numpy(np.asarray(im).copy()>0)[None,None,None]
        head=mask&torch.from_numpy(geom['raster']['part']==1)[None,None,None]
        h,w=canvas.canvas_hw;support=torch.zeros(1,6,h,w,dtype=torch.bool);boxes=torch.zeros(1,6,4)
        for region,slot in enumerate((0,2,3)):
            boxes[0,slot]=torch.from_numpy(local['boxes'][0,region])
            support[0,slot]=torch.from_numpy(local['regions'][0][region]['valid'])&mask[0,0,0]
        support &= ~(support.sum(1,keepdim=True)>1)
        valid=support.flatten(2).any(-1)&(boxes[...,2:]>boxes[...,:2]).all(-1)
        metadata=dict(schema=DETAIL_SCHEMA,role='primary_source_only',source_rgb_sha256=sha256(source['image']),
            source_canvas_id=canvas.fingerprint(),source_person_ids=[source['person_id'],'__inactive__'])
        detail=NativeDetailReferencesV5(boxes,support,valid,torch.tensor([[0,-1]]),(metadata,)).validate()
        entry=NativeAppearanceSourceV5(i,(source['person_id'],),i,canvas.metadata(),sha256(source['image']),
            sha256(source['params_npz']),mask,head,features,detail).validate()
        bank.append(entry)
        record=write_record(output/f'source_features_{i}.npz',features,schema='native_local_reference_v4')
        reports.append(dict(source=entry.record(),features=record,arcface_valid=features.arcface_valid.tolist(),
            face_texture_valid=features.face_dino_valid.tolist(),hand_texture_valid=features.hand_dino_valid.tolist(),detail_valid=valid.tolist()))
        gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'source_bank.json',{'schema':'abc_native_sources_v1','sources':reports})
    return NativeSourceBankV5(tuple(bank),torch.tensor([[0,1]]),torch.tensor([[0,0]]),(('source_A','source_B'),)).validate()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf8'));settings=request['settings'];out=Path(request['output_dir'])
    scene=json.loads(Path(request['scene']).read_text(encoding='utf8'));enabled=request['mode']=='native_v5'
    original_prompt=request['prompt'];diagnostic=request.get('diagnostic','none')
    if diagnostic=='photo_prompt':
        effective=out/'diagnostic_prompt.txt'
        effective.write_text(Path(original_prompt).read_text(encoding='utf8').strip()+PHOTO_SUFFIX,encoding='utf8')
        request={**request,'prompt':str(effective)}
    elif diagnostic=='monochrome_guides':
        from PIL import Image,ImageOps
        guides={}
        for role,path in scene['guides'].items():
            destination=out/('monochrome_'+role+'.png')
            with Image.open(path) as im:ImageOps.grayscale(im).convert('RGB').save(destination)
            guides[role]=str(destination)
        scene={**scene,'guides':guides}
    import torch
    from diffusers import QwenImage21Pipeline
    from src.data.canvas_contract import CanvasSpec
    from abc_native_assets import target_canvas
    from src.pose_control.qwen21_full.native_full_assets import read_record
    from src.pose_control.qwen21_full.native_full_data import collate_native_full_v4
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition,encode_pair
    from src.pose_control.qwen21_full.native_v5.generation import generate_native
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.integration import install_native_full_v5
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    torch.set_num_threads(4);torch.cuda.init();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    runtime=None;installation=None;bank=None
    canvas=target_canvas(scene)
    sample={'source_token_hw':CanvasSpec.from_record(scene['sources'][0]['canvas']).token_hw,'target_token_hw':canvas.token_hw}
    batch={'valid_image_mask':CanvasSpec.from_record(scene['sources'][0]['canvas']).valid_mask()[None]}
    if enabled:
        saved=torch.load(request['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        runtime=NativeFullRuntimeV5.from_checkpoint(saved).eval().requires_grad_(False)
        if runtime.config.verification_only:raise ValueError('Verification-only checkpoint cannot run workflow inference')
        cached=Path(scene.get('native_geometry_index','__not_prepared__'))
        index=json.loads(cached.read_text(encoding='utf8')) if cached.is_file() else None
        if index is None or index.get('dual_config')!=asdict(runtime.base.config) or index.get('scene_sha256')!=sha256(request['scene']):
            prep={**request,'native_geometry_dir':str(out/'native_geometry'),'dual_config':asdict(runtime.base.config)}
            atomic_json(out/'native_geometry_request.json',prep)
            subprocess.run([settings['geometry_python'],'-u',str(Path(__file__).with_name('abc_native_assets.py')),
                '--request',str(out/'native_geometry_request.json')],cwd=ROOT,check=True)
            index=json.loads((out/'native_geometry/index.json').read_text(encoding='utf8'))
        raw=read_record(index['target'],schema='abc_native_target_geometry_v1')
        sample=raw['sample'];sample['local_condition_v4']=raw['local']
        batch=collate_native_full_v4([sample]);bank=prepare_bank(scene,index,settings,out,
            feature_cache=Path(request['run_dir'])/'appearance_cache')
        batch['source_bank_v5']=bank
    spec=condition_spec(request,scene);atomic_json(out/'qwen_condition.json',spec)
    spec=read_condition({'path':str(out/'qwen_condition.json'),'sha256':sha256(out/'qwen_condition.json')})
    pipe=QwenImage21Pipeline.from_pretrained(settings['qwen_model'],dtype=torch.bfloat16,local_files_only=True)
    for component in (pipe.transformer,pipe.text_encoder,pipe.vae):component.eval().requires_grad_(False)
    if any(getattr(c,'peft_config',{}) for c in (pipe.transformer,pipe.text_encoder)):
        raise ValueError('Unexpected LoRA/PEFT in Qwen base')
    # Load text/VAE first; release them before moving the transformer to CUDA.
    pipe.text_encoder.to('cuda');pipe.vae.to('cuda')
    pair=encode_pair(pipe,sample,spec,include_target=False)
    source_latents={};source_proof=[];offset=0
    for i in range(2):
        h,w=pair['reference_token_hws'][i];latent=pair['reference_latents'][:,offset:offset+h*w].cpu().contiguous();offset+=h*w
        source_latents['source_'+str('AB'[i])]=latent
        source_proof.append(dict(qwen_image_index=i,shape=list(latent.shape),dtype=str(latent.dtype),
            input_sha256=spec['images'][i]['file']['sha256'],canvas_id=spec['images'][i]['canvas_id'],
            latent_sha256=hashlib.sha256(latent.view(torch.uint8).numpy().tobytes()).hexdigest()))
    torch.save({'schema':'abc_frozen_qwen_sources_v1','sources':source_latents,'provenance':source_proof},out/'qwen_source64.pt')
    pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
    pipe.transformer.to('cuda')
    controls,base_controls=control_records(request.get('variant','full'))
    options=NativeControlOptionsV5(**controls);base=DualControlOptions(**base_controls)
    from src.pose_control.qwen21_full.native_v5.observation import ResidualObserver
    observer=ResidualObserver() if enabled and request.get('observe_native',False) else None
    sampling={}
    if enabled:
        runtime.to('cuda');installation=install_native_full_v5(pipe.transformer,runtime)
    with torch.inference_mode():
        image=generate_native(pipe,runtime,batch,pair,options=options,base_options=base,
            seed=settings['seed'],steps=settings['steps'],enabled=enabled,observer=observer,sampling_report=sampling)
    if observer is not None:
        atomic_json(out/'native_residuals.json',dict(**observer.report(),controls=controls,base_controls=base_controls,
            source_binding=bank.metadata,condition_validity=sampling.get('condition_validity'),
            contribution_space_note='Local modality RMS is measured before shared output projection; face/hand RMS is the actual injected hidden residual.'))
    if image.size!=(canvas.canvas_hw[1],canvas.canvas_hw[0]):raise ValueError('Generated target Canvas differs')
    image=image.convert('RGB');image.save(out/'final_canvas.png')
    source_canvas=CanvasSpec.from_record(scene['sources'][0]['canvas'])
    x,y=source_canvas.offset_xy;h,w=source_canvas.resized_hw
    final=image.crop((x,y,x+w,y+h));final.save(out/'final_image.png')
    atomic_json(out/'generation.json',dict(schema='abc_generation_v1',mode=request['mode'],architecture='native_full_v5' if enabled else 'qwen21_base',
        interaction_geometry_source=scene['interaction'].get('source','estimated_MultiHMR_SMPLX'),
        guide_style=scene['diagnostics']['render'].get('guide_style','dwpose'),
        official_asset_sha256=scene['interaction'].get('official_asset_sha256'),
        contact_transfer=scene['diagnostics'].get('contact_candidates'),
        adapter_loaded=enabled,native_forward_executed=enabled,checkpoint=request.get('checkpoint'),
        checkpoint_sha256=sha256(request['checkpoint']) if enabled else None,source_images_in_qwen=2,
        checkpoint_trained_source_count=saved['runtime_config'].get('source_images_in_qwen') if enabled else None,reference_images=[
            {'index':i+1,'role':entry['role'],'input_file':entry['file'],'canvas':entry['canvas'],
             'canvas_id':entry['canvas_id'],'space':entry['space']} for i,entry in enumerate(spec['images'])],
        qwen_profile=PROFILE,reference_token_hws=pair['reference_token_hws'],target_token_hw=canvas.token_hw,
        qwen_source64={'path':str(out/'qwen_source64.pt'),'sha256':sha256(out/'qwen_source64.pt'),'sources':source_proof},
        target_rgb_loaded=False,source_bank=None if bank is None else bank.metadata,
        native_installation=installation,controls=asdict(options) if enabled else None,
        base_controls=asdict(base) if enabled else None,variant=request.get('variant','full'),diagnostic=diagnostic,
        original_prompt_sha256=sha256(original_prompt),sampling=sampling,
        residual_observation=None if observer is None else {'path':str(out/'native_residuals.json'),'sha256':sha256(out/'native_residuals.json')},
        seed=settings['seed'],steps=settings['steps'],canvas_size=list(image.size),final_size=list(final.size),
        prompt_sha256=sha256(request['prompt']),seconds=time.perf_counter()-started,
        peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3))

if __name__=='__main__':main()
