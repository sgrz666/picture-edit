#!/usr/bin/env python3
"""Native Full V5 single-primary-reference inference. Not executed at code delivery."""
from pathlib import Path
import argparse,json,sys
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.pose_control.qwen21_full.native_v5.cli import add_controls,options_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--pair-index',type=int,default=0);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prompt',default='Change the poses of the existing people. Preserve identities, clothing, faces, hands, and background.')
    add_controls(p);args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new V5 inference directory')
    if args.steps<=0 or args.pair_index<0: raise ValueError('Positive steps and valid pair index required')
    import torch
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5,collate_native_full_v5
    from src.pose_control.qwen21_full.native_v5.integration import install_native_full_v5
    from src.pose_control.qwen21_full.native_v5.pipeline import bind_native_full_pipeline_v5
    from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
    runtime=NativeFullRuntimeV5.from_checkpoint(torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)).eval().to('cuda')
    if runtime.config.verification_only: raise ValueError('Verification checkpoints cannot run production inference')
    sample=NativeFullDatasetV5(args.manifest,load_target=False,data_profile=args.data_profile)[args.pair_index]
    batch=collate_native_full_v5([sample]);options,base=options_for(args,batch['native_scene_kind'])
    pipe=QwenImage21Pipeline.from_pretrained(str(args.model),dtype=torch.bfloat16,local_files_only=True).to('cuda')
    install=install_native_full_v5(pipe.transformer,runtime);h,w=sample['target_canvas']['canvas_hw'];sh,sw=sample['source_canvas']['canvas_hw']
    if sh!=sw: raise ValueError('Current real Qwen uses a square reference Canvas; target may be rectangular')
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.no_grad(),bind_native_full_pipeline_v5(pipe.transformer,runtime,batch,options=options,base_options=base) as binding:
        image=pipe(image=[dataset_image_to_pil(sample['src_image'])],prompt=args.prompt,width=w,height=h,output_resolution=sh,
            num_inference_steps=args.steps,true_cfg_scale=1.,use_kv_cache=False,generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
    if image.size!=(w,h): raise ValueError('Generated image grid differs')
    image.save(args.output/'image.png')
    with (args.output/'request.json').open('x') as f:
        json.dump({'architecture':'native_full_v5','data_profile':sample['data_profile'],'pair':sample['pair_metadata'],
            'controls':asdict(options),'base':asdict(base),'installation':install,'binding':binding,
            'target_rgb_loaded':False,'steps':args.steps,'seed':args.seed},f,indent=2)


if __name__=='__main__': main()
