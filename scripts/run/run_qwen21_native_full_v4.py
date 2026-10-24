#!/usr/bin/env python3
"""Explicit Native Full V4 inference. Architecture delivered without execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json
from dataclasses import asdict
from scripts.run.run_qwen21_native_dual import add_control_arguments,control_options


def add_local_arguments(parser):
    parser.add_argument('--data-profile',choices=('auto','full_smplx','hybrid_hi4d'),default='auto',help='Assert manifest profile; absent profile is legacy full_smplx')
    parser.add_argument('--no-face',action='store_true');parser.add_argument('--no-hand',action='store_true')
    parser.add_argument('--face-strength',type=float,default=1.);parser.add_argument('--hand-strength',type=float,default=1.)


def options_for(args,kind):
    from src.pose_control.qwen21_full.native_full_runtime import FullControlOptionsV4
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    if kind=='single':
        base=DualControlOptions(depth_mode=args.depth_mode,detail=args.depth_mode=='full_geometry' and not args.no_detail,
            interaction=False,raster=False,part_contact=False,identity=not args.no_identity,
            depth_strength=args.depth_strength,identity_strength=args.identity_strength,
            relation_strength=0.,interaction_strength=0.)
    else: base=control_options(args)
    return FullControlOptionsV4(not args.no_face,not args.no_hand,args.face_strength,args.hand_strength),base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--pair-index',type=int,default=0);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prompt',default='Change the poses of the existing people. Preserve their identities, clothing, faces, hands, and background.')
    add_control_arguments(p);add_local_arguments(p);args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.steps<=0 or args.pair_index<0: raise ValueError('Positive steps and valid pair index required')
    import torch
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_full_runtime import NativeFullRuntimeV4
    from src.pose_control.qwen21_full.native_full_data import NativeFullDatasetV4,collate_native_full_v4
    from src.pose_control.qwen21_full.native_full_integration import install_native_full_v4
    from src.pose_control.qwen21_full.native_full_pipeline import bind_native_full_pipeline_v4
    from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    runtime=NativeFullRuntimeV4.from_checkpoint(saved).eval().to('cuda')
    if runtime.config.verification_only: raise ValueError('Verification-only checkpoints cannot run production inference')
    sample=NativeFullDatasetV4(args.manifest,load_target=False,data_profile=args.data_profile)[args.pair_index];batch=collate_native_full_v4([sample])
    options,base=options_for(args,batch['native_scene_kind'])
    pipe=QwenImage21Pipeline.from_pretrained(str(args.model),dtype=torch.bfloat16,local_files_only=True).to('cuda')
    install=install_native_full_v4(pipe.transformer,runtime)
    h,w=sample['target_canvas']['canvas_hw'];source_h,source_w=sample['source_canvas']['canvas_hw']
    if source_h!=source_w: raise ValueError('The current real Qwen pipeline requires a square reference Canvas; target may be rectangular')
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.no_grad(),bind_native_full_pipeline_v4(pipe.transformer,runtime,batch,options=options,base_options=base) as binding:
        image=pipe(image=[dataset_image_to_pil(sample['src_image'])],prompt=args.prompt,width=w,height=h,
            output_resolution=source_h,num_inference_steps=args.steps,true_cfg_scale=1.,use_kv_cache=False,
            generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
    if image.size!=(w,h): raise ValueError('Generated canvas differs')
    image.save(args.output/'image.png')
    with (args.output/'request.json').open('x',encoding='utf8') as stream:
        json.dump({'architecture':'native_full_v4','data_profile':sample['data_profile'],'pair':sample['pair_metadata'],'local_options':asdict(options),
            'base_options':asdict(base),'installation':install,'binding':binding,'target_rgb_loaded':False,
            'seed':args.seed,'steps':args.steps},stream,indent=2)


if __name__=='__main__': main()
