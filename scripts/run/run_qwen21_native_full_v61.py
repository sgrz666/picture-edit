#!/usr/bin/env python3
"""Native Full V6.1 explicit multi-reference inference."""
from pathlib import Path
import argparse,json,sys,math
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.pose_control.qwen21_full.native_v5.cli import add_controls,options_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--pair-index',type=int,default=0);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prompt',default=None,help='Optional assertion; must equal the locked per-pair Prompt')
    p.add_argument('--poselink-strength',type=float,default=1.)
    add_controls(p);args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new V6.1 inference directory')
    if args.steps<=0 or args.pair_index<0: raise ValueError('Positive steps and valid pair index required')
    if not math.isfinite(args.poselink_strength) or args.poselink_strength<0: raise ValueError('Finite nonnegative PoseLink strength required')
    import torch
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v61.data import NativeFullDatasetV61,collate_native_full_v61
    from src.pose_control.qwen21_full.native_v61.integration import install_native_full_v61
    from src.pose_control.qwen21_full.native_v6.conditioning import read_condition,encode_pair
    from src.pose_control.qwen21_full.native_v61.generation import generate_native
    runtime=NativeFullRuntimeV61.from_checkpoint(torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)).eval().to('cuda')
    if runtime.config.verification_only: raise ValueError('Verification checkpoints cannot run production inference')
    reader=NativeFullDatasetV61(args.manifest,load_target=False,data_profile=args.data_profile)
    sample=reader[args.pair_index]
    batch=collate_native_full_v61([sample]);options,base=options_for(args,batch['native_scene_kind'])
    pipe=QwenImage21Pipeline.from_pretrained(str(args.model),dtype=torch.bfloat16,local_files_only=True).to('cuda')
    install=install_native_full_v61(pipe.transformer,runtime);h,w=sample['target_canvas']['canvas_hw']
    condition=reader.rows[args.pair_index].get('qwen_condition')
    if condition is not None:
        spec=read_condition(condition,sample)
        if args.prompt is not None and args.prompt!=spec['prompt']['text']:raise ValueError('Explicit Prompt differs from locked per-pair Plan A')
        pair=encode_pair(pipe,sample,spec,include_target=False)
        protocol=spec['profile'];prompt_sha=spec['prompt']['sha256']
    else:
        raise ValueError('V6.1 requires explicit 2Ref/4Ref/5Ref qwen_condition; legacy Source-only has no PoseLink tokens')
    pipe.text_encoder.to('cpu')
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        image=generate_native(pipe,runtime,batch,pair,options=options,base_options=base,seed=args.seed,steps=args.steps,poselink_strength=args.poselink_strength)
    if image.size!=(w,h): raise ValueError('Generated image grid differs')
    image.save(args.output/'image.png')
    with (args.output/'request.json').open('x') as f:
        json.dump({'architecture':'native_full_v61','pose_config':runtime.pose_config.to_dict(),'poselink_strength':args.poselink_strength,'data_profile':sample['data_profile'],'pair':sample['pair_metadata'],
            'controls':asdict(options),'base':asdict(base),'installation':install,'qwen_condition_profile':protocol,'prompt_sha256':prompt_sha,
            'target_rgb_loaded':False,'steps':args.steps,'seed':args.seed},f,indent=2)


if __name__=='__main__': main()
