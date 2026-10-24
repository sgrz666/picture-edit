#!/usr/bin/env python3
"""V4 grouped freezes/LR interface, existing velocity MSE + flat AdamW baseline.

This entry has not been run. No automatic phase switching or implicit unfreeze.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,math,time
from dataclasses import asdict
from scripts.run.run_qwen21_native_dual import add_control_arguments
from scripts.run.run_qwen21_native_full_v4 import add_local_arguments,options_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--profile',choices=('face-heads','face-full','hand-heads','hand-full','joint','all-native'),default='joint')
    p.add_argument('--freeze-group',action='append',default=[]);p.add_argument('--unfreeze-group',action='append',default=[])
    p.add_argument('--group-lr',action='append',default=[],help='Explicit group=value, no phase scheduler')
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--steps',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-samples',type=int,default=8)
    p.add_argument('--prompt',default='Change the poses of the existing people. Preserve their identities, clothing, faces, hands, and background.')
    add_control_arguments(p);add_local_arguments(p);args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new V4 experiment directory')
    if args.steps<=0 or args.preflight_samples<=0 or not math.isfinite(args.lr) or args.lr<=0: raise ValueError('Positive steps/count/LR required')
    if set(args.freeze_group)&set(args.unfreeze_group): raise ValueError('A group cannot be both frozen and unfrozen')
    if args.profile.startswith('face-') and (args.no_face or args.face_strength==0): raise ValueError('Face profile requires an active Face branch')
    if args.profile.startswith('hand-') and (args.no_hand or args.hand_strength==0): raise ValueError('Hand profile requires an active Hand branch')
    import torch
    from torch.nn import functional as F
    from src.pose_control.qwen21_full.native_full_runtime import NativeFullRuntimeV4
    from src.pose_control.qwen21_full.native_full_data import NativeFullDatasetV4,collate_native_full_v4,identity_from_reference_v4
    from src.pose_control.qwen21_full.native_depth_runtime import native_velocity_loss
    from src.pose_control.qwen21_full.native_full_assets import sha256
    dataset=NativeFullDatasetV4(args.manifest,load_target=True)
    facts=[]
    for i in range(min(args.preflight_samples,len(dataset))):
        sample=dataset[i];facts.append({'pair':sample['pair_metadata'],'canvas':sample['target_canvas'],'local':sample['local_condition_v4'].metadata})
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'preflight.json').open('x',encoding='utf8') as stream: json.dump({'samples':facts,'model_constructed':False},stream,indent=2,default=str)
    if args.preflight_only: return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for explicitly requested training')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    system=NativeFullRuntimeV4.from_checkpoint(saved)
    if system.config.verification_only: raise ValueError('Verification-only checkpoints cannot run production training')
    overrides={**{k:False for k in args.freeze_group},**{k:True for k in args.unfreeze_group}}
    system.set_trainability(args.profile,overrides=overrides)
    rates={}
    for item in args.group_lr:
        key,value=item.split('=',1)
        if key in rates: raise ValueError('Duplicate LR group')
        rates[key]=float(value)
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_full_integration import install_native_full_v4,bind_native_full_v4
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache,dataset_image_to_pil,transformer_inputs
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    pipe=QwenImage21Pipeline.from_pretrained(str(args.model),dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.requires_grad_(False).to('cuda');pipe.vae.requires_grad_(False).to('cuda')
    cache=Qwen21IPERCache(args.output/'cache',resolution='native',prompt=args.prompt,canvas_policy='aspect_preserving')
    for i in range(len(dataset)):
        sample=dataset[i];cache.source(pipe,dataset_image_to_pil(sample['src_image']));cache.target(pipe,dataset_image_to_pil(sample['tgt_image']))
    pipe.text_encoder.to('cpu');pipe.vae.to('cpu');torch.cuda.empty_cache();pipe.transformer.to('cuda')
    system.to('cuda').train();install=install_native_full_v4(pipe.transformer,system);pipe.transformer.enable_gradient_checkpointing()
    groups=system.optimizer_groups(default_lr=args.lr,learning_rates=rates)
    if not groups: raise ValueError('No active parameters')
    optimizer=torch.optim.AdamW(groups,lr=args.lr);parameters=[p for group in groups for p in group['params']]
    metadata={'args':vars(args),'trainability':system.trainability,'install':install,'manifest_sha256':sha256(args.manifest),
        'checkpoint_sha256':sha256(args.checkpoint),'optimizer_resumed':False,'loss':'canvas_coverage_weighted_velocity_mse','scheduler':None}
    with (args.output/'run_manifest.json').open('x',encoding='utf8') as stream: json.dump(metadata,stream,indent=2,default=str)
    started=time.perf_counter()
    with (args.output/'metrics.jsonl').open('x',encoding='utf8') as stream:
        for step in range(args.steps):
            index=int(torch.randint(len(dataset),(1,),generator=torch.Generator().manual_seed(args.seed+step)))
            batch=collate_native_full_v4([dataset[index]]);options,base=options_for(args,batch['native_scene_kind'])
            pair=cache.pair(pipe,batch,torch.device('cuda'));grid=tuple(pair['target_token_hw'])
            if grid!=batch['target_token_hw']: raise ValueError('Actual Qwen grid differs')
            identity=identity_from_reference_v4(batch,pair['source_latent'],pair['source_token_hw']) if base.identity and base.identity_strength else None
            coverage=F.avg_pool2d(batch['valid_image_mask'].cuda().float(),16).flatten(1)
            target=pair['target_latent'];generator=torch.Generator(device='cuda').manual_seed(args.seed+step)
            sigma=torch.rand(1,device='cuda',generator=generator);noise=torch.randn(target.shape,device='cuda',dtype=target.dtype,generator=generator)
            noisy=(1-sigma[:,None,None])*target.float()+sigma[:,None,None]*noise.float()
            optimizer.zero_grad(set_to_none=True)
            prepared=system.encode(batch,target_token_hw=grid,identity=identity,options=options,base_options=base)
            with bind_native_full_v4(pipe.transformer,prepared,options=options,valid_mask=coverage,conv_input_gate=True):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=pipe.transformer(**transformer_inputs(pair,noisy.to(target.dtype),sigma))[0][:,-target.shape[1]:]
                loss=native_velocity_loss(prediction,noise.float()-target.float(),coverage)
            loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm==0: raise FloatingPointError('Invalid Native gradient/loss')
            gradients={k:[float(p.grad.norm()) for _,p in rows if p.grad is not None] for k,rows in system.named_parameter_groups().items()}
            optimizer.step();row={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad_norm),'group_gradients':gradients,'elapsed':time.perf_counter()-started}
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    payload=system.checkpoint_payload(metadata=json.loads(json.dumps({**metadata,'step':args.steps},default=str)))
    payload['optimizer_state_dict']=optimizer.state_dict()
    with (args.output/'checkpoint_last.pt').open('xb') as stream: torch.save(payload,stream)


if __name__=='__main__': main()
