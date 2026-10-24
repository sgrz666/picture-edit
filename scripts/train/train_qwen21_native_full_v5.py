#!/usr/bin/env python3
"""Explicit V5 profiles/local objectives; flat AdamW, no implicit phase changes."""
from pathlib import Path
import argparse,json,math,sys,time
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.pose_control.qwen21_full.native_v5.cli import add_controls,options_for,profile_controls


def eligibility(sample,profile):
    import torch
    c=sample['local_condition_v4'];s=sample['training_supervision_v5'];r=sample['local_references_v4']
    if not s.paired.all(): return False,'unpaired'
    if profile.startswith('face-'):
        from src.pose_control.qwen21_full.native_local_condition import RIGID_FACE_SLOTS
        pv=c.face_point_valid.clone();parameters=c.jaw_valid|c.expression_valid
        for row,meta in enumerate(c.metadata):
            for person in range(2):
                if meta['expression_mode']=='off' or (meta['expression_mode']=='source' and not meta.get('expression_retargeted',[False,False])[person]):
                    mask=torch.zeros(72,dtype=torch.bool);mask[list(RIGID_FACE_SLOTS)]=True
                    pv[row,person]&=mask;parameters[row,person]=False
        valid=c.face_roi_valid & (pv.any(-1)|parameters|r.arcface_valid.any(-1).gather(1,c.source_indices.clamp_min(0))|r.face_dino_valid.any(-1).gather(1,c.source_indices.clamp_min(0)))
        return bool(valid.any() and s.weights[:,:2].any()),'face_condition_and_supervision'
    if profile.startswith('hand-'):
        from src.pose_control.qwen21_full.native_face import gather_source
        has_geometry=c.hand_point_valid.any(-1)|c.hand_pose_valid|c.mesh_valid
        has_texture=gather_source(r.hand_dino_valid,c).any(-1)
        return bool((c.hand_roi_valid&(has_geometry|has_texture)).any() and s.weights[:,2:].any()),'hand_condition_and_supervision'
    if profile.startswith('detail-'):
        detail=sample['detail_references_v5'];available=False
        for region,p in enumerate((0,1,0,0,1,1)):
            if not c.person_valid[0,p]: continue
            src=int(c.source_indices[0,p]);sr=src if region<2 else 2+2*src+(region-2)%2
            available |= bool(detail.valid[0,sr] and c.region_valid[0,region] and s.weights[:,region].any())
        return available,'source_detail_and_target_supervision'
    if profile.startswith('interaction-'): return sample['native_scene_kind']=='dual','dual_interaction'
    return True,'paired_joint_or_all_native'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--profile',choices=('face-heads','face-full','hand-heads','hand-full','detail-heads','detail-full','interaction-heads','interaction-full','joint','all-native'),default='joint')
    p.add_argument('--loss-profile',choices=('velocity_only','local_velocity','local_aux'),default='velocity_only')
    p.add_argument('--auxiliary-every',type=int,default=4);p.add_argument('--auxiliary-sigma-max',type=float,default=.35)
    p.add_argument('--evaluators',type=Path,help='JSON identity/perceptual file references with SHA')
    p.add_argument('--freeze-group',action='append',default=[]);p.add_argument('--unfreeze-group',action='append',default=[])
    p.add_argument('--group-lr',action='append',default=[]);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--seed',type=int,default=42);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--max-attempts',type=int,help='Bound scheduled-off attempts; default 10 times requested optimizer updates')
    p.add_argument('--prompt',default='Change the poses of the existing people. Preserve identities, clothing, faces, hands, and background.')
    add_controls(p);args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new V5 training directory')
    if args.steps<=0 or not math.isfinite(args.lr) or args.lr<=0 or (args.max_attempts is not None and args.max_attempts<args.steps): raise ValueError('Positive steps/LR and sufficient attempt bound required')
    if set(args.freeze_group)&set(args.unfreeze_group): raise ValueError('Conflicting freeze overrides')
    if args.profile.startswith('face-') and (args.no_face or args.face_strength==0 or max(args.face_identity_gain,args.face_texture_gain,args.face_geometry_gain)==0):
        raise ValueError('Face profile requires an enabled Face modality')
    if args.profile.startswith('hand-') and (args.no_hand or args.hand_strength==0 or max(args.hand_geometry_gain,args.hand_texture_gain)==0):
        raise ValueError('Hand profile requires an enabled Hand modality')
    if args.profile.startswith('detail-') and (args.no_source_detail or args.detail_strength==0 or ((args.no_face or args.face_strength==0) and (args.no_hand or args.hand_strength==0))):
        raise ValueError('Detail profile requires an enabled local Source Detail region')
    if args.profile.startswith('interaction-') and (args.no_interaction or args.interaction_strength==0): raise ValueError('Interaction profile requires Interaction')
    import torch
    from torch.nn import functional as F
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5,collate_native_full_v5
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.losses import LocalLossConfig,local_velocity_loss,auxiliary_loss,frozen_evaluators
    from src.pose_control.qwen21_full.native_full_data import identity_from_reference_v4
    from src.pose_control.qwen21_full.native_full_assets import sha256
    config=LocalLossConfig(args.loss_profile,args.auxiliary_every,args.auxiliary_sigma_max)
    dataset=NativeFullDatasetV5(args.manifest,load_target=True,data_profile=args.data_profile)
    facts=[];eligible=[];aux_available=False
    for i in range(len(dataset)):
        sample=dataset[i];ok,reason=eligibility(sample,args.profile)
        facts.append({'index':i,'pair_id':sample['pair_metadata']['pair_id'],'eligible':ok,'reason':reason,'data_profile':sample['data_profile']})
        if ok: eligible.append(i);aux_available |= bool(sample['training_supervision_v5'].auxiliary_valid.any())
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'preflight.json').open('x') as f: json.dump({'all_rows':facts,'eligible_indices':eligible,'model_constructed':False},f,indent=2)
    if args.preflight_only: return
    if not eligible: raise ValueError('No eligible rows; original manifest was retained in the report')
    runtime=NativeFullRuntimeV5.from_checkpoint(torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True))
    if runtime.config.verification_only: raise ValueError('Verification checkpoints cannot run production training')
    runtime.set_trainability(args.profile,overrides={**{k:False for k in args.freeze_group},**{k:True for k in args.unfreeze_group}})
    if runtime.config.local_mode=='static' and args.profile.endswith('heads') and not runtime.trainability['native_body'] and not any(h.weight.detach().count_nonzero() for h in runtime.body.output_heads):
        raise ValueError('Static local-only heads cannot train through a frozen zero Native Body output; use joint or a trained Body')
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_v5.integration import install_native_full_v5,bind_native_full_v5
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache,dataset_image_to_pil,transformer_inputs
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    pipe=QwenImage21Pipeline.from_pretrained(str(args.model),dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.requires_grad_(False).to('cuda');pipe.vae.requires_grad_(False).to('cuda')
    cache=Qwen21IPERCache(args.output/'cache',resolution='native',prompt=args.prompt,canvas_policy='aspect_preserving')
    for i in eligible:
        sample=dataset[i];cache.source(pipe,dataset_image_to_pil(sample['src_image']));cache.target(pipe,dataset_image_to_pil(sample['tgt_image']))
    pipe.text_encoder.to('cpu')
    evaluators=None
    if args.loss_profile=='local_aux' and aux_available:
        if args.evaluators is None: raise ValueError('Explicit pinned evaluator job required for auxiliary losses')
        job=json.loads(args.evaluators.read_text());paths=[job[k]['path'] for k in ('identity','perceptual')]
        evaluators=frozen_evaluators(*paths,device='cuda',sha=[job[k]['sha256'] for k in ('identity','perceptual')])
    else: pipe.vae.to('cpu')
    pipe.transformer.to('cuda');runtime.to('cuda').train();installation=install_native_full_v5(pipe.transformer,runtime)
    pipe.transformer.enable_gradient_checkpointing()
    rates={}
    for item in args.group_lr:
        key,value=item.split('=',1)
        if key in rates: raise ValueError('Duplicate LR group')
        rates[key]=float(value)
    groups=runtime.optimizer_groups(default_lr=args.lr,learning_rates=rates)
    if not groups: raise ValueError('No trainable V5 groups')
    optimizer=torch.optim.AdamW(groups,lr=args.lr);parameters=[p for g in groups for p in g['params']]
    metadata={'args':vars(args),'loss_config':asdict(config),'coefficients':config.coefficients,'manifest_sha256':sha256(args.manifest),
        'checkpoint_sha256':sha256(args.checkpoint),'trainability':runtime.trainability,'optimizer_resumed':False,'scheduler':None,'installation':installation,
        'evaluator_job':job if evaluators is not None else None,'evaluator_job_sha256':sha256(args.evaluators) if evaluators is not None else None}
    with (args.output/'run_manifest.json').open('x') as f: json.dump(metadata,f,indent=2,default=str)
    started=time.perf_counter();updates=0;attempt=0;max_attempts=args.max_attempts or args.steps*10
    with (args.output/'metrics.jsonl').open('x') as f:
        while updates<args.steps and attempt<max_attempts:
            step=updates;draw=args.seed+attempt;attempt+=1
            index=eligible[int(torch.randint(len(eligible),(1,),generator=torch.Generator().manual_seed(draw)))]
            batch=collate_native_full_v5([dataset[index]]);options,base=options_for(args,batch['native_scene_kind']);options=profile_controls(args.profile,options)
            if args.profile.startswith(('face-','hand-','detail-')):
                from dataclasses import replace
                base=replace(base,interaction=False,raster=False,part_contact=False)
            pair=cache.pair(pipe,batch,torch.device('cuda'));grid=tuple(pair['target_token_hw'])
            identity=identity_from_reference_v4(batch,pair['source_latent'],pair['source_token_hw']) if base.identity and base.identity_strength else None
            coverage=F.avg_pool2d(batch['valid_image_mask'].cuda().float(),16).flatten(1)
            target=pair['target_latent'];generator=torch.Generator(device='cuda').manual_seed(draw)
            sigma=torch.rand(1,device='cuda',generator=generator);noise=torch.randn(target.shape,device='cuda',dtype=target.dtype,generator=generator)
            branches=('body','interaction','detail','face_identity','face_texture','face_geometry','hand_geometry','hand_texture')
            relevant=('face_identity','face_texture','face_geometry') if args.profile.startswith('face-') else (
                ('hand_geometry','hand_texture') if args.profile.startswith('hand-') else (('detail',) if args.profile.startswith('detail-') else (('interaction',) if args.profile.startswith('interaction-') else branches)))
            if not any(runtime.strength(b,layer,sigma,options).any() for b in relevant for layer in runtime.config.injection_layers):
                f.write(json.dumps({'attempt':attempt,'optimizer_updated':False,'reason':'all_profile_schedules_zero','sigma':float(sigma[0]),'pair_index':index})+'\n');f.flush();continue
            noisy=(1-sigma[:,None,None])*target.float()+sigma[:,None,None]*noise.float()
            optimizer.zero_grad(set_to_none=True)
            prepared=runtime.encode(batch,target_token_hw=grid,identity=identity,source_latent=pair['source_latent'],options=options,base_options=base)
            with bind_native_full_v5(pipe.transformer,prepared,options=options,valid_mask=coverage):
                with torch.autocast('cuda',dtype=torch.bfloat16): prediction=pipe.transformer(**transformer_inputs(pair,noisy.to(target.dtype),sigma))[0][:,-target.shape[1]:]
                supervision=batch['training_supervision_v5'].to('cuda')
                loss,parts=local_velocity_loss(prediction,noise.float()-target.float(),coverage,supervision,config)
                extra,aux=auxiliary_loss(pipe,noisy,prediction,sigma,(batch['tgt_image'].cuda().float()+1)*.5,supervision,config,step=step+1,evaluators=evaluators)
                loss=loss+extra
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm==0: raise FloatingPointError('Invalid V5 gradient/loss; no implicit unfreeze or Head filling')
            gradients={k:[float(p.grad.norm()) for _,p in rows if p.grad is not None] for k,rows in runtime.named_parameter_groups().items()}
            optimizer.step();updates+=1;row={'step':updates,'attempt':attempt,'optimizer_updated':True,'pair_index':index,'loss':float(loss.detach()),'grad_norm':float(norm),
                'parts':{k:float(v.detach()) for k,v in parts.items()},'aux':{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in aux.items()},
                'group_gradients':gradients,'elapsed':time.perf_counter()-started}
            f.write(json.dumps(row)+'\n');f.flush()
    payload=runtime.checkpoint_payload(metadata=json.loads(json.dumps({**metadata,'step':updates,'attempts':attempt,'requested_updates':args.steps,'max_attempts':max_attempts},default=str)))
    payload['optimizer_state_dict']=optimizer.state_dict()
    with (args.output/'checkpoint_last.pt').open('xb') as f: torch.save(payload,f)
    if updates!=args.steps: raise RuntimeError('Scheduled-off attempt bound reached; partial checkpoint/report preserved without claiming requested updates complete')


if __name__=='__main__': main()
