"""Bounded Native Adapter Flow-GRPO using actual generated-image SDPose rewards.

The policy is the Native runtime; Qwen stays frozen. Every selected stochastic
transition is stored at its rollout dtype and replayed, with a fixed SFT reference.
RL snapshots have their own schema and cannot be mistaken for shared SFT resume.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))
PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
SDPY='/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python'
SCHEMA='native_region_grpo_training_v1'
ALGORITHM={'name':'native_flow_grpo_v1','group_size':4,'rollout_steps':10,'sde_indices':[1,2,3],
    'noise_level':.7,'ppo_clip':1e-4,'reference_beta':1e-3,
    'reward_weights':{'body':.45,'hands':.35,'identity':.15,'background':.05},
    'reference':'fixed_RN_SFT_parent','reference_penalty':'same_variance_transition_gaussian_kl',
    'advantage':'per_component_group_standardization_then_combined_standardization',
    'evaluation':'held-out fixed-seed 20-step ODE; never used as optimization reward'}


def read(path):return json.loads(Path(path).read_text())


def clone_policy_state(state):
    import torch
    return {key:value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
            for key,value in state.items()}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def evaluate(root,manifest,folder):
    script=Path(__file__).with_name('evaluate.py')
    pose,full=folder/'sdpose.json',folder/'metrics.json'
    subprocess.run([SDPY,str(script),'pose','--root',str(root),'--manifest',str(manifest),'--output',str(pose)],check=True)
    subprocess.run([PY,str(script),'identity','--root',str(root),'--input',str(pose),'--output',str(full)],check=True)
    return read(full)


@contextmanager
def weights(runtime,reference,current):
    import torch
    try:
        with torch.no_grad():runtime.load_state_dict(reference,strict=True)
        yield
    finally:
        with torch.no_grad():runtime.load_state_dict(current,strict=True)


def predict(backend,batch,pair,route,sigma,latent):
    import torch
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    options,base=route_options(route,batch['native_scene_kind'][0] if isinstance(batch['native_scene_kind'],list) else batch['native_scene_kind'])
    safe_pair={k:v for k,v in pair.items() if k!='target_latent'}
    time=torch.tensor([sigma],device=backend.device,dtype=torch.float32)
    velocity,_=backend.predict_velocity(batch,safe_pair,route,time,latent,options,base)
    return velocity


def decode(pipe,latent,hw):
    import torch
    device=next(pipe.transformer.parameters()).device;old=next(pipe.vae.parameters()).device
    try:
        pipe.vae.to(device)
        unpacked=pipe._unpack_latents(latent,hw[0]*16,hw[1]*16,pipe.vae_scale_factor).to(pipe.vae.dtype)
        mean=torch.tensor(pipe.vae.config.latents_mean,device=device,dtype=unpacked.dtype).view(1,64,1,1,1)
        std=torch.tensor(pipe.vae.config.latents_std,device=device,dtype=unpacked.dtype).view(1,64,1,1,1)
        image=pipe.vae.decode(unpacked*std+mean,return_dict=False)[0][:,:,0]
        return pipe.image_processor.postprocess(image,output_type='pil')[0]
    finally:pipe.vae.to(old)


def rollout(backend,batch,pair,route,seed,sde_index,path):
    import numpy as np
    import torch
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift,retrieve_timesteps
    from src.pose_control.qwen21_full.region_rl import flow_sde_transition
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    th,tw=pair['target_token_hw'];device=backend.device;dtype=pair['prompt_embeds'].dtype
    generator=torch.Generator(device=device).manual_seed(seed)
    initial=torch.randn((1,1,64,th,tw),device=device,dtype=dtype,generator=generator)
    latent=backend.pipe._pack_latents(initial,1,64,th,tw)
    scheduler=type(backend.pipe.scheduler).from_config(backend.pipe.scheduler.config)
    mu=calculate_shift(th*tw,scheduler.config.get('base_image_seq_len',256),scheduler.config.get('max_image_seq_len',4096),
        scheduler.config.get('base_shift',.5),scheduler.config.get('max_shift',1.15))
    timesteps,_=retrieve_timesteps(scheduler,ALGORITHM['rollout_steps'],device,
        sigmas=np.linspace(1.,1/ALGORITHM['rollout_steps'],ALGORITHM['rollout_steps']),mu=mu)
    sigmas=scheduler.sigmas.detach().float().cpu().tolist();replay=None
    transition_generator=torch.Generator(device=device).manual_seed((seed*1664525+1013904223)%(2**63-1))
    with torch.no_grad():
        for index in range(len(timesteps)):
            sigma,next_sigma=float(sigmas[index]),float(sigmas[index+1])
            v=predict(backend,batch,pair,route,sigma,latent)
            if index==sde_index:
                before=latent
                after,log_prob,mean,std=flow_sde_transition(before,v,sigma,next_sigma,
                    noise_level=ALGORITHM['noise_level'],generator=transition_generator)
                replay={'latent':before.cpu(),'next_latent':after.cpu(),'old_log_prob':log_prob.cpu(),
                    'old_velocity':v.cpu(),'sigma':sigma,'next_sigma':next_sigma,'std':std,'sde_index':index,'seed':seed}
                latent=after
            else:latent=(latent.float()+(next_sigma-sigma)*v.float()).to(dtype)
        image=decode(backend.pipe,latent,(th,tw))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:image.save(f,format='PNG')
    if replay is None:raise ValueError('No interior SDE transition was sampled')
    return replay,{'path':str(path),'sha256':file_sha(path),'seed':seed,'steps':ALGORITHM['rollout_steps'],
        'target_rgb_used_as_condition':False,'stochastic_steps':[sde_index]}


def audit_rewards(root,pairs,folder):
    from PIL import Image,ImageChops
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    records=[]
    for pair in pairs[:2]:
        spec=CanvasSpec.from_record(pair['target_canvas'])
        with Image.open(pair['target_rgb']['path']) as im:gt=spec.image(im.convert('RGB'))
        with Image.open(pair['source_rgb']['path']) as im:copy=spec.image(im.convert('RGB'))
        variants={'GT':gt,'copy_source':copy,'blank':Image.new('RGB',gt.size),
                  'translated_GT':ImageChops.offset(gt,round(gt.width*.12),0)}
        for name,image in variants.items():
            path=folder/(pair['pair_id']+'_'+name+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:image.save(f,format='PNG')
            records.append({'pair_id':pair['pair_id'],'path':str(path),'sha256':file_sha(path),'label':name})
    manifest=folder/'manifest.json';write(manifest,{'label':'reward_controls','records':records})
    report=evaluate(root,manifest,folder)
    gt=[r['pose']['body']['reward'] for r in report['records'] if r['label']=='GT']
    blank=[r['pose']['body']['reward'] for r in report['records'] if r['label']=='blank']
    if not gt or min(gt)<=.5 or max(blank)>=min(gt)-.2:
        raise RuntimeError('SDPose reward controls do not distinguish GT from blank reliably')
    write(folder/'audit.json',{'passed':True,'GT_body':gt,'blank_body':blank,
        'scope':'train-split controls; copy/translated/source identity diagnostics retained; no val reward tuning'})
    return folder/'audit.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--updates',type=int,default=10)
    p.add_argument('--resume',type=Path);args=p.parse_args()
    if args.report.exists():raise FileExistsError('A terminal RL report already exists')
    import torch
    fraction=float(os.environ.get('REGION_PILOT_MEMORY_FRACTION','.9'))
    if not 0<fraction<=.95:raise ValueError('GPU memory fraction must be in (0,.95]')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(fraction)
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,NativeV7Trainer
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha,verify,verify_prefix
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import capture_rng,restore_rng
    from src.pose_control.qwen21_full.native_v62.training.validation import frozen_digest
    from src.pose_control.qwen21_full.region_rl import flow_sde_transition,group_advantages,grpo_loss
    config=TrainingConfigV7.from_dict(read(args.config))
    parent={'path':str(args.parent.resolve()),'sha256':file_sha(args.parent)}
    if config.initialization!=parent:raise ValueError('RL and reference must start from the exact RN SFT checkpoint')
    trainer=NativeV7Trainer(config);ctx=trainer._start('advance-stage',parent)
    backend=ctx['backend'];backend.runtime.train();optimizer=ctx['optimizer']
    original=torch.load(args.parent,map_location='cpu',mmap=True,weights_only=True)
    reference_state=original['state_dict'];base_update=ctx['global_update'];completed=0;attempts=0;replay_verified=False
    if config.stage.optimizer.scheduler!='constant':raise ValueError('This bounded RL pilot requires the explicit constant schedule')
    policy_identity={'algorithm':ALGORITHM,'config_sha256':config.signature(),'parent':parent,
        'rl_script_sha256':file_sha(__file__),'evaluation_script_sha256':file_sha(Path(__file__).with_name('evaluate.py')),
        'backend_identity':backend.identity,'catalog_sha256':backend.catalog.signature(),
        'reward_models':{name:file_sha(args.root/f'report/model-identity-{name}.json') for name in ('sdpose','identity')}}
    identity_hash=hashlib.sha256(json.dumps(policy_identity,sort_keys=True).encode()).hexdigest()
    if args.resume:
        marker=read(args.resume.with_suffix('.commit.json'));verify(marker['checkpoint'])
        if marker['checkpoint']!={'path':str(args.resume.resolve()),'sha256':file_sha(args.resume)}:
            raise ValueError('RL checkpoint/commit path or SHA differs')
        saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=True)
        if saved.get('training_schema')!=SCHEMA or saved['policy_state']['identity_sha256']!=identity_hash:
            raise ValueError('RL resume algorithm/data/code/reference differs')
        state=saved['policy_state'];verify_prefix(state['journal'])
        backend.runtime.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
        ctx['sampler'].load_state_dict(state['sampler_state']);restore_rng(state['rng'])
        completed=state['completed_updates'];attempts=state['attempts'];replay_verified=state['replay_verified'];del saved
    fixed=frozen_digest(backend)
    frozen_qwen=[(parameter,parameter._version) for name,parameter in backend.pipe.transformer.named_parameters()
                  if name in backend.base_state_names]
    if any(p.requires_grad for p,_ in frozen_qwen):raise RuntimeError('Qwen must remain frozen')
    if {id(p) for p,_ in frozen_qwen}&{id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Qwen parameters entered the Native optimizer')
    pairs=read(args.root/'data/evaluation.json');training_pairs=[x for x in pairs if x['split']=='train']
    run_id=uuid.uuid4().hex[:12];audit_folder=args.root/f'report/grpo/audit_{run_id}'
    reward_audit=audit_rewards(args.root,training_pairs,audit_folder)
    last=None
    def save():
        if frozen_digest(backend)!=fixed or any(p._version!=version for p,version in frozen_qwen):
            raise RuntimeError('Frozen Native/Qwen state changed during RL')
        payload=backend.runtime.checkpoint_payload(metadata={'training_interface':SCHEMA})
        payload['training_schema']=SCHEMA
        payload['policy_state']={'identity':policy_identity,'identity_sha256':identity_hash,
            'completed_updates':completed,'global_update':base_update+completed,'attempts':attempts,'replay_verified':replay_verified,
            'optimizer_state':optimizer.state_dict(),'sampler_state':ctx['sampler'].state_dict(),
            'rng':capture_rng(),'journal':ctx['journal'].state_dict(),'frozen_state':fixed}
        folder=Path(config.output_root)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(f'grpo_{completed:06d}_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        checkpoint={'path':str(path.resolve()),'sha256':file_sha(path)}
        write(path.with_suffix('.commit.json'),{'schema':SCHEMA.replace('_training_v1','_commit_v1'),
            'checkpoint':checkpoint,'identity_sha256':identity_hash,'completed_updates':completed})
        return checkpoint
    while completed<args.updates:
        if attempts>=args.updates*5:raise RuntimeError('Too many zero-variance or inapplicable reward groups')
        route_name,source,index=ctx['sampler'].draw();route={r.name:r for r in config.stage.routes}[route_name]
        sample,batch,pair=backend.pair(source,index);pair_id=sample['pair_metadata']['pair_id'];attempts+=1
        folder=args.root/f'output/grpo/rollouts/update_{completed:04d}_attempt_{attempts:04d}_{run_id}'
        replays=[];images=[];sde_index=ALGORITHM['sde_indices'][(attempts-1)%3]
        for candidate in range(ALGORITHM['group_size']):
            seed=420000+1000*(base_update+completed)+100*attempts+candidate
            replay,image=rollout(backend,batch,pair,route,seed,sde_index,folder/f'candidate_{candidate}.png')
            replays.append(replay);images.append({**image,'pair_id':pair_id,'label':'grpo_candidate'})
        with (folder/'replay.pt').open('xb') as f:torch.save({'source':source,'index':index,'pair_id':pair_id,'steps':replays},f)
        manifest=folder/'manifest.json';write(manifest,{'label':'grpo_rollout','records':images})
        measured=evaluate(args.root,manifest,folder);components={'body':[],'hands':[],'identity':[],'background':[]}
        applicable={}
        for r in measured['records']:
            for name in ('body','hands'):
                applicable[name]=r['pose'][name]['applicable'];components[name].append(r['pose'][name]['reward'] or 0.)
            applicable['identity']=r['identity']['applicable'];components['identity'].append(r['identity']['identity_reward'] or 0.)
            applicable['background']=r['pixel']['background_reward'] is not None;components['background'].append(r['pixel']['background_reward'] or 0.)
        advantage=group_advantages({k:torch.tensor(v,device=backend.device) for k,v in components.items()},ALGORITHM['reward_weights'],applicable)
        if advantage.abs().max()==0:
            ctx['journal'].append({'kind':'rl_degenerate_reward_group','attempt':attempts,'pair_id':pair_id,'rewards':components});continue
        current=clone_policy_state(backend.runtime.state_dict())
        optimizer.zero_grad(set_to_none=True);stats=[];delta=[]
        for candidate,replay in enumerate(replays):
            x=replay['latent'].to(backend.device);y=replay['next_latent'].to(backend.device)
            old=replay['old_log_prob'].to(backend.device);sigma=replay['sigma'];next_sigma=replay['next_sigma']
            with torch.no_grad():
                with weights(backend.runtime,reference_state,current):
                    ref_velocity=predict(backend,batch,pair,route,sigma,x)
                    _,_,ref_mean,_=flow_sde_transition(x,ref_velocity,sigma,next_sigma,next_latent=y)
            velocity=predict(backend,batch,pair,route,sigma,x)
            _,new,mean,std=flow_sde_transition(x,velocity,sigma,next_sigma,next_latent=y)
            if not replay_verified:delta.append(float((new.detach()-old).abs().max()))
            loss,info=grpo_loss(new,old,advantage[candidate:candidate+1],mean,ref_mean,
                clip=ALGORITHM['ppo_clip'],beta=ALGORITHM['reference_beta'],variance=std**2)
            (loss/len(replays)).backward();stats.append(info)
        if not replay_verified:
            result={'max_abs_log_prob_delta':max(delta),'passed':max(delta)<=1e-4,
                    'stored_transition_dtype':str(replays[0]['latent'].dtype),'fixed_reference_sha256':parent['sha256']}
            write(audit_folder/'replay.json',result)
            if not result['passed']:raise RuntimeError('No-update stochastic replay did not match')
            replay_verified=True
        active=[backend.catalog.parameters[n] for n in sorted(backend.active)]
        norm=torch.nn.utils.clip_grad_norm_(active,config.stage.optimizer.clip_norm)
        if not torch.isfinite(norm) or norm==0:raise RuntimeError('RL policy gradient is not finite/nonzero')
        optimizer.step();completed+=1
        changed=sum(not torch.equal(current[k],v.detach().cpu()) for k,v in backend.runtime.state_dict().items() if torch.is_tensor(v))
        if not changed:raise RuntimeError('RL optimizer did not update the Native policy')
        if any(p._version!=version for p,version in frozen_qwen):raise RuntimeError('Frozen Qwen changed')
        ctx['journal'].append({'kind':'flow_grpo_update','update':completed,'global_update':base_update+completed,
            'pair_id':pair_id,'attempt':attempts,'rewards':components,'applicable':applicable,
            'advantages':advantage.cpu().tolist(),'gradient_norm':float(norm),'changed_native_tensors':changed,
            'candidate_images':images,'policy_stats':stats,'frozen_qwen_unchanged':True})
        print(json.dumps({'rl_update':completed,'requested':args.updates,'gradient_norm':float(norm),'changed_native_tensors':changed}),flush=True)
        optimizer.zero_grad(set_to_none=True);del current
        if completed%5==0:last=save()
    if last is None or completed%5:last=save()
    write(args.report,{'schema':SCHEMA,'passed':True,'completed_updates':completed,'requested_updates':args.updates,
        'attempts':attempts,'checkpoint':last,'source_checkpoint':parent,'algorithm':ALGORITHM,
        'reward_audit':str(reward_audit),'replay_verified':replay_verified,'candidate_images':attempts*ALGORITHM['group_size'],
        'frozen_qwen_unchanged':True,'scope':'bounded Native Adapter RL pilot; extra rollout cost reported separately'})


if __name__=='__main__':main()
