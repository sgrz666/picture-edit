"""Bounded Native Adapter Flow-GRPO using actual generated-image SDPose rewards.

The policy is the Native runtime; Qwen stays frozen. Every selected stochastic
transition is stored at its rollout dtype and replayed, with a fixed SFT reference.
RL snapshots have their own schema and cannot be mistaken for shared SFT resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))
SDPY='/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python'
SCHEMA='native_region_grpo_training_v1'
ALGORITHM={'name':'native_flow_grpo_v2','group_size':4,'rollout_steps':10,'sde_indices':[1,2,3],
    'noise_level':.7,'ppo_clip':1e-4,'reference_beta':1e-3,
    'reward_weights':{'body':.45,'hands':.35,'identity':.15,'background':.05},
    'reference':'fixed_RN_SFT_parent','reference_penalty':'same_variance_transition_gaussian_kl',
    'advantage':'gdpo-guarded','std_floor':.001,
    'evaluation':'held-out fixed-seed 20-step ODE; never used as optimization reward'}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def evaluate(root,manifest,folder,evaluator,targets):
    script=Path(__file__).with_name('evaluate.py')
    pose,full=folder/'sdpose.json',folder/'metrics.json'
    try:
        subprocess.run([SDPY,str(script),'pose','--root',str(root),'--manifest',str(manifest),'--output',str(pose)],check=True)
        report = evaluator.enrich(read(pose), None, fixed=targets)
    except Exception as exc:
        write(folder/'evaluation-failed.json', {'status':'evaluator_error','message':str(exc)})
        raise
    write(full, report)
    return report


def render_candidate(policy, seed, path):
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    image, replay = policy.generate(seed, stochastic=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:image.save(f,format='PNG')
    return replay,{'path':str(path),'sha256':file_sha(path),'seed':seed,'steps':ALGORITHM['rollout_steps'],
        'target_rgb_used_as_condition':False,'stochastic_steps':[policy.sde_step]}


def score_columns(records, eligible):
    columns = {name: [] for name, applicable in eligible.items() if applicable}
    for row in records:
        values = {name: row['pose'][name]['reward'] for name in ('body', 'hands')}
        values.update(identity=row['identity']['identity_reward'], background=row['pixel']['background_reward'])
        if {name for name, score in values.items() if score is not None} != set(columns):
            raise ValueError('Candidate rewards disagree with fixed target eligibility')
        for name in columns:
            columns[name].append(values[name])
    return columns


def audit_rewards(root,pairs,folder,evaluator):
    from PIL import Image,ImageChops
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    records=[]
    for pair in pairs[:2]:
        spec=CanvasSpec.from_record(pair['target_canvas'])
        with Image.open(pair['target_rgb']['path']) as im:gt=spec.image(im.convert('RGB'))
        with Image.open(pair['source_rgb']['path']) as im:
            copy=CanvasSpec.from_record(pair['source_canvas']).image(im.convert('RGB')).resize(gt.size)
        variants={'GT':gt,'copy_source':copy,'blank':Image.new('RGB',gt.size),
                  'translated_GT':ImageChops.offset(gt,round(gt.width*.12),0)}
        for name,image in variants.items():
            path=folder/(pair['pair_id']+'_'+name+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:image.save(f,format='PNG')
            records.append({'pair_id':pair['pair_id'],'path':str(path),'sha256':file_sha(path),'label':name})
    manifest=folder/'manifest.json';write(manifest,{'label':'reward_controls','records':records})
    fixed = {pair['pair_id']: evaluator.prepare(pair) for pair in pairs[:2]}
    report=evaluate(root,manifest,folder,evaluator,fixed)
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
    if args.updates < 1:raise ValueError('Positive optimizer update budget required')
    if args.report.exists():raise FileExistsError('A terminal RL report already exists')
    import torch
    fraction=float(os.environ.get('REGION_PILOT_MEMORY_FRACTION','.9'))
    if not 0<fraction<=.95:raise ValueError('GPU memory fraction must be in (0,.95]')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(fraction)
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,NativeV7Trainer
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha,verify,verify_prefix
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import capture_rng,restore_rng
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.training.validation import frozen_digest
    from src.pose_control.qwen21_full.native_rl.objective import advantages
    from src.pose_control.qwen21_full.native_rl.runner import (FrozenReference, NativeRollout, RunCounters,
                                                              audit_replay, fixed_reference_means, restrict_optimizer,
                                                              update_policy)
    from evaluate import IdentityPixelEvaluator
    config=TrainingConfigV7.from_dict(read(args.config))
    parent={'path':str(args.parent.resolve()),'sha256':file_sha(args.parent)}
    if config.initialization!=parent:raise ValueError('RL and reference must start from the exact RN SFT checkpoint')
    original=torch.load(args.parent,map_location='cpu',mmap=True,weights_only=True)
    state=original['training_state']
    print(json.dumps({'sft_parent':parent,'sft_stage':state['config']['stage']['stage_id'],
                      'sft_global_updates':state['global_update'],'sft_stage_updates':state['stage_update'],
                      'sft_ancestor':state['parent'],'stage_origin':state['stage_origin']}),flush=True)
    del original,state
    trainer=NativeV7Trainer(config);ctx=trainer._start('advance-stage',parent)
    backend=ctx['backend'];backend.runtime.eval();optimizer=ctx['optimizer']
    active=[backend.catalog.parameters[n] for n in sorted(backend.active)]
    restrict_optimizer(optimizer,active)
    active_ids={id(p) for p in active}
    frozen=[p for p in dict.fromkeys([*backend.catalog.parameters.values(), *backend.pipe.transformer.parameters()])
            if id(p) not in active_ids]
    reference=FrozenReference(active,storage_device='cpu');base_update=ctx['global_update'];counters=RunCounters();replay_verified=False
    if config.stage.optimizer.scheduler!='constant':raise ValueError('This bounded RL pilot requires the explicit constant schedule')
    evaluator=IdentityPixelEvaluator(args.root)
    evaluation_manifest=args.root/'data/evaluation.json'
    policy_identity={'algorithm':ALGORITHM,'config_sha256':config.signature(),'parent':parent,
        'rl_script_sha256':file_sha(__file__),'evaluation_script_sha256':file_sha(Path(__file__).with_name('evaluate.py')),
        'evaluation_manifest_sha256':file_sha(evaluation_manifest),
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
        counters=RunCounters(**state['counters']);replay_verified=state['replay_verified'];del saved
    fixed=frozen_digest(backend)
    frozen_qwen=[(parameter,parameter._version) for name,parameter in backend.pipe.transformer.named_parameters()
                  if name in backend.base_state_names]
    if any(p.requires_grad for p,_ in frozen_qwen):raise RuntimeError('Qwen must remain frozen')
    if {id(p) for p,_ in frozen_qwen}&{id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Qwen parameters entered the Native optimizer')
    pairs=read(evaluation_manifest);training_pairs=[x for x in pairs if x['split']=='train']
    train_by_id={row['pair_id']:row for row in training_pairs}
    run_id=uuid.uuid4().hex[:12];audit_folder=args.root/f'report/grpo/audit_{run_id}'
    reward_audit=audit_rewards(args.root,training_pairs,audit_folder,evaluator)
    last=None
    def save():
        completed=counters.optimizer_updates
        if frozen_digest(backend)!=fixed or any(p._version!=version for p,version in frozen_qwen):
            raise RuntimeError('Frozen Native/Qwen state changed during RL')
        payload=backend.runtime.checkpoint_payload(metadata={'training_interface':SCHEMA})
        payload['training_schema']=SCHEMA
        payload['policy_state']={'identity':policy_identity,'identity_sha256':identity_hash,
            'completed_updates':completed,'global_update':base_update+completed,'attempts':counters.rollout_batches,
            'counters':counters.record(),'replay_verified':replay_verified,
            'optimizer_state':optimizer.state_dict(),'sampler_state':ctx['sampler'].state_dict(),
            'rng':capture_rng(),'journal':ctx['journal'].state_dict(),'frozen_state':fixed}
        folder=Path(config.output_root)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(f'grpo_{completed:06d}_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        checkpoint={'path':str(path.resolve()),'sha256':file_sha(path)}
        write(path.with_suffix('.commit.json'),{'schema':SCHEMA.replace('_training_v1','_commit_v1'),
            'checkpoint':checkpoint,'identity_sha256':identity_hash,'completed_updates':completed})
        return checkpoint
    while counters.optimizer_updates<args.updates:
        attempts,completed=counters.rollout_batches,counters.optimizer_updates
        if attempts>=args.updates*5:raise RuntimeError('Too many zero-variance or inapplicable reward groups')
        route_name,source,index=ctx['sampler'].draw();route={r.name:r for r in config.stage.routes}[route_name]
        sample,batch,pair=backend.pair(source,index);pair_id=sample['pair_metadata']['pair_id'];attempts+=1
        targets=evaluator.prepare(train_by_id[pair_id]);eligible=targets['eligible']
        folder=args.root/f'output/grpo/rollouts/update_{completed:04d}_attempt_{attempts:04d}_{run_id}'
        replays=[];images=[];sde_index=ALGORITHM['sde_indices'][(attempts-1)%len(ALGORITHM['sde_indices'])]
        options,base=route_options(route,sample['native_scene_kind'])
        policy=NativeRollout(backend,batch,pair,route,options,base,steps=ALGORITHM['rollout_steps'],
                             sde_step=sde_index,noise_level=ALGORITHM['noise_level'])
        for candidate in range(ALGORITHM['group_size']):
            seed=420000+1000*(base_update+completed)+100*attempts+candidate
            replay,image=render_candidate(policy,seed,folder/f'candidate_{candidate}.png')
            replays.append(replay);images.append({**image,'pair_id':pair_id,'label':'grpo_candidate'})
        counters.rollout(len(images))
        stored=[{k:v.cpu() if torch.is_tensor(v) else v for k,v in vars(replay).items()} for replay in replays]
        with (folder/'replay.pt').open('xb') as f:torch.save({'source':source,'index':index,'pair_id':pair_id,'steps':stored},f)
        manifest=folder/'manifest.json';write(manifest,{'label':'grpo_rollout','records':images})
        measured=evaluate(args.root,manifest,folder,evaluator,{pair_id:targets})
        if [r['sha256'] for r in measured['records']]!=[r['sha256'] for r in images]:
            raise ValueError('Reward output differs from rollout candidate order')
        components=score_columns(measured['records'],eligible)
        advantage, reward_stats=advantages([components],ALGORITHM['reward_weights'],mode=ALGORITHM['advantage'],std_floor=ALGORITHM['std_floor'])
        advantage=advantage[0]
        grouped=[(replays,advantage,policy.predict)]
        if advantage.abs().max()==0:
            audit_replay(grouped,noise_level=ALGORITHM['noise_level'])
            ctx['journal'].append({'kind':'rl_degenerate_reward_group','attempt':attempts,'pair_id':pair_id,
                                  'rewards':components,**counters.record()});continue
        means=fixed_reference_means(reference,grouped,noise_level=ALGORITHM['noise_level'])
        before=[p._version for p in active]
        stats=update_policy(optimizer,grouped,active,frozen,reference_means=means,noise_level=ALGORITHM['noise_level'],
                            clip=ALGORITHM['ppo_clip'],beta=ALGORITHM['reference_beta'],max_grad=config.stage.optimizer.clip_norm)
        counters.updated(len(replays));completed=counters.optimizer_updates
        if not replay_verified:
            result={'max_ratio_error':stats['max_ratio_error'],'passed':True,
                    'stored_transition_dtype':str(replays[0].state.dtype),'fixed_reference_sha256':parent['sha256']}
            write(audit_folder/'replay.json',result)
            replay_verified=True
        changed=sum(p._version!=v for p,v in zip(active,before))
        if not changed:raise RuntimeError('RL optimizer did not update the Native policy')
        if any(p._version!=version for p,version in frozen_qwen):raise RuntimeError('Frozen Qwen changed')
        ctx['journal'].append({'kind':'flow_grpo_update','update':completed,'global_update':base_update+completed,
            'pair_id':pair_id,'attempt':attempts,'rewards':components,'applicable':eligible,'reward_stats':reward_stats,
            'advantages':advantage.tolist(),'gradient_norm':stats['grad_norm'],'updated_parameter_tensors':changed,
            'candidate_images':images,'policy_stats':stats,'frozen_qwen_unchanged':True,**counters.record()})
        print(json.dumps({'rl_update':completed,'requested':args.updates,'gradient_norm':stats['grad_norm'],**counters.record()}),flush=True)
        if completed%5==0:last=save()
    completed=counters.optimizer_updates
    if last is None or completed%5:last=save()
    write(args.report,{'schema':SCHEMA,'passed':True,'completed_updates':completed,'requested_updates':args.updates,
        'attempts':counters.rollout_batches,'checkpoint':last,'source_checkpoint':parent,'algorithm':ALGORITHM,
        'reward_audit':str(reward_audit),'replay_verified':replay_verified,'candidate_images':counters.generated_images,**counters.record(),
        'frozen_qwen_unchanged':True,'scope':'bounded Native Adapter RL pilot; extra rollout cost reported separately'})


if __name__=='__main__':main()
