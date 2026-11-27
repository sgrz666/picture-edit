"""Bounded Native Adapter Flow-GRPO using actual generated-image SDPose rewards.

The policy is the Native runtime; Qwen stays frozen. Every selected stochastic
transition is stored at its rollout dtype and replayed, with a fixed SFT reference.
RL snapshots have their own schema and cannot be mistaken for shared SFT resume.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))
from src.pose_control.qwen21_full.native_rl.regional_cli import DEFAULT_SDE_INDICES, ROLLOUT_STEPS

SDPY='/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python'
SCHEMA='native_region_grpo_training_v1'
ALGORITHM={'name':'native_flow_grpo_v5','group_size':4,'rollout_steps':ROLLOUT_STEPS,'sde_indices':list(DEFAULT_SDE_INDICES),
    'noise_level':.7,'ppo_clip':1e-4,'reference_beta':1e-3,
    'reward_weights':{'body':.45*.85/.95,'hands':.35*.85/.95,'identity':.15*.85/.95,'background':.15},
    'background_metric':'masked_source_l1_v1',
    'reference':'fixed_RN_SFT_parent','reference_penalty':'same_variance_transition_gaussian_kl',
    'advantage':'gdpo-guarded','std_floor':.001,
    'evaluation':'held-out fixed-seed ODE regression gate'}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def evaluate(root,manifest,folder,evaluator,targets,*,pose_device=None):
    script=Path(__file__).with_name('evaluate.py')
    pose,full=folder/'sdpose.json',folder/'metrics.json'
    try:
        command=[SDPY,str(script),'pose','--root',str(root),'--manifest',str(manifest),'--output',str(pose)]
        if pose_device is not None:
            command.extend(('--device',pose_device))
        subprocess.run(command,check=True)
        report = evaluator.enrich(read(pose), None, fixed=targets)
    except Exception as exc:
        write(folder/'evaluation-failed.json', {'status':'evaluator_error','message':str(exc)})
        raise
    write(full, report)
    return report


def ode_metrics(summary):
    body, hands = (summary['pose'][name] for name in ('body', 'hands'))
    metrics = {'pose': body['reward'], 'hands': hands.get('training_reward'), 'hands_gesture': hands.get('gesture_reward'),
               'identity': summary['identity']['identity_reward']['mean'],
               'background': summary['pixel']['background_reward']['mean']}
    return {name: value for name, value in metrics.items() if value is not None}


def ode_cohort(data, evaluation_rows, *, limit, seed):
    if len({row['pair_id'] for row in evaluation_rows}) != len(evaluation_rows):
        raise ValueError('Regional evaluation manifest contains duplicate pair IDs')
    pairs = {row['pair_id'] for row in evaluation_rows if row['split'] == 'val'}
    keys = sorted({(row['source_id'], row['index'], row['pair_id']) for row in data.records
                   if row['role'] == 'validation' and row['eligible']})
    if not keys or any(pair not in pairs for _, _, pair in keys) or len({pair for _, _, pair in keys}) != len(keys):
        raise ValueError('Regional ODE requires an explicit, unique held-out cohort')
    return [{'source': source, 'index': index, 'pair_id': pair_id, 'seed': seed + i,
             'condition': data.conditions[source, index][0]}
            for i, (source, index, pair_id) in enumerate(keys[:limit])]


def evaluate_ode(root, backend, route, cohort, folder, evaluator, targets, *, pose_device):
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import capture_rng, restore_rng
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    from src.pose_control.qwen21_full.native_rl.runner import NativeRollout

    rng = capture_rng()
    records = []
    try:
        for item in cohort:
            sample, batch, pair = backend.pair(item['source'], item['index'])
            if (pair['condition_sha256'] != item['condition']['sha256']
                    or sample['pair_metadata']['pair_id'] != item['pair_id']):
                raise ValueError('Regional ODE condition differs from the fixed held-out cohort')
            options, base = route_options(route, sample['native_scene_kind'])
            steps = backend.config.generation_steps
            policy = NativeRollout(backend, batch, pair, route, options, base, steps=steps, sde_step=1,
                                   noise_level=ALGORITHM['noise_level'])
            image, _ = policy.generate(item['seed'])
            path = folder / f'{len(records):04d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                image.save(handle, format='PNG')
            records.append({**item, 'path': str(path), 'sha256': file_sha(path), 'steps': steps})
        manifest = folder / 'manifest.json'
        write(manifest, {'label': 'regional_rl_ode', 'records': records})
        report = evaluate(root, manifest, folder, evaluator, targets, pose_device=pose_device)
        if [row['sha256'] for row in report['records']] != [row['sha256'] for row in records]:
            raise ValueError('ODE reward output differs from held-out image order')
        return ode_metrics(report['summary'])
    finally:
        restore_rng(rng)


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
        values['hands'] = row['pose']['hands'].get('training_reward', values['hands'])
        values.update(identity=row['identity']['identity_reward'], background=row['pixel']['background_reward'])
        values['spatial'] = (row.get('spatial') or {}).get('reward')
        if {name for name, score in values.items() if score is not None} != set(columns):
            raise ValueError('Candidate rewards disagree with fixed target eligibility')
        for name in columns:
            columns[name].append(values[name])
    return columns


def score_spatial_group(worker, target, records, folder):
    try:
        rows = worker.score_group(target, records)
        report = [{'file': {'path': record['path'], 'sha256': record['sha256']}, **row}
                  for record, row in zip(records, rows, strict=True)]
        write(folder / 'spatial.json', {'target': target.record(), 'candidates': report})
    except Exception as exc:
        write(folder / 'evaluation-failed.json', {'status': 'evaluator_error', 'evaluator': 'spatialreward',
                                                 'message': str(exc)})
        raise
    for record, row in zip(records, rows, strict=True):
        record['spatial'] = row


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
    from src.pose_control.qwen21_full.native_rl.regional_cli import add_arguments, configure_arguments

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--updates',type=int,default=10)
    p.add_argument('--resume',type=Path)
    add_arguments(p)
    args=p.parse_args()
    args.reward_arguments=configure_arguments(p,args)
    if args.updates < 1:raise ValueError('Positive optimizer update budget required')
    if args.report.exists():raise FileExistsError('A terminal RL report already exists')
    with ExitStack() as resources:
        train(args, resources)


def train(args, resources):
    import torch
    fraction=float(os.environ.get('REGION_PILOT_MEMORY_FRACTION','.9'))
    if not 0<fraction<=.95:raise ValueError('GPU memory fraction must be in (0,.95]')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(fraction)
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,NativeV7Trainer
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha,verify,verify_prefix,write_json
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import capture_rng,environment,restore_rng
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.training.validation import frozen_digest
    from src.pose_control.qwen21_full.native_rl.objective import advantages
    from src.pose_control.qwen21_full.native_rl.rewards import reward_weights
    from src.pose_control.qwen21_full.region_rewards import HAND_REWARD
    from src.pose_control.qwen21_full.native_rl.spatial_reward import spatial_arguments, spatial_target, with_spatial_weight
    from src.pose_control.qwen21_full.native_rl.spatial_worker import SpatialWorker
    from src.pose_control.qwen21_full.native_rl.validation import regression_limits, regressions
    from src.pose_control.qwen21_full.native_rl.runner import (FrozenReference, NativeRollout, RunCounters,
                                                              audit_replay, fixed_reference_means, restrict_optimizer,
                                                              rollout_sde_step, update_policy)
    from evaluate import IdentityPixelEvaluator
    config=TrainingConfigV7.from_dict(read(args.config))
    if len(config.stage.routes) != 1:
        raise ValueError('Regional RL ODE requires one fixed Native route')
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
    evaluator=IdentityPixelEvaluator(args.root, adaface_model=args.adaface_model,
                                    adaface_rpe_extension=args.adaface_rpe_extension)
    validation_evaluator = IdentityPixelEvaluator(args.root) if args.adaface_model is not None else evaluator
    if args.reward_profile == 'hands42':
        weights=reward_weights('single', pose_only=False, profile='hands42', spatial_weight=args.spatial_weight)
        weights={'body' if name=='pose' else name:value for name,value in weights.items()}
    else:
        weights=with_spatial_weight(ALGORITHM['reward_weights'], args.spatial_weight)
    algorithm={**ALGORITHM, 'sde_indices':args.sde_indices, 'reward_profile':args.reward_profile,
               'reward_weights':weights, 'hand_reward':HAND_REWARD, 'identity_backend':evaluator.identity_backend}
    spatial=None
    if args.spatial_model is not None:
        spatial=resources.enter_context(SpatialWorker.from_args(args))
        algorithm.update(spatial_worker=spatial.identity, spatial_std_floor=args.spatial_std_floor)
    floors={'spatial':args.spatial_std_floor} if args.spatial_std_floor is not None else None
    evaluation_manifest=args.root/'data/evaluation.json'
    manifest_ref=verify({'path':str(evaluation_manifest.resolve())})
    pairs=read(evaluation_manifest)
    training_pairs=[row for row in pairs if row['split']=='train']
    train_by_id={row['pair_id']:row for row in training_pairs}
    val_by_id={row['pair_id']:row for row in pairs if row['split']=='val'}
    cohort=ode_cohort(backend.data,pairs,limit=args.val_limit,seed=config.generation_seed)
    limits=regression_limits(args)
    limits['background']=args.background_drop
    ode_policy={'cohort':cohort,'steps':config.generation_steps,'validate_every':args.validate_every,
                'patience':args.patience,'limits':limits,'sdpose_device':args.ode_pose_device,
                'identity_backend':validation_evaluator.identity_backend}
    policy_identity={'algorithm':algorithm,'config_sha256':config.signature(),'parent':parent,
        'requested_updates':args.updates,'ode_policy':ode_policy,
        'rl_script_sha256':file_sha(__file__),'evaluation_script_sha256':file_sha(Path(__file__).with_name('evaluate.py')),
        'evaluation_manifest_sha256':manifest_ref['sha256'],
        'background_region_index':evaluator.background_index,
        'backend_identity':backend.identity,'catalog_sha256':backend.catalog.signature(),
        'training_environment':{**environment(),'python_path':sys.executable,'prefix':sys.prefix},
        'reward_models':{'sdpose':file_sha(args.root/'report/model-identity-sdpose.json'),
                         'identity':evaluator.model_identity['sha256'],
                         'ode_identity':validation_evaluator.model_identity['sha256']}}
    identity_hash=hashlib.sha256(json.dumps(policy_identity,sort_keys=True).encode()).hexdigest()
    stop=None
    last=None
    selected=parent
    selected_update=0
    selected_report=None
    if args.resume:
        marker=read(args.resume.with_suffix('.commit.json'))
        last=verify(marker['checkpoint'])
        if last['path']!=str(args.resume.resolve()):
            raise ValueError('RL checkpoint/commit path or SHA differs')
        saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=True)
        if saved.get('training_schema')!=SCHEMA or saved['policy_state']['identity_sha256']!=identity_hash:
            raise ValueError('RL resume algorithm/data/code/reference differs')
        state=saved['policy_state'];verify_prefix(state['journal'])
        if (marker['identity_sha256']!=identity_hash or marker['stop']!=state['stop'] or
                marker['approved_current']!=state['approved_current'] or
                marker['completed_updates']!=state['counters']['optimizer_updates'] or
                marker['rollout_batches']!=state['counters']['rollout_batches']):
            raise ValueError('Regional RL commit state differs from its checkpoint')
        backend.runtime.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
        ctx['sampler'].load_state_dict(state['sampler_state']);restore_rng(state['rng'])
        counters=RunCounters(**state['counters']);replay_verified=state['replay_verified']
        stop=state['stop']
        validation=state['validation']
        verify(validation['baseline_report'])
        if validation['last_report'] is not None:
            verify(validation['last_report'])
        if state['approved_current']:
            selected,selected_update=last,counters.optimizer_updates
            selected_report=validation['last_report']
        else:
            selected,selected_update=verify(state['selected']),state['selected_update']
            selected_report=state['selected_report']
            if selected_report is not None:
                verify(selected_report)
        del saved
    fixed=frozen_digest(backend)
    frozen_qwen=[(parameter,parameter._version) for name,parameter in backend.pipe.transformer.named_parameters()
                  if name in backend.base_state_names]
    if any(p.requires_grad for p,_ in frozen_qwen):raise RuntimeError('Qwen must remain frozen')
    if {id(p) for p,_ in frozen_qwen}&{id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Qwen parameters entered the Native optimizer')
    run_id=uuid.uuid4().hex[:12];audit_folder=args.root/f'report/grpo/audit_{run_id}'
    ode_root=args.root/f'report/grpo/ode_{run_id}'
    held_out = {}
    if stop is None:
        held_out={item['pair_id']:validation_evaluator.prepare(val_by_id[item['pair_id']],
                  sample=backend.data.sample(item['source'],item['index'])) for item in cohort}
        reward_audit=str(audit_rewards(args.root,training_pairs,audit_folder,evaluator))
    else:
        reward_audit=state['reward_audit']

    def measure(folder):
        verify(manifest_ref)
        return evaluate_ode(args.root,backend,config.stage.routes[0],cohort,folder,validation_evaluator,held_out,
                            pose_device=args.ode_pose_device)

    if args.resume is None:
        baseline=measure(ode_root/'sft')
        if not {'pose','identity'} <= baseline.keys():
            raise ValueError('Regional ODE requires applicable held-out Pose and glintr100 ID rewards')
        baseline_report=write_json(ode_root/'sft'/'gate.json',{'sft_reference':parent,'policy':ode_policy,'metrics':baseline})
        validation={'baseline':baseline,'baseline_report':baseline_report,'last_report':None,'last_update':0,'streak':0}

    def validate():
        folder=ode_root/f'update_{counters.optimizer_updates:06d}'
        metrics=measure(folder)
        failed=regressions(metrics,validation['baseline'],limits)
        streak=validation['streak']+1 if failed else 0
        report=write_json(folder/'gate.json',{'sft_reference':parent,'baseline':validation['baseline_report'],
            'metrics':metrics,'regressed_metrics':failed,'consecutive_regressions':streak,**counters.record()})
        validation.update(last_report=report,last_update=counters.optimizer_updates,streak=streak)
        ctx['journal'].append({'kind':'rl_ode_validation','report':report,'regressed_metrics':failed,**counters.record()})
        return not failed

    def save(*, approved=False):
        nonlocal selected,selected_update,selected_report
        completed=counters.optimizer_updates
        if frozen_digest(backend)!=fixed or any(p._version!=version for p,version in frozen_qwen):
            raise RuntimeError('Frozen Native/Qwen state changed during RL')
        payload=backend.runtime.checkpoint_payload(metadata={'training_interface':SCHEMA})
        payload['training_schema']=SCHEMA
        payload['policy_state']={'identity':policy_identity,'identity_sha256':identity_hash,
            'completed_updates':completed,'global_update':base_update+completed,'attempts':counters.rollout_batches,
            'counters':counters.record(),'replay_verified':replay_verified,
            'validation':validation,'approved_current':approved,'selected':selected,'selected_update':selected_update,
            'selected_report':selected_report,
            'stop':stop,'reward_audit':reward_audit,
            'optimizer_state':optimizer.state_dict(),'sampler_state':ctx['sampler'].state_dict(),
            'rng':capture_rng(),'journal':ctx['journal'].state_dict(),'frozen_state':fixed}
        folder=Path(config.output_root)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(f'grpo_{completed:06d}_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        checkpoint={'path':str(path.resolve()),'sha256':file_sha(path)}
        write_json(path.with_suffix('.commit.json'),{'schema':SCHEMA.replace('_training_v1','_commit_v1'),
            'checkpoint':checkpoint,'identity_sha256':identity_hash,'completed_updates':completed,
            'rollout_batches':counters.rollout_batches,'stop':stop,'approved_current':approved})
        if approved:
            selected,selected_update=checkpoint,completed
            selected_report=validation['last_report']
        return checkpoint
    while stop is None:
        attempts,completed=counters.rollout_batches,counters.optimizer_updates
        if attempts>=args.updates*5:
            stop='rollout_budget' if completed else 'no_reward_signal'
            if completed:
                approved=validate() if completed>validation['last_update'] else False
                if validation['streak']>=args.patience:
                    stop='ode_regression'
                last=save(approved=approved)
            break
        route_name,source,index=ctx['sampler'].draw();route={r.name:r for r in config.stage.routes}[route_name]
        sample,batch,pair=backend.pair(source,index);pair_id=sample['pair_metadata']['pair_id'];attempts+=1
        targets=evaluator.prepare(train_by_id[pair_id],sample=sample);eligible=targets['eligible']
        if spatial is not None:
            condition_ref, condition=backend.data.conditions[source,index]
            if condition_ref['sha256'] != pair['condition_sha256']:
                raise ValueError('SpatialReward condition differs from rollout condition')
            visual_target=spatial_target(condition, spatial.references)
            eligible={**eligible, 'spatial':True}
        folder=args.root/f'output/grpo/rollouts/update_{completed:04d}_attempt_{attempts:04d}_{run_id}'
        replays=[];images=[];sde_index=rollout_sde_step(algorithm['sde_indices'], counters.rollout_batches)
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
        if spatial is not None:
            score_spatial_group(spatial, visual_target, measured['records'], folder)
        components=score_columns(measured['records'],eligible)
        diagnostics={}
        advantage, reward_stats=advantages([components],algorithm['reward_weights'],mode=algorithm['advantage'],
                                           std_floor=algorithm['std_floor'],std_floors=floors,diagnostics=diagnostics)
        advantage=advantage[0]
        write_json(folder/'reward-diagnostics.json',{'reward_stats':reward_stats,
            'reward_diagnostics':diagnostics,'advantages':advantage.tolist()})
        grouped=[(replays,advantage,policy.predict)]
        if advantage.abs().max()==0:
            audit_replay(grouped,noise_level=ALGORITHM['noise_level'])
            ctx['journal'].append({'kind':'rl_degenerate_reward_group','attempt':attempts,'pair_id':pair_id,
                                  'rewards':components,'reward_diagnostics':diagnostics,**counters.record()});continue
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
            'reward_diagnostics':diagnostics,
            'advantages':advantage.tolist(),'gradient_norm':stats['grad_norm'],'updated_parameter_tensors':changed,
            'candidate_images':images,'policy_stats':stats,'frozen_qwen_unchanged':True,**counters.record()})
        print(json.dumps({'rl_update':completed,'requested':args.updates,'gradient_norm':stats['grad_norm'],**counters.record()}),flush=True)
        if completed%args.validate_every==0 or completed==args.updates:
            approved=validate()
            if validation['streak']>=args.patience:
                stop='ode_regression'
            elif completed==args.updates:
                stop='budget'
            last=save(approved=approved)
        elif completed%5==0:
            last=save()
    completed=counters.optimizer_updates
    write_json(args.report,{'schema':SCHEMA,'passed':selected_update>0,'stop':stop,
        'completed_updates':completed,'requested_updates':args.updates,'selected_update':selected_update,
        'selected_ode_report':selected_report if selected_report is not None else validation['baseline_report'],
        'attempts':counters.rollout_batches,'checkpoint':selected,'last_checkpoint':last,'source_checkpoint':parent,
        'algorithm':algorithm,'ode_policy':ode_policy,'ode_validation':validation,
        'spatial_arguments':spatial_arguments(args),
        'reward_arguments':args.reward_arguments,
        'reward_audit':reward_audit,'replay_verified':replay_verified,'candidate_images':counters.generated_images,**counters.record(),
        'frozen_qwen_unchanged':True,'scope':'bounded Native Adapter RL with held-out rollback'})


if __name__=='__main__':main()
