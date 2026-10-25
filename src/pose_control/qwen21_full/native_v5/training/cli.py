"""Safe default prepare. Model/optimizer imports are behind explicit actions."""
from pathlib import Path
from dataclasses import replace
import argparse,json
from .config import TrainingConfig,SourceSpec,LossSpec,stage_preset,canonical

def ref(path):return None if path is None else {'path':str(Path(path).resolve()),'sha256':None}

def parse_config(a):
    if a.config:
        if a.profile or a.phase:raise ValueError('Choose config stage or legacy profile/phase')
        cfg=TrainingConfig.from_dict(json.loads(a.config.read_text(encoding='utf8')))
    elif a.action=='resume':
        from .checkpointing import load_committed
        saved,_,_=load_committed(ref(a.checkpoint));cfg=TrainingConfig.from_dict(saved['training_state']['config'])
    else:
        phase=a.phase or ('S3' if a.profile in ('joint','all-native') else 'S1' if a.profile and a.profile.endswith('heads') else 'S2' if a.profile else 'S0')
        branch=a.branch or (a.profile.split('-')[0] if a.profile and a.profile not in ('joint','all-native') else None)
        stage=stage_preset(phase,branch,budget=a.steps)
        if a.profile=='all-native':stage=replace(stage,unfreeze_geometry=True)
        sources=[]
        if a.manifest:sources.append(SourceSpec('train',manifest=ref(a.manifest),groups=ref(a.groups),weight=1.))
        if a.validation_manifest:sources.append(SourceSpec('validation',role='validation',manifest=ref(a.validation_manifest),groups=ref(a.validation_groups),weight=1.))
        cfg=TrainingConfig(stage=stage,sources=tuple(sources))
    updates={}
    for arg,key in (('output','output_root'),('model','model_path'),('cache_root','cache_root')):
        if getattr(a,arg) is not None:updates[key]=str(getattr(a,arg).resolve())
    for arg in ('model_manifest','cache_identity','gradient_report','body_baseline_report'):
        if getattr(a,arg) is not None:updates[arg]=ref(getattr(a,arg))
    if a.checkpoint is not None and a.action!='resume':updates['initialization']=ref(a.checkpoint)
    if a.seed is not None:updates['seed']=a.seed
    if a.evaluators:updates['evaluators']=json.loads(a.evaluators.read_text(encoding='utf8'))
    stage=cfg.stage
    if a.steps is not None:stage=replace(stage,budget_updates=a.steps)
    if a.accumulation is not None:stage=replace(stage,accumulation=a.accumulation)
    if a.lr is not None:stage=replace(stage,optimizer=replace(stage.optimizer,preset='legacy_flat',multiplier=a.lr/1e-4))
    if a.freeze_group:stage=replace(stage,exclude_groups=tuple(a.freeze_group))
    if a.unfreeze_group:stage=replace(stage,include_groups=tuple(a.unfreeze_group))
    if a.group_lr:
        values=dict(stage.optimizer.group_lrs)
        for item in a.group_lr:
            k,v=item.split('=',1)
            if k in values:raise ValueError('Duplicate LR override')
            values[k]=float(v)
        stage=replace(stage,optimizer=replace(stage.optimizer,group_lrs=values))
    if a.loss_profile:
        routes=[]
        for r in stage.routes:
            f=.5 if r.branch in ('face','detail','joint') else 0.;h=.5 if r.branch in ('hand','detail','joint') else 0.
            loss=LossSpec() if a.loss_profile=='velocity_only' else LossSpec(face=f,hand=h,identity=.1 if a.loss_profile=='local_aux' else 0.,perceptual=.05 if a.loss_profile=='local_aux' else 0.)
            routes.append(replace(r,loss=loss))
        stage=replace(stage,routes=tuple(routes))
    return replace(cfg,stage=stage,**updates)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',default='prepare',choices=('prepare','preflight','prepare-cache','train','resume','advance-stage'))
    p.add_argument('--level',choices=('data','grad'),default='data');p.add_argument('--preflight-only',action='store_true')
    for name in ('config','checkpoint','manifest','validation-manifest','groups','validation-groups','model','model-manifest','cache-root','cache-identity','output','report','gradient-report','body-baseline-report','evaluators','cache-output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--profile',choices=('face-heads','face-full','hand-heads','hand-full','detail-heads','detail-full','interaction-heads','interaction-full','joint','all-native'))
    p.add_argument('--phase',choices=('S0','S1','S2','S3'));p.add_argument('--branch',choices=('face','hand','detail','interaction'))
    p.add_argument('--steps',type=int);p.add_argument('--accumulation',type=int);p.add_argument('--seed',type=int);p.add_argument('--lr',type=float)
    p.add_argument('--freeze-group',action='append');p.add_argument('--unfreeze-group',action='append');p.add_argument('--group-lr',action='append')
    p.add_argument('--loss-profile',choices=('velocity_only','local_velocity','local_aux'));p.add_argument('--stop-after-update',type=int)
    a=p.parse_args(argv)
    if a.preflight_only:a.action='preflight';a.level='data'
    if a.action in ('resume','advance-stage') and a.checkpoint is None:p.error('Explicit committed --checkpoint required')
    cfg=parse_config(a)
    from .trainer import NativeV5Trainer
    trainer=NativeV5Trainer(cfg)
    if a.action=='prepare':print(canonical(trainer.prepare(a.report)));return
    if a.action=='preflight':
        if a.report is None:p.error('--report required')
        result=trainer.preflight_data(a.report) if a.level=='data' else trainer.preflight_grad(a.report)
        print(canonical({'level':a.level,'passed':result['passed'],'report':str(a.report)}));return
    if a.action=='prepare-cache':
        if a.cache_output is None:p.error('--cache-output must be a new directory')
        print(canonical(trainer.cache_prepare(a.cache_output)));return
    print(canonical(trainer.run(mode=a.action,checkpoint=ref(a.checkpoint),stop_after_update=a.stop_after_update)))
