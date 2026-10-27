#!/usr/bin/env python3
"""Audit committed training ancestry and optimizer counters without using CUDA."""
from pathlib import Path
import argparse
import json
import math
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def parameter_step(saved,name):
    opt=saved['training_state']['optimizer_state']
    for group in opt['param_groups']:
        for n,i in zip(group['names'],group['params']):
            if n==name:
                value=opt['state'].get(i)
                if value is None:return None
                return {'step':float(value['step']),'exp_avg_dtype':str(value['exp_avg'].dtype),
                    'exp_avg_norm':float(value['exp_avg'].norm()),'exp_avg_sq_dtype':str(value['exp_avg_sq'].dtype)}
    raise ValueError('Audited parameter missing from optimizer superset: '+name)


def main():
    import torch
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v5.training.io import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError('Use a new immutable summary')
    if not a.output.resolve().is_relative_to('/data/shangguanrz'):raise ValueError('Reports belong on persistent storage')
    markers=[(json.loads(f.read_text()),f.stat().st_mtime_ns) for f in (a.root/'training/checkpoints').glob('*.commit.json')]
    marker,_=max(markers,key=lambda x:(x[0]['global_update'],x[1]))
    saved,reference,tail=load_committed(marker['checkpoint']);final=saved
    if saved['training_state']['global_update']!=300:raise ValueError('This completion report requires exactly300 updates')
    ancestors=[];updates=[];parents=[];stages={};frozen=[];s2=None
    probes=[];generations=[];successful_records=[]
    while True:
        state=saved['training_state'];phase=state['config']['stage']['phase'];frozen.append(state['frozen_state'])
        ancestors.append({'checkpoint':reference,'phase':phase,'global_update':state['global_update'],
            'stage_update':state['stage_update'],'micro_position':state['micro_position'],'log_tail':tail,
            'committed_journal':state['journal']})
        if state['micro_position']!=0:raise ValueError('Saved partial accumulation window')
        raw=Path(state['journal']['path']).read_text().splitlines()[:state['journal']['sequence']]
        records=[json.loads(line)['record'] for line in raw]
        successful=[r for r in records if r.get('optimizer_updated')]
        probes.extend({'phase':phase,**r} for r in records if r.get('kind')=='same_scene_observation_loss')
        generations.extend({'phase':phase,**r} for r in records if r.get('kind')=='fixed_seed_generation')
        successful_records.extend({'phase':phase,'global_update':r['global_update'],
            'gradient_norm':r['gradient_norm'],'window':[{'route':v['route'],'source':v['source'],'sigma':v['sigma']} for v in r['window']],
            'velocity':[v.get('velocity') for v in r['losses']]} for r in successful)
        updates += [r['global_update'] for r in successful]
        bucket=stages.setdefault(phase,{'updates':0,'loss_velocity':[],'gradient_norm':[],'sigma':[],'routes':{}})
        bucket['updates']+=len(successful)
        for row in successful:
            if not math.isfinite(row['gradient_norm']):raise ValueError('Nonfinite recorded gradient')
            bucket['gradient_norm'].append(row['gradient_norm'])
            for item in row['window']:
                bucket['sigma'].append(item['sigma']);bucket['routes'][item['route']]=bucket['routes'].get(item['route'],0)+1
            for item in row['losses']:
                if 'velocity' in item:bucket['loss_velocity'].append(item['velocity'])
        if state['global_update']==240:s2=saved
        if state['parent'] is None:
            initialization=state['stage_origin'];break
        saved,reference,tail=load_committed(state['parent'])
    if sorted(updates)!=list(range(1,301)):raise ValueError('Committed ancestry has missing or duplicate optimizer updates')
    if {k:v['updates'] for k,v in stages.items()}!={'S0':60,'S1':60,'S2':120,'S3':60}:raise ValueError('Stage budgets differ')
    comparisons={}
    if s2 is None:raise ValueError('S2 parent missing')
    for name in ('base.body.output_heads.0.weight','face_controller.output_heads.0.weight','hand_controller.output_heads.0.weight'):
        before=parameter_step(s2,name);after=parameter_step(final,name)
        if before is None or after is None or after['step']-before['step']!=60:
            raise ValueError('Optimizer update counter did not continue through S3: '+name)
        comparisons[name]={'at_240':before,'at_300':after,'counter_increment':60}
    for key in ('qwen','frozen_auxiliary'):
        if len({f[key] for f in frozen})!=1:raise ValueError('Frozen Qwen/VAE/text state changed between stages')
    for value in stages.values():
        for key in ('loss_velocity','gradient_norm','sigma'):
            numbers=value[key]
            value[key]=None if not numbers else {'count':len(numbers),'mean':statistics.mean(numbers),'min':min(numbers),'max':max(numbers)}
    report={'schema':'native_v5_fixedpair_training_completion_audit_v2','passed':True,'completed_updates':300,
        'checkpoint':marker['checkpoint'],'initialization':initialization,'stages':stages,'committed_ancestry':ancestors,
        'optimizer_counter_continuity':comparisons,'frozen_qwen_and_auxiliary_across_stages':True,
        'optimizer_or_model_forward_executed':False,'cuda_used':False,
        'successful_update_records':sorted(successful_records,key=lambda r:r['global_update']),
        'committed_observation_probes':sorted(probes,key=lambda r:r['update']),
        'committed_generation_records':sorted(generations,key=lambda r:r['update']),
        'loss_comparison_limit':'random sigma and differing routes; this is not a generation-quality score'}
    write_json(a.output,report)
    print(json.dumps({'passed':True,'updates':300,'stage_updates':{k:v['updates'] for k,v in stages.items()},
        'optimizer_counter_continuity':True,'frozen_qwen':True,'report':str(a.output)}))


if __name__=='__main__':main()
