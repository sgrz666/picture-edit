#!/usr/bin/env python3
"""User-authorized continuous 300-update pilot, with real gates at every stage."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import argparse
import gc
import json
import sys
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan
    from src.pose_control.qwen21_full.native_v5.training.backend import NativeBackend
    from src.pose_control.qwen21_full.native_v5.training.validation import gradient_preflight
    from src.pose_control.qwen21_full.native_v5.training.observations import generation_probe
    from src.pose_control.qwen21_full.native_v5.training.io import write_json,file_sha
    from src.pose_control.qwen21_full.native_v5.training.checkpointing import load_committed
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('fixedpair_prep',ROOT/'scripts/data/prepare_qwen21_fixedpair_v5.py')
    h=module_from_spec(spec);spec.loader.exec_module(h)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('accept-only','accept-and-train'))
    p.add_argument('--kind',choices=('tiktok','hi4d','both'),default='both')
    p.add_argument('--reuse-tiktok-baseline',type=Path,help='Explicit immutable baseline report from the same locked initialization/conditions')
    p.add_argument('--continue-tiktok-from',type=Path,help='Committed completed-stage checkpoint; preserve AdamW state and finish the remaining stages')
    args=p.parse_args();torch.set_num_threads(4)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    initialization=h.ref(h.TROOT/'initialization/native_v5_init_v2.pt')
    for kind,root in (('tiktok',h.TROOT),('hi4d',h.HROOT)):
        if args.kind not in ('both',kind):continue
        log=root/'logs/orchestration.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        def event(**values):
            row={'time':datetime.now(timezone.utc).isoformat(),'run_id':run_id,'kind':kind,**values}
            with log.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps(row,ensure_ascii=False),flush=True)
        selected=json.loads((root/'data/selection.json').read_text());authorization=selected['continuous_stage_authorization']
        previous=None;s0_gradient=None;body_gate=None
        event(status='starting',action=args.action,initialization=initialization,updates=300)
        try:
            phases=('S0','S1','S2','S3')
            if kind=='tiktok' and args.continue_tiktok_from is not None:
                saved,parent,_=load_committed(h.ref(args.continue_tiktok_from))
                state=saved['training_state'];old=TrainingConfig.from_dict(state['config'])
                expected={60:'S0',120:'S1',240:'S2'}
                completed=state['global_update']
                if expected.get(completed)!=old.stage.phase or state['stage_update']!=old.stage.budget_updates:
                    raise ValueError('Continue from a fully committed completed-stage boundary')
                current=TrainingConfig.from_dict(json.loads((root/'configs/S0_prepare.json').read_text()))
                if old.sources!=current.sources or old.model_manifest!=current.model_manifest or old.seed!=current.seed:
                    raise ValueError('Continuation data/model/seed differs from the locked experiment')
                if old.cache_identity!=h.ref(root/'data/qwen_frozen_v2/index.json'):
                    raise ValueError('Continuation frozen condition cache differs')
                ancestor=saved
                while ancestor['training_state']['parent'] is not None:
                    ancestor,_,_=load_committed(ancestor['training_state']['parent'])
                if ancestor['training_state']['stage_origin']!=initialization:
                    raise ValueError('Continuation initialization lineage differs')
                previous={'checkpoint':parent,'commit':h.ref(Path(parent['path']).with_suffix('.commit.json'))}
                body_gate=old.body_baseline_report
                phases=phases[phases.index(old.stage.phase)+1:]
                event(status='continuing',global_update=completed,remaining_updates=300-completed,
                    parent_checkpoint=parent,remaining_stages=phases,optimizer_state_retained=True)
                del saved,state,ancestor;gc.collect()
            for phase in phases:
                template=root/'configs'/('S0_prepare.json' if phase=='S0' else phase+'_template.json')
                cfg=TrainingConfig.from_dict(json.loads(template.read_text()))
                cfg=replace(cfg,initialization=initialization if previous is None else previous['checkpoint'],
                    cache_identity=h.ref(root/'data/qwen_frozen_v2/index.json'),
                    continuous_stage_authorization=authorization,body_baseline_report=body_gate)
                if phase=='S3' and selected['profile']=='single_2ref':
                    inactive={'interaction_heads','interaction_controller','relation_raster','relation_tokens','strength_interaction'}
                    cfg=replace(cfg,stage=replace(cfg.stage,exclude_groups=tuple(sorted(set(cfg.stage.exclude_groups)|inactive))))
                cfg.require_ready()
                report_dir=root/'report'/run_id;report_dir.mkdir(parents=True,exist_ok=True)
                config_file=root/'configs'/(phase+'_'+run_id+'.json')
                write_json(config_file,cfg.record())
                event(status='gradient_preflight_start',stage=phase,config=str(config_file))
                data=DataPlan(cfg);write_json(report_dir/(phase+'_data.json'),data.report())
                backend=NativeBackend(cfg,data,purpose='grad')
                torch.cuda.reset_peak_memory_stats()
                report=gradient_preflight(backend,cfg,data)
                report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
                grad_ref=write_json(report_dir/(phase+'_grad.json'),report)
                if not report['production_passed']:raise RuntimeError('Real gradient gate failed: '+str(grad_ref))
                event(status='gradient_preflight_passed',stage=phase,peak_cuda_bytes=report['peak_cuda_allocated_bytes'])
                if phase=='S0':
                    s0_gradient=grad_ref
                    if kind=='tiktok' and args.reuse_tiktok_baseline is not None:
                        baseline=json.loads(args.reuse_tiktok_baseline.read_text())
                        old_run=args.reuse_tiktok_baseline.parent.name
                        old=TrainingConfig.from_dict(json.loads((root/'configs'/('S0_'+old_run+'.json')).read_text()))
                        fields=('initialization','model_manifest','cache_identity','sources','qwen_condition_profile','generation_seed','generation_steps')
                        if any(getattr(old,k)!=getattr(cfg,k) for k in fields):raise ValueError('Explicit reused baseline belongs to another experiment')
                        if len(baseline['records'])!=4 or any(file_sha(r['path'])!=r['sha256'] for r in baseline['records']):
                            raise ValueError('Reused baseline image digest/count differs')
                        baseline={**baseline,'explicit_reuse_from':h.ref(args.reuse_tiktok_baseline),
                            'reuse_reason':'same weights, conditioning, seed and sampler; only training readiness gate changed'}
                    else:baseline=generation_probe(backend,cfg,data,0,root/'output'/run_id/'images',baseline=True)
                    write_json(report_dir/'baseline_generation.json',baseline)
                    event(status='baseline_generated',images=len(baseline['records']))
                if args.action=='accept-only':
                    del backend,data;gc.collect();torch.cuda.empty_cache();break
                cfg=replace(cfg,gradient_report=grad_ref)
                del backend,data;gc.collect();torch.cuda.empty_cache()
                event(status='training_start',stage=phase,stage_budget=cfg.stage.budget_updates)
                result=NativeV5Trainer(cfg).run(mode='train' if previous is None else 'advance-stage',
                    checkpoint=None if previous is None else previous['checkpoint'])
                saved,parent,_=load_committed(result['checkpoint'])
                state=saved['training_state'];completed=state['global_update']
                event(status='stage_completed',stage=phase,global_update=completed,checkpoint=result['checkpoint'])
                if state['stage_update']!=cfg.stage.budget_updates:raise RuntimeError('Incomplete stage budget')
                if phase=='S0':
                    body_gate=write_json(report_dir/'body_technical_gate.json',{'schema':'native_v5_body_pilot_gate_v1',
                        'body_accepted':True,'manual_acknowledgment':False,'acceptance_kind':'technical_pilot_gate',
                        'continuous_stage_authorization':authorization,'s0_gradient_report':s0_gradient,
                        'checkpoint_sha256':parent['sha256'],'completed_updates':completed,
                        'generation_quality_claimed':False,'automatic_optimizer_stage_change':False})
                previous=result
                del saved,state;gc.collect();torch.cuda.empty_cache()
            else:
                if completed!=300:raise RuntimeError('Pilot must finish exactly 300 updates')
                # Additional seeds and single-branch inference controls are
                # observations only, and never advance the optimizer counter.
                final_cfg=replace(cfg,initialization=previous['checkpoint'])
                data=DataPlan(final_cfg);saved,_,_=load_committed(previous['checkpoint'])
                backend=NativeBackend(final_cfg,data,saved=saved,purpose='grad')
                extra=generation_probe(backend,final_cfg,data,300,root/'output'/run_id/'final_extra_seeds',seeds=(2026,3407))
                ablations=generation_probe(backend,final_cfg,data,300,root/'output'/run_id/'final_controls',ablations=True)
                write_json(report_dir/'final_extra_seeds.json',extra);write_json(report_dir/'final_controls.json',ablations)
                write_json(report_dir/'completed.json',{'completed_updates':300,'checkpoint':previous,
                    'initialization':initialization,'scope':'one_training_pair+one_same_scene_observation',
                    'same_scene_generalization_only':True,'automatic_stage_authorization':authorization})
                event(status='completed',global_update=300,checkpoint=previous['checkpoint'])
                del backend,data,saved;gc.collect();torch.cuda.empty_cache()
        except Exception as error:
            event(status='failed',error=str(error),traceback=traceback.format_exc(),last_committed_checkpoint=previous)
            raise


if __name__=='__main__':main()
