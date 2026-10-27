#!/usr/bin/env python3
"""Pin Qwen files, audit active routes and emit explicit fixed-pair configs."""
from pathlib import Path
from dataclasses import replace
import argparse
import importlib.util
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,SourceSpec,RouteSpec,LossSpec,stage_preset
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan,sample_status
    spec=importlib.util.spec_from_file_location('fixedpair_prep',ROOT/'scripts/data/prepare_qwen21_fixedpair_v5.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('tiktok','hi4d','both'),default='both');args=p.parse_args()
    model=(ROOT/'models/Qwen-Image-2.1').resolve();manifest=h.TROOT/'source/qwen_model_files.json'
    if not manifest.exists():
        files=[h.ref(f) for f in sorted(model.rglob('*')) if f.is_file() and f.suffix in ('.safetensors','.bin','.json','.model','.txt','.tiktoken')]
        h.write(manifest,files)
    init=h.ref(h.TROOT/'initialization/native_v5_init.pt')
    for kind,root,profile,scene in (('tiktok',h.TROOT,'full_smplx','single'),('hi4d',h.HROOT,'hybrid_hi4d','dual')):
        if args.kind not in ('both',kind):continue
        data=root/'data/native_v5_v1';selection=json.loads((root/'data/selection.json').read_text())
        sources=tuple(SourceSpec(role,role=role,manifest=h.ref(data/(role+'.jsonl')),groups=h.ref(data/(role+'_groups.json')),
            data_profile=profile,scene_kind=scene,weight=1.) for role in ('train','observation'))
        cfg=TrainingConfig(stage=stage_preset('S0',budget=60),sources=sources,initialization=init,
            model_path=str(model),model_manifest=h.ref(manifest),cache_root=str(root/'data/qwen_frozen_v2'),
            output_root=str(root/'training'),experiment_mode='fixed_pair_overfit',qwen_condition_profile=selection['profile'],
            save_every_updates=30,probe_every_updates=30,probe_samples=1,generation_every_updates=30,generation_steps=20,generation_seed=42)
        plan=DataPlan(cfg);sample=plan.sample('train',0);status=sample_status(sample,RouteSpec('joint','joint',1.),cfg)
        routes=[b for b in ('face','hand','detail','interaction') if status['branch_usable'][b]]
        if not routes:raise ValueError('No usable V5 local/interaction route in selected sample')
        out=root/'configs';out.mkdir(parents=True,exist_ok=True)
        h.write(out/'S0_prepare.json',cfg.record())
        h.write(root/'report/data_preflight_S0.json',plan.report())
        h.write(root/'report/active_routes.json',{'routes':routes,'status':status,'inactive':
            [b for b in ('face','hand','detail','interaction') if b not in routes],'branch_loss':'velocity_only'})
        for phase,budget in (('S1',60),('S2',120),('S3',60)):
            stage=stage_preset(phase,budget=budget)
            if phase in ('S1','S2'):
                stage=replace(stage,routes=tuple(RouteSpec(b,b,1.,loss=LossSpec()) for b in routes))
            h.write(out/(phase+'_template.json'),replace(cfg,stage=stage).record())
        print(json.dumps({'kind':kind,'active_routes':routes,'train_and_observation':len(plan.records),
            'target_grid':sample['target_token_hw'],'source_grid':sample['source_token_hw'],'data_passed':True}),flush=True)


if __name__=='__main__':main()
