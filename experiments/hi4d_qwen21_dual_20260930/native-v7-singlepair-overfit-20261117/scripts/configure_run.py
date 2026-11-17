"""Create explicit, SHA-bound fixed-pair V7 configurations; no model loading."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def main(argv=None):
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,stage_preset_v7,DataPlanV7
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec,RouteSpec,LossSpec
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha,write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache-index',type=Path)
    parser.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    def ref(path):return {'path':str(path.resolve()),'sha256':file_sha(path)}
    prepared=json.loads((args.data/'prepared.json').read_text())['manifests']
    sources=tuple(SourceSpec(role,role=role,manifest=prepared[role],groups=prepared[role+'_groups'],
                            weight=1.,data_profile='hybrid_hi4d',scene_kind='dual') for role in ('train','observation'))
    old=Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
    authorization={'source':'direct_user_message','text':'补齐一下训练试试,你觉得我们需要单样本过拟合实验吗🤔'}
    cfg=TrainingConfigV7(stage=stage_preset_v7('S0',budget=30),sources=sources,
        initialization=ref(args.root/'initialization/v7.pt'),model_path='/data/shangguanrz/pic-edit-models/Qwen-Image-2.1',
        model_manifest=ref(old/'source/qwen_model_files.json'),cache_root=str(args.root/'cache/native-4ref'),
        cache_identity=None if args.cache_index is None else ref(args.cache_index),
        evaluators={'identity':ref(PROJECT/'models/face_eval/identity_encoder.ts')},
        output_root=str(args.root/'output/training'),experiment_mode='fixed_pair_overfit',
        qwen_condition_profile='dual_4ref',save_every_updates=30,probe_every_updates=30,probe_samples=16,
        generation_every_updates=0,generation_steps=20,generation_seed=42,diagnostics=True,
        continuous_stage_authorization=authorization)
    out=args.root/'configs'/args.name
    out.mkdir(parents=True,exist_ok=False)
    statuses={}
    for phase,budget in (('S0',30),('S1',30),('S2',120),('S3',120)):
        stage=stage_preset_v7(phase,None if phase!='S2' else 'poselink',budget=budget)
        if phase=='S1':
            stage=replace(stage,routes=tuple(RouteSpec(branch,branch,1.,loss=LossSpec(identity=.1))
                                           for branch in ('face','hand','interaction','poselink')))
        current=replace(cfg,stage=stage)
        plan=DataPlanV7(current)
        statuses[phase]={'active_routes':[route.branch for route in current.stage.routes],
                         'paper_eligibility':plan.paper_eligibility,'budget':budget}
        write_json(out/(phase+'.json'),current.record())
        write_json(out/(phase+'-data.json'),plan.report())
    write_json(out/'selection.json',{'authorization':authorization,'stage_updates':[30,30,120,120],
        'total_updates':300,'source_prepared':ref(args.data/'prepared.json'),'stages':statuses,
        'gpu_memory_fraction':.5,'canvas_policy':'original_native_pad','training_started':False})
    print(json.dumps({'configs':str(out),'stages':statuses,'missing':cfg.missing()}),flush=True)


if __name__=='__main__':main()
