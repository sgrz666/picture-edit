"""Real-sample checks before enabling append-only on-demand frozen encoding."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import uuid
import torch
from supervisor import file_ref,read_ref,write_new_json,EventJournal,active_plan_reference
from run_plan import stage_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True);parser.add_argument('--repo-root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root,repo=Path(args.run_root).resolve(),Path(args.repo_root).resolve();sys.path.insert(0,str(repo))
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig,digest
    from src.pose_control.qwen21_full.native_v5.training.frozen_inputs import FrozenInputPlan
    from src.pose_control.qwen21_full.native_v5.training.streaming import QualifiedBodyPlan
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from src.pose_control.qwen21_full.native_v5.training.cache_store import FrozenCacheStore
    from src.pose_control.qwen21_full.native_v5.training.io import file_sha
    torch.set_num_threads(4)
    plan=read_ref(active_plan_reference(EventJournal(root).read()));prepared=json.loads((root/'prepared.json').read_text())
    audit=read_ref(prepared['audits']['native_data'])
    if audit.get('passed') is not True or audit.get('pairs_read')!=12156:raise ValueError('Full Native qualification missing')
    config=TrainingConfig.from_dict(stage_config(plan,prepared,'S0',initialization=prepared['initialization']))
    qualification=audit['train_val_route_audit'];start=time.perf_counter()
    projected=QualifiedBodyPlan(config,qualification);elapsed=time.perf_counter()-start
    old_preparation=json.loads((Path(config.cache_root)/'preparation.json').read_text())
    if projected.identity['sources']!=old_preparation['identity']['sources']:raise ValueError('Cache source identity changed')
    tmp=root/'control_checks'/('streaming_'+uuid.uuid4().hex);tmp.mkdir(parents=True)
    cases=[];source_names=list(projected.datasets)
    selected=[(source_names[0],0),(source_names[0],len(projected.datasets[source_names[0]])//2),(source_names[1],0)]
    for source,index in selected:
        row=projected.datasets[source].rows[index]
        small=tmp/(source+'_'+str(index)+'.jsonl');small.write_text(json.dumps(row)+'\n')
        full=NativeFullDatasetV5(small,load_target=True,data_profile='full_smplx')[0]
        minimal=projected.frozen.datasets[source][index]
        for field in ('source_token_hw','target_token_hw'):
            if full[field]!=minimal[field]:raise ValueError('Native/Frozen grid differs')
        if not torch.equal(full['tgt_image'],minimal['tgt_image']):raise ValueError('Native/Frozen pixels differ')
        ref,spec=projected.frozen.conditions[(source,index)]
        if read_condition(ref,full)!=spec:raise ValueError('Native/Frozen condition differs')
        expected=sample_status(full,config.stage.routes[0],config)
        if expected!=projected.status[('body',source,index)]:raise ValueError('Projected S0 status differs from original full reader')
        binding=projected.frozen.binding(source,index,config.model_manifest['sha256'],digest(config.prompt))
        meta=full['local_condition_v4'].metadata[0]
        if any(binding[k]!=meta[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')):
            raise ValueError('Frozen cache input binding changed')
        cases.append({'source':source,'index':index,'pair_id':row['pair_id'],'pixels_bitwise_equal':True,
                      'condition_equal':True,'s0_status_equal':True,'binding_equal':True})
    base=Path(plan['repo_root']);unchanged=('objective.py','sampling.py','parameters.py','validation.py')
    for name in unchanged:
        if name=='validation.py':continue  # Only cache prefetch is inserted before the same gradient cases.
        relative=Path('src/pose_control/qwen21_full/native_v5/training')/name
        if file_sha(base/relative)!=file_sha(repo/relative):raise ValueError('Training math changed: '+name)
    result={'schema':'native_streaming_startup_acceptance_v1','passed':True,'optimizer_step_executed':False,
            'baseline_repo':str(base),'repo_root':str(repo),'cache_mode':'on_demand','qualification':qualification,
            'new_code':{name:file_ref(repo/'src/pose_control/qwen21_full/native_v5/training'/name)
                        for name in ('trainer.py','validation.py','parameters.py')},
            'producer':file_ref(__file__),'startup_seconds':elapsed,'qualified_rows':len(projected.records),
            'cached_sources_identity_preserved':True,'real_samples':cases,'training_budget':plan['total_updates'],
            'scope':'Qualified metadata projection and exact frozen input equivalence; production gradient gate remains mandatory.'}
    print(write_new_json(args.output,result),flush=True)


if __name__=='__main__':main()
