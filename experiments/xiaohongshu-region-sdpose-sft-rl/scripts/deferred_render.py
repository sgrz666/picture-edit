"""Render paired seeds after optimization; reuse one loaded Qwen per checkpoint."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from prepare import read,ref,write

FINAL_LABELS=('initial','uniform_S3','regional_S3','continue_sft','grpo')
INTERMEDIATE_LABELS=tuple(arm+'_'+phase for arm in ('uniform','regional') for phase in ('S0','S1','S2'))
PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
SDPY='/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python'


def image_tasks(prepared):
    tasks=[]
    for label in (*FINAL_LABELS,*INTERMEDIATE_LABELS):
        for seed in ((42,123) if label in FINAL_LABELS else (42,)):
            tasks.append({'label':label,'split':'val','seed':seed,'pairs':prepared['val_pairs']})
        if label in FINAL_LABELS:
            for seed in (42,123,2026):
                tasks.append({'label':label,'split':'test','seed':seed,'pairs':prepared['test_pairs']})
    return tasks


def output_folder(root,label,split,seed):
    if seed==42:return root/f'output/generation/{label}' if split=='val' else root/f'output/generation/test/{label}'
    return root/f'output/generation/multi-seed/seed_{seed}/{split}/{label}'


def metrics_path(root,label,split,seed):
    if seed==42:return root/f'report/evaluation/{label if split=="val" else "test_"+label}/metrics.json'
    return root/f'report/evaluation/multi-seed/seed_{seed}/{split}/{label}/metrics.json'


def config_path(root,label):
    return root/'configs/cache-ready/uniform/S0.json' if label=='initial' else root/f'configs/generation/{label}.json'


def verify_records(records,cfg,split,seed,expected):
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    if len(records)!=len(expected) or {r['pair_id'] for r in records}!=expected:raise ValueError('Render pair set differs')
    for row in records:
        if (row['seed']!=seed or row['split']!=split or row['model_sha256']!=cfg.initialization['sha256']
                or row['config_sha256']!=cfg.signature() or file_sha(row['path'])!=row['sha256']):
            raise ValueError('Saved render identity differs')


def render(root,repo,label):
    import torch
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,DataPlanV7,NativeBackendV7
    from src.pose_control.qwen21_full.native_v62.training.config import RouteSpec
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.generation import generate_native
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.75)
    cfg=TrainingConfigV7.from_dict(read(config_path(root,label)))
    prepared=read(root/'prepared.json');tasks=[t for t in image_tasks(prepared) if t['label']==label]
    pairs=read(root/'data/evaluation.json');pending=[]
    for task in tasks:
        manifest=output_folder(root,label,task['split'],task['seed'])/'manifest.json'
        expected={r['pair_id'] for r in pairs if r['split']==task['split']}
        if manifest.exists():verify_records(read(manifest)['records'],cfg,task['split'],task['seed'],expected)
        else:pending.append(task)
    if not pending:return
    data=DataPlanV7(cfg);backend=NativeBackendV7(cfg,data,purpose='train');backend.runtime.eval()
    route=RouteSpec('fixed_generation','joint',1.,controls={'interaction':False},base_controls={'interaction':False,'raster':False,'part_contact':False})
    records=defaultdict(list)
    with torch.no_grad():
        for split in ('val','test'):
            selected=[t for t in pending if t['split']==split]
            if not selected:continue
            role='validation' if split=='val' else 'test'
            source=next(s.source_id for s in cfg.sources if s.role==role)
            for index in range(len(data.datasets[source])):
                sample,batch,pair=backend.pair(source,index);pid=sample['pair_metadata']['pair_id']
                options,base=route_options(route,sample['native_scene_kind'])
                for task in selected:
                    seed=task['seed'];folder=output_folder(root,label,split,seed);folder.mkdir(parents=True,exist_ok=True)
                    path=folder/f'{pid}_seed{seed}.png';marker=path.with_suffix('.json')
                    if path.exists() and marker.exists():
                        row=read(marker);verify_records([row],cfg,split,seed,{pid})
                    else:
                        if path.exists() or marker.exists():path=folder/(path.stem+'_attempt_'+uuid.uuid4().hex[:10]+'.png');marker=path.with_suffix('.json')
                        image=generate_native(backend.pipe,backend.runtime,batch,pair,options=options,base_options=base,
                            seed=seed,steps=20,enabled=True,poselink_strength=route.poselink_strength)
                        with path.open('xb') as f:image.save(f,format='PNG')
                        row={'pair_id':pid,'path':str(path),'sha256':ref(path)['sha256'],'seed':seed,'steps':20,'label':label,
                            'split':split,'source_id':source,'target_rgb_used_as_condition':False,
                            'config_sha256':cfg.signature(),'model_sha256':cfg.initialization['sha256']}
                        write(marker,row)
                    records[(split,seed)].append(row)
                    print(json.dumps({'generated':label,'split':split,'seed':seed,'pair':index+1}),flush=True)
    for task in pending:
        split,seed=task['split'],task['seed'];write(output_folder(root,label,split,seed)/'manifest.json',
            {'schema':'region_multiseed_images_v1','label':label,'split':split,'seed':seed,'records':records[(split,seed)]})


def run(root,repo,common):
    from src.pose_control.qwen21_full.native_v62.training.config import digest
    evaluation=__import__('importlib.util',fromlist=['util'])
    spec=evaluation.spec_from_file_location('frozen_image_evaluation',common/'evaluate.py')
    evaluator=evaluation.module_from_spec(spec);spec.loader.exec_module(evaluator)
    prepared=read(root/'prepared.json');tasks=image_tasks(prepared)
    plan=root/'report/deferred-image-plan.json'
    if plan.exists():
        if read(plan)['tasks']!=tasks:raise ValueError('Frozen image plan differs')
    else:write(plan,{'tasks':tasks,'expected_images':sum(t['pairs'] for t in tasks),'training_core':str(repo),
        'training_precision_changed':False,'optimization_budget_changed':False,'test_used_for_optimization':False})
    for label in (*FINAL_LABELS,*INTERMEDIATE_LABELS):
        subprocess.run([PY,str(Path(__file__)),'render','--root',str(root),'--repo',str(repo),'--label',label],check=True)
    pending=[];all_reports=[];manifests={}
    for task in tasks:
        key=(task['label'],task['split'],task['seed']);manifest=output_folder(root,*key)/'manifest.json'
        manifests[key]=ref(manifest);destination=metrics_path(root,*key)
        if destination.exists():all_reports.append({'task':task,'metrics':ref(destination)});continue
        pending.extend(read(manifest)['records'])
    if pending:
        identifier=digest([ref(r['path']) for r in pending])[:16];folder=root/f'report/multi-seed-evaluation/{identifier}'
        manifest=folder/'generated.json';pose=folder/'sdpose.json';full=folder/'metrics.json'
        if not manifest.exists():write(manifest,{'label':'paired_multi_seed','records':pending})
        if not pose.exists():subprocess.run([SDPY,str(common/'evaluate.py'),'pose','--root',str(root),'--manifest',str(manifest),'--output',str(pose)],check=True)
        if not full.exists():subprocess.run([PY,str(common/'evaluate.py'),'identity','--root',str(root),'--input',str(pose),'--output',str(full)],check=True)
        measured=read(full);groups=defaultdict(list)
        for row in measured['records']:groups[(row['label'],row['split'],row['seed'])].append(row)
        for key,records in groups.items():
            destination=metrics_path(root,*key)
            if not destination.exists():write(destination,{**{k:v for k,v in measured.items() if k not in ('records','summary')},
                'label':key[0],'split':key[1],'seed':key[2],'records':records,'summary':evaluator.aggregate(records)})
            all_reports.append({'task':{'label':key[0],'split':key[1],'seed':key[2]},'metrics':ref(destination)})
    result=read(root/'optimization-complete.json');result['test_manifests']={label:manifests[(label,'test',42)] for label in FINAL_LABELS}
    result['multi_seed_generation']={'images':sum(t['pairs'] for t in tasks),'reports':all_reports,'plan':ref(plan)}
    if not (root/'complete.json').exists():write(root/'complete.json',result)
    write(root/'report/deferred-render-complete.json',{'images':sum(t['pairs'] for t in tasks),'reports':all_reports,
        'val_images':sum(t['pairs'] for t in tasks if t['split']=='val'),'test_images':sum(t['pairs'] for t in tasks if t['split']=='test')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','render'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--common-dir',type=Path);p.add_argument('--label');a=p.parse_args();sys.path.insert(0,str(a.repo))
    os.environ['PICTUREEDIT_REPO']=str(a.repo)
    if a.action=='render':render(a.root,a.repo,a.label)
    else:run(a.root,a.repo,a.common_dir)


if __name__=='__main__':main()
