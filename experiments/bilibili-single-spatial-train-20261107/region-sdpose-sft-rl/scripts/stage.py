"""Explicit cache, data/gradient, SFT and generation actions on the shared Trainer."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import uuid

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))
from src.pose_control.qwen21_full.native_v62.training.io import write_json,file_sha


def read(path):return json.loads(Path(path).read_text())
def reference(path):return {'path':str(Path(path).resolve()),'sha256':file_sha(path)}


def configure_gpu(fraction=.9):
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(fraction)
    free,total=torch.cuda.mem_get_info()
    print(json.dumps({'free_gib':free/2**30,'total_gib':total/2**30,'memory_fraction':fraction}),flush=True)


def generate(config,root,label,output):
    import torch
    from src.pose_control.qwen21_full.native_v7.training import DataPlanV7,NativeBackendV7
    from src.pose_control.qwen21_full.native_v62.training.config import RouteSpec
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.generation import generate_native
    data=DataPlanV7(config);backend=NativeBackendV7(config,data,purpose='train');backend.runtime.eval()
    evaluation={x['pair_id']:x for x in read(root/'data/evaluation.json')}
    validation=[s.source_id for s in config.sources if s.role=='validation']
    if len(validation)!=1:raise ValueError('One explicit held-out validation source required')
    validation_source=validation[0]
    chosen=list(range(len(data.datasets[validation_source])))
    records=[];route=RouteSpec('fixed_generation','joint',1.,controls={'interaction':False},
        base_controls={'interaction':False,'raster':False,'part_contact':False})
    with torch.no_grad():
        for index in chosen:
            sample,batch,pair=backend.pair(validation_source,index)
            options,base=route_options(route,sample['native_scene_kind'])
            path=output/(sample['pair_metadata']['pair_id']+'_seed42.png')
            path.parent.mkdir(parents=True,exist_ok=True)
            marker=path.with_suffix('.json')
            if path.exists() and marker.exists():
                existing=read(marker)
                if (existing['sha256']!=file_sha(path) or existing['config_sha256']!=config.signature() or
                        existing['model_sha256']!=config.initialization['sha256']):
                    raise ValueError('Saved generation provenance differs')
                records.append(existing);continue
            if path.exists() or marker.exists():
                path=path.with_name(path.stem+'_attempt_'+uuid.uuid4().hex[:12]+'.png');marker=path.with_suffix('.json')
            image=generate_native(backend.pipe,backend.runtime,batch,pair,options=options,base_options=base,
                seed=42,steps=20,enabled=True,poselink_strength=route.poselink_strength)
            with path.open('xb') as f:image.save(f,format='PNG')
            record={'pair_id':sample['pair_metadata']['pair_id'],'path':str(path),'sha256':file_sha(path),
                    'seed':42,'steps':20,'label':label,'target_rgb_used_as_condition':False,
                    'config_sha256':config.signature(),'model_sha256':config.initialization['sha256']}
            write_json(marker,record);records.append(record)
            print(json.dumps({'generated':str(path),'label':label}),flush=True)
    return write_json(output/'manifest.json',{'schema':'bilibili_region_generated_images_v1','label':label,'records':records})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['data','cache','grad','train','generate'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--report',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--parent',type=Path);ap.add_argument('--resume',type=Path);ap.add_argument('--label',default='baseline')
    args=ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,NativeV7Trainer
    config=TrainingConfigV7.from_dict(read(args.config));trainer=NativeV7Trainer(config)
    if args.action=='data':
        report=trainer.preflight_data(args.report)
        print(json.dumps({'passed':report['passed'],'rows':len(report['rows']),
            'region_hand_rows':sum(x.get('region_eligible',{}).get('hands',False) for x in report['rows'])}),flush=True)
        return
    configure_gpu()
    if args.action=='cache':
        result=trainer.cache_prepare(args.root/'cache')
        cache_ref=result if isinstance(result,dict) and 'path' in result else reference(args.root/'cache/index.json')
        for arm in ('uniform','regional'):
            for template in sorted((args.root/f'configs/templates/{arm}').glob('*.json')):
                raw=read(template);raw['cache_identity']=cache_ref
                write_json(args.root/f'configs/cache-ready/{arm}/{template.name}',TrainingConfigV7.from_dict(raw).record())
        write_json(args.root/'cache_ready.json',{'cache':cache_ref,'target_pose_backend':'sdpose'})
        print(json.dumps({'cache_ready':cache_ref}),flush=True)
    elif args.action=='grad':
        report=trainer.preflight_grad(args.report)
        if report.get('passed') is not True or report.get('production_passed') is not True:
            raise RuntimeError('Real gradient preflight did not pass')
        print(json.dumps({'gradient_passed':True,'report':str(args.report)}),flush=True)
    elif args.action=='train':
        result=trainer.run(mode='resume' if args.resume else 'advance-stage' if args.parent else 'train',
            checkpoint=reference(args.resume) if args.resume else reference(args.parent) if args.parent else None)
        write_json(args.report,result);print(json.dumps(result),flush=True)
    else:
        result=generate(config,args.root,args.label,args.output)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
