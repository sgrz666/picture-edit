#!/usr/bin/env python3
"""Explicit strict V4 -> V5 migration. No optimizer, inference, or source overwrite."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v4',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--warm-start-job',type=Path)
    p.add_argument('--allow-fresh-hand',action='store_true');p.add_argument('--allow-fresh-detail',action='store_true')
    p.add_argument('--no-native-local-warm-start',action='store_true');args=p.parse_args()
    if args.output.exists() or args.report.exists(): raise FileExistsError('Use new V5 checkpoint and report paths')
    if not str(args.output.resolve()).startswith('/data/shangguanrz/'): raise ValueError('New weights belong under /data/shangguanrz/')
    import torch
    from src.pose_control.qwen21_full.native_v5.migration import migrate_v4_to_v5
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_full_assets import sha256
    saved=torch.load(args.v4,map_location='cpu',weights_only=True,mmap=True)
    config=NativeConfigV5.from_dict(json.loads(args.config.read_text())) if args.config else None
    jobs=[]
    if args.warm_start_job:
        for raw in json.loads(args.warm_start_job.read_text()):
            ref=raw['source']
            if sha256(ref['path'])!=ref['sha256']: raise ValueError('Warm-start source SHA differs')
            jobs.append({'checkpoint':torch.load(ref['path'],map_location='cpu',weights_only=True,mmap=True),
                'state_key':raw['state_key'],'mapping':raw['mapping'],'source_sha256':ref['sha256']})
    runtime=migrate_v4_to_v5(saved,config=config,allow_fresh_hand=args.allow_fresh_hand,allow_fresh_detail=args.allow_fresh_detail,
        warm_start_native_local=not args.no_native_local_warm_start,warm_start=jobs)
    report={**runtime.inheritance_report,'source_path':str(args.v4.resolve()),'source_sha256':sha256(args.v4)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f: torch.save(runtime.checkpoint_payload(metadata={'inheritance':report}),f)
    report['destination_sha256']=sha256(args.output)
    with args.report.open('x') as f: json.dump(report,f,indent=2)


if __name__=='__main__': main()
