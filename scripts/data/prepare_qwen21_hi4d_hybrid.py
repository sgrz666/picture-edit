#!/usr/bin/env python3
"""Explicit, offline Hi4D Hybrid tools. No automatic detection or extraction."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('import-observations','detect-missing','reference-jobs','build'):
        p=sub.add_parser(name)
        p.add_argument('--v3-manifest',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True,help='New directory under /data/shangguanrz/')
        p.add_argument('--config',type=Path,help='HybridConfig JSON; defaults are unaccepted engineering values')
        if name=='import-observations':
            p.add_argument('--generation-manifest',type=Path,required=True)
            p.add_argument('--layout',choices=('dwpose134','coco_wholebody133'),required=True)
        else: p.add_argument('--observations',type=Path,required=True)
        if name=='detect-missing':
            p.add_argument('--requests',type=Path,required=True,help='JSON list of [pair,action,camera_id,frame_id]')
            p.add_argument('--detector-config',type=Path,required=True)
        if name=='reference-jobs':
            p.add_argument('--appearance-config',type=Path,required=True,help='modalities and model SHA declarations')
            p.add_argument('--reference-side',type=int,default=512)
        if name=='build':
            p.add_argument('--references-mode',choices=('explicit_empty','manifest'),required=True)
            p.add_argument('--references-manifest',type=Path)
            p.add_argument('--max-pixels',type=int,default=512*512)
            p.add_argument('--reference-side',type=int,default=512)
    args=parser.parse_args()
    # Lazy project imports: --help never loads Torch, a detector, or a model.
    from src.pose_control.qwen21_full.native_hybrid_assets import HybridConfig
    from src.pose_control.qwen21_full.native_hybrid_cache import import_observations,detect_missing,reference_jobs,build_cache
    def read(path): return json.loads(path.read_text(encoding='utf8'))
    config=HybridConfig(**read(args.config)) if args.config else HybridConfig()
    if args.command=='import-observations':
        import_observations(args.v3_manifest,args.generation_manifest,args.output,layout=args.layout,config=config)
    elif args.command=='detect-missing':
        detect_missing(args.v3_manifest,args.observations,read(args.requests),read(args.detector_config),args.output,config=config)
    elif args.command=='reference-jobs':
        appearance=read(args.appearance_config)
        reference_jobs(args.v3_manifest,args.observations,args.output,modalities=appearance['modalities'],
            models=appearance.get('models',{}),config=config,reference_side=args.reference_side)
    else:
        build_cache(args.v3_manifest,args.observations,args.output,references_mode=args.references_mode,
            references_manifest=args.references_manifest,config=config,max_pixels=args.max_pixels,reference_side=args.reference_side)


if __name__=='__main__': main()
