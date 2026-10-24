#!/usr/bin/env python3
"""Explicit V3 + legacy encoders -> V4. No optimizer resume or implicit upgrade."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('v3','face','semantics','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hand',type=Path);p.add_argument('--allow-fresh-hand',action='store_true')
    p.add_argument('--face-prefix',default='face_preparer.');p.add_argument('--hand-prefix',default='hand_preparer.')
    p.add_argument('--face-state-key',default='state_dict');p.add_argument('--hand-state-key',default='state_dict')
    p.add_argument('--discard-prefix',action='append',default=[]);p.add_argument('--warm-start-attention',action='store_true')
    p.add_argument('--local-mode',choices=('dynamic','static'),default='dynamic');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trusted-legacy-pickle',action='store_true',help='Only for explicitly trusted old checkpoint containers')
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.migration.json').exists(): raise FileExistsError(args.output)
    if args.hand is None and not args.allow_fresh_hand: p.error('Unconfirmed Hand requires explicit --allow-fresh-hand')
    import torch
    from src.pose_control.qwen21_full.native_full_migration import migrate_v3_to_v4
    from src.pose_control.qwen21_full.native_local_control import NativeLocalConfigV4
    from src.pose_control.qwen21_full.native_full_assets import sha256
    torch.manual_seed(args.seed)
    def load(path,legacy=False): return torch.load(path,map_location='cpu',mmap=True,weights_only=not (legacy and args.trusted_legacy_pickle))
    def encoder(path,key):
        saved=load(path,True)
        if key not in saved: raise ValueError('Missing explicitly selected state container: '+key)
        return {**saved,'state_dict':saved[key]}
    result=migrate_v3_to_v4(load(args.v3),encoder(args.face,args.face_state_key),hand=None if args.hand is None else encoder(args.hand,args.hand_state_key),
        face_prefix=args.face_prefix,hand_prefix=args.hand_prefix,allow_fresh_hand=args.allow_fresh_hand,
        semantics=json.loads(args.semantics.read_text(encoding='utf8')),config=NativeLocalConfigV4(local_mode=args.local_mode),
        warm_start_attention=args.warm_start_attention,extra_discard_prefixes=tuple(args.discard_prefix))
    sources={k:{'path':str(v.resolve()),'sha256':sha256(v)} for k,v in (('v3',args.v3),('face',args.face),('hand',args.hand)) if v is not None}
    payload=result.checkpoint_payload(metadata={'sources':sources,'seed':args.seed,'acceptance':'not_run'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream: torch.save(payload,stream)
    with args.output.with_suffix('.migration.json').open('x',encoding='utf8') as stream: json.dump(result.inheritance_report,stream,indent=2)
    print(json.dumps({'architecture':payload['architecture'],'sources':sources,'output':str(args.output),'acceptance':'not_run'}))


if __name__=='__main__': main()
