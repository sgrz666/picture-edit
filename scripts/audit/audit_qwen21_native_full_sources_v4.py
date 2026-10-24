#!/usr/bin/env python3
"""Metadata-only checkpoint audit. Does not construct a model or migrate weights."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trusted-legacy-pickle',action='store_true');p.add_argument('--state-key',default='state_dict');args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from src.pose_control.qwen21_full.native_full_assets import sha256
    with FakeTensorMode(): saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=not args.trusted_legacy_pickle)
    state=saved.get(args.state_key)
    if not isinstance(state,dict): raise ValueError('Expected explicitly selected state container '+args.state_key+'; available='+str(list(saved)))
    tensors={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items() if torch.is_tensor(v)}
    report={'path':str(args.checkpoint.resolve()),'sha256':sha256(args.checkpoint),'architecture':saved.get('architecture'),'state_key':args.state_key,
        'tensor_keys':tensors,'non_tensor_keys':[k for k in state if k not in tensors],
        'recorded_step':saved.get('step',saved.get('metadata',{}).get('step')),
        'face_candidate_history':'3000-step candidate requires external log binding; no V4 quality claim',
        'hand_training_source':'unconfirmed unless separately evidenced','model_constructed':False,'migration_executed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as stream: json.dump(report,stream,indent=2)


if __name__=='__main__': main()
