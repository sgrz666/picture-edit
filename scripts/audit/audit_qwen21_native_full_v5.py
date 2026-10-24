#!/usr/bin/env python3
"""Header/shape-only checkpoint inventory; no model construction or migration."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--state-key',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from src.pose_control.qwen21_full.native_full_assets import sha256
    with FakeTensorMode(): saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state=saved[args.state_key]
    if not isinstance(state,dict): raise ValueError('Explicit state container required')
    tensors={k:{'shape':list(v.shape),'dtype':str(v.dtype),'elements':v.numel()} for k,v in state.items() if torch.is_tensor(v)}
    report={'source_path':str(args.checkpoint.resolve()),'sha256':sha256(args.checkpoint),'architecture':saved.get('architecture'),
        'recorded_step':saved.get('step',saved.get('metadata',{}).get('step')),'state_key':args.state_key,'tensors':tensors,
        'non_tensor_keys':sorted(set(state)-set(tensors)),'training_evidence':saved.get('metadata',{}).get('module_training_evidence',{}),
        'claims':{'structure':'header_and_shapes_only','trained_weights':'requires_bound_logs_and_actual_weight_audit',
            'runtime':'not_run','control':'not_run','quality':'not_run'},'model_constructed':False,'migration_executed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(report,f,indent=2,default=str)


if __name__=='__main__': main()
