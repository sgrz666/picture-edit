#!/usr/bin/env python3
"""Bound real Qwen checkpoint recomputation diagnostics, never an optimizer."""
from pathlib import Path
from functools import partial
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from torch.utils.checkpoint import checkpoint
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan
    from src.pose_control.qwen21_full.native_v5.training.backend import NativeBackend
    from src.pose_control.qwen21_full.native_v5.training.validation import gradient_preflight
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    cfg=TrainingConfig.from_dict(json.loads(args.config.read_text()));data=DataPlan(cfg);backend=NativeBackend(cfg,data,purpose='grad')
    backend.pipe.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint,use_reentrant=False,debug=True))
    report=gradient_preflight(backend,cfg,data)
    print(json.dumps({'passed':report['production_passed'],'optimizer_created':False}))


if __name__=='__main__':main()
