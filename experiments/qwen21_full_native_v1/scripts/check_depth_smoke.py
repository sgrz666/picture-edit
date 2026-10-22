"""Verify the saved three-step Native checkpoint and its actual learning signals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0,str(args.project))
    import torch
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    torch.set_num_threads(4)
    path = args.run/"checkpoint_last.pt"
    saved = torch.load(path,map_location="cpu",mmap=True,weights_only=True)
    model = NativeDepthRuntime.from_native_checkpoint(saved).eval()
    rows = [json.loads(line) for line in (args.run/"metrics.jsonl").read_text().splitlines()]
    assert len(rows)==3 and all(r["grad_norm"]>0 for r in rows)
    assert rows[0]["depth_head_grad"]==0 and rows[1]["depth_head_grad"]>0
    assert rows[2]["depth_conv_grad"]>0
    assert model.depth.projection.weight.abs().max()>0
    assert saved["metadata"]["step"]==3
    result = {"checkpoint":str(path),"bytes":path.stat().st_size,"weights_only_load":True,
              "rebuild_without_full_checkpoint":True,"architecture":saved["architecture"],
              "steps":3,"distinct_training_pairs":len({r["pair_id"] for r in rows}),
              "target_token_hw":rows[0]["target_token_hw"],"metrics":rows,
              "depth_projection_max_abs":float(model.depth.projection.weight.detach().abs().max()),
              "full_checkpoint_sha256":saved["metadata"]["full_checkpoint_sha256"],
              "quality_claim":False}
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
