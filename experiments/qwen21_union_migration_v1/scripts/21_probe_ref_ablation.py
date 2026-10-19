#!/usr/bin/env python
"""诊断探针（不属于 48 张矩阵）：scale 1.0 把骨架线画进画面，是不是「骨架图同时当参考图」造成的。

矩阵里发现的偏差：hard_02 在 control_context_scale=1.0 时输出里出现白色火柴人线条（scale 0.5 干净）。
当前工作流的输入把同一张骨架图既作为 reference_images 之一喂给 Qwen3-VL + VAE，又作为 Union 控制图。
本探针只改这一个变量——参考图列表里去掉骨架图，只留源人物照片——其余（提示词、控制图、seed、步数、
画布）与矩阵完全一致，跑两张：

  A) refs=[source] + 控制 scale 1.0   → 看线条是否消失、姿态是否仍跟骨架
  B) refs=[source] + 控制 scale 0.0   → 没有姿态引导时的样子，用来判断 A 的姿态改善是不是控制带来的

产物写在 $WORK/runs/probe_skeleton_ref_ablation_<时间戳>/，不覆盖矩阵运行。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from PIL import Image

from union_env import WORK, bootstrap, build_pipeline, verify_videox_checkout  # noqa: F401

EXP = pathlib.Path(__file__).resolve().parent.parent
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("run_matrix_mod", pathlib.Path(__file__).with_name("07_run_matrix.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)  # 复用矩阵的 sample_inputs / to_tensor，保证输入完全一致


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="hard_02")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    bootstrap()      # to_tensor 里要 import videox_fun，必须先挂好固定版本源码
    manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))
    entry = next(s for s in manifest["samples"] if s["sample"] == args.sample)
    refs, ctrl, prompt, h, w = rm.sample_inputs(entry)
    if entry["kind"] == "single":
        kept = [refs[0]]           # 只留源人物照片，去掉骨架参考图
    else:
        kept = [refs[0], refs[1]]  # 双人：只留 A/B 两张照片，去掉 per-person 骨架参考图
    ctrl_t = rm.to_tensor(ctrl, h, w)

    name = f"probe_skeleton_ref_ablation_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = WORK / "runs" / name
    out_dir.mkdir(parents=True)
    pipe, audit = build_pipeline(kind="reference", memory_mode="model_group_offload")
    (out_dir / "weight_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")

    recs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for tag, scale in (("norefs_ctrl10", 1.0), ("norefs_ctrl00", 0.0)):
            free, total = torch.cuda.mem_get_info()
            t0 = time.time()
            img = pipe(prompt=prompt, reference_images=kept, control_image=ctrl_t,
                       control_context_scale=scale, height=h, width=w, num_inference_steps=args.steps,
                       true_cfg_scale=1.0, use_kv_cache=False,
                       generator=torch.Generator(device=pipe._execution_device).manual_seed(seed)).images[0]
            p = out_dir / f"{entry['kind']}__{args.sample}__{tag}__seed{seed}.png"
            img.save(p)
            recs.append({"tag": tag, "scale": scale, "seed": seed, "out": str(p),
                         "n_refs": len(kept), "canvas_hw": [h, w],
                         "seconds": round(time.time() - t0, 1),
                         "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         "free_GiB_at_start": round(free / 2**30, 2)})
            print("[probe] " + json.dumps(recs[-1], ensure_ascii=False), flush=True)

    summary = {"run": str(out_dir), "sample": args.sample, "steps": args.steps,
               "videox_commit": verify_videox_checkout(),
               "changed_variable": "reference_images 去掉骨架参考图，其余与矩阵一致",
               "records": recs}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
