#!/usr/bin/env python
"""官方复现关：用 Union 仓库自带的官方素材与官方入口跑 Pose / Depth / Inpaint。

模型卡没有随图发布每个 demo 的提示词（README 里只有 "your prompt describing the whole target
image"），所以这里的提示词是**按控制图内容自拟**的，只用于验证"控制输入是否有效"，
不用于逐像素重现卡面图片。Inpaint 的语义用数值判定：白色区域应当被重绘，黑色区域应当保持。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from union_env import VIDEOX, WORK, build_pipeline, verify_videox_checkout  # noqa: F401

UNION_DIR = WORK / "models/Qwen-Image-2.1-Fun-Controlnet-Union"

# 来自模型卡 demo 表格的列顺序（Canny/Depth/Grayscale/HED/Lineart/MLSD/Pose/Scribble）
OFFICIAL = {
    "pose": UNION_DIR / "asset/control_4_00000931.png",
    "depth": UNION_DIR / "asset/control_5_00005389.png",
    "inpaint_image": UNION_DIR / "asset/inpaint_source.png",
    "inpaint_mask": UNION_DIR / "asset/inpaint_mask.png",
    "inpaint_control": UNION_DIR / "asset/inpaint_control.jpg",
}
OFFICIAL_RESULT = {
    "pose": UNION_DIR / "results/control_4_00000931.png",
    "depth": UNION_DIR / "results/control_5_00005389.png",
    "inpaint": UNION_DIR / "results/inpaint.png",
}

PROMPTS = {
    "pose": "一位年轻女子站在海边，亮紫色长发被海风吹起，身穿纯白色无袖连衣裙，单手叉腰，"
            "背景是湛蓝的海面与晴朗的天空。",
    "depth": "一位长发女子在室内侧身而坐，双手在胸前合十，人物位于画面右侧前景且离镜头很近，"
             "背景是虚化的房间、柜子和家具，景深明显。",
    "inpaint": "一位年轻女子穿着红色皮夹克，内搭深色T恤，双手在胸前合十，坐在室内，"
               "背景是虚化的房间与家具。",
}


def canvas_for(path: pathlib.Path, long_side: int) -> list[int]:
    """按控制图自身宽高比取画布，两边都对齐到 32（官方要求 16 的倍数，这里取更严的 32）。"""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    s = (long_side ** 2 * (w / h)) ** 0.5
    tw = max(32, int(round(s / 32)) * 32)
    th = max(32, int(round(s / (h / w) / 32)) * 32)
    return [th, tw]


def run_one(pipeline, out_dir: pathlib.Path, *, task, prompt, control=None, mask=None, image=None,
            sample_size=None, scale=1.0, seed=43, steps=40):
    import torch
    from videox_fun.utils import get_image_latent

    dev = pipeline._execution_device
    kwargs = dict(prompt=prompt, height=sample_size[0], width=sample_size[1],
                  num_inference_steps=steps, true_cfg_scale=1.0,
                  generator=torch.Generator(device=dev).manual_seed(seed),
                  control_context_scale=scale, use_kv_cache=False)
    if control is not None:
        kwargs["control_image"] = get_image_latent(str(control), sample_size=list(sample_size))[:, :, 0]
    if image is not None:
        kwargs["image"] = get_image_latent(str(image), sample_size=list(sample_size))[:, :, 0]
    if mask is not None:
        kwargs["mask_image"] = get_image_latent(str(mask), sample_size=list(sample_size))[:, :1, 0]

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    img = pipeline(**kwargs).images[0]
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{task}_scale{scale}_seed{seed}_{steps}step.png"
    img.save(out_dir / name)
    rec = {"task": task, "scale": scale, "seed": seed, "steps": steps, "canvas_hw": list(sample_size),
           "seconds": round(time.time() - t0, 1), "peak_GiB": round(peak, 2), "out": str(out_dir / name),
           "control": str(control) if control else None, "image": str(image) if image else None,
           "mask": str(mask) if mask else None}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


def inpaint_semantics(out_path: str, source_path: pathlib.Path, mask_path: pathlib.Path) -> dict:
    """白色=重绘、黑色=保持：比较输出与底图在 mask 内外的平均变化。"""
    import numpy as np
    from PIL import Image
    with Image.open(out_path) as im:
        out = np.asarray(im.convert("RGB"), dtype=np.float32)
    with Image.open(source_path) as im:
        src = np.asarray(im.convert("RGB").resize((out.shape[1], out.shape[0])), dtype=np.float32)
    with Image.open(mask_path) as im:
        m = np.asarray(im.convert("L").resize((out.shape[1], out.shape[0])), dtype=np.float32) / 255.0
    white = m >= 0.5
    d = np.abs(out - src).mean(axis=2)
    inside = float(d[white].mean()) if white.any() else None
    outside = float(d[~white].mean()) if (~white).any() else None
    ok = inside is not None and outside is not None and inside > outside
    return {"mask_white_fraction": round(float(white.mean()), 4),
            "mean_abs_change_inside_white": None if inside is None else round(inside, 2),
            "mean_abs_change_outside_black": None if outside is None else round(outside, 2),
            "white_is_regenerated": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    ap.add_argument("--tasks", default="pose,depth,inpaint")
    ap.add_argument("--scales", default="0.0,1.0")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    import torch
    name = args.name or f"official_{args.mode}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = WORK / "runs" / name
    steps = 8 if args.mode == "smoke" else 40
    long_side = 512 if args.mode == "smoke" else 1024
    scales = [float(s) for s in args.scales.split(",")]

    missing = {k: str(v) for k, v in OFFICIAL.items() if not v.exists()}
    if missing:
        print("[assets] 缺少官方素材，相关任务会被标记为阻塞而不是伪造:", json.dumps(missing, ensure_ascii=False))
    have = {k: v for k, v in OFFICIAL.items() if v.exists()}

    pipeline, audit = build_pipeline(kind="official",
                                     memory_mode="model_group_offload" if torch.cuda.is_available() else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")

    records, blocked = [], []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task in ("pose", "depth"):
            ctrl = have.get(task)
            if ctrl is None:
                blocked.append({"task": task, "reason": "官方素材缺失"})
                continue
            size = canvas_for(ctrl, long_side) if args.mode == "formal" else [long_side, long_side]
            for scale in scales:
                records.append(run_one(pipeline, out_dir, task=task, prompt=PROMPTS[task], control=ctrl,
                                       sample_size=size, scale=scale, seed=43, steps=steps))
        elif task == "inpaint":
            src, msk = have.get("inpaint_image"), have.get("inpaint_mask")
            ctrl = have.get("inpaint_control")
            if src is None or msk is None:
                blocked.append({"task": task, "reason": "官方 inpaint 底图或 mask 缺失"})
                continue
            size = canvas_for(src, long_side) if args.mode == "formal" else [long_side, long_side]
            rec = run_one(pipeline, out_dir, task="inpaint", prompt=PROMPTS["inpaint"], control=ctrl,
                          mask=msk, image=src, sample_size=size, scale=1.0, seed=43, steps=steps)
            rec["semantic_check"] = inpaint_semantics(rec["out"], src, msk)
            print("[inpaint semantics]", json.dumps(rec["semantic_check"], ensure_ascii=False), flush=True)
            records.append(rec)

    for task, ref in OFFICIAL_RESULT.items():
        if ref.exists():
            import shutil
            shutil.copy2(ref, out_dir / f"official_reference_{task}.png")

    summary = {"mode": args.mode, "run": str(out_dir), "steps": steps, "seed": 43,
               "cfg": 1.0, "kv_cache": False, "videox_commit": verify_videox_checkout(),
               "prompts_are_official": False,
               "prompt_note": "模型卡未随 demo 发布提示词，此处提示词按控制图内容自拟",
               "assets_used": {k: str(v) for k, v in have.items()},
               "weight_gate": None if audit is None else audit.get("gate_pass"),
               "records": records, "blocked": blocked}
    pose_recs = [r for r in records if r["task"] == "pose"]
    depth_recs = [r for r in records if r["task"] == "depth"]
    inp_recs = [r for r in records if r["task"] == "inpaint"]
    summary["gate_pass"] = bool(pose_recs or depth_recs) and all(
        r.get("semantic_check", {}).get("white_is_regenerated", True) for r in inp_recs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1, ensure_ascii=False))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
