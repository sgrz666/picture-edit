#!/usr/bin/env python
"""补齐官方复现关里缺的一项量化证据：Pose / Depth 控制强度 0 → 1 时输出确实受控制影响。

04_official_samples.py 的 gate 只验了 inpaint 的白色重绘语义和「pose/depth 出图了」，
没有量化「控制有没有起作用」。这里对同一 seed、同一提示词、只差 control_context_scale 的两张图
做像素级对照：平均绝对差、变化像素占比，并出一张 控制图 | scale0 | scale1 | 官方参考图 的面板。
结果写进该运行的 response_check.json（新增文件，不改写既有产物）。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw

WORK = pathlib.Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")


def pair_stats(a_path: pathlib.Path, b_path: pathlib.Path) -> dict:
    with Image.open(a_path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32)
    with Image.open(b_path) as im:
        b = np.asarray(im.convert("RGB").resize((a.shape[1], a.shape[0])), dtype=np.float32)
    d = np.abs(a - b)
    return {"shape_a": list(a.shape[:2]), "shape_b": list(b.shape[:2]),
            "mean_abs_diff_0_255": round(float(d.mean()), 2),
            "max_abs_diff": round(float(d.max()), 2),
            "changed_pixel_fraction": round(float((d.mean(axis=2) > 8).mean()), 4)}


def panel(out: pathlib.Path, cells):
    ims = []
    for label, p in cells:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((300, 300))
        ims.append((label, im))
    w = sum(i.width for _, i in ims) + 10 * (len(ims) + 1)
    h = max(i.height for _, i in ims) + 34
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    x = 10
    for label, im in ims:
        canvas.paste(im, (x, 26))
        d.text((x, 8), label, fill="black")
        x += im.width + 10
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = WORK / "runs" / args.run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assets = {k: pathlib.Path(v) for k, v in summary.get("assets_used", {}).items()}

    out = {"run": str(run), "seed": summary.get("seed"), "steps": summary.get("steps"),
           "cfg": summary.get("cfg"), "checks": {}, "panels": {}}
    for task in ("pose", "depth"):
        p0 = run / f"{task}_scale0.0_seed43_{summary['steps']}step.png"
        p1 = run / f"{task}_scale1.0_seed43_{summary['steps']}step.png"
        if not (p0.exists() and p1.exists()):
            out["checks"][task] = {"available": False, "reason": f"缺少 {p0.name} 或 {p1.name}"}
            continue
        st = pair_stats(p0, p1)
        st.update({"available": True, "control_responded": bool(st["changed_pixel_fraction"] > 0.2),
                   "note": "同 seed 同提示词，只有 control_context_scale 从 0 变 1"})
        out["checks"][task] = st
        cells = [(f"{task} 控制图", assets.get(task, p0))]
        ref = run / f"official_reference_{task}.png"
        if ref.exists():
            cells.append(("官方发布示例", ref))
        cells += [("scale 0（不注入控制）", p0), ("scale 1（完整控制）", p1)]
        png = run / f"panel_control_response_{task}.png"
        panel(png, cells)
        out["panels"][task] = str(png)

    inp = [r for r in summary.get("records", []) if r["task"] == "inpaint"]
    out["inpaint_semantics"] = inp[0].get("semantic_check") if inp else None
    out["gate_pass"] = bool(out["checks"]) and all(
        c.get("control_responded") for c in out["checks"].values() if c.get("available")) and \
        bool((out["inpaint_semantics"] or {}).get("white_is_regenerated"))

    p = run / "response_check.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
