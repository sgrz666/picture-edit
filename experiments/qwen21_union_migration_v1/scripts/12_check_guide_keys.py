#!/usr/bin/env python
"""guide 键选择核对：keypoints_B_px vs keypoints_generation_px。

判据很硬：拿工作流自己的 final_canvas 当被测量，正确的 guide 应当给出高 PCK，
错误的 guide 会把 pck 压到 0。同时打印工作流当时记录的 measurements.json 作为参照。
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
from src.evaluation.no_gt_metrics import pose_error_stats, torso_px  # noqa: E402
from src.pose_edit.dwpose import detect_people  # noqa: E402

CODE = "/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1"
REPO = "/home/shangguanrz/project/pic-edit"
CFG = json.load(open(f"{REPO}/workflow-multiple/config.json"))
THR = CFG["keypoint_score_threshold"]
RUN = "hard_02"

z = np.load(f"{CODE}/input/single/{RUN}/requested_keypoints.npz")
print("npz keys:", z.files)
own = f"{REPO}/workflow/runs/scale_anchor_20261013/{RUN}/generation/final_canvas.png"
rgb = np.asarray(Image.open(own).convert("RGB"))
size = (rgb.shape[1], rgb.shape[0])
ppl = detect_people(CFG, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
pts = np.asarray(ppl[0]["points"], dtype=float)
sc = np.asarray(ppl[0]["scores"], dtype=float)
keep = np.isfinite(pts).all(axis=1) & (sc >= THR)
print(f"measuring the workflow's own canvas {size}, det_torso={round(torso_px(pts),1)}")

for key, vkey in [("keypoints_B_px", "valid_B"), ("keypoints_generation_px", "valid_generation")]:
    if key not in z.files:
        print(f"  {key}: 不存在")
        continue
    g, gv = z[key], z[vkey]
    ys = g[gv][:, 1]
    xs = g[gv][:, 0]
    st = pose_error_stats(g, gv, np.where(keep[:, None], pts, np.nan), keep, size)
    b = st["groups"]["body"]
    print(f"  {key}: x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] "
          f"torso={round(torso_px(g),1)} -> pck@0.2={b.get('pck@0.2')} "
          f"pck@0.5={b.get('pck@0.5')} mpjpe_norm={st['summary']['body_mpjpe_norm']}")

m = f"{REPO}/workflow/runs/scale_anchor_20261013/{RUN}/evaluation_dwpose/measurements.json"
try:
    d = json.load(open(m))
    print("\n工作流当时记录的 measurements.json（节选）:")
    print(json.dumps({k: v for k, v in d.items() if not isinstance(v, (list, dict))}, ensure_ascii=False)[:600])
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:300]}")
except Exception as e:
    print("measurements.json 读不到:", e)
