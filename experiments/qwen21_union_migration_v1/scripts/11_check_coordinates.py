#!/usr/bin/env python
"""坐标核对：同一套 guide 去比"工作流自己产出的 final_canvas"和"本实验 native 臂的出图"。

如果连工作流自己的产物都算出 PCK≈0，问题在 guide 的坐标系/尺度，不在生成质量。
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
from src.evaluation.no_gt_metrics import PCK_THRESHOLDS, pose_error_stats, torso_px  # noqa: E402
from src.pose_edit.dwpose import detect_people  # noqa: E402

W = "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1"
CODE = "/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1"
REPO = "/home/shangguanrz/project/pic-edit"
CFG = json.load(open(f"{REPO}/workflow-multiple/config.json"))
THR = CFG["keypoint_score_threshold"]

print("repo PCK_THRESHOLDS:", PCK_THRESHOLDS, " score_threshold:", THR)

for sid in ["hard_02", "hard_05_torso"]:
    z = np.load(f"{CODE}/input/single/{sid}/requested_keypoints.npz")
    g, gv = z["keypoints_B_px"], z["valid_B"]
    print(f"\n### {sid}")
    print("  guide valid joints:", int(gv.sum()), " torso_px:", round(torso_px(g), 2))
    xs = g[gv][:, 0]
    ys = g[gv][:, 1]
    print(f"  guide x range [{xs.min():.1f},{xs.max():.1f}]  y range [{ys.min():.1f},{ys.max():.1f}]")

    targets = {
        "workflow own final_canvas": f"{REPO}/workflow/runs/scale_anchor_20261013/{sid}/generation/final_canvas.png",
        "workflow own final_image": f"{REPO}/workflow/runs/scale_anchor_20261013/{sid}/generation/final_image.png",
        "our native arm seed42": f"{W}/runs/matrix_20261015T134829/single__{sid}__native__seed42.png",
    }
    for label, path in targets.items():
        try:
            im = Image.open(path)
        except Exception as e:
            print(f"  {label}: 打不开 {type(e).__name__}: {e}")
            continue
        rgb = np.asarray(im.convert("RGB"))
        size = (rgb.shape[1], rgb.shape[0])
        ppl = detect_people(CFG, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ppl:
            print(f"  {label}: size={size} 无人检出")
            continue
        pts = np.asarray(ppl[0]["points"], dtype=float)
        sc = np.asarray(ppl[0]["scores"], dtype=float)
        keep = np.isfinite(pts).all(axis=1) & (sc >= THR)
        st = pose_error_stats(g, gv, np.where(keep[:, None], pts, np.nan), keep, size)
        b = st["groups"]["body"]
        s = st["summary"]
        print(f"  {label}: size={size} detected={len(ppl)} det_torso={round(torso_px(pts),1)} "
              f"pck@0.2={b.get('pck@0.2')} strict={b.get('pck@0.2_strict')} "
              f"mpjpe_norm={s.get('body_mpjpe_norm')} compared={st['joints_compared']}/{st['joints_requested']}")
