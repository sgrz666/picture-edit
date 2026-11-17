#!/usr/bin/env python3
"""Cross-check SDPose-134 against the dataset's existing DWPose-134 pose JSONs.

Both are expressed in ``original_pixel`` space of the same frame image, so the per-joint pixel
distance is a direct agreement metric. Only joints that are valid on both sides are compared
(SDPose: ``valid_mask134``; DWPose: ``confidence >= --conf``), which keeps ankle points that
fall outside a cropped frame out of the mean instead of silently pulling it up.

Usage::

    python scripts/audit/check_sdpose_vs_dwpose_agreement.py \
        --kp-root /data/shangguanrz/datasets/bilibili/geometry/skeleton/sdpose_2d_kp \
        --dwpose-root /data/shangguanrz/datasets/bilibili/frames/poses

Reports per-joint median pixel error, torso-diagonal-normalised PCK, and the frames with the
worst agreement so they can be eyeballed individually.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

# 134-slot geometry order: body 0..17, feet 18..23, face 24..91, hands 92..133.
BODY_NAMES = [
    "nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
    "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle",
    "R_eye", "R_ear", "L_eye", "L_ear",
]
BODY_SLOTS = list(range(18))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp-root", default="/data/shangguanrz/datasets/bilibili/geometry/skeleton/sdpose_2d_kp")
    ap.add_argument("--dwpose-root", default="/data/shangguanrz/datasets/bilibili/frames/poses")
    ap.add_argument("--conf", type=float, default=0.30, help="DWPose confidence floor for comparison")
    ap.add_argument("--dwpose-order", choices=("openpose", "geometry"), default="openpose",
                    help="eye/ear slot order used by the DWPose JSONs. Measured on this dataset: "
                         "frames/poses is openpose order (15=L_eye, 16=R_ear), i.e. slots 15<->16 are "
                         "swapped relative to the SDPose geometry order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worst", type=int, default=8, help="how many worst-agreeing frames to list")
    return ap.parse_args()


def torso_diag(pts: np.ndarray) -> float:
    """Shoulder-mid to hip-mid distance, used to normalise the pixel error."""
    sh = 0.5 * (pts[2] + pts[5])
    hp = 0.5 * (pts[8] + pts[11])
    return float(np.linalg.norm(sh - hp))


def main() -> int:
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.kp_root, "*", "*_keypoints.json")))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("no sdpose keypoint json found")
        return 1

    per_joint: dict[int, list[float]] = {i: [] for i in BODY_SLOTS}
    norm_err: list[float] = []
    frames: list[tuple[float, str, int]] = []
    skipped_no_dwpose = skipped_no_diag = 0

    for jf in files:
        d = json.load(open(jf, encoding="utf-8"))
        bvid, fid = d["bvid"], d["frame_id"]
        dw_path = os.path.join(args.dwpose_root, bvid, f"{fid}.json")
        if not os.path.isfile(dw_path):
            skipped_no_dwpose += 1
            continue
        dw = json.load(open(dw_path, encoding="utf-8"))
        dw_pts = np.asarray(dw["candidate"][:18], dtype=float)
        dw_conf = np.asarray(dw["confidence"][:18], dtype=float)
        if args.dwpose_order == "openpose":
            # geometry order is 14=R_eye,15=R_ear,16=L_eye,17=L_ear; openpose swaps 15<->16.
            dw_pts[[15, 16]] = dw_pts[[16, 15]]
            dw_conf[[15, 16]] = dw_conf[[16, 15]]
        sp_pts = np.asarray(
            [[np.nan, np.nan] if p[0] is None else p for p in d["points134_raw"][:18]], dtype=float
        )
        sp_valid = np.asarray(d["valid_mask134"][:18], dtype=bool)

        ok = sp_valid & (dw_conf >= args.conf) & np.isfinite(sp_pts).all(axis=1)
        if ok.sum() < 6:
            continue
        dist = np.linalg.norm(sp_pts[ok] - dw_pts[ok], axis=1)
        diag = torso_diag(sp_pts)
        if diag <= 1e-6:
            skipped_no_diag += 1
            continue
        for slot, dd in zip(np.asarray(BODY_SLOTS)[ok], dist):
            per_joint[int(slot)].append(float(dd))
        norm_err.append(float(np.median(dist) / diag))
        frames.append((float(np.mean(dist)), f"{bvid}/{fid}", int(ok.sum())))

    print(f"frames_compared={len(frames)} total_json={len(files)} "
          f"skipped_no_dwpose={skipped_no_dwpose} skipped_no_diag={skipped_no_diag}")
    pooled = np.concatenate([np.asarray(v) for v in per_joint.values() if v])
    print(f"pooled_body18_px: median={np.median(pooled):.2f} mean={pooled.mean():.2f} "
          f"p90={np.percentile(pooled, 90):.2f} p99={np.percentile(pooled, 99):.2f} n={pooled.size}")
    ne = np.asarray(norm_err)
    print(f"torso_normalised_median_err: median={np.median(ne):.4f} mean={ne.mean():.4f} "
          f"p90={np.percentile(ne, 90):.4f}")
    for thr in (0.02, 0.05, 0.10):
        print(f"  PCK@{thr:.2f} (body18, torso-diag normalised) = {(ne <= thr).mean() * 100:.2f}%")
    print("per-joint median px error:")
    for slot in BODY_SLOTS:
        v = np.asarray(per_joint[slot])
        if v.size:
            print(f"  {slot:2d} {BODY_NAMES[slot]:<11s} median={np.median(v):7.2f} "
                  f"mean={v.mean():7.2f} n={v.size}")
    frames.sort(key=lambda t: -t[0])
    print(f"worst {args.worst} frames by mean body18 px err:")
    for mean_px, key, n in frames[: args.worst]:
        print(f"  {key}  mean={mean_px:7.2f}px  joints={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
