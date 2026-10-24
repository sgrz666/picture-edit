#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_hi4d_dwpose_1500.py

Batch render 3D-aligned DWPose WholeBody skeletons (134 keypoints: body + 68 face + 42 hands)
for all 1,500 target frames in Hi4D pilot dataset at 512x512 resolution.

Workflow:
1. Iterate over datasets/Hi4D_pilot_v1/processed_frames_1500.jsonl (1,500 frames across 30 sequences).
2. For each frame, load or detect WholeBody DWPose 134 keypoints for Subject A and Subject B:
   - Reuse existing native detections from native_interaction_v2/dwpose_native if present.
   - For missing frames, perform full-frame detection and/or person-mask ROI recovery on raw 1280x940 frames.
   - Save native npz cache to ensure 100% completion.
3. Transform native keypoints to 512x512 canvas using calibrated sequence affine matrix (A_original_to_512).
4. Render 512x512 RGB images with standard OpenPose rainbow palette, face dots, and 21-point finger bones:
   - pose_dwpose/A/<stem>.png
   - pose_dwpose/B/<stem>.png
   - pose_dwpose/combined/<stem>.png
   - pose_dwpose/A_npy/<stem>.npy (134, 3: [x_512, y_512, score])
   - pose_dwpose/B_npy/<stem>.npy (134, 3: [x_512, y_512, score])
5. Generate upgraded final manifests:
   - datasets/Hi4D_pilot_v1/final_manifests/pairs_1500_dwpose_{train,val,test}.jsonl
   - datasets/Hi4D_pilot_v1/dwpose_1500_summary.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path("/home/shangguanrz/project/pic-edit")
HI4D_ROOT = ROOT / "datasets/Hi4D_pilot_v1"
DWROOT = ROOT / "add/tools/dwpose_mimicmotion_official"
DET_MODEL = DWROOT / "models/yolox_l.onnx"
POSE_MODEL = DWROOT / "models/dw-ll_ucoco_384.onnx"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DWROOT))

from dwpose.wholebody import Wholebody
from src.pose_edit.geometry import render_poseguide


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def combine_guides(img_A: Image.Image, img_B: Image.Image) -> Image.Image:
    """Combine two person skeletons using maximum intensity projection on black canvas."""
    arr_A = np.asarray(img_A, dtype=np.uint16)
    arr_B = np.asarray(img_B, dtype=np.uint16)
    arr_comb = np.where(arr_B > arr_A, arr_B, arr_A).astype(np.uint8)
    return Image.fromarray(arr_comb)


class Hi4DDWPoseRenderer:
    def __init__(self, device: str = "cpu"):
        print(f"[Renderer] Initializing Wholebody detector on {device}...")
        self.model = Wholebody(str(DET_MODEL), str(POSE_MODEL), device=device)
        self.meta_cache = {}
        print("[Renderer] Wholebody detector ready.")

    def get_meta(self, pair: str, action: str):
        key = f"{pair}/{action}"
        if key not in self.meta_cache:
            meta_path = HI4D_ROOT / f"processed_512_1500/{pair}/{action}/sequence_meta.json"
            meta = load_json(meta_path)
            self.meta_cache[key] = {
                "A_to_512": np.array(meta["A_original_to_512"], dtype=np.float32),
                "crop": meta.get("crop", {}),
            }
        return self.meta_cache[key]

    def detect_roi(self, img: np.ndarray, mask: np.ndarray, margin: int = 30):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return np.zeros((134, 2), dtype=np.float32), np.zeros((134,), dtype=np.float32)
        H, W = img.shape[:2]
        x0, x1 = max(0, int(xs.min()) - margin), min(W, int(xs.max()) + margin)
        y0, y1 = max(0, int(ys.min()) - margin), min(H, int(ys.max()) + margin)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return np.zeros((134, 2), dtype=np.float32), np.zeros((134,), dtype=np.float32)
        
        cand, sub = self.model(roi)
        if cand is None or len(cand) == 0:
            return np.zeros((134, 2), dtype=np.float32), np.zeros((134,), dtype=np.float32)
        
        pts = cand[0].copy().astype(np.float32)
        scores = sub[0].copy().astype(np.float32) if sub is not None else np.ones((134,), dtype=np.float32)
        pts[:, 0] += x0
        pts[:, 1] += y0
        return pts, scores

    def get_keypoints(self, row: dict):
        pair = row["pair"]
        action = row["action"]
        stem = row["frame_stem"]

        dw_A_path = HI4D_ROOT / f"native_interaction_v2/dwpose_native/{pair}/{action}/{stem}_A.npz"
        dw_B_path = HI4D_ROOT / f"native_interaction_v2/dwpose_native/{pair}/{action}/{stem}_B.npz"

        # 1. Fast path: reuse existing npz
        if dw_A_path.is_file() and dw_B_path.is_file():
            try:
                with np.load(dw_A_path) as zA, np.load(dw_B_path) as zB:
                    pts_A = zA["xy_native"].copy().astype(np.float32)
                    sc_A = zA["scores"].copy().astype(np.float32)
                    pts_B = zB["xy_native"].copy().astype(np.float32)
                    sc_B = zB["scores"].copy().astype(np.float32)
                    if pts_A.shape == (134, 2) and pts_B.shape == (134, 2):
                        return pts_A, sc_A, pts_B, sc_B, "reused"
            except Exception:
                pass

        # 2. Detect path
        raw_rgb_path = Path(row["rgb"])
        img_raw = cv2.imread(str(raw_rgb_path))
        if img_raw is None:
            raise RuntimeError(f"Failed to read raw image: {raw_rgb_path}")
        H, W = img_raw.shape[:2]

        mask_A_path = Path(row["person_mask_A"])
        mask_B_path = Path(row["person_mask_B"])
        mask_A = cv2.imread(str(mask_A_path), cv2.IMREAD_GRAYSCALE) > 0
        mask_B = cv2.imread(str(mask_B_path), cv2.IMREAD_GRAYSCALE) > 0

        cand, sub = self.model(img_raw)
        pts_A, sc_A = None, None
        pts_B, sc_B = None, None

        if cand is not None and len(cand) >= 2:
            scores_per_cand = []
            for c in cand:
                v = c[(c[:, 0] > 0) & (c[:, 1] > 0) & (c[:, 0] < W) & (c[:, 1] < H)]
                in_A = sum(mask_A[int(y), int(x)] for x, y in v) / max(len(v), 1)
                in_B = sum(mask_B[int(y), int(x)] for x, y in v) / max(len(v), 1)
                scores_per_cand.append((in_A, in_B))
            
            best_A_idx = int(np.argmax([s[0] for s in scores_per_cand]))
            best_B_idx = int(np.argmax([s[1] for s in scores_per_cand]))
            if best_A_idx != best_B_idx and scores_per_cand[best_A_idx][0] > 0.25 and scores_per_cand[best_B_idx][1] > 0.25:
                pts_A = cand[best_A_idx].copy().astype(np.float32)
                sc_A = sub[best_A_idx].copy().astype(np.float32)
                pts_B = cand[best_B_idx].copy().astype(np.float32)
                sc_B = sub[best_B_idx].copy().astype(np.float32)

        method = "full_detected"
        if pts_A is None or (sc_A > 0.3).sum() < 10:
            pts_A, sc_A = self.detect_roi(img_raw, mask_A)
            method = "roi_recovered"
        if pts_B is None or (sc_B > 0.3).sum() < 10:
            pts_B, sc_B = self.detect_roi(img_raw, mask_B)
            method = "roi_recovered"

        # Cache newly detected npz
        dw_A_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dw_A_path, xy_native=pts_A, scores=sc_A, width=W, height=H)
        np.savez_compressed(dw_B_path, xy_native=pts_B, scores=sc_B, width=W, height=H)

        return pts_A, sc_A, pts_B, sc_B, method

    def render_frame(self, row: dict):
        pair = row["pair"]
        action = row["action"]
        stem = row["frame_stem"]

        pts_A, sc_A, pts_B, sc_B, method = self.get_keypoints(row)
        meta = self.get_meta(pair, action)
        A_to_512 = meta["A_to_512"]

        def transform_to_512(pts, sc):
            pts = pts.copy()
            valid = (sc > 0.3) & np.isfinite(pts).all(axis=1)
            pts[~valid] = np.nan
            h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
            p512 = (A_to_512 @ h.T).T[:, :2]
            p512[~valid] = np.nan
            return p512

        pts_A_512 = transform_to_512(pts_A, sc_A)
        pts_B_512 = transform_to_512(pts_B, sc_B)

        img_A = render_poseguide(pts_A_512, (512, 512))
        img_B = render_poseguide(pts_B_512, (512, 512))
        img_comb = combine_guides(img_A, img_B)

        # Output paths
        base_dir = HI4D_ROOT / f"processed_512_1500/{pair}/{action}/pose_dwpose"
        out_A_png = base_dir / f"A/{stem}.png"
        out_B_png = base_dir / f"B/{stem}.png"
        out_comb_png = base_dir / f"combined/{stem}.png"
        out_A_npy = base_dir / f"A_npy/{stem}.npy"
        out_B_npy = base_dir / f"B_npy/{stem}.npy"

        out_A_png.parent.mkdir(parents=True, exist_ok=True)
        out_B_png.parent.mkdir(parents=True, exist_ok=True)
        out_comb_png.parent.mkdir(parents=True, exist_ok=True)
        out_A_npy.parent.mkdir(parents=True, exist_ok=True)
        out_B_npy.parent.mkdir(parents=True, exist_ok=True)

        img_A.save(out_A_png)
        img_B.save(out_B_png)
        img_comb.save(out_comb_png)

        # Save (134, 3) keypoints [x, y, score]
        feat_A = np.concatenate([pts_A_512, sc_A[:, None]], axis=1)
        feat_B = np.concatenate([pts_B_512, sc_B[:, None]], axis=1)
        np.save(out_A_npy, feat_A)
        np.save(out_B_npy, feat_B)

        return {
            "pair": pair,
            "action": action,
            "stem": stem,
            "method": method,
            "valid_A": int((sc_A > 0.3).sum()),
            "valid_B": int((sc_B > 0.3).sum()),
            "mean_score_A": float(sc_A.mean()),
            "mean_score_B": float(sc_B.mean()),
            "pose_dwpose_A": str(out_A_png),
            "pose_dwpose_B": str(out_B_png),
            "pose_dwpose_combined": str(out_comb_png),
            "pose_dwpose_A_npy": str(out_A_npy),
            "pose_dwpose_B_npy": str(out_B_npy),
        }


def update_manifests():
    """Build upgraded final manifests pointing to pose_dwpose."""
    print("[Manifests] Updating final manifests to point to pose_dwpose...")
    manifest_dir = HI4D_ROOT / "final_manifests"
    
    for split in ["train", "val", "test"]:
        src_path = manifest_dir / f"pairs_1500_{split}.jsonl"
        dst_path = manifest_dir / f"pairs_1500_dwpose_{split}.jsonl"
        
        rows = load_jsonl(src_path)
        updated_rows = []
        for r in rows:
            pair = r["pair"]
            action = r["action"]
            target_id = r["target_frame_id"]
            stem = f"{target_id:06d}"
            
            base = f"/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1/processed_512_1500/{pair}/{action}/pose_dwpose"
            r["target_pose_A"] = f"{base}/A/{stem}.png"
            r["target_pose_B"] = f"{base}/B/{stem}.png"
            r["target_pose_combined"] = f"{base}/combined/{stem}.png"
            r["target_pose_A_npy"] = f"{base}/A_npy/{stem}.npy"
            r["target_pose_B_npy"] = f"{base}/B_npy/{stem}.npy"
            r["pose_representation"] = "DWPose_WholeBody_134"
            updated_rows.append(r)
        
        with dst_path.open("w", encoding="utf-8") as f:
            for r in updated_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Manifests] Wrote {len(updated_rows)} pairs to {dst_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Hi4D 1500-frame DWPose WholeBody Batch Renderer")
    parser.add_argument("--device", type=str, default="cpu", help="Device for Wholebody detector (cpu or cuda:0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames for dry run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rendering if combined png exists")
    args = parser.parse_args()

    frames_jsonl = HI4D_ROOT / "processed_frames_1500.jsonl"
    rows = load_jsonl(frames_jsonl)
    if args.limit:
        rows = rows[:args.limit]
    total_frames = len(rows)
    print(f"[Main] Loaded {total_frames} frames from {frames_jsonl.name}")

    renderer = Hi4DDWPoseRenderer(device=args.device)

    started_at = time.time()
    results = []
    method_counts = {"reused": 0, "full_detected": 0, "roi_recovered": 0}
    
    for i, row in enumerate(tqdm(rows, desc="Rendering DWPose")):
        pair = row["pair"]
        action = row["action"]
        stem = row["frame_stem"]
        
        comb_png = HI4D_ROOT / f"processed_512_1500/{pair}/{action}/pose_dwpose/combined/{stem}.png"
        if args.skip_existing and comb_png.is_file():
            results.append({"pair": pair, "action": action, "stem": stem, "method": "skipped"})
            continue

        res = renderer.render_frame(row)
        results.append(res)
        method_counts[res["method"]] = method_counts.get(res["method"], 0) + 1

    elapsed = time.time() - started_at
    rate = total_frames / max(elapsed, 0.001)

    summary = {
        "total_frames": total_frames,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(rate, 2),
        "methods": method_counts,
        "mean_valid_A": float(np.mean([r["valid_A"] for r in results if "valid_A" in r])),
        "mean_valid_B": float(np.mean([r["valid_B"] for r in results if "valid_B" in r])),
        "mean_score_A": float(np.mean([r["mean_score_A"] for r in results if "mean_score_A" in r])),
        "mean_score_B": float(np.mean([r["mean_score_B"] for r in results if "mean_score_B" in r])),
        "output_format": {
            "resolution": 512,
            "keypoints": 134,
            "components": ["body_18", "face_68", "hands_42", "feet_6"],
            "palette": "OpenPose_WholeBody_Rainbow",
            "files": ["pose_dwpose/A/*.png", "pose_dwpose/B/*.png", "pose_dwpose/combined/*.png"],
        }
    }

    summary_path = HI4D_ROOT / "dwpose_1500_summary.json"
    write_json(summary_path, summary)
    print(f"\n[Summary] Render complete in {elapsed:.1f}s ({rate:.1f} fps):")
    print(json.dumps(summary, indent=2))

    # Update manifests
    update_manifests()


if __name__ == "__main__":
    main()
