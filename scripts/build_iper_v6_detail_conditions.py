#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build DeepGen V6.4 Detail Branch Conditions for iPER dataset.

Extracts and precomputes:
1. smplx_detail: [70, 103] float32 tensor
   (expression[10] + jaw_pose[3] + left_hand_pose[45] + right_hand_pose[45])
2. face_keypoints: [70, 68, 3] float32 tensor in [0, 1]
3. hand_keypoints: [70, 2, 21, 3] float32 tensor in [0, 1]
4. boxes: [70, 3, 4] float32 tensor (xyxy normalized to [0, 1])
5. region_valid: [70, 3] bool tensor
6. reference_crops: [70, 3, 3, 224, 224] float32 tensor in [-1, 1]
7. stem_to_idx: dict mapping frame stem to frame index 0..69

Saves bundle in iper_assets_512_nvdiffrast/{appearance}/v6_detail_conditions.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V6.4 detail conditions for iPER appearances")
    parser.add_argument(
        "--sampled_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
        help="Root directory of iPER sampled images and keypoints",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
        help="Root directory of iPER preprocessed assets",
    )
    parser.add_argument(
        "--appearance",
        type=str,
        default="024_6",
        help="Specific appearance to process, or 'all'",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing condition files",
    )
    return parser.parse_args()


def extract_detail_keypoints_and_boxes(
    kp_data: Dict[str, np.ndarray],
    img_w: float = 1024.0,
    img_h: float = 1024.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract face (68) and hand (2x21) keypoints and bounding boxes according to V6.4 spec:
    - 0:18: body
    - 18:24: foot
    - 24:92: face 68 pts
    - 92:113: left hand 21 pts
    - 113:134: right hand 21 pts

    Rules:
    - xy divided by img_w, img_h, normalized to [0, 1].
    - score clamped to [0, 1].
    - score < 0.30 set to (0, 0, 0).
    - face valid if >= 48/68 valid points; hand valid if >= 12/21 valid points.
    - face box expanded by 15%, hand box expanded by 25%, square-ified, clipped to [0, 1].
    - On 1024 resolution, min side length is forced to 64 px (64/1024 = 0.0625).
    """
    kp134 = kp_data["keypoints_134_px"].astype(np.float32).copy()
    sc134 = np.clip(kp_data["scores_134"].astype(np.float32).copy(), 0.0, 1.0)

    # Actual width and height from npz if present
    if "image_width" in kp_data and "image_height" in kp_data:
        w = float(kp_data["image_width"])
        h = float(kp_data["image_height"])
    else:
        w, h = img_w, img_h

    # Normalize xy
    xy_norm = kp134.copy()
    xy_norm[:, 0] /= max(w, 1.0)
    xy_norm[:, 1] /= max(h, 1.0)

    # Threshold score < 0.30 -> (0, 0, 0)
    invalid = sc134 < 0.30
    xy_norm[invalid] = 0.0
    sc134[invalid] = 0.0

    kps_3d = np.concatenate([xy_norm, sc134[:, None]], axis=-1)  # [134, 3]

    face_kps = kps_3d[24:92].copy()       # [68, 3]
    left_hand_kps = kps_3d[92:113].copy() # [21, 3]
    right_hand_kps = kps_3d[113:134].copy()# [21, 3]
    hand_kps = np.stack([left_hand_kps, right_hand_kps], axis=0) # [2, 21, 3]

    # Validity checks
    face_valid_cnt = int(np.sum(face_kps[:, 2] >= 0.30))
    lh_valid_cnt = int(np.sum(left_hand_kps[:, 2] >= 0.30))
    rh_valid_cnt = int(np.sum(right_hand_kps[:, 2] >= 0.30))

    face_valid = face_valid_cnt >= 48
    lh_valid = lh_valid_cnt >= 12
    rh_valid = rh_valid_cnt >= 12
    region_valid = np.array([face_valid, lh_valid, rh_valid], dtype=bool)

    # Box computation helper
    def _compute_square_box(
        pts_3d: np.ndarray,
        expand_ratio: float,
        is_valid: bool,
        min_size_px: float = 64.0,
    ) -> np.ndarray:
        if not is_valid:
            return np.zeros(4, dtype=np.float32)

        valid_mask = pts_3d[:, 2] >= 0.30
        valid_pts = pts_3d[valid_mask, :2]
        if len(valid_pts) == 0:
            return np.zeros(4, dtype=np.float32)

        pts_px = valid_pts * np.array([w, h], dtype=np.float32)
        x1, y1 = np.min(pts_px, axis=0)
        x2, y2 = np.max(pts_px, axis=0)

        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)

        x1_exp = x1 - bw * expand_ratio
        x2_exp = x2 + bw * expand_ratio
        y1_exp = y1 - bh * expand_ratio
        y2_exp = y2 + bh * expand_ratio

        cx = (x1_exp + x2_exp) * 0.5
        cy = (y1_exp + y2_exp) * 0.5
        side = max(x2_exp - x1_exp, y2_exp - y1_exp, float(min_size_px))
        half_side = side * 0.5

        sq_x1 = max(0.0, min(cx - half_side, w - 1.0))
        sq_y1 = max(0.0, min(cy - half_side, h - 1.0))
        sq_x2 = max(sq_x1 + 1.0, min(cx + half_side, w))
        sq_y2 = max(sq_y1 + 1.0, min(cy + half_side, h))

        box_norm = np.array([sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h], dtype=np.float32)
        return box_norm

    face_box = _compute_square_box(face_kps, 0.15, face_valid)
    lh_box = _compute_square_box(left_hand_kps, 0.25, lh_valid)
    rh_box = _compute_square_box(right_hand_kps, 0.25, rh_valid)

    boxes = np.stack([face_box, lh_box, rh_box], axis=0) # [3, 4]
    return face_kps, hand_kps, boxes, region_valid


def crop_region_reference(
    img_rgb: np.ndarray,
    box_norm: np.ndarray,
    is_valid: bool,
    crop_size: int = 224,
) -> np.ndarray:
    """Crop, resize to 224x224 and normalize to [-1, 1]. Output: [3, 224, 224] float32."""
    if not is_valid:
        return np.zeros((3, crop_size, crop_size), dtype=np.float32)

    h, w = img_rgb.shape[:2]
    x1 = int(round(box_norm[0] * w))
    y1 = int(round(box_norm[1] * h))
    x2 = int(round(box_norm[2] * w))
    y2 = int(round(box_norm[3] * h))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((3, crop_size, crop_size), dtype=np.float32)

    resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(np.float32) / 127.5) - 1.0 # [-1, 1]
    return np.transpose(norm, (2, 0, 1)) # [3, H, W]


def process_appearance_detail(
    app: str,
    sampled_root: Path,
    assets_root: Path,
    overwrite: bool = False,
) -> bool:
    """Process all 70 frames of one appearance and save v6_detail_conditions.pt."""
    app_sampled = sampled_root / app
    app_assets = assets_root / app

    out_path = app_assets / "v6_detail_conditions.pt"
    if out_path.exists() and not overwrite:
        print(f"[{app}] v6_detail_conditions.pt exists, skipping.")
        return True

    fmap_path = app_sampled / "smplestx" / "frame_map.json"
    params_path = app_sampled / "smplestx" / "params.pt"

    if not fmap_path.exists() or not params_path.exists():
        print(f"[{app}] Missing frame_map.json or params.pt, skipping.")
        return False

    with open(fmap_path, "r", encoding="utf-8") as f:
        fmap = json.load(f)

    frames = fmap.get("frames", [])
    if len(frames) != 70:
        print(f"[{app}] Frames count {len(frames)} != 70, skipping.")
        return False

    params = torch.load(params_path, map_location="cpu", weights_only=False)

    # SMPL-X detail: expression[10] + jaw_pose[3] + left_hand_pose[45] + right_hand_pose[45] = 103
    expr = params["expression"].float()       # [70, 10]
    jaw = params["jaw_pose"].float()          # [70, 3]
    lh_pose = params["left_hand_pose"].float()# [70, 45]
    rh_pose = params["right_hand_pose"].float()# [70, 45]
    smplx_detail = torch.cat([expr, jaw, lh_pose, rh_pose], dim=-1) # [70, 103]

    face_kps_list = []
    hand_kps_list = []
    boxes_list = []
    region_valid_list = []
    ref_crops_list = []
    stem_to_idx = {}

    for idx, fr in enumerate(frames):
        stem = fr["stem"]
        role = fr["role"]
        stem_to_idx[stem] = idx

        # Load keypoints
        kp_file = app_sampled / "keypoints" / role / f"{stem}.npz"
        if not kp_file.exists():
            print(f"[{app}] Missing keypoints for {stem}, skipping.")
            return False

        kp_data = np.load(kp_file)
        f_kps, h_kps, bxs, r_val = extract_detail_keypoints_and_boxes(kp_data)

        # Load RGB image to extract crops
        rgb_rel = fr.get("rgb_1024", f"rgb_1024/{role}/{stem}.png")
        rgb_path = app_sampled / rgb_rel
        if not rgb_path.exists():
            # Fallback to 512
            rgb_path = app_assets / "rgb_512" / role / f"{stem}.png"

        img_rgb = cv2.imread(str(rgb_path))
        if img_rgb is not None:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            crops = np.stack(
                [
                    crop_region_reference(img_rgb, bxs[r], r_val[r])
                    for r in range(3)
                ],
                axis=0,
            ) # [3, 3, 224, 224]
        else:
            crops = np.zeros((3, 3, 224, 224), dtype=np.float32)

        face_kps_list.append(torch.from_numpy(f_kps))
        hand_kps_list.append(torch.from_numpy(h_kps))
        boxes_list.append(torch.from_numpy(bxs))
        region_valid_list.append(torch.from_numpy(r_val))
        ref_crops_list.append(torch.from_numpy(crops))

    bundle = {
        "smplx_detail": smplx_detail,                                 # [70, 103] float32
        "face_keypoints": torch.stack(face_kps_list, dim=0),          # [70, 68, 3] float32
        "hand_keypoints": torch.stack(hand_kps_list, dim=0),          # [70, 2, 21, 3] float32
        "boxes": torch.stack(boxes_list, dim=0),                      # [70, 3, 4] float32
        "region_valid": torch.stack(region_valid_list, dim=0),        # [70, 3] bool
        "reference_crops": torch.stack(ref_crops_list, dim=0),        # [70, 3, 3, 224, 224] float32
        "stem_to_idx": stem_to_idx,
    }

    app_assets.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"[{app}] Successfully generated and saved: {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB)")
    return True


def main() -> None:
    args = parse_args()
    sampled_root = Path(args.sampled_root)
    assets_root = Path(args.assets_root)

    if args.appearance == "all":
        apps = sorted([d.name for d in sampled_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    else:
        apps = [args.appearance]

    success = 0
    for app in apps:
        if process_appearance_detail(app, sampled_root, assets_root, args.overwrite):
            success += 1

    print(f"\nCompleted {success}/{len(apps)} appearances successfully.")


if __name__ == "__main__":
    main()
