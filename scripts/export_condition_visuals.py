#!/usr/bin/env python3
"""Export condition visual assets (DWPose, Normal, Depth, Part Map) for selected showcase samples."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/home/shangguanrz/project/pic-edit")
SHOWCASE_DIR = PROJECT_ROOT / "experiments" / "iper_v6_comparative_3000" / "multistage_evolution_showcase"
SAMPLED_ROOT = PROJECT_ROOT / "datasets" / "iPER" / "iper_sampled_6src64tgt"
ASSETS_ROOT = PROJECT_ROOT / "datasets" / "iPER" / "iper_assets_512_nvdiffrast"

PART_PALETTE = np.array([
    [0, 0, 0],         # 0: bg
    [255, 215, 0],     # 1: head / face (gold)
    [30, 144, 255],    # 2: torso (dodger blue)
    [50, 205, 50],     # 3: left upper arm (lime)
    [34, 139, 34],     # 4: right upper arm (forest green)
    [255, 140, 0],     # 5: left forearm (dark orange)
    [255, 69, 0],      # 6: right forearm (orange red)
    [255, 105, 180],   # 7: left hand (hot pink)
    [199, 21, 133],    # 8: right hand (violet red)
    [138, 43, 226],    # 9: left thigh (blue violet)
    [75, 0, 130],      # 10: right thigh (indigo)
    [0, 206, 209],     # 11: left calf (dark turquoise)
    [0, 139, 139],     # 12: right calf (dark cyan)
    [255, 20, 147],    # 13: left foot (deep pink)
    [255, 0, 0],       # 14: right foot (red)
], dtype=np.uint8)

def main():
    pairs_file = SHOWCASE_DIR / "selected_showcase_pairs.jsonl"
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    for idx, p in enumerate(pairs, 1):
        app = p["appearance"]
        tgt_stem = p["target"]["stem"]
        src_stem = p["source"]["stem"]

        # 1. DWPose
        dw_path = SAMPLED_ROOT / app / "dwpose_512" / "target" / f"{tgt_stem}.png"
        if dw_path.exists():
            dw_img = Image.open(dw_path).convert("RGB")
        else:
            dw_img = Image.new("RGB", (512, 512), (0, 0, 0))
        dw_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_cond_dwpose.png")

        # 2. Normal
        norm_path = ASSETS_ROOT / app / "normal_512" / "target" / f"{tgt_stem}.png"
        if norm_path.exists():
            norm_img = Image.open(norm_path).convert("RGB")
        else:
            norm_img = Image.new("RGB", (512, 512), (128, 128, 255))
        norm_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_cond_normal.png")

        # 3. Depth (render with colormap)
        depth_path = ASSETS_ROOT / app / "depth_npy_512" / "target" / f"{tgt_stem}.npy"
        if depth_path.exists():
            depth_arr = np.load(depth_path).astype(np.float32)
            valid = (depth_arr > 0) & np.isfinite(depth_arr)
            if valid.any():
                d_min, d_max = depth_arr[valid].min(), depth_arr[valid].max()
                d_norm = np.clip((depth_arr - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
                # Invert so near is warm/bright, far is cool/dark
                d_u8 = ((1.0 - d_norm) * 255).astype(np.uint8)
                d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
                d_color[~valid] = [0, 0, 0]
                depth_img = Image.fromarray(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB))
            else:
                depth_img = Image.new("RGB", (512, 512), (0, 0, 0))
        else:
            depth_img = Image.new("RGB", (512, 512), (0, 0, 0))
        depth_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_cond_depth.png")

        # 4. Part Map (14-color palette)
        part_path = ASSETS_ROOT / app / "part_512" / "target" / f"{tgt_stem}.png"
        if part_path.exists():
            part_arr = np.array(Image.open(part_path))
            # Clip indices to palette range
            part_arr = np.clip(part_arr, 0, len(PART_PALETTE) - 1)
            part_rgb = PART_PALETTE[part_arr]
            part_img = Image.fromarray(part_rgb)
        else:
            part_img = Image.new("RGB", (512, 512), (0, 0, 0))
        part_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_cond_part.png")

        # 5. Source Image
        src_path = SAMPLED_ROOT / app / p["source"]["rgb_1024"]
        if not src_path.exists():
            src_path = SAMPLED_ROOT / app / "rgb_1024" / p["source"]["role"] / f"{src_stem}.png"
        if src_path.exists():
            src_img = Image.open(src_path).convert("RGB")
        else:
            src_img = Image.new("RGB", (512, 512), (0, 0, 0))
        src_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_source.png")

        # 6. Target GT Image
        tgt_path = SAMPLED_ROOT / app / p["target"]["rgb_1024"]
        if not tgt_path.exists():
            tgt_path = SAMPLED_ROOT / app / "rgb_1024" / "target" / f"{tgt_stem}.png"
        if tgt_path.exists():
            tgt_img = Image.open(tgt_path).convert("RGB")
        else:
            tgt_img = Image.new("RGB", (512, 512), (0, 0, 0))
        tgt_img.save(SHOWCASE_DIR / f"sample_{idx}_{app}_target_gt.png")

        print(f"Exported condition visuals for sample {idx} ({app}: {tgt_stem})")

if __name__ == "__main__":
    main()
