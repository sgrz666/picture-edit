#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate visual inspection collage for V6.3 preprocessed conditions.
Displays:
[Target RGB | Normal Map | Depth Map | 14-Class Part Map | 25ch Pose Heatmap Max-Proj | Mask]
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path

from src.data.iper_dataset import IPERPoseDataset


# Distinct palette for 14 body parts (0 is black background)
PART_PALETTE = np.array([
    [0, 0, 0],         # 0: bg
    [255, 0, 0],       # 1: head (red)
    [0, 255, 0],       # 2: torso (green)
    [0, 0, 255],       # 3: left_upper_arm (blue)
    [255, 255, 0],     # 4: left_lower_arm (cyan)
    [255, 0, 255],     # 5: left_hand (magenta)
    [0, 255, 255],     # 6: right_upper_arm (yellow)
    [128, 0, 255],     # 7: right_lower_arm (purple)
    [255, 128, 0],     # 8: right_hand (orange)
    [0, 128, 255],     # 9: left_upper_leg (sky blue)
    [128, 255, 0],     # 10: left_lower_leg (lime)
    [0, 255, 128],     # 11: left_foot (spring green)
    [255, 0, 128],     # 12: right_upper_leg (rose)
    [128, 128, 255],   # 13: right_lower_leg (light blue)
    [255, 255, 128],   # 14: right_foot (light yellow)
], dtype=np.uint8)


def main():
    sampled_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt"
    assets_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"
    pairs_jsonl = f"{sampled_root}/splits/train_pairs.jsonl"
    out_dir = Path("/home/shangguanrz/project/pic-edit/experiments/v6_condition_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = IPERPoseDataset(
        pairs_jsonl=pairs_jsonl,
        sampled_root=sampled_root,
        assets_root=assets_root,
        resolution=512,
        augment=False,
    )
    
    # Pick 4 diverse samples across different appearances
    indices = [0, 500, 2000, 5000]
    collages = []
    
    for idx in indices:
        item = dataset[idx]
        app = item["appearance"]
        stem = item["target_stem"]
        
        # Target RGB [0, 255]
        tgt_rgb = ((item["tgt_image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        
        # Normal [0, 255]
        normal_v6 = item["normal"].permute(1, 2, 0).numpy()
        normal_vis = ((normal_v6 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        
        # Depth [0, 255] colormap
        depth_vis = (item["depth"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        # Part Map colored
        part_onehot = item["part_onehot"].numpy() # [14, H, W]
        part_idx = np.argmax(part_onehot, axis=0) + 1 # 1..14
        part_bg = (part_onehot.sum(axis=0) == 0)
        part_idx[part_bg] = 0
        part_colored = PART_PALETTE[part_idx]
        
        # Pose Heatmap Max Projection colored
        heatmap_max = item["pose_heatmap"].max(dim=0)[0].numpy() # [H, W]
        heatmap_vis = (heatmap_max * 255).clip(0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_VIRIDIS)
        
        # Mask
        mask_vis = (item["human_mask"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_color = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
        
        # Row collage: [Target RGB | Normal | Depth | Part Map | Heatmap | Mask]
        row = np.concatenate([tgt_rgb, normal_vis, depth_color, part_colored, heatmap_color, mask_color], axis=1)
        collages.append(row)
        
    full_collage = np.concatenate(collages, axis=0)
    out_file = out_dir / "v6_condition_multimodal_sample.png"
    cv2.imwrite(str(out_file), cv2.cvtColor(full_collage, cv2.COLOR_RGB2BGR))
    print(f"Visual condition collage saved to {out_file} (shape: {full_collage.shape})")


if __name__ == "__main__":
    main()
