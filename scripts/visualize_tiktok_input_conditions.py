#!/usr/bin/env python3
"""Generate a comprehensive visual showcase of all input conditions for TikTok single-sample training.

Visualizes:
1. Source Frame (0014) RGB (Appearance, clothing, identity)
2. Source Face Crop (with ArcFace 512D & DINOv2 256-patch annotation)
3. Target Frame 3D Surface Normal (0074)
4. Target Frame 25-Joint Pose Heatmap (0074)
5. Target Frame 14-Class Body Part Segmentation (0074)
6. Target Frame Foreground Silhouette Mask (0074)
7. Target Frame 72-Point 3D Face Landmarks & ROI Bounding Box
8. Global Condition Metadata & Text Prompt
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tiktok_dataset import TikTokV65FaceOverfitLoader


def generate_input_conditions_figure():
    loader = TikTokV65FaceOverfitLoader(
        raw_root="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
        assets_root="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
        sequence="00001",
        source_stem="0014",
        target_stem="0074",
        resolution=512,
    )
    overfit_inputs = loader.load()
    b = overfit_inputs.batch
    face_cond = overfit_inputs.face_condition
    face_refs = overfit_inputs.face_references

    # 1. Source Image RGB
    src_rgb = ((b["src_image"][0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    # 2. Source Face Crop
    sbox = face_cond.source_boxes[0, 0].numpy()  # [x1, y1, x2, y2] normalized
    H, W = src_rgb.shape[:2]
    sx1, sy1, sx2, sy2 = int(sbox[0] * W), int(sbox[1] * H), int(sbox[2] * W), int(sbox[3] * H)
    src_face = src_rgb[max(0, sy1):min(H, sy2), max(0, sx1):min(W, sx2)]
    src_face = cv2.resize(src_face, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    # Add ArcFace / DINOv2 annotation banner
    cv2.putText(src_face, "ArcFace: 512D Embedding", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(src_face, "DINOv2: 256 Patches x 1536D", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 3. 3D Surface Normal
    normal_rgb = ((b["normal"][0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    # 4. Human Foreground Mask
    mask_gray = (b["human_mask"][0, 0].numpy().clip(0, 1) * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

    # 5. 25-Joint Pose Heatmap
    pose_max = b["pose_heatmap"][0].max(dim=0)[0].numpy().clip(0, 1)
    pose_color = cv2.applyColorMap((pose_max * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    pose_rgb = cv2.cvtColor(pose_color, cv2.COLOR_BGR2RGB)

    # 6. 14-Class Part Segmentation
    part_idx = b["part_onehot"][0].argmax(dim=0).numpy().astype(np.uint8)
    part_vis = ((part_idx.astype(np.float32) / 14.0) * 255).astype(np.uint8)
    part_color = cv2.applyColorMap(part_vis, cv2.COLORMAP_TURBO)
    part_color[part_idx == 0] = [20, 20, 20]
    part_rgb = cv2.cvtColor(part_color, cv2.COLOR_BGR2RGB)

    # 7. 72-Point Face Landmarks & Target Bounding Box
    tgt_rgb = ((b["tgt_image"][0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    lms_canvas = tgt_rgb.copy()
    # Darken background slightly to emphasize landmarks
    lms_canvas = (lms_canvas.astype(np.float32) * 0.45).astype(np.uint8)
    tbox = face_cond.target_boxes[0, 0].numpy()
    tx1, ty1, tx2, ty2 = int(tbox[0] * W), int(tbox[1] * H), int(tbox[2] * W), int(tbox[3] * H)
    cv2.rectangle(lms_canvas, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
    cv2.putText(lms_canvas, "Target Face ROI", (tx1, max(20, ty1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    lms = face_cond.landmarks[0, 0].numpy()  # [72, 3], x,y in [0, 1]
    for i, pt in enumerate(lms):
        px, py = int(pt[0] * W), int(pt[1] * H)
        if 0 <= px < W and 0 <= py < H:
            # Different colors for jaw, brows, eyes, nose, mouth
            if i < 17:
                color = (0, 200, 255)      # Jawline (Cyan)
            elif i < 27:
                color = (0, 255, 100)      # Eyebrows (Green)
            elif i < 36:
                color = (255, 200, 0)      # Nose (Orange)
            elif i < 48:
                color = (255, 80, 80)      # Eyes (Red/Pink)
            else:
                color = (200, 50, 255)     # Lips/Mouth (Purple)
            cv2.circle(lms_canvas, (px, py), 3, color, -1)

    # 8. Create Matplotlib High-Resolution Layout (2x4 Grid)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=200)
    fig.patch.set_facecolor("#16181d")

    panels = [
        (src_rgb, "[Condition 1] Source Image (0014)\nAppearance, Identity & Clothing", "#4cc9f0"),
        (src_face, "[Condition 2] Source Face Identity Features\nArcFace 512D + DINOv2 256-Patch Grid", "#7209b7"),
        (normal_rgb, "[Condition 3] 3D SMPL-X Surface Normal (0074)\nMuscle Contour & Spatial Orientation", "#4895ef"),
        (mask_rgb, "[Condition 4] Human Silhouette Mask (0074)\nForeground / Background Boundary", "#4361ee"),
        (pose_rgb, "[Condition 5] 25 OpenPose/DWPose Heatmap (0074)\n25 Skeletal Joint Probability Distribution", "#f72585"),
        (part_rgb, "[Condition 6] 14-Class Part Segmentation (0074)\nSMPL-X Body Part Semantic Mask", "#b5179e"),
        (lms_canvas, "[Condition 7] 72 Face Landmarks & ROI (0074)\nEyes, Eyebrows, Nose, Lips Geometry", "#f77f00"),
    ]

    for ax, (img, title, color) in zip(axes.flat[:7], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=8)
        ax.axis("off")

    # Panel 8: Parameter Table and Text Conditioning Summary
    ax8 = axes[1, 3]
    ax8.set_facecolor("#1e222b")
    ax8.axis("off")

    smplx_global = b["smplx_global"][0].numpy()
    text_content = (
        "=== [Condition 8] Global Controls ===\n\n"
        "• Text Prompt:\n"
        "  \"Change person pose to match target.\n"
        "   Preserve identity, clothing & bg.\"\n\n"
        "• 26D SMPL-X Global Parameters:\n"
        f"  - Betas (Shape 10D)   : mean={smplx_global[:10].mean():.3f}\n"
        f"  - Rot6D (Orientation) : norm={np.linalg.norm(smplx_global[10:16]):.3f}\n"
        f"  - Translation (3D)    : [{smplx_global[16]:.2f}, {smplx_global[17]:.2f}, {smplx_global[18]:.2f}]\n"
        f"  - Camera Ext/Int (7D) : [{smplx_global[19]:.2f}, {smplx_global[20]:.2f}, ...]\n\n"
        "• Task Identifier: TaskType.SINGLE (0)\n"
        "• Target Face Box: [0.24, 0.20, 0.56, 0.51]\n"
        "• Resolution: 512 x 512 | Device: CUDA (BF16)"
    )

    ax8.text(
        0.05,
        0.92,
        text_content,
        transform=ax8.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        color="#e0e0e0",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#262b36", edgecolor="#3f4756", alpha=0.9),
    )
    ax8.set_title("[Condition 8] Text Prompt & Global Vectors\n26D SMPL-X + Task ID + Camera", fontsize=12, fontweight="bold", color="#a8dadc", pad=8)

    fig.suptitle(
        "TikTok Sequence 00001 (Source 0014 -> Target 0074): Complete Input Conditioning Schema for V6.5 Face Adapter",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    plt.tight_layout()
    out_dir = Path("/home/shangguanrz/project/pic-edit/experiments/tiktok_v65_pure_inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_input_conditions_showcase.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated input conditions showcase: {out_path}")


if __name__ == "__main__":
    generate_input_conditions_figure()
