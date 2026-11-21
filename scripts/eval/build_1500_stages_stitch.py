#!/usr/bin/env python3
"""Evaluation, pure inference, and multi-stage stitching script for TikTok 1500-step training.

Stitches together:
1. All 16 progressive stages (Step 0 to 1500) + Source + Target GT into a timeline grid
2. Facial ROI progression across stages
3. Hand ROI progression across stages
4. 1500-step training loss curves
5. Pure 28-step diffusion inference panels (Fullbody, Face, Hands)
6. Quantitative Scorecard table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tiktok_dataset import TikTokV65FaceOverfitLoader
from src.pose_control.v6.checkpoint import load_v65_checkpoint
from src.pose_control.v6.conditions import AdapterIdentityCondition
from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from train_tiktok_v65_face import get_cached_conditions


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = (
        ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "msyhbd.ttc", "simhei.ttf"]
        if bold
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf", "msyh.ttc", "simsun.ttc"]
    )
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label_banner(
    img: Image.Image,
    title: str,
    subtitle: str | None = None,
    font_size: int = 16,
    bg_color: Tuple[int, int, int, int] = (15, 23, 42, 230),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    f_title = get_font(font_size, bold=True)
    f_sub = get_font(max(10, font_size - 4), bold=False)

    tb = draw.textbbox((0, 0), title, font=f_title)
    th = tb[3] - tb[1]
    banner_h = th + 14
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=f_sub)
        banner_h += (sb[3] - sb[1]) + 6

    banner = Image.new("RGBA", (img.width, banner_h), bg_color)
    img.paste(banner, (0, 0), banner)
    d = ImageDraw.Draw(img)
    tw = tb[2] - tb[0]
    d.text(((img.width - tw) // 2, 6), title, font=f_title, fill=text_color)
    if subtitle:
        sw = sb[2] - sb[0]
        d.text(((img.width - sw) // 2, th + 10), subtitle, font=f_sub, fill=(203, 213, 225))
    return img.convert("RGB")


def compute_image_metrics(pred_pil: Image.Image, gt_pil: Image.Image) -> dict[str, float]:
    pred = np.array(pred_pil.convert("RGB")).astype(np.float32) / 255.0
    gt = np.array(gt_pil.convert("RGB")).astype(np.float32) / 255.0
    if pred.shape != gt.shape:
        gt = np.array(gt_pil.convert("RGB").resize((pred.shape[1], pred.shape[0]))).astype(np.float32) / 255.0

    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-10)))
    l1 = float(np.mean(np.abs(pred - gt)))

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = np.var(pred)
    sigma_y = np.var(gt)
    sigma_xy = np.mean((pred - mu_x) * (gt - mu_y))
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))
    ssim = max(0.0, min(1.0, ssim))

    return {"mse": mse, "psnr": psnr, "l1": l1, "ssim": ssim}


def get_hand_crop_box(kps: torch.Tensor, img_size: int = 512, margin: float = 0.35) -> Tuple[int, int, int, int]:
    if kps.ndim > 2:
        kps = kps.view(-1, 3)
    valid = kps[:, 2] > 0.05
    if int(valid.sum()) < 2:
        return (0, 0, img_size, img_size)
    pts = kps[valid, :2].detach().cpu().numpy()
    min_x, min_y = float(pts[:, 0].min()), float(pts[:, 1].min())
    max_x, max_y = float(pts[:, 0].max()), float(pts[:, 1].max())

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half_side = max(max_x - min_x, max_y - min_y) * (0.5 + margin) + 0.02

    bx1 = max(0.0, cx - half_side)
    by1 = max(0.0, cy - half_side)
    bx2 = min(1.0, cx + half_side)
    by2 = min(1.0, cy + half_side)

    return (int(bx1 * img_size), int(by1 * img_size), int(bx2 * img_size), int(by2 * img_size))


def plot_loss_curves_1500(metrics_path: Path, out_path: Path) -> None:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", [])
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    totals = [m["total"] for m in metrics]
    flows = [m["flow"] for m in metrics]
    outsides = [m.get("outside", 0.0) for m in metrics]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=220)

    # Subplot 1: Total Loss & Flow Loss
    ax1 = axes[0]
    ax1.plot(steps, totals, label="Total Loss", color="#1E3A8A", linewidth=2.0)
    ax1.plot(steps, flows, label="Flow Matching Loss", color="#0284C7", linewidth=1.6, linestyle="--")
    ax1.set_title("TikTok 00001: 1500-Step Deep Overfit Convergence", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Loss Value", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    reduction = ((totals[0] - totals[-1]) / totals[0]) * 100.0
    ax1.annotate(
        f"Initial: {totals[0]:.4f}",
        xy=(steps[0], totals[0]),
        xytext=(steps[0] + 50, totals[0] + 0.03),
        arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.2),
        fontweight="bold",
        fontsize=10,
    )
    ax1.annotate(
        f"Step 1500: {totals[-1]:.4f}\n(-{reduction:.1f}%)",
        xy=(steps[-1], totals[-1]),
        xytext=(steps[-1] - 400, totals[-1] + 0.08),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
        fontweight="bold",
        color="#059669",
        fontsize=10,
    )

    # Subplot 2: Outside Area Regularization
    ax2 = axes[1]
    ax2.plot(steps, outsides, label="Outside Area Regularization", color="#D97706", linewidth=1.8)
    ax2.set_title("Outside Face-Mask Regularization Dynamics (1500 Steps)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Regularization Value", fontsize=12)
    ax2.legend(loc="lower right", frameon=True, fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Full Adapter Single-Sample Overfit Dynamics across 1500 Steps (Sequence 00001, Frame 0014 -> 0074)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved: {out_path}")


def stitch_all_stages(
    previews_dir: Path,
    out_path: Path,
    src_pil: Image.Image,
    tgt_pil: Image.Image,
) -> None:
    """Stitch all progressive stages:
    Row 1: [Source 0014] -> Step 0 -> Step 100 -> Step 200 -> Step 300 -> Step 400 -> Step 500 -> Step 600 -> Step 700
    Row 2: Step 800 -> Step 900 -> Step 1000 -> Step 1100 -> Step 1200 -> Step 1300 -> Step 1400 -> Step 1500 -> [Target GT 0074]
    """
    row1_steps = [0, 100, 200, 300, 400, 500, 600, 700]
    row2_steps = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

    cell_size = 320
    pad = 10
    cols = 9
    rows = 2
    header_h = 75

    canvas_w = cols * cell_size + (cols + 1) * pad
    canvas_h = header_h + rows * cell_size + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    f_main = get_font(24, bold=True)
    f_sub = get_font(13, bold=False)
    title_text = "TikTok 00001: Complete 1500-Step Progressive Evolution Timeline (All Stages Stitched)"
    sub_text = "Tracking Model Generation from Initial Pose Condition (Step 0) Through 1500 Iterations to Target Ground Truth"
    tb = draw.textbbox((0, 0), title_text, font=f_main)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((canvas_w - (tb[2] - tb[0])) // 2, 16), title_text, font=f_main, fill=(15, 23, 42))
    draw.text(((canvas_w - (sb[2] - sb[0])) // 2, 48), sub_text, font=f_sub, fill=(71, 85, 105))

    # Helper to load preview
    def load_step_img(s: int) -> Image.Image:
        p = previews_dir / f"step_{s:06d}_face_on.png"
        if p.exists():
            return Image.open(p).convert("RGB").resize((cell_size, cell_size))
        return Image.new("RGB", (cell_size, cell_size), (200, 200, 200))

    # Assemble Row 1: Source + Steps 0..700
    row1_items = [(src_pil.resize((cell_size, cell_size)), "Source (0014)", "Input Identity", (30, 41, 59, 235))]
    for s in row1_steps:
        lbl = "Initial (Step 0)" if s == 0 else f"Step {s}"
        row1_items.append((load_step_img(s), lbl, f"Stage {s//100}", (15, 23, 42, 220)))

    # Assemble Row 2: Steps 800..1500 + Target GT
    row2_items = []
    for s in row2_steps:
        row2_items.append((load_step_img(s), f"Step {s}", f"Stage {s//100}", (15, 23, 42, 220)))
    row2_items.append((tgt_pil.resize((cell_size, cell_size)), "Target GT (0074)", "Ground Truth Target", (5, 150, 105, 235)))

    # Paste Row 1
    y1 = header_h + pad
    for c, (img, t, sub, bg) in enumerate(row1_items):
        x = pad + c * (cell_size + pad)
        labeled = add_label_banner(img, t, sub, font_size=15, bg_color=bg)
        canvas.paste(labeled, (x, y1))

    # Paste Row 2
    y2 = header_h + pad + cell_size + pad
    for c, (img, t, sub, bg) in enumerate(row2_items):
        x = pad + c * (cell_size + pad)
        labeled = add_label_banner(img, t, sub, font_size=15, bg_color=bg)
        canvas.paste(labeled, (x, y2))

    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def stitch_face_roi_stages(
    previews_dir: Path,
    out_path: Path,
    src_pil: Image.Image,
    tgt_pil: Image.Image,
) -> None:
    """Stitch Facial ROI close-up across all 16 stages (Steps 0, 100, ..., 1500) + Source + GT."""
    steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    crop_tgt = (115, 95, 295, 275)
    crop_src = (120, 100, 300, 280)
    z = 240
    pad = 8
    cols = 9
    rows = 2
    header_h = 70

    canvas_w = cols * z + (cols + 1) * pad
    canvas_h = header_h + rows * z + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    f_main = get_font(22, bold=True)
    f_sub = get_font(13, bold=False)
    title_text = "Facial ROI Multi-Stage Evolution (Steps 0 - 1500 vs Source and Target GT)"
    sub_text = "Track Face Detail, Expression, Hair Texture, and Identity Convergence Across 1500 Training Iterations"
    tb = draw.textbbox((0, 0), title_text, font=f_main)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((canvas_w - (tb[2] - tb[0])) // 2, 14), title_text, font=f_main, fill=(15, 23, 42))
    draw.text(((canvas_w - (sb[2] - sb[0])) // 2, 44), sub_text, font=f_sub, fill=(71, 85, 105))

    def crop_step_face(s: int) -> Image.Image:
        p = previews_dir / f"step_{s:06d}_face_on.png"
        if p.exists():
            return Image.open(p).convert("RGB").resize((512, 512)).crop(crop_tgt).resize((z, z), Image.Resampling.LANCZOS)
        return Image.new("RGB", (z, z), (200, 200, 200))

    items_r1 = [(src_pil.resize((512, 512)).crop(crop_src).resize((z, z)), "Source", "0014", (30, 41, 59, 235))]
    for s in steps[:8]:
        lbl = "Initial" if s == 0 else f"Step {s}"
        items_r1.append((crop_step_face(s), lbl, f"{s} iters", (15, 23, 42, 220)))

    items_r2 = []
    for s in steps[8:]:
        items_r2.append((crop_step_face(s), f"Step {s}", f"{s} iters", (15, 23, 42, 220)))
    items_r2.append((tgt_pil.resize((512, 512)).crop(crop_tgt).resize((z, z)), "Target GT", "0074 GT", (5, 150, 105, 235)))

    y1 = header_h + pad
    for c, (img, t, sub, bg) in enumerate(items_r1):
        x = pad + c * (z + pad)
        canvas.paste(add_label_banner(img, t, sub, font_size=14, bg_color=bg), (x, y1))

    y2 = header_h + pad + z + pad
    for c, (img, t, sub, bg) in enumerate(items_r2):
        x = pad + c * (z + pad)
        canvas.paste(add_label_banner(img, t, sub, font_size=14, bg_color=bg), (x, y2))

    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def stitch_hand_roi_stages(
    previews_dir: Path,
    out_path: Path,
    src_pil: Image.Image,
    tgt_pil: Image.Image,
    lh_box: Tuple[int, int, int, int],
    rh_box: Tuple[int, int, int, int],
) -> None:
    """Stitch Left & Right Hand ROIs across all 16 stages (Steps 0, 100, ..., 1500) + Source + GT."""
    steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    z = 240
    pad = 8
    cols = 9
    header_h = 75
    section_h = 36

    canvas_w = cols * z + (cols + 1) * pad
    canvas_h = header_h + 2 * (section_h + 2 * z + 3 * pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    f_main = get_font(22, bold=True)
    f_sub = get_font(13, bold=False)
    f_section = get_font(16, bold=True)
    title_text = "Hand ROI Multi-Stage Progressive Evolution (Steps 0 - 1500 vs Source & Target GT)"
    sub_text = "Tracking Left Hand & Right Hand Detail, Palm Alignment, and Finger Convergence across 1500 Iterations"
    tb = draw.textbbox((0, 0), title_text, font=f_main)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((canvas_w - (tb[2] - tb[0])) // 2, 14), title_text, font=f_main, fill=(15, 23, 42))
    draw.text(((canvas_w - (sb[2] - sb[0])) // 2, 44), sub_text, font=f_sub, fill=(71, 85, 105))

    def crop_step_hand(s: int, box: Tuple[int, int, int, int]) -> Image.Image:
        p = previews_dir / f"step_{s:06d}_face_on.png"
        if p.exists():
            return Image.open(p).convert("RGB").resize((512, 512)).crop(box).resize((z, z), Image.Resampling.LANCZOS)
        return Image.new("RGB", (z, z), (200, 200, 200))

    cur_y = header_h + pad
    for hand_name, hand_box in [("Left Hand", lh_box), ("Right Hand", rh_box)]:
        draw.text((pad + 10, cur_y + 6), f"• {hand_name} (Target Crop Box: {hand_box})", font=f_section, fill=(30, 41, 59))
        cur_y += section_h

        items_r1 = [(src_pil.resize((512, 512)).crop(hand_box).resize((z, z)), "Source", "0014", (30, 41, 59, 235))]
        for s in steps[:8]:
            lbl = "Initial" if s == 0 else f"Step {s}"
            items_r1.append((crop_step_hand(s, hand_box), lbl, f"{s} iters", (15, 23, 42, 220)))

        items_r2 = []
        for s in steps[8:]:
            items_r2.append((crop_step_hand(s, hand_box), f"Step {s}", f"{s} iters", (15, 23, 42, 220)))
        items_r2.append((tgt_pil.resize((512, 512)).crop(hand_box).resize((z, z)), "Target GT", "0074 GT", (5, 150, 105, 235)))

        for c, (img, t, sub, bg) in enumerate(items_r1):
            x = pad + c * (z + pad)
            canvas.paste(add_label_banner(img, t, sub, font_size=14, bg_color=bg), (x, cur_y))
        cur_y += z + pad

        for c, (img, t, sub, bg) in enumerate(items_r2):
            x = pad + c * (z + pad)
            canvas.paste(add_label_banner(img, t, sub, font_size=14, bg_color=bg), (x, cur_y))
        cur_y += z + pad * 2

    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def render_scorecard_1500(quant_results: dict[str, Any], out_path: Path) -> None:
    b = quant_results["full_body"]
    face = quant_results["face_roi"]
    lh = quant_results["left_hand_roi"]
    rh = quant_results["right_hand_roi"]

    def fmt_delta(val, unit="", invert=False):
        sign = "+" if val > 0 else ""
        text = f"{sign}{val:.2f}{unit}" if abs(val) >= 0.01 else f"{sign}{val:.4f}{unit}"
        is_pos = (val > 0) if not invert else (val < 0)
        color = (5, 150, 105) if is_pos else ((220, 38, 38) if abs(val) > 0.05 else (71, 85, 105))
        return text, color

    rows = [
        ("Full Body (512x512)", "PSNR (dB)", f"{b['baseline']['psnr']:.2f}", f"{b['full_adapter']['psnr']:.2f}", fmt_delta(b['full_adapter']['psnr'] - b['baseline']['psnr'], " dB")),
        ("Full Body (512x512)", "SSIM", f"{b['baseline']['ssim']:.4f}", f"{b['full_adapter']['ssim']:.4f}", fmt_delta(b['full_adapter']['ssim'] - b['baseline']['ssim'])),
        ("Face ROI (280x280)", "PSNR (dB)", f"{face['baseline']['psnr']:.2f}", f"{face['full_adapter']['psnr']:.2f}", fmt_delta(face['full_adapter']['psnr'] - face['baseline']['psnr'], " dB")),
        ("Face ROI (280x280)", "L1 Error", f"{face['baseline']['l1']:.4f}", f"{face['full_adapter']['l1']:.4f}", fmt_delta(face['full_adapter']['l1'] - face['baseline']['l1'], invert=True)),
        ("Left Hand ROI", "PSNR (dB)", f"{lh['baseline']['psnr']:.2f}", f"{lh['full_adapter']['psnr']:.2f}", fmt_delta(lh['full_adapter']['psnr'] - lh['baseline']['psnr'], " dB")),
        ("Left Hand ROI", "L1 Error", f"{lh['baseline']['l1']:.4f}", f"{lh['full_adapter']['l1']:.4f}", fmt_delta(lh['full_adapter']['l1'] - lh['baseline']['l1'], invert=True)),
        ("Right Hand ROI", "PSNR (dB)", f"{rh['baseline']['psnr']:.2f}", f"{rh['full_adapter']['psnr']:.2f}", fmt_delta(rh['full_adapter']['psnr'] - rh['baseline']['psnr'], " dB")),
        ("Right Hand ROI", "L1 Error", f"{rh['baseline']['l1']:.4f}", f"{rh['full_adapter']['l1']:.4f}", fmt_delta(rh['full_adapter']['l1'] - rh['baseline']['l1'], invert=True)),
    ]

    width, height = 1300, 720
    canvas = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    f_title = get_font(24, bold=True)
    f_sub = get_font(13, bold=False)
    f_th = get_font(15, bold=True)
    f_td = get_font(14, bold=False)
    f_td_bold = get_font(14, bold=True)

    title_text = "Quantitative Evaluation Scorecard: TikTok Sample 00001 (1500 Steps Full Adapter Overfit)"
    sub_text = "Pure 28-Step Generative Diffusion (Zero GT Leakage) - Baseline vs 1500-Step Trained Model"
    tb = draw.textbbox((0, 0), title_text, font=f_title)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((width - (tb[2] - tb[0])) // 2, 22), title_text, font=f_title, fill=(15, 23, 42))
    draw.text(((width - (sb[2] - sb[0])) // 2, 58), sub_text, font=f_sub, fill=(71, 85, 105))

    table_x = 50
    table_y = 100
    table_w = width - 100
    col_widths = [320, 180, 240, 240, 220]
    headers = ["Evaluation Region", "Metric", "Baseline (Face OFF)", "Full Adapter 1500 Steps", "Delta / Improvement"]

    header_h = 55
    draw.rectangle([table_x, table_y, table_x + table_w, table_y + header_h], fill=(15, 23, 42))
    cur_x = table_x
    for i, h in enumerate(headers):
        w = col_widths[i]
        hb = draw.textbbox((0, 0), h, font=f_th)
        draw.text((cur_x + (w - (hb[2] - hb[0])) // 2, table_y + (header_h - (hb[3] - hb[1])) // 2), h, font=f_th, fill=(255, 255, 255))
        cur_x += w

    row_y = table_y + header_h
    row_h = 56
    for idx, r in enumerate(rows):
        bg = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
        draw.rectangle([table_x, row_y, table_x + table_w, row_y + row_h], fill=bg)
        draw.line([table_x, row_y + row_h, table_x + table_w, row_y + row_h], fill=(226, 232, 240), width=1)

        cur_x = table_x
        w0 = col_widths[0]
        b0 = draw.textbbox((0, 0), r[0], font=f_td_bold)
        draw.text((cur_x + 25, row_y + (row_h - (b0[3] - b0[1])) // 2), r[0], font=f_td_bold, fill=(30, 41, 59))
        cur_x += w0

        w1 = col_widths[1]
        b1 = draw.textbbox((0, 0), r[1], font=f_td)
        draw.text((cur_x + (w1 - (b1[2] - b1[0])) // 2, row_y + (row_h - (b1[3] - b1[1])) // 2), r[1], font=f_td, fill=(51, 65, 85))
        cur_x += w1

        w2 = col_widths[2]
        b2 = draw.textbbox((0, 0), r[2], font=f_td)
        draw.text((cur_x + (w2 - (b2[2] - b2[0])) // 2, row_y + (row_h - (b2[3] - b2[1])) // 2), r[2], font=f_td, fill=(15, 23, 42))
        cur_x += w2

        w3 = col_widths[3]
        b3 = draw.textbbox((0, 0), r[3], font=f_td_bold)
        draw.text((cur_x + (w3 - (b3[2] - b3[0])) // 2, row_y + (row_h - (b3[3] - b3[1])) // 2), r[3], font=f_td_bold, fill=(15, 23, 42))
        cur_x += w3

        w4 = col_widths[4]
        d_text, d_color = r[4]
        b4 = draw.textbbox((0, 0), d_text, font=f_td_bold)
        draw.text((cur_x + (w4 - (b4[2] - b4[0])) // 2, row_y + (row_h - (b4[3] - b4[1])) // 2), d_text, font=f_td_bold, fill=d_color)

        row_y += row_h

    draw.rectangle([table_x, table_y, table_x + table_w, row_y], outline=(203, 213, 225), width=2)
    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="00001")
    parser.add_argument("--source-stem", default="0014")
    parser.add_argument("--target-stem", default="0074")
    parser.add_argument("--raw-root", default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset")
    parser.add_argument("--assets-root", default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native")
    parser.add_argument("--model-path", default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
    parser.add_argument("--training-dir", default="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_1500")
    parser.add_argument("--output-dir", default="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_1500/final_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    training_dir = Path(args.training_dir)
    if not training_dir.is_absolute():
        training_dir = (PROJECT_ROOT / training_dir).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STARTING 1500-STEP COMPREHENSIVE EVALUATION & MULTI-STAGE STITCHING")
    print(f"Training Directory: {training_dir}")
    print(f"Output Directory  : {out_dir}")
    print("=" * 72)

    # 1. Plot Loss Curves
    print("\n[1/6] Plotting 1500-step training dynamics...", flush=True)
    metrics_json = training_dir / "metrics.json"
    plot_loss_curves_1500(metrics_json, out_dir / "01_full_adapter_1500_loss_curves.png")

    # 2. Load TikTok Data
    print("\n[2/6] Loading TikTok input pair...", flush=True)
    loader = TikTokV65FaceOverfitLoader(
        raw_root=args.raw_root,
        assets_root=args.assets_root,
        sequence=args.sequence,
        source_stem=args.source_stem,
        target_stem=args.target_stem,
        resolution=512,
    )
    overfit_inputs = loader.load()
    batch = overfit_inputs.batch

    src_raw = (batch["src_image"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    src_pil = Image.fromarray((src_raw * 255.0 + 0.5).astype(np.uint8))
    tgt_raw = (batch["tgt_image"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    tgt_pil = Image.fromarray((tgt_raw * 255.0 + 0.5).astype(np.uint8))
    norm_raw = (batch["normal"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    norm_pil = Image.fromarray((norm_raw * 255.0 + 0.5).astype(np.uint8))

    src_pil.save(out_dir / "input_source_0014.png")
    tgt_pil.save(out_dir / "target_gt_reference_0074.png")
    norm_pil.save(out_dir / "input_target_normal_0074.png")

    # 3. Stitch All Progressive Stages
    print("\n[3/6] Stitching all 16 progressive stages into timeline grids...", flush=True)
    kps_hand = overfit_inputs.hand_detail_condition.hand_keypoints[0, 0]
    left_hand_box = get_hand_crop_box(kps_hand[0], 512, margin=0.35)
    right_hand_box = get_hand_crop_box(kps_hand[1], 512, margin=0.35)

    stitch_all_stages(training_dir / "previews", out_dir / "02_all_stages_stitched_evolution.png", src_pil, tgt_pil)
    stitch_face_roi_stages(training_dir / "previews", out_dir / "03_all_stages_face_roi_evolution.png", src_pil, tgt_pil)
    stitch_hand_roi_stages(training_dir / "previews", out_dir / "03b_all_stages_hand_roi_evolution.png", src_pil, tgt_pil, left_hand_box, right_hand_box)

    # 4. Pure Diffusion Inference from Gaussian noise
    print("\n[4/6] Executing pure 28-step diffusion inference from Gaussian noise...", flush=True)
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe, condition_use_depth=False).to(device=device, dtype=dtype)

    ckpt_path = training_dir / "checkpoints" / "checkpoint_step_001500.pt"
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_v65_checkpoint(adapter, checkpoint_data)
    adapter.eval()

    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)

    cache_dir = out_dir / "condition_cache"
    prompt = "Change the person's pose to match the target pose. Preserve identity, face, hair, clothing, hands, body proportions, lighting, and background."
    ref_latent, _, _ = get_cached_conditions(pipe, batch, prompt, str(cache_dir), device, dtype)

    normal = batch["normal"].to(device, dtype=dtype)
    pose = batch["pose_heatmap"].to(device, dtype=dtype)
    part = batch["part_onehot"].to(device, dtype=dtype)
    smplx_global = batch["smplx_global"].to(device, dtype=dtype)
    human_mask = batch["human_mask"].to(device, dtype=dtype)
    task_id = batch["task_id"].to(device, dtype=torch.long)

    geometry_bundle = adapter.condition_injector(
        normal_a=normal,
        pose_heatmap_a=pose,
        part_onehot_a=part,
        smplx_global_a=smplx_global,
        human_mask_a=human_mask,
        task_id=task_id,
    )
    source_person_latents = torch.zeros(
        1, 2, ref_latent.shape[1], ref_latent.shape[-2], ref_latent.shape[-1], device=device, dtype=dtype
    )
    source_person_latents[:, 0] = ref_latent[:1]
    identity_condition = AdapterIdentityCondition(
        source_person_latents=source_person_latents,
        source_indices=torch.tensor([[0, 1]], device=device),
    )

    detail_condition = overfit_inputs.hand_detail_condition.to(device=device, dtype=dtype)
    detail_references = overfit_inputs.hand_detail_references.to(device=device, dtype=dtype)
    face_condition = overfit_inputs.face_condition.to(device=device, dtype=dtype)
    face_references = overfit_inputs.face_references.to(device=device, dtype=dtype)

    # Baseline (Face OFF)
    print("  -> Generating Baseline: Face OFF...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        res_off = controlled(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            hand_condition=detail_condition,
            hand_references=detail_references,
            face_condition=face_condition,
            face_references=face_references,
            prompt=prompt,
            negative_prompt="",
            image=src_pil,
            height=512,
            width=512,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=0.0,
        )
    img_off = res_off.images[0]
    img_off.save(out_dir / "pure_inference_face_off.png")
    print(f"     Done in {time.time() - t0:.2f}s")

    # Full Adapter (Face ON, 1500 Steps)
    print("  -> Generating Full Adapter Trained (1500 Steps, Face ON)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        res_on = controlled(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            hand_condition=detail_condition,
            hand_references=detail_references,
            face_condition=face_condition,
            face_references=face_references,
            prompt=prompt,
            negative_prompt="",
            image=src_pil,
            height=512,
            width=512,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=1.0,
        )
    img_on = res_on.images[0]
    img_on.save(out_dir / "pure_inference_face_on.png")
    print(f"     Done in {time.time() - t0:.2f}s")

    # 5. Extract ROIs & Metrics
    print("\n[5/6] Extracting ROIs and computing metrics...", flush=True)
    metrics_body_off = compute_image_metrics(img_off, tgt_pil)
    metrics_body_on = compute_image_metrics(img_on, tgt_pil)

    crop_box_tgt = (115, 95, 295, 275)
    crop_box_src = (120, 100, 300, 280)
    z = 320
    face_src = src_pil.crop(crop_box_src).resize((z, z), Image.Resampling.LANCZOS)
    face_tgt = tgt_pil.crop(crop_box_tgt).resize((z, z), Image.Resampling.LANCZOS)
    face_off = img_off.crop(crop_box_tgt).resize((z, z), Image.Resampling.LANCZOS)
    face_on = img_on.crop(crop_box_tgt).resize((z, z), Image.Resampling.LANCZOS)

    metrics_face_off = compute_image_metrics(face_off, face_tgt)
    metrics_face_on = compute_image_metrics(face_on, face_tgt)

    kps_hand = overfit_inputs.hand_detail_condition.hand_keypoints[0, 0]
    left_hand_box = get_hand_crop_box(kps_hand[0], 512, margin=0.35)
    right_hand_box = get_hand_crop_box(kps_hand[1], 512, margin=0.35)

    lh_src = src_pil.crop(left_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    lh_tgt = tgt_pil.crop(left_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    lh_off = img_off.crop(left_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    lh_on = img_on.crop(left_hand_box).resize((z, z), Image.Resampling.LANCZOS)

    rh_src = src_pil.crop(right_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    rh_tgt = tgt_pil.crop(right_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    rh_off = img_off.crop(right_hand_box).resize((z, z), Image.Resampling.LANCZOS)
    rh_on = img_on.crop(right_hand_box).resize((z, z), Image.Resampling.LANCZOS)

    metrics_lh_off = compute_image_metrics(lh_off, lh_tgt)
    metrics_lh_on = compute_image_metrics(lh_on, lh_tgt)
    metrics_rh_off = compute_image_metrics(rh_off, rh_tgt)
    metrics_rh_on = compute_image_metrics(rh_on, rh_tgt)

    quant_results = {
        "full_body": {"baseline": metrics_body_off, "full_adapter": metrics_body_on},
        "face_roi": {"baseline": metrics_face_off, "full_adapter": metrics_face_on},
        "left_hand_roi": {"baseline": metrics_lh_off, "full_adapter": metrics_lh_on},
        "right_hand_roi": {"baseline": metrics_rh_off, "full_adapter": metrics_rh_on},
    }
    with open(out_dir / "quantitative_metrics_1500.json", "w", encoding="utf-8") as f:
        json.dump(quant_results, f, indent=2)

    # 6. Render Panels
    print("\n[6/6] Generating composite panels and scorecard...", flush=True)

    # Panel 4: Fullbody comparison panel
    fullbody_panels = [
        (src_pil.resize((512, 512)), "Source Frame (0014)", "Input Identity"),
        (norm_pil.resize((512, 512)), "Target 3D Normal (0074)", "Pose & Shape Guide"),
        (tgt_pil.resize((512, 512)), "Target GT Reference", "Ground Truth Target"),
        (img_off.resize((512, 512)), "Baseline: Face OFF", f"PSNR: {metrics_body_off['psnr']:.2f} dB"),
        (img_on.resize((512, 512)), "Full Adapter: 1500 Steps", f"PSNR: {metrics_body_on['psnr']:.2f} dB (Trained)"),
    ]
    labeled_full = [add_label_banner(img, t, s, font_size=18) for img, t, s in fullbody_panels]
    total_w = sum(img.width for img in labeled_full) + (len(labeled_full) - 1) * 12 + 40
    total_h = 512 + 100
    canvas_fb = Image.new("RGB", (total_w, total_h), (241, 245, 249))
    draw_fb = ImageDraw.Draw(canvas_fb)
    f_main = get_font(24, bold=True)
    f_sub = get_font(13, bold=False)
    t_text = "TikTok 00001: Pure 28-Step Generative Diffusion After 1500 Steps Full Adapter Overfit"
    s_text = "Comparing Input Source, Target Normal, Ground Truth, Baseline (Face OFF), and Full Adapter Trained (Face ON)"
    tb = draw_fb.textbbox((0, 0), t_text, font=f_main)
    sb = draw_fb.textbbox((0, 0), s_text, font=f_sub)
    draw_fb.text(((total_w - (tb[2] - tb[0])) // 2, 16), t_text, font=f_main, fill=(15, 23, 42))
    draw_fb.text(((total_w - (sb[2] - sb[0])) // 2, 50), s_text, font=f_sub, fill=(71, 85, 105))
    x_off = 20
    for img in labeled_full:
        canvas_fb.paste(img, (x_off, 80))
        x_off += img.width + 12
    canvas_fb.save(out_dir / "04_genuine_inference_fullbody_1500.png", quality=95)

    # Panel 5: Face Zoom
    face_items = [
        (face_src, "Source Frame (0014)", "Input Identity"),
        (face_tgt, "Target GT Reference", "Ground Truth Target"),
        (face_off, "Baseline: Face OFF", f"PSNR: {metrics_face_off['psnr']:.2f} dB | SSIM: {metrics_face_off['ssim']:.4f}"),
        (face_on, "Full Adapter: 1500 Steps", f"PSNR: {metrics_face_on['psnr']:.2f} dB | SSIM: {metrics_face_on['ssim']:.4f}"),
    ]
    labeled_face = [add_label_banner(img, t, s, font_size=16) for img, t, s in face_items]
    w_face = len(labeled_face) * z + (len(labeled_face) - 1) * 12 + 40
    h_face = z + 95
    canvas_face = Image.new("RGB", (w_face, h_face), (241, 245, 249))
    draw_face = ImageDraw.Draw(canvas_face)
    t_text = "Facial ROI Zoom-In Comparison (1500-Step Full Adapter Overfit)"
    s_text = "Evaluated on cropped facial bounding box [115, 95, 295, 275]"
    tb = draw_face.textbbox((0, 0), t_text, font=f_main)
    sb = draw_face.textbbox((0, 0), s_text, font=f_sub)
    draw_face.text(((w_face - (tb[2] - tb[0])) // 2, 14), t_text, font=f_main, fill=(15, 23, 42))
    draw_face.text(((w_face - (sb[2] - sb[0])) // 2, 45), s_text, font=f_sub, fill=(71, 85, 105))
    x_off = 20
    for img in labeled_face:
        canvas_face.paste(img, (x_off, 72))
        x_off += img.width + 12
    canvas_face.save(out_dir / "05_genuine_inference_face_roi_1500.png", quality=95)

    # Panel 6: Hand Zoom
    hand_rows = [
        ("Left Hand (Target Box: [127, 414, 257, 512])", [
            (lh_src, "Source LH", "Input Frame 0014"),
            (lh_tgt, "Target GT LH", "Ground Truth Reference"),
            (lh_off, "Baseline LH", f"PSNR: {metrics_lh_off['psnr']:.2f} dB | L1: {metrics_lh_off['l1']:.4f}"),
            (lh_on, "Full Adapter 1500 Steps", f"PSNR: {metrics_lh_on['psnr']:.2f} dB | L1: {metrics_lh_on['l1']:.4f}"),
        ]),
        ("Right Hand (Target Box: [18, 70, 151, 202])", [
            (rh_src, "Source RH", "Input Frame 0014"),
            (rh_tgt, "Target GT RH", "Ground Truth Reference"),
            (rh_off, "Baseline RH", f"PSNR: {metrics_rh_off['psnr']:.2f} dB | L1: {metrics_rh_off['l1']:.4f}"),
            (rh_on, "Full Adapter 1500 Steps", f"PSNR: {metrics_rh_on['psnr']:.2f} dB | L1: {metrics_rh_on['l1']:.4f}"),
        ]),
    ]
    total_hw = 4 * z + 3 * 12 + 40
    total_hh = 2 * z + 2 * 45 + 100
    canvas_hand = Image.new("RGB", (total_hw, total_hh), (241, 245, 249))
    draw_hand = ImageDraw.Draw(canvas_hand)
    f_section = get_font(15, bold=True)
    t_text = "Hand ROI Zoom-In Comparison: Left & Right Hands (1500 Steps Full Adapter Overfit)"
    s_text = "SMPL-X Keypoints extracted from joint_proj_model[25:65] + Left/Right Hand Pose parameters"
    tb = draw_hand.textbbox((0, 0), t_text, font=f_main)
    sb = draw_hand.textbbox((0, 0), s_text, font=f_sub)
    draw_hand.text(((total_hw - (tb[2] - tb[0])) // 2, 14), t_text, font=f_main, fill=(15, 23, 42))
    draw_hand.text(((total_hw - (sb[2] - sb[0])) // 2, 45), s_text, font=f_sub, fill=(71, 85, 105))
    y_off = 75
    for row_title, items in hand_rows:
        draw_hand.text((20, y_off), row_title, font=f_section, fill=(30, 41, 59))
        y_off += 28
        x_off = 20
        for img, t, s in items:
            lbl = add_label_banner(img, t, s, font_size=15)
            canvas_hand.paste(lbl, (x_off, y_off))
            x_off += z + 12
        y_off += z + 22
    canvas_hand.save(out_dir / "06_genuine_inference_hand_roi_1500.png", quality=95)

    # Panel 7: Scorecard
    render_scorecard_1500(quant_results, out_dir / "07_quantitative_evaluation_summary_1500.png")

    print("\n" + "=" * 72)
    print("ALL 1500-STEP PANELS AND EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved in: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
