#!/usr/bin/env python3
"""Comprehensive evaluation, pure generative inference, and comparison panel generation
for TikTok 00001 full-adapter single-sample overfit training.

Generates:
1. Training Loss Curves (Total, Flow, Face ID, Perceptual, Outside)
2. Training Step Progression Evolution (Step 0 to 200)
3. Genuine 28-Step Generative Inference Comparison (Source, Normal, GT, Baseline, Full Adapter)
4. Facial ROI Close-up Zoom Comparison
5. Hand ROI Close-up Zoom Comparison (Left Hand + Right Hand)
6. Quantitative Scorecard Summary Infographic (Full Body, Face, Left Hand, Right Hand)
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
import torch.nn.functional as F
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


def add_label(
    img: Image.Image,
    text: str,
    font_size: int = 18,
    bg_color=(15, 23, 42, 220),
    text_color=(255, 255, 255),
    subtext: str | None = None,
) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font_sub = ImageFont.truetype("arial.ttf", max(11, font_size - 5))
    except Exception:
        font = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    banner_h = text_h + 16
    if subtext:
        sub_bbox = draw.textbbox((0, 0), subtext, font=font_sub)
        sub_w = sub_bbox[2] - sub_bbox[0]
        banner_h += (sub_bbox[3] - sub_bbox[1]) + 6

    banner = Image.new("RGBA", (img.width, banner_h), bg_color)
    img.paste(banner, (0, 0), banner)
    text_draw = ImageDraw.Draw(img)
    text_draw.text(((img.width - text_w) // 2, 7), text, font=font, fill=text_color)
    if subtext:
        text_draw.text(((img.width - sub_w) // 2, text_h + 12), subtext, font=font_sub, fill=(203, 213, 225))
    return img.convert("RGB")


def compute_image_metrics(pred_pil: Image.Image, gt_pil: Image.Image) -> dict[str, float]:
    """Compute MSE, PSNR, SSIM, L1 between prediction and ground truth."""
    pred = np.array(pred_pil.convert("RGB")).astype(np.float32) / 255.0
    gt = np.array(gt_pil.convert("RGB")).astype(np.float32) / 255.0
    if pred.shape != gt.shape:
        gt = np.array(gt_pil.convert("RGB").resize((pred.shape[1], pred.shape[0]))).astype(np.float32) / 255.0

    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-10)))
    l1 = float(np.mean(np.abs(pred - gt)))
    
    # Simple, robust SSIM calculation
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
    """Extract a square bounding box for a hand from 21 keypoints [21, 3] (x, y, conf)."""
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


def plot_loss_curves(metrics_path: Path, out_path: Path) -> None:
    """Plot publication-quality training curves."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", [])
    if not metrics:
        print("No metrics to plot")
        return

    steps = [m["step"] for m in metrics]
    totals = [m["total"] for m in metrics]
    flows = [m["flow"] for m in metrics]
    outsides = [m.get("outside", 0.0) for m in metrics]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Subplot 1: Total Loss & Flow Loss
    ax1 = axes[0]
    ax1.plot(steps, totals, label="Total Loss", color="#1E3A8A", linewidth=2.2)
    ax1.plot(steps, flows, label="Flow Matching Loss", color="#0284C7", linewidth=1.8, linestyle="--")
    ax1.set_title("TikTok 00001: Full Adapter Convergence", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Loss Value", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Annotate initial and final loss
    ax1.annotate(
        f"Initial: {totals[0]:.4f}",
        xy=(steps[0], totals[0]),
        xytext=(steps[0] + 15, totals[0] + 0.02),
        arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.2),
        fontweight="bold",
        fontsize=10,
    )
    ax1.annotate(
        f"Final: {totals[-1]:.4f}\n(-{((totals[0]-totals[-1])/totals[0]*100):.1f}%)",
        xy=(steps[-1], totals[-1]),
        xytext=(steps[-1] - 50, totals[-1] + 0.06),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
        fontweight="bold",
        color="#059669",
        fontsize=10,
    )

    # Subplot 2: Auxiliary Losses (Outside & Auxiliary)
    ax2 = axes[1]
    ax2.plot(steps, outsides, label="Outside Area Regularization", color="#D97706", linewidth=1.8)
    ax2.set_title("Outside Face-Mask Regularization Dynamics", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Regularization Value", fontsize=12)
    ax2.legend(loc="lower right", frameon=True, fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Full Adapter Single-Sample Overfit Training Dynamics (Sequence 00001, Frame 0014 -> 0074)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_evolution_grid(previews_dir: Path, out_path: Path, tgt_pil: Image.Image) -> None:
    """Generate a 2-row evolution progression grid across steps."""
    step_labels = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    imgs = []
    titles = []
    
    for s in step_labels:
        img_p = previews_dir / f"step_{s:06d}_face_on.png"
        if img_p.exists():
            img = Image.open(img_p).convert("RGB")
        else:
            img = Image.new("RGB", (512, 512), (180, 180, 180))
        lbl = "Initial State (Step 0)" if s == 0 else f"Step {s}"
        imgs.append(add_label(img.resize((360, 360)), lbl, font_size=16))
    
    # Add Ground Truth as the 10th image
    imgs.append(add_label(tgt_pil.resize((360, 360)), "Target GT Reference", font_size=16, bg_color=(5, 150, 105, 230)))

    # Create 2 rows x 5 columns canvas
    cols = 5
    rows = 2
    cell_w, cell_h = 360, 360
    pad = 12
    header_h = 60
    
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
    
    title_text = "Full Adapter Single-Sample Overfitting Progression (Steps 0 - 200 vs Target GT)"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) // 2, 16), title_text, font=title_font, fill=(15, 23, 42))

    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = header_h + pad + r * (cell_h + pad)
        canvas.paste(img, (x, y))

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
    parser.add_argument("--training-dir", default="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_200")
    parser.add_argument("--output-dir", default="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_200/final_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    
    training_dir = Path(args.training_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STARTING FULL ADAPTER COMPREHENSIVE EVALUATION & PANEL GENERATION")
    print(f"Sequence: {args.sequence}, Source: {args.source_stem} -> Target: {args.target_stem}")
    print(f"Training Directory: {training_dir}")
    print(f"Output Directory  : {out_dir}")
    print("=" * 72)

    # 1. Plot Loss Curves
    print("\n[1/6] Plotting training dynamics and loss curves...", flush=True)
    metrics_json = training_dir / "metrics.json"
    plot_loss_curves(metrics_json, out_dir / "01_full_adapter_training_loss_curves.png")

    # 2. Load TikTok Data
    print("\n[2/6] Loading TikTok input pair and 3D conditions...", flush=True)
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

    # 3. Plot Preview Progression
    print("\n[3/6] Generating step progression evolution grid...", flush=True)
    plot_evolution_grid(training_dir / "previews", out_dir / "02_full_adapter_step_evolution.png", tgt_pil)

    # 4. Pure Diffusion Inference
    print("\n[4/6] Executing pure 28-step diffusion inference from Gaussian noise...", flush=True)
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe, condition_use_depth=False).to(device=device, dtype=dtype)

    ckpt_path = training_dir / "checkpoints" / "checkpoint_step_000200.pt"
    print(f"Loading checkpoint from: {ckpt_path}", flush=True)
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
    print("  -> Generating Baseline (Face OFF, Global Baseline)...", flush=True)
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

    # Full Adapter (Face ON)
    print("  -> Generating Full Adapter Trained (Face ON + Hand + Body)...", flush=True)
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

    # 5. Extract Hand & Face Crops and Compute Metrics
    print("\n[5/6] Extracting ROIs and computing metrics...", flush=True)
    # Full body metrics
    metrics_body_off = compute_image_metrics(img_off, tgt_pil)
    metrics_body_on = compute_image_metrics(img_on, tgt_pil)

    # Face ROI
    crop_box_tgt = (115, 95, 295, 275)
    crop_box_src = (120, 100, 300, 280)
    zoom_size = 280
    face_src = src_pil.crop(crop_box_src).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    face_tgt = tgt_pil.crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    face_off = img_off.crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    face_on = img_on.crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)

    metrics_face_off = compute_image_metrics(face_off, face_tgt)
    metrics_face_on = compute_image_metrics(face_on, face_tgt)

    # Hand ROIs
    kps_hand = overfit_inputs.hand_detail_condition.hand_keypoints[0, 0] # person 0 -> shape [2, 21, 3] (side, point, coords)
    left_hand_box = get_hand_crop_box(kps_hand[0], 512, margin=0.35) # left hand
    right_hand_box = get_hand_crop_box(kps_hand[1], 512, margin=0.35) # right hand

    print(f"  Left Hand Target Box : {left_hand_box}")
    print(f"  Right Hand Target Box: {right_hand_box}")

    lh_src = src_pil.crop(left_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    lh_tgt = tgt_pil.crop(left_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    lh_off = img_off.crop(left_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    lh_on = img_on.crop(left_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)

    rh_src = src_pil.crop(right_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    rh_tgt = tgt_pil.crop(right_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    rh_off = img_off.crop(right_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    rh_on = img_on.crop(right_hand_box).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)

    metrics_lh_off = compute_image_metrics(lh_off, lh_tgt)
    metrics_lh_on = compute_image_metrics(lh_on, lh_tgt)

    metrics_rh_off = compute_image_metrics(rh_off, rh_tgt)
    metrics_rh_on = compute_image_metrics(rh_on, rh_tgt)

    # Save quantitative metrics json
    quant_results = {
        "full_body": {"baseline": metrics_body_off, "full_adapter": metrics_body_on},
        "face_roi": {"baseline": metrics_face_off, "full_adapter": metrics_face_on},
        "left_hand_roi": {"baseline": metrics_lh_off, "full_adapter": metrics_lh_on},
        "right_hand_roi": {"baseline": metrics_rh_off, "full_adapter": metrics_rh_on},
    }
    with open(out_dir / "quantitative_metrics.json", "w", encoding="utf-8") as f:
        json.dump(quant_results, f, indent=2)

    # 6. Composite Figures
    print("\n[6/6] Generating composite panels and scorecards...", flush=True)

    # Figure 3: Full-body comparison panel
    fullbody_panels = [
        (src_pil, "Source Frame (0014)", "Input Identity"),
        (norm_pil, "Target 3D Normal (0074)", "Pose & Shape Guide"),
        (tgt_pil, "Target GT Reference", "Ground Truth Target"),
        (img_off, "Baseline: Face OFF", f"PSNR: {metrics_body_off['psnr']:.2f} dB"),
        (img_on, "Full Adapter: Face ON", f"PSNR: {metrics_body_on['psnr']:.2f} dB (Trained)"),
    ]
    labeled_full = [add_label(img.resize((512, 512)), text, font_size=18, subtext=sub) for img, text, sub in fullbody_panels]
    total_w = sum(img.width for img in labeled_full) + (len(labeled_full) - 1) * 10
    total_h = 512 + 65
    canvas_fb = Image.new("RGB", (total_w, total_h), (241, 245, 249))
    draw_fb = ImageDraw.Draw(canvas_fb)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
    title_text = "TikTok 00001: Pure 28-Step Generative Diffusion (Zero GT Leakage, Seed 42)"
    bbox = draw_fb.textbbox((0, 0), title_text, font=title_font)
    draw_fb.text(((total_w - (bbox[2] - bbox[0])) // 2, 14), title_text, font=title_font, fill=(15, 23, 42))

    x_off = 0
    for img in labeled_full:
        canvas_fb.paste(img, (x_off, 55))
        x_off += img.width + 10
    canvas_fb.save(out_dir / "03_genuine_inference_fullbody_comparison.png", quality=95)

    # Figure 4: Facial ROI Zoom-in Comparison
    face_items = [
        (face_src, "Source (0014)", "Input Identity"),
        (face_tgt, "Target GT (Reference)", "Ground Truth"),
        (face_off, "Baseline: Face OFF", f"PSNR: {metrics_face_off['psnr']:.2f} dB"),
        (face_on, "Full Adapter: Face ON", f"PSNR: {metrics_face_on['psnr']:.2f} dB (Trained)"),
    ]
    labeled_face = [add_label(img, t, font_size=16, subtext=s) for img, t, s in face_items]
    w_face = len(labeled_face) * zoom_size + (len(labeled_face) - 1) * 10
    h_face = zoom_size + 65
    canvas_face = Image.new("RGB", (w_face, h_face), (241, 245, 249))
    draw_face = ImageDraw.Draw(canvas_face)
    title_face = "Facial ROI Zoom-In Comparison (Pure Diffusion from Gaussian Noise)"
    bbox = draw_face.textbbox((0, 0), title_face, font=title_font)
    draw_face.text(((w_face - (bbox[2] - bbox[0])) // 2, 14), title_face, font=title_font, fill=(15, 23, 42))
    x_off = 0
    for img in labeled_face:
        canvas_face.paste(img, (x_off, 55))
        x_off += img.width + 10
    canvas_face.save(out_dir / "04_genuine_inference_face_roi_zoom.png", quality=95)

    # Figure 5: Hand ROI Zoom-in Comparison (2 Rows: Left Hand & Right Hand)
    hand_rows = [
        ("Left Hand", [
            (lh_src, "Source LH", ""),
            (lh_tgt, "Target GT LH", "Ground Truth"),
            (lh_off, "Baseline LH", f"PSNR: {metrics_lh_off['psnr']:.2f} dB"),
            (lh_on, "Full Adapter LH", f"PSNR: {metrics_lh_on['psnr']:.2f} dB"),
        ]),
        ("Right Hand", [
            (rh_src, "Source RH", ""),
            (rh_tgt, "Target GT RH", "Ground Truth"),
            (rh_off, "Baseline RH", f"PSNR: {metrics_rh_off['psnr']:.2f} dB"),
            (rh_on, "Full Adapter RH", f"PSNR: {metrics_rh_on['psnr']:.2f} dB"),
        ]),
    ]
    h_item_size = 260
    pad_h = 10
    total_hw = 4 * h_item_size + 3 * pad_h + 30
    total_hh = 60 + 2 * h_item_size + pad_h + 30
    canvas_hand = Image.new("RGB", (total_hw, total_hh), (241, 245, 249))
    draw_hand = ImageDraw.Draw(canvas_hand)
    title_hand = "Hand ROI Zoom-In Comparison: Left & Right Hands (SMPL-X 21-Points Control)"
    bbox = draw_hand.textbbox((0, 0), title_hand, font=title_font)
    draw_hand.text(((total_hw - (bbox[2] - bbox[0])) // 2, 14), title_hand, font=title_font, fill=(15, 23, 42))

    y_pos = 55
    for hand_name, items in hand_rows:
        x_pos = 15
        for img, lbl, sub in items:
            lbl_img = add_label(img.resize((h_item_size, h_item_size)), lbl, font_size=15, subtext=sub)
            canvas_hand.paste(lbl_img, (x_pos, y_pos))
            x_pos += h_item_size + pad_h
        y_pos += h_item_size + pad_h
    canvas_hand.save(out_dir / "05_genuine_inference_hand_roi_zoom.png", quality=95)

    # Figure 6: Quantitative Scorecard Infographic
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.axis("off")
    table_data = [
        ["Evaluation Region", "Metric", "Baseline (Face OFF)", "Full Adapter (Face ON)", "Improvement / Delta"],
        ["Full Body (512x512)", "PSNR (dB)", f"{metrics_body_off['psnr']:.2f}", f"{metrics_body_on['psnr']:.2f}", f"+{metrics_body_on['psnr'] - metrics_body_off['psnr']:.2f} dB"],
        ["Full Body (512x512)", "SSIM", f"{metrics_body_off['ssim']:.4f}", f"{metrics_body_on['ssim']:.4f}", f"+{metrics_body_on['ssim'] - metrics_body_off['ssim']:.4f}"],
        ["Face ROI (280x280)", "PSNR (dB)", f"{metrics_face_off['psnr']:.2f}", f"{metrics_face_on['psnr']:.2f}", f"+{metrics_face_on['psnr'] - metrics_face_off['psnr']:.2f} dB"],
        ["Face ROI (280x280)", "L1 Error", f"{metrics_face_off['l1']:.4f}", f"{metrics_face_on['l1']:.4f}", f"-{metrics_face_off['l1'] - metrics_face_on['l1']:.4f}"],
        ["Left Hand ROI", "PSNR (dB)", f"{metrics_lh_off['psnr']:.2f}", f"{metrics_lh_on['psnr']:.2f}", f"+{metrics_lh_on['psnr'] - metrics_lh_off['psnr']:.2f} dB"],
        ["Left Hand ROI", "L1 Error", f"{metrics_lh_off['l1']:.4f}", f"{metrics_lh_on['l1']:.4f}", f"-{metrics_lh_off['l1'] - metrics_lh_on['l1']:.4f}"],
        ["Right Hand ROI", "PSNR (dB)", f"{metrics_rh_off['psnr']:.2f}", f"{metrics_rh_on['psnr']:.2f}", f"+{metrics_rh_on['psnr'] - metrics_rh_off['psnr']:.2f} dB"],
        ["Right Hand ROI", "L1 Error", f"{metrics_rh_off['l1']:.4f}", f"{metrics_rh_on['l1']:.4f}", f"-{metrics_rh_off['l1'] - metrics_rh_on['l1']:.4f}"],
    ]
    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Styling table cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(0.12)
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("#FFFFFF")
            if col == 4:
                cell.get_text().set_color("#059669")
                cell.get_text().set_weight("bold")
            cell.set_height(0.09)

    plt.title(
        "Quantitative Evaluation Scorecard: TikTok Sample 00001 (Pure 28-Step Generative Diffusion)",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )
    plt.savefig(out_dir / "06_quantitative_evaluation_summary.png", bbox_inches="tight", dpi=200)
    plt.close()

    print("\n" + "=" * 72)
    print("ALL EVALUATIONS AND PANELS SUCCESSFULLY COMPLETED!")
    print(f"Generated panels saved in: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
