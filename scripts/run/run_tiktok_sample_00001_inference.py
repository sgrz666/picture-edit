#!/usr/bin/env python3
"""Run genuine, pure diffusion inference (from 100% random Gaussian noise) on TikTok sample 00001.

Validates end-to-end generation WITHOUT any ground-truth target latent leakage.
Compares:
1. Source Frame (0014)
2. Target Ground Truth (0074, for reference only)
3. Pure 28-step Generative Inference: Face OFF (V6.3/V6.4 Base Baseline)
4. Pure 28-step Generative Inference: Face ON (V6.5 Face Adapter Trained Checkpoint)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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


def add_label(img: Image.Image, text: str, font_size: int = 18, bg_color=(0, 0, 0, 200), text_color=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    banner = Image.new("RGBA", (img.width, text_h + 16), bg_color)
    img.paste(banner, (0, 0), banner)
    text_draw = ImageDraw.Draw(img)
    text_draw.text(((img.width - text_w) // 2, 8), text, font=font, fill=text_color)
    return img.convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pure Inference for TikTok V6.5 Face Adapter")
    parser.add_argument("--sequence", default="00001")
    parser.add_argument("--source-stem", default="0014")
    parser.add_argument("--target-stem", default="0074")
    parser.add_argument("--condition-override", help="Audited single-target pose/part .pt")
    parser.add_argument(
        "--raw-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
    )
    parser.add_argument(
        "--assets-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
    )
    parser.add_argument(
        "--model-path",
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/shangguanrz/project/pic-edit/experiments/tiktok_v65_face_overfit_00001/checkpoints/checkpoint_step_000200.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/shangguanrz/project/pic-edit/experiments/tiktok_v65_pure_inference",
    )
    parser.add_argument(
        "--prompt",
        default="Change the person pose to match the target pose. Preserve identity, clothing, and background.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--face-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.face_strength <= 1.0:
        raise ValueError("--face-strength must be in [0,1]")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TikTok V6.5 Pure Generative Diffusion Inference (From 100% Random Noise)")
    print(f"Sequence        : {args.sequence}")
    print(f"Source Stem     : {args.source_stem} -> Target Stem: {args.target_stem}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Steps / CFG     : {args.num_inference_steps} steps, scale={args.guidance_scale}")
    print(f"Seed            : {args.seed}")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/5] Loading TikTok condition batch via TikTokV65FaceOverfitLoader...", flush=True)
    loader = TikTokV65FaceOverfitLoader(
        raw_root=args.raw_root,
        assets_root=args.assets_root,
        sequence=args.sequence,
        source_stem=args.source_stem,
        target_stem=args.target_stem,
        resolution=args.resolution,
        condition_override=args.condition_override,
    )
    overfit_inputs = loader.load()
    batch = overfit_inputs.batch

    # Convert source image tensor to PIL Image for DeepGen VLM prompt ingestion
    src_raw = (batch["src_image"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    src_pil = Image.fromarray((src_raw * 255.0 + 0.5).astype(np.uint8))
    src_pil.save(out_dir / "input_source_0014.png")

    tgt_raw = (batch["tgt_image"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    tgt_pil = Image.fromarray((tgt_raw * 255.0 + 0.5).astype(np.uint8))
    tgt_pil.save(out_dir / "target_gt_reference_0074.png")

    norm_raw = (batch["normal"][0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    norm_pil = Image.fromarray((norm_raw * 255.0 + 0.5).astype(np.uint8))
    norm_pil.save(out_dir / "input_target_normal_0074.png")

    # 2. Load Pipeline and Adapter
    print("\n[2/5] Loading DeepGen pipeline and UnifiedSMPLXAdapterV6...", flush=True)
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)

    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe, condition_use_depth=False).to(
        device=device, dtype=dtype
    )

    print(f"Loading trained V6.5 checkpoint from {args.checkpoint}...", flush=True)
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_v65_checkpoint(adapter, checkpoint_data)
    adapter.eval()

    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)

    # 3. Prepare Conditions
    print("\n[3/5] Precomputing pipeline and adapter conditioning inputs...", flush=True)
    cache_dir = out_dir / "condition_cache"
    ref_latent, _, _ = get_cached_conditions(pipe, batch, args.prompt, str(cache_dir), device, dtype)

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

    # 4. Pure Diffusion Inference
    print("\n[4/5] Executing Genuine Generative Inference from 100% Random Noise...", flush=True)

    # Run A: Face OFF (Baseline)
    print("  -> Running Full Generative Diffusion: Face OFF (Baseline)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        result_off = controlled(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            hand_condition=detail_condition,
            hand_references=detail_references,
            face_condition=face_condition,
            face_references=face_references,
            prompt=args.prompt,
            negative_prompt="",
            image=src_pil,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=0.0,  # Face OFF
        )
    img_face_off = result_off.images[0]
    img_face_off.save(out_dir / "pure_inference_face_off.png")
    print(f"     Finished Face OFF in {time.time() - t0:.2f}s")

    # Run B: Face ON (V6.5 Trained Face Adapter)
    print("  -> Running Full Generative Diffusion: Face ON (V6.5 Adapter)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        result_on = controlled(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            hand_condition=detail_condition,
            hand_references=detail_references,
            face_condition=face_condition,
            face_references=face_references,
            prompt=args.prompt,
            negative_prompt="",
            image=src_pil,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=args.face_strength,  # Face ON
        )
    img_face_on = result_on.images[0]
    img_face_on.save(out_dir / "pure_inference_face_on.png")
    print(f"     Finished Face ON in {time.time() - t0:.2f}s")

    # 5. Composite Comparison Image Generation
    print("\n[5/5] Generating Comprehensive Comparison Panels...", flush=True)

    # Panel 1: Full-Body Comparison
    panels = [
        (src_pil, "Source Frame (0014)"),
        (norm_pil, "Target 3D Normal (0074)"),
        (tgt_pil, "Target GT (0074 Reference)"),
        (img_face_off, "Pure Inference: Face OFF"),
        (img_face_on, "Pure Inference: Face ON (V6.5)"),
    ]
    labeled = [add_label(img.resize((512, 512)), label, font_size=20) for img, label in panels]
    total_w = sum(img.width for img in labeled) + (len(labeled) - 1) * 10
    total_h = 512 + 65
    canvas = Image.new("RGB", (total_w, total_h), (242, 244, 246))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
    title_text = f"TikTok 00001: 100% Pure Generative Diffusion Inference from Random Gaussian Noise (28 Steps, Seed {args.seed})"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    draw.text(((total_w - (bbox[2] - bbox[0])) // 2, 14), title_text, font=title_font, fill=(20, 25, 35))

    x_offset = 0
    for img in labeled:
        canvas.paste(img, (x_offset, 55))
        x_offset += img.width + 10
    panel_path = out_dir / "genuine_inference_fullbody_comparison.png"
    canvas.save(panel_path, quality=95)
    print(f"  Saved: {panel_path}")

    # Panel 2: Face ROI Close-Up Zoom Comparison
    crop_box_tgt = (115, 95, 295, 275)
    crop_box_src = (120, 100, 300, 280)
    zoom_size = 280
    src_crop = src_pil.resize((512, 512)).crop(crop_box_src).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    tgt_crop = tgt_pil.resize((512, 512)).crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    off_crop = img_face_off.resize((512, 512)).crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
    on_crop = img_face_on.resize((512, 512)).crop(crop_box_tgt).resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)

    roi_items = [
        (src_crop, "Source (0014)"),
        (tgt_crop, "Target GT (Reference)"),
        (off_crop, "Pure Inference: Face OFF"),
        (on_crop, "Pure Inference: Face ON (V6.5)"),
    ]
    roi_labeled = [add_label(img, label, font_size=18) for img, label in roi_items]
    roi_w = len(roi_labeled) * zoom_size + (len(roi_labeled) - 1) * 10
    roi_h = zoom_size + 65
    roi_canvas = Image.new("RGB", (roi_w, roi_h), (245, 246, 248))
    roi_draw = ImageDraw.Draw(roi_canvas)
    roi_title = "Facial ROI Zoom-In Comparison (Pure Diffusion from Gaussian Noise)"
    bbox = roi_draw.textbbox((0, 0), roi_title, font=title_font)
    roi_draw.text(((roi_w - (bbox[2] - bbox[0])) // 2, 14), roi_title, font=title_font, fill=(20, 25, 35))

    x_offset = 0
    for img in roi_labeled:
        roi_canvas.paste(img, (x_offset, 55))
        x_offset += img.width + 10
    roi_path = out_dir / "genuine_inference_face_roi_zoom.png"
    roi_canvas.save(roi_path, quality=95)
    print(f"  Saved: {roi_path}")

    print("\n" + "=" * 70)
    print("Pure Generative Inference Successfully Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
