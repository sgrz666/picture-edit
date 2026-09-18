#!/usr/bin/env python3
"""Evaluate trained UnifiedSMPLXAdapterV6 on iPER validation or test sets.

Loads V6 adapter checkpoint, runs ControlledDeepGenPipeline inference on pairs,
computes quantitative metrics (MAE, MSE, PSNR, SSIM), and generates visual comparison strips.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Project root configuration
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from diffusers import DiffusionPipeline
from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
from train_iper_v6_native import get_cached_conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Unified SMPL-X Adapter V6 on iPER")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
        help="Path to pretrained DeepGen diffusers model",
    )
    parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint .pt file",
    )
    parser.add_argument(
        "--pairs_jsonl",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits/dev_val_pairs.jsonl",
        help="Path to evaluation pairs jsonl (dev_val or test)",
    )
    parser.add_argument(
        "--sampled_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
        help="Root directory of iPER sampled images",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
        help="Root directory of iPER preprocessed assets",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/shangguanrz/project/pic-edit/experiments/iper_adapter_finetune_v1/condition_cache",
        help="Condition cache directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/shangguanrz/project/pic-edit/experiments/iper_v6_eval_results",
        help="Output directory for evaluations and images",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=50, help="Max number of pairs to evaluate")
    parser.add_argument("--inference_steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--geometry_strength", type=float, default=1.0)
    parser.add_argument("--condition_use_depth", action="store_true", default=False)
    parser.add_argument("--condition_backend", type=str, default="native", choices=["native", "champ"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Change the person pose to match the target pose. Preserve identity, clothing, and background.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, low quality, low resolution, distorted, deformed, "
            "broken content, missing parts, damaged details, artifacts, "
            "glitch, noise, extra fingers, missing fingers, mutated hands, "
            "bad composition, wrong proportion, unfinished"
        ),
    )
    return parser.parse_args()


def compute_metrics(generated: Image.Image, target: Image.Image) -> Dict[str, float]:
    """Compute MAE, MSE, PSNR, SSIM between two PIL Images."""
    gen_arr = np.asarray(generated.convert("RGB"), dtype=np.float32)
    tgt_arr = np.asarray(target.convert("RGB"), dtype=np.float32)
    if gen_arr.shape != tgt_arr.shape:
        raise ValueError(f"Shape mismatch: {gen_arr.shape} vs {tgt_arr.shape}")

    diff = gen_arr - tgt_arr
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse / 255.0**2))

    gen_gray = np.mean(gen_arr, axis=2)
    tgt_gray = np.mean(tgt_arr, axis=2)
    mu_g = gen_gray.mean()
    mu_t = tgt_gray.mean()
    sig_g = gen_gray.std()
    sig_t = tgt_gray.std()
    sig_gt = np.mean((gen_gray - mu_g) * (tgt_gray - mu_t))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = float(
        ((2 * mu_g * mu_t + c1) * (2 * sig_gt + c2))
        / ((mu_g**2 + mu_t**2 + c1) * (sig_g**2 + sig_t**2 + c2) + 1e-10)
    )

    try:
        from skimage.metrics import structural_similarity
        ssim_skimage = float(structural_similarity(
            gen_arr, tgt_arr, channel_axis=2, data_range=255.0
        ))
    except Exception:
        ssim_skimage = ssim

    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim, "ssim_skimage": ssim_skimage}


def make_comparison_strip(
    panels: List[Image.Image],
    labels: List[str],
    size: int = 256,
) -> Image.Image:
    """Create a horizontal comparison strip."""
    canvas = Image.new("RGB", (size * len(panels), size + 30), color=(25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for i, (p_img, p_lbl) in enumerate(zip(panels, labels)):
        resized = p_img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        canvas.paste(resized, (i * size, 30))
        draw.text((i * size + 6, 6), p_lbl, fill=(240, 240, 240))
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("==================================================================", flush=True)
    print("   Unified SMPL-X Adapter V6.3 Evaluation                          ", flush=True)
    print("==================================================================", flush=True)

    # 1. Pipeline & Adapter Loading
    print(f"Loading base DeepGen model from {args.model_path}...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    pipe._load_extras(attn_implementation="sdpa")

    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(
        pipe,
        condition_use_depth=args.condition_use_depth,
        normal_backend=args.condition_backend,
        depth_backend=args.condition_backend,
    ).to(device=device, dtype=dtype)

    print(f"Loading adapter weights from {args.adapter_checkpoint}...", flush=True)
    ckpt = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    adapter_state = ckpt.get("adapter_state_dict", ckpt)
    adapter.load_state_dict(adapter_state)
    adapter.eval()

    controlled_pipeline = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)

    # 2. Dataset Setup
    dataset = IPERPoseDataset(
        pairs_jsonl=args.pairs_jsonl,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        augment=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded evaluation pairs: {len(dataset)}", flush=True)

    all_metrics_on: List[Dict[str, float]] = []
    all_metrics_off: List[Dict[str, float]] = []

    for idx, batch in enumerate(dataloader):
        if idx >= args.max_samples:
            break

        app = batch["appearance"][0]
        src_stem = batch["source_stem"][0]
        tgt_stem = batch["target_stem"][0]
        print(f"Evaluating sample {idx + 1}/{min(args.max_samples, len(dataset))}: {app}_{src_stem}->{tgt_stem}...", flush=True)

        normal_a = batch["normal"].to(device, dtype=dtype)
        part_onehot_a = batch["part_onehot"].to(device, dtype=dtype)
        pose_heatmap_a = batch["pose_heatmap"].to(device, dtype=dtype)
        smplx_global_a = batch["smplx_global"].to(device, dtype=dtype)
        human_mask_a = batch["human_mask"].to(device, dtype=dtype)
        task_id = batch["task_id"].to(device, dtype=torch.long)
        depth_a = batch["depth"].to(device, dtype=dtype) if args.condition_use_depth else None

        bundle = adapter.condition_injector(
            normal_a=normal_a,
            pose_heatmap_a=pose_heatmap_a,
            part_onehot_a=part_onehot_a,
            smplx_global_a=smplx_global_a,
            human_mask_a=human_mask_a,
            task_id=task_id,
            depth_a=depth_a,
        )

        ref_latent, _, _ = get_cached_conditions(pipe, batch, args.prompt, args.cache_dir, device, dtype)
        src_person_latents = torch.zeros(
            1, 2, 16, ref_latent.shape[-2], ref_latent.shape[-1], device=device, dtype=dtype
        )
        src_person_latents[:, 0] = ref_latent[:1]
        identity = AdapterIdentityCondition(
            source_person_latents=src_person_latents,
            source_indices=torch.tensor([[0, 1]], device=device),
        )

        src_raw = (batch["src_image"][0].float() * 0.5 + 0.5).clamp(0, 1)
        tgt_raw = (batch["tgt_image"][0].float() * 0.5 + 0.5).clamp(0, 1)
        src_pil = to_pil_image(src_raw.cpu())
        tgt_pil = to_pil_image(tgt_raw.cpu())

        # Generate Adapter ON
        gen_on = controlled_pipeline(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=ref_latent[:1],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=src_pil,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=args.geometry_strength,
            interaction_strength=0.0,
        ).images[0]

        # Generate Adapter OFF
        gen_off = controlled_pipeline(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=ref_latent[:1],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=src_pil,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            geometry_strength=0.0,
            interaction_strength=0.0,
        ).images[0]

        # Metrics against Target GT
        m_on = compute_metrics(gen_on, tgt_pil)
        m_on["appearance"] = app
        m_on["source_stem"] = src_stem
        m_on["target_stem"] = tgt_stem
        m_on["pair_idx"] = idx

        m_off = compute_metrics(gen_off, tgt_pil)
        m_off["appearance"] = app
        m_off["source_stem"] = src_stem
        m_off["target_stem"] = tgt_stem
        m_off["pair_idx"] = idx

        all_metrics_on.append(m_on)
        all_metrics_off.append(m_off)

        # Build comparison strip: [Source, Target GT, Adapter OFF, Adapter ON]
        strip = make_comparison_strip(
            [src_pil, tgt_pil, gen_off, gen_on],
            ["Source", "Ground Truth", "Adapter OFF", "Adapter ON (V6.3)"],
            size=384,
        )
        strip_path = images_dir / f"eval_{idx:03d}_{app}_{src_stem}_{tgt_stem}.png"
        strip.save(strip_path)

    # Average metrics
    def avg_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
        if not items:
            return {}
        metric_keys = [k for k in ("mae", "mse", "psnr", "ssim", "ssim_skimage") if k in items[0]]
        return {k: float(np.mean([d[k] for d in items])) for k in metric_keys}

    summary_on = avg_dict(all_metrics_on)
    summary_off = avg_dict(all_metrics_off)

    # Output structure strictly compatible with baseline eval_metrics.json schema
    summary = {
        "summary": {
            "num_samples": len(all_metrics_on),
            "avg_mae": summary_on.get("mae", 0.0),
            "avg_mse": summary_on.get("mse", 0.0),
            "avg_psnr": summary_on.get("psnr", 0.0),
            "avg_ssim": summary_on.get("ssim", 0.0),
            "avg_ssim_skimage": summary_on.get("ssim_skimage", 0.0),
        },
        "per_sample": all_metrics_on,
        "summary_adapter_off": {
            "num_samples": len(all_metrics_off),
            "avg_mae": summary_off.get("mae", 0.0),
            "avg_mse": summary_off.get("mse", 0.0),
            "avg_psnr": summary_off.get("psnr", 0.0),
            "avg_ssim": summary_off.get("ssim", 0.0),
            "avg_ssim_skimage": summary_off.get("ssim_skimage", 0.0),
        },
        "per_sample_adapter_off": all_metrics_off,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "pairs_jsonl": str(args.pairs_jsonl),
        "metrics_adapter_on": summary_on,
        "metrics_adapter_off": summary_off,
    }

    metrics_file = output_dir / "eval_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================================", flush=True)
    print("   EVALUATION SUMMARY                                             ", flush=True)
    print("==================================================================", flush=True)
    print(f"Adapter ON  -> PSNR: {summary_on.get('psnr', 0):.2f} dB, SSIM: {summary_on.get('ssim', 0):.4f}, MAE: {summary_on.get('mae', 0):.2f}")
    print(f"Adapter OFF -> PSNR: {summary_off.get('psnr', 0):.2f} dB, SSIM: {summary_off.get('ssim', 0):.4f}, MAE: {summary_off.get('mae', 0):.2f}")
    print(f"Results saved to: {metrics_file}")


if __name__ == "__main__":
    main()
