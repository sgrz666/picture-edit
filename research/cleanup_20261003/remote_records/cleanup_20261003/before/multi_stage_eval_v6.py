#!/usr/bin/env python3
"""Multi-stage checkpoint inference and comparison visualization for UnifiedSMPLXAdapterV6.

Loads intermediate checkpoints (Step 500, 1000, 1500, 2000, 2500, 3000)
and evaluates them on identical test sample pairs with fixed random seeds.
Generates an evolution comparison grid showing progressive refinement
and compares results against the 130.28M baseline adapter.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
if DEFAULT_PROJECT_ROOT.name == "scripts":
    DEFAULT_PROJECT_ROOT = DEFAULT_PROJECT_ROOT.parent
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from diffusers import DiffusionPipeline
from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
from train_iper_v6_native import get_cached_conditions


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-stage checkpoint evaluation for V6.3 Adapter")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "models" / "DeepGen-1.0-diffusers"),
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "experiments" / "iper_v6_comparative_3000"),
    )
    parser.add_argument(
        "--baseline_exp_dir",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "experiments" / "iper_adapter_finetune_v1"),
        help="Path to baseline 130.28M experiment directory for comparison",
    )
    parser.add_argument(
        "--sampled_root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "datasets" / "iPER" / "iper_sampled_6src64tgt"),
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "datasets" / "iPER" / "iper_assets_512_nvdiffrast"),
    )
    parser.add_argument(
        "--test_pairs",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "datasets" / "iPER" / "iper_sampled_6src64tgt" / "splits" / "test_pairs.jsonl"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "experiments" / "iper_adapter_finetune_v1" / "condition_cache"),
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[500, 1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--num_samples", type=int, default=3)
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


def compute_metrics(generated: Image.Image, target: Image.Image) -> dict:
    gen_arr = np.asarray(generated.convert("RGB"), dtype=np.float32)
    tgt_arr = np.asarray(target.convert("RGB"), dtype=np.float32)
    diff = gen_arr - tgt_arr
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0 ** 2 / (mse + 1e-10)))

    # Baseline-identical grayscale SSIM formula (c1 = (0.01*255)^2, c2 = (0.03*255)^2)
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

    from skimage.metrics import structural_similarity
    ssim_skimage = float(structural_similarity(
        gen_arr, tgt_arr, channel_axis=2, data_range=255.0
    ))
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim, "ssim_skimage": ssim_skimage}


def select_diverse_pairs(test_pairs_path: str, count: int = 3) -> list[dict]:
    with open(test_pairs_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    appearances = {}
    for p in pairs:
        app = p["appearance"]
        if app not in appearances:
            appearances[app] = []
        appearances[app].append(p)

    selected = []
    for app, app_pairs in list(appearances.items())[:count]:
        app_pairs.sort(
            key=lambda p: abs(p["target"]["frame_index0"] - p["source"]["frame_index0"]),
            reverse=True,
        )
        selected.append(app_pairs[0])

    if len(selected) < count:
        selected = pairs[:count]
    return selected


def main():
    args = parse_args()
    exp_path = Path(args.exp_dir)
    out_dir = exp_path / "stage_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading DeepGen pipeline from {args.model_path}...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    pipe.vae.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe._load_extras(attn_implementation="sdpa")

    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe).to(device=device, dtype=dtype)
    adapter.eval()

    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)

    dataset = IPERPoseDataset(
        pairs_jsonl=args.test_pairs,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        augment=False,
    )

    pairs = select_diverse_pairs(args.test_pairs, count=args.num_samples)
    print(f"Selected {len(pairs)} diverse test pairs for evolution visualization:", flush=True)
    pair_indices = []
    for idx, p in enumerate(pairs):
        app = p["appearance"]
        tgt_stem = p["target"]["stem"]
        for d_idx, sample in enumerate(dataset.samples):
            if sample["appearance"] == app and sample["target"]["stem"] == tgt_stem:
                pair_indices.append(d_idx)
                break
        print(f"  Pair {idx+1}: {app} Source={p['source']['stem']} -> Target={tgt_stem}", flush=True)

    pair_data = []
    for p_idx in pair_indices:
        item = dataset[p_idx]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}

        ref_latent, _, _ = get_cached_conditions(pipe, batch, args.prompt, args.cache_dir, device, dtype)

        normal_a = batch["normal"].to(device, dtype=dtype)
        part_onehot_a = batch["part_onehot"].to(device, dtype=dtype)
        pose_heatmap_a = batch["pose_heatmap"].to(device, dtype=dtype)
        smplx_global_a = batch["smplx_global"].to(device, dtype=dtype)
        human_mask_a = batch["human_mask"].to(device, dtype=dtype)
        task_id = batch["task_id"].to(device, dtype=torch.long)

        bundle = adapter.condition_injector(
            normal_a=normal_a,
            pose_heatmap_a=pose_heatmap_a,
            part_onehot_a=part_onehot_a,
            smplx_global_a=smplx_global_a,
            human_mask_a=human_mask_a,
            task_id=task_id,
        )

        src_person_latents = torch.zeros(1, 2, 16, ref_latent.shape[-2], ref_latent.shape[-1], device=device, dtype=dtype)
        src_person_latents[:, 0] = ref_latent[:1]
        identity = AdapterIdentityCondition(
            source_person_latents=src_person_latents,
            source_indices=torch.tensor([[0, 1]], device=device),
        )

        src_raw = (item["src_image"].float() * 0.5 + 0.5).clamp(0, 1)
        tgt_raw = (item["tgt_image"].float() * 0.5 + 0.5).clamp(0, 1)
        src_pil = to_pil_image(src_raw.cpu())
        tgt_pil = to_pil_image(tgt_raw.cpu())

        dwpose_tensor = item["control_map"][4:7].permute(1, 2, 0).numpy()
        dwpose_pil = Image.fromarray((np.clip(dwpose_tensor * 255, 0, 255)).astype(np.uint8))

        pair_data.append({
            "appearance": item["appearance"],
            "source_stem": item["source_stem"],
            "target_stem": item["target_stem"],
            "src_pil": src_pil,
            "tgt_pil": tgt_pil,
            "dwpose_pil": dwpose_pil,
            "bundle": bundle,
            "identity": identity,
            "ref_latent": ref_latent[:1],
        })

    step_results = {}
    stage_metrics = {step: [] for step in args.steps}

    for step in args.steps:
        ckpt_file = exp_path / f"adapter_step_{step}.pt"
        if not ckpt_file.exists():
            ckpt_file = exp_path / f"checkpoint_step_{step}.pt"
        if not ckpt_file.exists():
            print(f"Warning: checkpoint for step {step} not found, skipping.", flush=True)
            continue

        print(f"\n--- Running V6.3 inference for Step {step} ({ckpt_file.name}) ---", flush=True)
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()

        step_results[step] = []
        for p_idx, data in enumerate(pair_data):
            t0 = time.time()
            with torch.no_grad():
                output = controlled(
                    condition_bundle=data["bundle"],
                    identity_condition=data["identity"],
                    source_scene_latents=data["ref_latent"],
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image=data["src_pil"],
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    geometry_strength=1.0,
                    interaction_strength=0.0,
                ).images[0]

            metrics = compute_metrics(output, data["tgt_pil"])
            duration = time.time() - t0
            metrics["step"] = step
            metrics["pair_idx"] = p_idx
            metrics["duration_sec"] = duration
            stage_metrics[step].append(metrics)

            save_name = f"pair{p_idx+1}_{data['appearance']}_step_{step}.png"
            output.save(out_dir / save_name)
            step_results[step].append(output)
            print(f"  [Pair {p_idx+1}] {data['appearance']} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} ({duration:.1f}s)", flush=True)

    metrics_summary = {}
    for step, m_list in stage_metrics.items():
        if m_list:
            metrics_summary[step] = {
                "avg_psnr": float(np.mean([m["psnr"] for m in m_list])),
                "avg_ssim": float(np.mean([m["ssim"] for m in m_list])),
                "avg_mae": float(np.mean([m["mae"] for m in m_list])),
                "details": m_list,
            }
    with open(out_dir / "stage_evolution_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {out_dir / 'stage_evolution_metrics.json'}", flush=True)

    # Build comparison figure
    col_headers = ["Source (Ref)", "Pose Guidance"] + [f"Step {s}" for s in args.steps if s in step_results] + ["Ground Truth"]

    def get_font(size: int, bold: bool = False):
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
        for fp in font_candidates:
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
        return ImageFont.load_default()

    tile_size = 256
    header_h = 42
    label_h = 30
    cell_h = tile_size + label_h
    num_cols = len(col_headers)
    num_rows = len(pair_data)

    total_w = num_cols * tile_size
    total_h = header_h + num_rows * cell_h

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    font_h = get_font(13, bold=True)
    font_sub = get_font(11, bold=True)
    font_m = get_font(10, bold=False)

    for c, title in enumerate(col_headers):
        x = c * tile_size
        bg_col = (30, 41, 59) if "Step" in title else ((20, 83, 45) if "Ground Truth" in title else (15, 23, 42))
        draw.rectangle([(x, 0), (x + tile_size - 1, header_h - 1)], fill=bg_col)
        bbox = draw.textbbox((0, 0), title, font=font_h)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (tile_size - tw) // 2, (header_h - th) // 2), title, fill=(248, 250, 252), font=font_h)

    for r, data in enumerate(pair_data):
        y_base = header_h + r * cell_h

        # Column 0: Source
        src_thumb = data["src_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(src_thumb, (0, y_base))
        draw.rectangle([(0, y_base + tile_size), (tile_size - 1, y_base + cell_h - 1)], fill=(15, 23, 42))
        app_label = f"{data['appearance']} Ref"
        bbox = draw.textbbox((0, 0), app_label, font=font_sub)
        draw.text(((tile_size - (bbox[2] - bbox[0])) // 2, y_base + tile_size + 7), app_label, fill=(148, 163, 184), font=font_sub)

        # Column 1: DWPose
        dw_thumb = data["dwpose_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(dw_thumb, (tile_size, y_base))
        draw.rectangle([(tile_size, y_base + tile_size), (tile_size * 2 - 1, y_base + cell_h - 1)], fill=(15, 23, 42))
        bbox = draw.textbbox((0, 0), "Target Pose", font=font_sub)
        draw.text((tile_size + (tile_size - (bbox[2] - bbox[0])) // 2, y_base + tile_size + 7), "Target Pose", fill=(148, 163, 184), font=font_sub)

        # Checkpoints columns
        c_idx = 2
        for s in args.steps:
            if s not in step_results:
                continue
            out_img = step_results[s][r]
            out_thumb = out_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            cx = c_idx * tile_size
            canvas.paste(out_thumb, (cx, y_base))

            m = stage_metrics[s][r]
            lbl = f"PSNR: {m['psnr']:.1f} dB | SSIM: {m['ssim']:.3f}"
            draw.rectangle([(cx, y_base + tile_size), (cx + tile_size - 1, y_base + cell_h - 1)], fill=(30, 41, 59))
            bbox = draw.textbbox((0, 0), lbl, font=font_m)
            draw.text((cx + (tile_size - (bbox[2] - bbox[0])) // 2, y_base + tile_size + 7), lbl, fill=(186, 230, 253), font=font_m)
            c_idx += 1

        # Last Column: Ground truth
        tgt_thumb = data["tgt_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        gx = (num_cols - 1) * tile_size
        canvas.paste(tgt_thumb, (gx, y_base))
        draw.rectangle([(gx, y_base + tile_size), (gx + tile_size - 1, y_base + cell_h - 1)], fill=(20, 83, 45))
        bbox = draw.textbbox((0, 0), "Ground Truth", font=font_sub)
        draw.text((gx + (tile_size - (bbox[2] - bbox[0])) // 2, y_base + tile_size + 7), "Ground Truth", fill=(187, 247, 208), font=font_sub)

    for c in range(1, num_cols):
        draw.line([(c * tile_size, 0), (c * tile_size, total_h)], fill=(51, 65, 85), width=1)
    for r in range(1, num_rows + 1):
        draw.line([(0, header_h + r * cell_h - 1), (total_w, header_h + r * cell_h - 1)], fill=(51, 65, 85), width=1)

    grid_path = exp_path / "multi_stage_evolution_grid.png"
    canvas.save(grid_path)
    canvas.save(out_dir / "multi_stage_evolution_grid.png")
    print(f"✓ Saved high-resolution evolution grid to {grid_path}", flush=True)


if __name__ == "__main__":
    main()
