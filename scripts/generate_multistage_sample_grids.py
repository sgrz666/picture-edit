#!/usr/bin/env python3
"""Generate multi-stage evolution sample strips and grids for UnifiedSMPLXAdapterV6.

Order per sample row:
[原图 (Source)] | [Ground Truth] | [Step 0 (Adapter OFF)] | [Step 500] | [Step 1000] | [Step 1500] | [Step 2000] | [Step 2500] | [Step 3000 (最终)]

Strictly matches user request for multi-sample evolution across training checkpoints.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
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
    parser = argparse.ArgumentParser(description="Multi-stage evolution generator")
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
        "--pairs_jsonl",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "datasets" / "iPER" / "iper_sampled_6src64tgt" / "splits" / "baseline_50_eval_pairs.jsonl"),
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
    mae = float(np.mean(np.abs(gen_arr - tgt_arr)))
    mse = float(np.mean((gen_arr - tgt_arr) ** 2))
    psnr = float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 1e-12 else 99.0

    gen_gray = 0.2989 * gen_arr[..., 0] + 0.5870 * gen_arr[..., 1] + 0.1140 * gen_arr[..., 2]
    tgt_gray = 0.2989 * tgt_arr[..., 0] + 0.5870 * tgt_arr[..., 1] + 0.1140 * tgt_arr[..., 2]
    mu_x = float(np.mean(gen_gray))
    mu_y = float(np.mean(tgt_gray))
    sigma_x2 = float(np.var(gen_gray))
    sigma_y2 = float(np.var(tgt_gray))
    sigma_xy = float(np.mean((gen_gray - mu_x) * (tgt_gray - mu_y)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)))

    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def get_font(size: int, bold: bool = False):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    exp_path = Path(args.exp_dir)
    out_dir = exp_path / "multistage_evolution_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline test pairs and select 5 diverse samples
    pairs_file = Path(args.pairs_jsonl)
    if not pairs_file.exists():
        pairs_file = Path(args.sampled_root) / "splits" / "test_pairs.jsonl"
    all_pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_pairs.append(json.loads(line))

    # Pick 5 high quality diverse appearances:
    # 1) 019_2: Cheer arms high in air
    # 2) 001_11: Side arm horizontal extension
    # 3) 001_18: Crossing arms / self-occlusion
    # 4) 001_19: Forward lean / twist
    # 5) 024_7: Action motion pose
    selected_targets = [
        ("019_2", "target_f001164"),
        ("001_11", "target_f001456"),
        ("001_18", "target_f001379"),
        ("001_19", "target_f001370"),
        ("024_7", "target_f001306"),
    ]

    selected_pairs = []
    for app_req, tgt_req in selected_targets:
        found = False
        for p in all_pairs:
            if p["appearance"] == app_req and p["target"]["stem"] == tgt_req:
                selected_pairs.append(p)
                found = True
                break
        if not found:
            for p in all_pairs:
                if p["appearance"] == app_req:
                    selected_pairs.append(p)
                    found = True
                    break

    while len(selected_pairs) < 5 and len(selected_pairs) < len(all_pairs):
        p = all_pairs[len(selected_pairs)]
        if p not in selected_pairs:
            selected_pairs.append(p)

    print(f"Selected {len(selected_pairs)} pairs for multi-stage evolution showcase:")
    for idx, p in enumerate(selected_pairs):
        print(f"  [{idx+1}] {p['appearance']}: {p['source']['stem']} -> {p['target']['stem']}")

    tmp_pairs_path = out_dir / "selected_showcase_pairs.jsonl"
    with open(tmp_pairs_path, "w", encoding="utf-8") as f:
        for p in selected_pairs:
            f.write(json.dumps(p) + "\n")

    dataset = IPERPoseDataset(
        pairs_jsonl=str(tmp_pairs_path),
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        augment=False,
    )

    # 2. Load Pipeline & Adapter
    print(f"\nLoading DeepGen pipeline from {args.model_path} ...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    pipe.vae.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe._load_extras(attn_implementation="sdpa")

    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe).to(device=device, dtype=dtype)
    adapter.eval()

    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)

    pair_data = []
    for d_idx in range(len(dataset)):
        item = dataset[d_idx]
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
            "idx": d_idx,
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

    all_stages = [0] + args.steps
    images_by_stage: Dict[int, List[Image.Image]] = {s: [] for s in all_stages}
    metrics_by_stage: Dict[int, List[dict]] = {s: [] for s in all_stages}

    # Step 0 (Adapter OFF / Base DiT)
    print("\n==========================================")
    print("Generating Step 0 (Adapter OFF / Base DiT)")
    print("==========================================")
    adapter.eval()
    for p_idx, data in enumerate(pair_data):
        save_file = out_dir / f"sample_{p_idx+1}_{data['appearance']}_step_0.png"
        if save_file.exists():
            print(f"  [Sample {p_idx+1}] Cached Step 0 exists, loading {save_file.name}")
            img_0 = Image.open(save_file).convert("RGB")
        else:
            t0 = time.time()
            with torch.no_grad():
                img_0 = controlled(
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
                    geometry_strength=0.0,
                    interaction_strength=0.0,
                ).images[0]
            img_0.save(save_file)
            print(f"  [Sample {p_idx+1}] Generated Step 0 in {time.time()-t0:.1f}s")

        m = compute_metrics(img_0, data["tgt_pil"])
        images_by_stage[0].append(img_0)
        metrics_by_stage[0].append(m)

    # Step 500 to 3000
    for step in args.steps:
        ckpt_file = exp_path / f"adapter_step_{step}.pt"
        if not ckpt_file.exists():
            ckpt_file = exp_path / f"checkpoint_step_{step}.pt"
        if not ckpt_file.exists():
            print(f"Warning: checkpoint for step {step} not found, skipping.")
            continue

        print(f"\n==========================================")
        print(f"Loading Step {step} ({ckpt_file.name})")
        print(f"==========================================")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.eval()

        for p_idx, data in enumerate(pair_data):
            save_file = out_dir / f"sample_{p_idx+1}_{data['appearance']}_step_{step}.png"
            if save_file.exists():
                print(f"  [Sample {p_idx+1}] Cached Step {step} exists, loading {save_file.name}")
                out_img = Image.open(save_file).convert("RGB")
            else:
                t0 = time.time()
                with torch.no_grad():
                    out_img = controlled(
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
                out_img.save(save_file)
                print(f"  [Sample {p_idx+1}] Generated Step {step} in {time.time()-t0:.1f}s")

            m = compute_metrics(out_img, data["tgt_pil"])
            images_by_stage[step].append(out_img)
            metrics_by_stage[step].append(m)

    # 3. Build Individual Sample Evolution Strips
    col_titles = ["Source (原图)", "Ground Truth (真值)", "Step 0 (无控制)"] + [f"Step {s}" for s in args.steps]

    tile_size = 280
    header_h = 44
    label_h = 32
    cell_h = tile_size + label_h
    num_cols = len(col_titles)

    font_title = get_font(13, bold=True)
    font_badge = get_font(11, bold=True)
    font_metric = get_font(10, bold=False)

    for p_idx, data in enumerate(pair_data):
        strip_w = num_cols * tile_size
        strip_h = header_h + cell_h
        strip_canvas = Image.new("RGB", (strip_w, strip_h), color=(15, 23, 42))
        strip_draw = ImageDraw.Draw(strip_canvas)

        for c, title in enumerate(col_titles):
            x = c * tile_size
            if c == 0:
                bg = (30, 41, 59)
            elif c == 1:
                bg = (20, 83, 45)
            elif c == 2:
                bg = (120, 53, 15)
            else:
                bg = (23, 37, 84)
            strip_draw.rectangle([(x, 0), (x + tile_size - 1, header_h - 1)], fill=bg)
            bbox = strip_draw.textbbox((0, 0), title, font=font_title)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            strip_draw.text((x + (tile_size - tw) // 2, (header_h - th) // 2), title, fill=(248, 250, 252), font=font_title)

        y_base = header_h

        # Column 0: Source
        src_t = data["src_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        strip_canvas.paste(src_t, (0, y_base))
        strip_draw.rectangle([(0, y_base + tile_size), (tile_size - 1, y_base + cell_h - 1)], fill=(30, 41, 59))
        strip_draw.text((12, y_base + tile_size + 8), f"{data['appearance']} Ref", fill=(148, 163, 184), font=font_badge)

        # Column 1: Ground Truth
        tgt_t = data["tgt_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        strip_canvas.paste(tgt_t, (tile_size, y_base))
        strip_draw.rectangle([(tile_size, y_base + tile_size), (tile_size * 2 - 1, y_base + cell_h - 1)], fill=(20, 83, 45))
        strip_draw.text((tile_size + 12, y_base + tile_size + 8), "Target GT Pose", fill=(187, 247, 208), font=font_badge)

        # Column 2 to N: Step 0, Step 500...
        for s_idx, s in enumerate(all_stages):
            c = s_idx + 2
            x = c * tile_size
            img = images_by_stage[s][p_idx]
            img_t = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            strip_canvas.paste(img_t, (x, y_base))

            m = metrics_by_stage[s][p_idx]
            metric_text = f"PSNR:{m['psnr']:.1f}dB | SSIM:{m['ssim']:.3f}"
            strip_draw.rectangle([(x, y_base + tile_size), (x + tile_size - 1, y_base + cell_h - 1)], fill=(15, 23, 42))
            strip_draw.text((x + 8, y_base + tile_size + 9), metric_text, fill=(186, 230, 253), font=font_metric)

        for c in range(1, num_cols):
            strip_draw.line([(c * tile_size, 0), (c * tile_size, strip_h)], fill=(51, 65, 85), width=1)

        single_strip_path = out_dir / f"sample_{p_idx+1}_{data['appearance']}_evolution_strip.png"
        strip_canvas.save(single_strip_path)
        print(f"Saved sample {p_idx+1} strip: {single_strip_path.name}")

    # 4. Build Master Multi-Sample Matrix Grid
    master_w = num_cols * tile_size
    master_h = header_h + len(pair_data) * cell_h
    master_canvas = Image.new("RGB", (master_w, master_h), color=(15, 23, 42))
    master_draw = ImageDraw.Draw(master_canvas)

    for c, title in enumerate(col_titles):
        x = c * tile_size
        if c == 0:
            bg = (30, 41, 59)
        elif c == 1:
            bg = (20, 83, 45)
        elif c == 2:
            bg = (120, 53, 15)
        else:
            bg = (23, 37, 84)
        master_draw.rectangle([(x, 0), (x + tile_size - 1, header_h - 1)], fill=bg)
        bbox = master_draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        master_draw.text((x + (tile_size - tw) // 2, (header_h - th) // 2), title, fill=(248, 250, 252), font=font_title)

    for p_idx, data in enumerate(pair_data):
        y_base = header_h + p_idx * cell_h

        src_t = data["src_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        master_canvas.paste(src_t, (0, y_base))
        master_draw.rectangle([(0, y_base + tile_size), (tile_size - 1, y_base + cell_h - 1)], fill=(30, 41, 59))
        master_draw.text((12, y_base + tile_size + 8), f"ID {data['appearance']} Ref", fill=(148, 163, 184), font=font_badge)

        tgt_t = data["tgt_pil"].resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        master_canvas.paste(tgt_t, (tile_size, y_base))
        master_draw.rectangle([(tile_size, y_base + tile_size), (tile_size * 2 - 1, y_base + cell_h - 1)], fill=(20, 83, 45))
        master_draw.text((tile_size + 12, y_base + tile_size + 8), "Target GT Pose", fill=(187, 247, 208), font=font_badge)

        for s_idx, s in enumerate(all_stages):
            c = s_idx + 2
            x = c * tile_size
            img = images_by_stage[s][p_idx]
            img_t = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            master_canvas.paste(img_t, (x, y_base))

            m = metrics_by_stage[s][p_idx]
            metric_text = f"PSNR:{m['psnr']:.1f}dB | SSIM:{m['ssim']:.3f}"
            master_draw.rectangle([(x, y_base + tile_size), (x + tile_size - 1, y_base + cell_h - 1)], fill=(15, 23, 42))
            master_draw.text((x + 8, y_base + tile_size + 9), metric_text, fill=(186, 230, 253), font=font_metric)

    for c in range(1, num_cols):
        master_draw.line([(c * tile_size, 0), (c * tile_size, master_h)], fill=(51, 65, 85), width=1)
    for r in range(1, len(pair_data) + 1):
        master_draw.line([(0, header_h + r * cell_h - 1), (master_w, header_h + r * cell_h - 1)], fill=(51, 65, 85), width=1)

    master_grid_path = out_dir / "v6_multistage_evolution_master_grid.png"
    master_canvas.save(master_grid_path)
    print(f"\n✓ Master multi-stage grid successfully generated: {master_grid_path}", flush=True)


if __name__ == "__main__":
    main()
