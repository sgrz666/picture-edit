#!/usr/bin/env python3
"""Generate Qwen21 adapter OFF/ON images on the fixed iPER comparison pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from src.data.iper_dataset import IPERPoseDataset
from src.data.tiktok_dataset import load_pose_dataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.runtime import MODEL_REVISION, PROMPT, dataset_image_to_pil


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    parser.add_argument("--checkpoint", type=Path, help="Omit to run the native Qwen21 baseline only")
    parser.add_argument("--pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl")
    parser.add_argument("--dataset-type", choices=("auto", "iper", "tiktok_v3"), default="auto")
    parser.add_argument("--align-mode", choices=("square_crop", "resize"), default="square_crop")
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "datasets/iPER/iper_assets_512_nvdiffrast")
    parser.add_argument("--tiktok-raw-root", type=Path, default=ROOT / "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset")
    parser.add_argument("--tiktok-assets-root", type=Path, default=ROOT / "datasets/TikTokDataset/TikTok_3d_assets_native")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/qwen21_iper_body/eval_50")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--disable-kv-cache", action="store_true")
    return parser.parse_args()


def composite_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def metrics(image: Image.Image, target: Image.Image) -> dict[str, float]:
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    b = np.asarray(target.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"evaluation image shapes differ: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    return {
        "mae": float(np.mean(np.abs(a - b))),
        "mse": mse,
        "psnr": float("inf") if mse == 0 else float(-10 * np.log10(mse / 255**2)),
        "ssim_skimage": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
    }


def comparison_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    size = 320
    canvas = Image.new("RGB", (size * len(images), size + 30), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((size, size)), (i * size, 30))
        draw.text((i * size + 7, 8), label, fill="white")
    return canvas


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    device = torch.device("cuda")
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    adapter = None
    if args.checkpoint is not None:
        adapter = Qwen21BodyAdapter().to(device=device, dtype=torch.float32)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(saved["adapter_state_dict"])
        adapter.eval()
    dataset = load_pose_dataset(
        args.pairs,
        dataset_type=args.dataset_type,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        tiktok_raw_root=args.tiktok_raw_root,
        tiktok_assets_root=args.tiktok_assets_root,
        resolution=args.resolution,
        align_mode=args.align_mode,
        augment=False,
    )
    rows = []
    for index in range(min(args.max_samples, len(dataset))):
        sample = dataset[index]
        batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value] for name, value in sample.items()}
        source = dataset_image_to_pil(sample["src_image"])
        target = dataset_image_to_pil(sample["tgt_image"])
        kwargs = dict(
            prompt=args.prompt, image=source, width=args.resolution, height=args.resolution, output_resolution=args.resolution,
            num_inference_steps=args.steps, true_cfg_scale=1.0,
            use_kv_cache=not args.disable_kv_cache,
        )
        generated = {}
        timing = {}
        memory = {}
        control_calls = None
        for mode in (("off", "on") if adapter is not None else ("off",)):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            began = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(args.seed)
            if mode == "on":
                with torch.no_grad():
                    grid = args.resolution // 16
                    geometry = adapter.encode_geometry(batch, (grid, grid))
                with Qwen21ControlContext(pipe.transformer, adapter, geometry) as control:
                    result = pipe(**kwargs, generator=generator)
                control_calls = control.call_counts
                if any(calls == 0 for calls in control_calls.values()):
                    raise RuntimeError("one or more Qwen21 control layers were not used during sampling")
            else:
                result = pipe(**kwargs, generator=generator)
            torch.cuda.synchronize()
            generated[mode] = composite_rgb(result.images[0])
            timing[mode] = time.perf_counter() - began
            memory[mode] = torch.cuda.max_memory_allocated() / 1024**3
        stem = f"{index:03d}_{sample['appearance']}_{sample['source_stem']}_{sample['target_stem']}"
        for mode, image in generated.items():
            image.save(image_dir / f"{stem}_{mode}.png")
        panels = [source, target, generated["off"]]
        labels = ["Source", "Target", "Qwen21 native"]
        if "on" in generated:
            panels.append(generated["on"])
            labels.append("Qwen21 body control")
        comparison_strip(panels, labels).save(image_dir / f"{stem}_strip.png")
        row = {
            "pair_idx": index, "appearance": sample["appearance"],
            "source_stem": sample["source_stem"], "target_stem": sample["target_stem"],
            "off": metrics(generated["off"], target),
            "on": metrics(generated["on"], target) if "on" in generated else None,
            "seconds": timing, "peak_cuda_gib": memory,
            "control_calls": control_calls,
            "image_stem": stem,
        }
        rows.append(row)
        (args.output / "per_sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{index + 1}/{min(args.max_samples, len(dataset))} {stem}: native PSNR {row['off']['psnr']:.2f} dB" + (f", controlled {row['on']['psnr']:.2f} dB" if row["on"] else ""), flush=True)
    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None, "model": str(args.model),
        "model_revision": MODEL_REVISION, "pairs": str(args.pairs),
        "steps": args.steps, "seed": args.seed, "prompt": args.prompt,
        "resolution": args.resolution,
        "kv_cache": not args.disable_kv_cache, "num_samples": len(rows),
        "timing_includes_control_geometry": True,
        "mean": {
            mode: {
                metric: float(np.mean([row[mode][metric] for row in rows]))
                for metric in ("mae", "mse", "psnr", "ssim_skimage")
            } for mode in (("off", "on") if adapter is not None else ("off",))
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
