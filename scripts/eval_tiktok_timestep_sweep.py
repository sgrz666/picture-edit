#!/usr/bin/env python3
"""Timestep sweep for the TikTok single-sample V6.5 Face experiment.

Diagnoses the train/inference mismatch behind the negative generative result.

The training script optimises the Face branch at a *single* fixed timestep
(t=150) with one denoising step.  At sampling time the branch is active for
every step with ``progress > face_schedule.start_progress`` (t < 650), i.e.
across a sigma range it has never been optimised for.  This sweep measures
OFF/ON behaviour at several timesteps to locate where the branch helps and
where it hurts.

Usage::

    python scripts/eval_tiktok_timestep_sweep.py \
        --output-dir experiments/tiktok_objective_eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from scripts.precompute_tiktok_face_features_v65 import load_v65_face_cache
from train_iper_v65_face import (
    _crop_batch,
    _feature_distance,
    _landmark_nme,
    _load_torchscript_evaluator,
    _pairwise_distance,
    checkpoint_float_dtype,
    decode_latents_to_pixels,
    make_fixed_training_state,
)
from train_tiktok_v65_face import get_cached_conditions

ASSETS = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native"
RAW = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset"
MODEL = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
EVAL = "/home/shangguanrz/project/pic-edit/models/face_eval"
DEFAULT_CKPT = (
    "/home/shangguanrz/project/pic-edit/experiments/"
    "tiktok_v65_face_overfit_00001/checkpoints/checkpoint_step_000200.pt"
)
DEFAULT_CACHE = (
    "/home/shangguanrz/project/pic-edit/experiments/"
    "tiktok_v65_face_overfit_00001/condition_cache"
)
PROMPT = (
    "Change the person pose to match the target pose. "
    "Preserve identity, clothing, and background."
)


def face_schedule_multiplier(progress: float) -> float:
    """Mirror StrengthScheduleConfig default for the face branch."""
    start, end = 0.35, 1.0
    if progress < start or progress > end:
        return 0.0
    position = (progress - start) / (end - start)
    return 0.5 * (1.0 - math.cos(math.pi * position))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="00001")
    parser.add_argument("--source-stem", default="0014")
    parser.add_argument("--target-stem", default="0074")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--condition-cache", default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", default="experiments/tiktok_objective_eval")
    parser.add_argument("--timesteps", default="80,150,250,350,450,550,640")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    timesteps = [int(v) for v in args.timesteps.split(",") if v.strip()]

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from diffusers import DiffusionPipeline

    from src.data.tiktok_dataset import TikTokV65FaceOverfitLoader
    from src.pose_control.v6.checkpoint import load_v65_checkpoint
    from src.pose_control.v6.conditions import AdapterIdentityCondition
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    cache = load_v65_face_cache(Path(ASSETS) / args.sequence / "v65_face_features.pt")
    inputs = TikTokV65FaceOverfitLoader(
        raw_root=RAW,
        assets_root=ASSETS,
        sequence=args.sequence,
        source_stem=args.source_stem,
        target_stem=args.target_stem,
        resolution=args.resolution,
    ).load()
    batch = inputs.batch

    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype, trust_remote_code=True)
    pipe.to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe).to(device=device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = dict(checkpoint)
    checkpoint.pop("optimizer_state_dict", None)
    migration_dtype = checkpoint_float_dtype(checkpoint)
    adapter.to(device=device, dtype=migration_dtype)
    load_v65_checkpoint(adapter, checkpoint)
    adapter.to(device=device, dtype=dtype)
    adapter.eval()

    face_condition = inputs.face_condition.to(device=device, dtype=dtype)
    face_references = inputs.face_references.to(device=device, dtype=dtype)
    detail_condition = inputs.hand_detail_condition.to(device=device, dtype=dtype)
    detail_references = inputs.hand_detail_references.to(device=device, dtype=dtype)

    target_pixels = batch["tgt_image"].to(device, dtype=dtype)
    target_latent = pipe.pixels_to_latents(target_pixels).detach()
    latent_h, latent_w = target_latent.shape[-2:]

    ref_latent, sequence, pooled = get_cached_conditions(
        pipe, batch, PROMPT, args.condition_cache, device, dtype
    )
    bundle = adapter.condition_injector(
        normal_a=batch["normal"].to(device, dtype),
        pose_heatmap_a=batch["pose_heatmap"].to(device, dtype),
        part_onehot_a=batch["part_onehot"].to(device, dtype),
        smplx_global_a=batch["smplx_global"].to(device, dtype),
        human_mask_a=batch["human_mask"].to(device, dtype),
        task_id=batch["task_id"].to(device=device, dtype=torch.long),
    )
    source_person = torch.zeros(1, 2, ref_latent.shape[1], latent_h, latent_w, device=device, dtype=dtype)
    source_person[:, 0] = ref_latent[:1]
    identity = AdapterIdentityCondition(
        source_person_latents=source_person,
        source_indices=torch.tensor([[0, 1]], device=device),
    )

    prepared = adapter.prepare_conditioning(
        condition_bundle=bundle,
        identity_condition=identity,
        source_scene_latents=ref_latent[:1],
        target_latent_hw=(latent_h, latent_w),
        detail_condition=detail_condition,
        detail_references=detail_references,
        face_condition=face_condition,
        face_references=face_references,
    )

    fixed = make_fixed_training_state(
        target_latent.shape, seed=42, max_timestep=650, fixed_timestep=150,
        device=device, dtype=dtype,
    )
    noise = fixed["noise"]

    identity_eval = _load_torchscript_evaluator(f"{EVAL}/identity_encoder.ts", label="identity", device=device)
    lpips_eval = _load_torchscript_evaluator(f"{EVAL}/lpips.ts", label="lpips", device=device)
    landmark_eval = _load_torchscript_evaluator(f"{EVAL}/landmark_encoder.ts", label="landmark", device=device)

    boxes = face_condition.target_boxes[:, 0].float()
    target_01 = (target_pixels.float() * 0.5 + 0.5).clamp(0, 1)
    target_crop = _crop_batch(target_01, boxes)

    def predict(noisy, timestep, progress, face_strength):
        control = adapter(
            target_latents=noisy,
            prepared=prepared,
            cond_hidden_states=[[ref_latent[0]]],
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            timestep=timestep,
            denoise_progress=progress,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            hand_strength=1.0,
            face_strength=face_strength,
        )
        return pipe.transformer(
            hidden_states=noisy,
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            cond_hidden_states=[[ref_latent[0]]],
            timestep=timestep,
            block_controlnet_hidden_states=[
                v.to(dtype=dtype) for v in control.block_controlnet_hidden_states
            ],
            return_dict=False,
        )[0]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for t in timesteps:
            sigma = t / 1000.0
            timestep = torch.full((1,), t, device=device, dtype=torch.long)
            progress = torch.tensor([1.0 - sigma], device=device, dtype=dtype)
            noisy = (1.0 - sigma) * target_latent + sigma * noise
            row: dict[str, Any] = {
                "timestep": t,
                "sigma": round(sigma, 3),
                "face_schedule_multiplier": round(face_schedule_multiplier(1.0 - sigma), 4),
            }
            for name, strength in (("off", 0.0), ("on", 1.0)):
                velocity = predict(noisy, timestep, progress, strength)
                x0 = noisy - sigma * velocity
                pixels = (decode_latents_to_pixels(pipe, x0) * 0.5 + 0.5).clamp(0, 1)
                crop = _crop_batch(pixels.float(), boxes)
                row[f"facesim_{name}"] = float(
                    (1.0 - _feature_distance(identity_eval, crop, target_crop, cosine=True)).mean().cpu()
                )
                row[f"lpips_{name}"] = float(
                    _pairwise_distance(lpips_eval, crop, target_crop).mean().cpu()
                )
                row[f"nme_{name}"] = float(
                    _landmark_nme(landmark_eval, crop, target_crop).mean().cpu()
                )
                row[f"roi_mse_{name}"] = float(F.mse_loss(crop, target_crop).cpu())
            row["facesim_delta"] = row["facesim_on"] - row["facesim_off"]
            row["lpips_delta"] = row["lpips_on"] - row["lpips_off"]
            rows.append(row)
            print(
                f"t={t:4d} sigma={sigma:.2f} sched={row['face_schedule_multiplier']:.3f} | "
                f"FaceSim off/on={row['facesim_off']:.5f}/{row['facesim_on']:.5f} "
                f"(d={row['facesim_delta']:+.5f}) | "
                f"LPIPS off/on={row['lpips_off']:.5f}/{row['lpips_on']:.5f} "
                f"(d={row['lpips_delta']:+.5f})",
                flush=True,
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "tiktok_v65_face_overfit_00001",
        "checkpoint": args.checkpoint,
        "training_timestep": 150,
        "note": "单步单样本扫描；每个 t 只做一次前向，不是多步采样。",
        "rows": rows,
    }
    path = output / "timestep_sweep.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
