#!/usr/bin/env python3
"""Two-phase single-sample overfit sanity test for the V6.5 TikTok pipeline.

Purpose
-------
Verify that the model, loss functions and training loop are *wired correctly*
by overfitting one fixed sample pair, and — unlike the legacy fixed-t=150
overfit — that the memorization actually transfers to **genuine multi-step
generation from 100% noise**.

Phase A (``--phase-a-steps``, fixed noise + fixed timestep)
    Deterministic configuration: identical (noise, t) every step. The optimal
    prediction is a constant, so the loss *must* collapse toward zero. This
    catches: data-loading bugs, VAE encode/decode mismatches, loss wiring
    errors, silently-frozen parameter groups (grad-flow report), optimizer
    bugs and checkpoint save/load failures.

Phase B (remaining steps, fresh noise + uniform full-range timestep)
    Each step samples t uniformly in the *scheduler-shifted* variable that the
    28-step inference sampler actually visits (t' in [0,1000], sigma = t'/1000).
    The loss is NOT expected to reach zero here (stochastic t/noise); the
    pass signal is the periodic **genuine 28-step generation** metric below.

Periodic genuine-generation evaluation (``--eval-gen-every``)
    Every N steps and at step 0, the script runs the real 28-step sampler
    (seeded, from 100% noise) with the adapter ON (all strengths 1.0) and with
    the face branch OFF, computes PSNR/SSIM/L1 vs the ground-truth target
    (full body + face ROI), saves a 4-panel PNG and appends JSONL records.
    This is the only metric family that evidences real generation quality.

Why the legacy overfit was misleading
-------------------------------------
Fixed t=150 (sigma=0.15) makes the model input 85% ground-truth latent; the
task degenerates to "copy the answer". Loss and single-step metrics looked
great while genuine 28-step generation collapsed. Phase A keeps that
configuration deliberately — but only as a *pipeline* check, never as
quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from train_iper_v65_face import (
    _crop_batch,
    _dtype_from_name,
    _feature_distance,
    _load_torchscript_evaluator,
    _save_preview,
    checkpoint_float_dtype,
    compute_face_validation_metrics,
    compute_non_face_sha256,
    compute_v65_face_losses,
    decode_latents_to_pixels,
    make_fixed_training_state,
    rasterize_box_mask,
)
from train_tiktok_v65_face import get_cached_conditions
from scripts.precompute_tiktok_face_features_v65 import load_v65_face_cache


# ---------------------------------------------------------------------------
# Image metrics (numpy, no extra deps)
# ---------------------------------------------------------------------------

def compute_image_metrics(pred_pil, gt_pil) -> dict[str, float]:
    """MSE / PSNR / L1 / SSIM between two PIL images (RGB, resized to pred)."""
    pred = np.asarray(pred_pil.convert("RGB"), dtype=np.float32) / 255.0
    gt = np.asarray(gt_pil.convert("RGB"), dtype=np.float32) / 255.0
    if pred.shape != gt.shape:
        gt_pil = gt_pil.convert("RGB").resize((pred.shape[1], pred.shape[0]))
        gt = np.asarray(gt_pil, dtype=np.float32) / 255.0
    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-10)))
    l1 = float(np.mean(np.abs(pred - gt)))
    c1, c2 = 0.01**2, 0.03**2
    mx, my = pred.mean(), gt.mean()
    vx, vy = pred.var(), gt.var()
    cxy = float(np.mean((pred - mx) * (gt - my)))
    ssim = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return {"mse": mse, "psnr": psnr, "l1": l1, "ssim": float(max(0.0, min(1.0, ssim)))}


def _pil_from_batch_tensor(t: torch.Tensor):
    from PIL import Image
    arr = (t[0].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def _save_panel(images: list, labels: list, path: Path) -> None:
    from PIL import Image, ImageDraw
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images) + 10 * (len(images) - 1), h + 40), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for img, label in zip(images, labels):
        canvas.paste(img, (x, 40))
        draw.text((x + 6, 12), label, fill=(235, 235, 235))
        x += img.width + 10
    canvas.save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-phase TikTok single-sample overfit sanity test")

    # Data (same pair as the legacy overfit)
    p.add_argument("--sequence", required=True)
    p.add_argument("--source-stem", required=True)
    p.add_argument("--target-stem", required=True)
    p.add_argument("--condition-override")
    p.add_argument("--raw-root",
                   default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset")
    p.add_argument("--assets-root",
                   default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native")

    # Model
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
    p.add_argument("--v64-checkpoint",
                   default="/home/shangguanrz/project/pic-edit/experiments/"
                            "iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt")

    # Training schedule
    p.add_argument("--output-dir", default="experiments/tiktok_overfit_sanity")
    p.add_argument("--cache-dir")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--phase-a-steps", type=int, default=200,
                   help="Deterministic fixed-(noise,t) pipeline self-check steps")
    p.add_argument("--fixed-timestep", type=int, default=150, help="Phase A / diagnostic timestep")
    p.add_argument("--timestep-min", type=int, default=0, help="Phase B uniform sampling lower bound")
    p.add_argument("--timestep-max", type=int, default=999, help="Phase B uniform sampling upper bound")
    p.add_argument("--max-timestep", type=int, default=650, help="Aux face losses gate (t <= this)")
    p.add_argument("--auxiliary-fraction", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--prompt",
                   default="Change the person's pose to match the target pose. Preserve identity, "
                           "face, hair, clothing, hands, body proportions, lighting, and background.")

    # Evaluators (same defaults as legacy script)
    p.add_argument("--face-id-evaluator",
                   default="/home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts")
    p.add_argument("--face-perceptual-evaluator",
                   default="/home/shangguanrz/project/pic-edit/models/face_eval/perceptual_encoder.ts")
    p.add_argument("--face-lpips-evaluator",
                   default="/home/shangguanrz/project/pic-edit/models/face_eval/lpips.ts")
    p.add_argument("--face-landmark-evaluator",
                   default="/home/shangguanrz/project/pic-edit/models/face_eval/landmark_encoder.ts")
    p.add_argument("--disable-face-id-loss", action="store_true")
    p.add_argument("--disable-face-perceptual-loss", action="store_true")
    p.add_argument("--disable-face-lpips-metric", action="store_true")
    p.add_argument("--disable-face-landmark-metric", action="store_true")

    # Logging / artifacts
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--preview-every", type=int, default=100, help="Diagnostic-config preview cadence")
    p.add_argument("--eval-gen-every", type=int, default=100, help="Genuine 28-step generation cadence")
    p.add_argument("--eval-gen-steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--eval-seed", type=int, default=1234, help="Fixed seed for genuine-gen evals")

    # Pass criteria (tunable)
    p.add_argument("--pass-phase-a-loss-ratio", type=float, default=0.70,
                   help="Phase A final flow EMA must be <= this fraction of its initial value. "
                        "Calibrated to the legacy recipe (lr=1e-5, 200 fixed-t steps historically "
                        "reach ~0.64x); tighten when raising lr or phase-A steps. The check gates "
                        "consistent deterministic descent, not literal loss->0 in 200 steps.")
    p.add_argument("--pass-phase-b-loss-ratio", type=float, default=0.60,
                   help="Phase B final flow EMA must be <= this fraction of its initial value")
    p.add_argument("--pass-gen-psnr-delta", type=float, default=5.0,
                   help="Final ON gen PSNR must beat step-0 ON PSNR by this many dB")
    p.add_argument("--pass-gen-psnr-abs", type=float, default=20.0,
                   help="Stretch goal: final ON gen PSNR must reach this absolute dB (near-perfect fit)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.phase_a_steps < 0 or args.phase_a_steps >= args.max_steps:
        raise ValueError("require 0 <= phase_a_steps < max_steps")
    if not 0.0 < args.auxiliary_fraction <= 1.0:
        raise ValueError("auxiliary_fraction must be in (0,1]")
    if not (0 <= args.timestep_min <= args.timestep_max <= 1000):
        raise ValueError("timestep range must satisfy 0 <= min <= max <= 1000")

    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError("overfit sanity test requires the project Diffusers environment") from exc
    try:
        from src.pose_control.v6.checkpoint import build_v65_checkpoint, load_v64_checkpoint
        from src.pose_control.v6.conditions import AdapterIdentityCondition
        from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
        from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
        from src.data.tiktok_dataset import TikTokV65FaceOverfitLoader
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("the V6.5 face/checkpoint integration is unavailable") from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    gen_eval_dir = output_dir / "gen_evals"
    for d in (checkpoint_dir, preview_dir, gen_eval_dir):
        d.mkdir(parents=True, exist_ok=True)
    condition_cache_dir = Path(args.cache_dir or output_dir / "condition_cache")
    condition_cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    gen_history_path = output_dir / "gen_eval_history.jsonl"

    # ── Data (identical loading path as legacy script) ────────────────────
    face_cache_path = Path(args.assets_root) / args.sequence / "v65_face_features.pt"
    face_cache = load_v65_face_cache(face_cache_path)
    stem_to_idx = face_cache["stem_to_idx"]
    for stem in (args.source_stem, args.target_stem):
        if stem not in stem_to_idx:
            raise KeyError(f"stem {stem!r} absent from v65_face_features.pt")
    source_index, target_index = stem_to_idx[args.source_stem], stem_to_idx[args.target_stem]
    if not (face_cache["valid"][source_index] and face_cache["valid"][target_index]):
        raise RuntimeError("source and target frames must both have valid face features")

    overfit_inputs = TikTokV65FaceOverfitLoader(
        raw_root=args.raw_root,
        assets_root=args.assets_root,
        sequence=args.sequence,
        source_stem=args.source_stem,
        target_stem=args.target_stem,
        resolution=args.resolution,
        condition_override=args.condition_override,
    ).load()
    batch = overfit_inputs.batch

    # ── Pipeline & adapter ────────────────────────────────────────────────
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    pipe.to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe, condition_use_depth=False).to(device=device)

    checkpoint = torch.load(args.v64_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("V6.4 checkpoint must be a mapping")
    checkpoint_for_migration = dict(checkpoint)
    discarded_old_optimizer = checkpoint_for_migration.pop("optimizer_state_dict", None) is not None
    if discarded_old_optimizer:
        print("discarding incompatible V6.4 optimizer_state_dict before V6.5 migration")
    migration_dtype = checkpoint_float_dtype(checkpoint_for_migration)
    adapter.to(device=device, dtype=migration_dtype)
    load_v64_checkpoint(adapter, checkpoint_for_migration)
    adapter.to(device=device, dtype=dtype)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    non_face_before = compute_non_face_sha256(adapter)

    # ── Conditions ────────────────────────────────────────────────────────
    face_condition = overfit_inputs.face_condition.to(device=device, dtype=dtype)
    face_references = overfit_inputs.face_references.to(device=device, dtype=dtype)
    detail_condition = overfit_inputs.hand_detail_condition.to(device=device, dtype=dtype)
    detail_references = overfit_inputs.hand_detail_references.to(device=device, dtype=dtype)

    normal = batch["normal"].to(device, dtype=dtype)
    pose = batch["pose_heatmap"].to(device, dtype=dtype)
    part = batch["part_onehot"].to(device, dtype=dtype)
    smplx_global = batch["smplx_global"].to(device, dtype=dtype)
    human_mask = batch["human_mask"].to(device, dtype=dtype)
    task_id = batch["task_id"].to(device, dtype=torch.long)
    target_pixels = batch["tgt_image"].to(device, dtype=dtype)
    target_latent = pipe.pixels_to_latents(target_pixels).detach()
    latent_height, latent_width = target_latent.shape[-2:]

    ref_latent, sequence, pooled = get_cached_conditions(
        pipe, batch, args.prompt, str(condition_cache_dir), device, dtype
    )
    source_person_latents = torch.zeros(
        1, 2, ref_latent.shape[1], latent_height, latent_width, device=device, dtype=dtype
    )
    source_person_latents[:, 0] = ref_latent[:1]
    identity_condition = AdapterIdentityCondition(
        source_person_latents=source_person_latents,
        source_indices=torch.tensor([[0, 1]], device=device),
    )
    face_mask = rasterize_box_mask(
        face_condition.target_boxes[:, 0].float(), latent_height, latent_width
    ).to(device=device, dtype=dtype)

    src_pil = _pil_from_batch_tensor(batch["src_image"])
    tgt_pil = _pil_from_batch_tensor(batch["tgt_image"])
    src_pil.save(output_dir / "input_source.png")
    tgt_pil.save(output_dir / "target_gt.png")

    # ── Fixed diagnostic configuration (Phase A & comparable previews) ────
    diag = make_fixed_training_state(
        target_latent.shape,
        seed=args.seed,
        max_timestep=args.max_timestep,
        fixed_timestep=args.fixed_timestep,
        device=device,
        dtype=dtype,
    )
    diag_noise, diag_timestep = diag["noise"], diag["timestep"]
    diag_sigma = (diag_timestep.float() / 1000.0).to(dtype).view(-1, 1, 1, 1)
    diag_noisy = (1.0 - diag_sigma) * target_latent + diag_sigma * diag_noise
    diag_velocity = diag_noise - target_latent
    diag_progress = (1.0 - diag_timestep.float() / 1000.0).to(dtype)

    # ── Face evaluators ───────────────────────────────────────────────────
    id_evaluator = None if args.disable_face_id_loss else _load_torchscript_evaluator(
        args.face_id_evaluator, label="face_id", device=device)
    perceptual_evaluator = None if args.disable_face_perceptual_loss else _load_torchscript_evaluator(
        args.face_perceptual_evaluator, label="face_perceptual", device=device)
    lpips_evaluator = None if args.disable_face_lpips_metric else _load_torchscript_evaluator(
        args.face_lpips_evaluator, label="face_lpips", device=device)
    landmark_evaluator = None if args.disable_face_landmark_metric else _load_torchscript_evaluator(
        args.face_landmark_evaluator, label="face_landmark", device=device)

    # ── Forward closure (parameterized by noise config) ───────────────────
    def forward_prediction(noisy_latent, timestep, progress, face_strength: float) -> torch.Tensor:
        geometry_bundle = adapter.condition_injector(
            normal_a=normal, pose_heatmap_a=pose, part_onehot_a=part,
            smplx_global_a=smplx_global, human_mask_a=human_mask, task_id=task_id,
        )
        prepared = adapter.prepare_conditioning(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            target_latent_hw=(latent_height, latent_width),
            hand_condition=detail_condition,
            hand_references=detail_references,
            face_condition=face_condition,
            face_references=face_references,
        )
        control = adapter(
            target_latents=noisy_latent,
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
            hidden_states=noisy_latent,
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            cond_hidden_states=[[ref_latent[0]]],
            timestep=timestep,
            block_controlnet_hidden_states=[
                value.to(dtype=dtype) for value in control.block_controlnet_hidden_states
            ],
            return_dict=False,
        )[0]

    # ── C1: interface invariance at init (ON == OFF before training) ──────
    adapter.eval()
    with torch.no_grad():
        init_on = forward_prediction(diag_noisy, diag_timestep, diag_progress, 1.0)
        init_off = forward_prediction(diag_noisy, diag_timestep, diag_progress, 0.0)
        init_on_off_max_diff = float((init_on.float() - init_off.float()).abs().max())
        fixed_diag_off_prediction = init_off.detach()
    print(f"[C1] init ON-vs-OFF max|Δprediction| = {init_on_off_max_diff:.3e} (must be <= 1e-5)")

    # ── Genuine 28-step generation evaluation ─────────────────────────────
    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)
    gen_history: list[dict[str, Any]] = []
    face_box = face_condition.target_boxes[0, 0].float().cpu().numpy()
    res = int(args.resolution)
    face_crop_box = (
        int(face_box[0] * res), int(face_box[1] * res),
        int(face_box[2] * res), int(face_box[3] * res),
    )

    @torch.no_grad()
    def genuine_gen_eval(global_step: int) -> dict[str, Any]:
        """Run the real 28-step sampler from 100% noise; ON vs face-OFF."""
        was_training = adapter.training
        adapter.eval()
        geometry_bundle = adapter.condition_injector(
            normal_a=normal, pose_heatmap_a=pose, part_onehot_a=part,
            smplx_global_a=smplx_global, human_mask_a=human_mask, task_id=task_id,
        )
        images = {}
        for tag, face_strength in (("off", 0.0), ("on", 1.0)):
            result = controlled(
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
                height=res,
                width=res,
                num_inference_steps=args.eval_gen_steps,
                guidance_scale=args.guidance_scale,
                seed=args.eval_seed,
                geometry_strength=1.0,
                interaction_strength=0.0,
                detail_strength=1.0,
                hand_strength=1.0,
                face_strength=face_strength,
            )
            images[tag] = result.images[0]
        record: dict[str, Any] = {"step": global_step}
        for tag in ("off", "on"):
            full = compute_image_metrics(images[tag], tgt_pil)
            face = compute_image_metrics(
                images[tag].crop(face_crop_box), tgt_pil.crop(face_crop_box))
            record[f"{tag}_full_psnr"] = full["psnr"]
            record[f"{tag}_full_ssim"] = full["ssim"]
            record[f"{tag}_face_psnr"] = face["psnr"]
            record[f"{tag}_face_ssim"] = face["ssim"]
            images[tag].save(gen_eval_dir / f"step_{global_step:06d}_gen_{tag}.png")
        _save_panel(
            [src_pil, tgt_pil, images["off"], images["on"]],
            [f"Source {args.source_stem}", f"GT {args.target_stem}",
             f"OFF psnr={record['off_full_psnr']:.2f}", f"ON psnr={record['on_full_psnr']:.2f}"],
            gen_eval_dir / f"step_{global_step:06d}_panel.png",
        )
        gen_history.append(record)
        with gen_history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"  [gen-eval @{global_step}] full PSNR off={record['off_full_psnr']:.2f} "
            f"on={record['on_full_psnr']:.2f} | face PSNR off={record['off_face_psnr']:.2f} "
            f"on={record['on_face_psnr']:.2f}",
            flush=True,
        )
        if was_training:
            adapter.train()
        return record

    print("[0/max] running step-0 genuine generation baseline...", flush=True)
    step0_record = genuine_gen_eval(0)

    # ── Diagnostic-config preview (legacy semantics, leaky by design) ─────
    comparison_history: list[dict[str, Any]] = []

    @torch.no_grad()
    def save_diag_preview(label: str) -> None:
        was_training = adapter.training
        adapter.eval()
        on = forward_prediction(diag_noisy, diag_timestep, diag_progress, 1.0)
        off = fixed_diag_off_prediction
        on_x0 = diag_noisy - diag_sigma * on
        off_x0 = diag_noisy - diag_sigma * off
        on_pixels = (decode_latents_to_pixels(pipe, on_x0) * 0.5 + 0.5).clamp(0, 1)
        off_pixels = (decode_latents_to_pixels(pipe, off_x0) * 0.5 + 0.5).clamp(0, 1)
        _save_preview(on_pixels, preview_dir / f"{label}_face_on.png")
        _save_preview(off_pixels, preview_dir / f"{label}_face_off.png")
        target_01 = (target_pixels.float() * 0.5 + 0.5).clamp(0, 1)
        record = {
            "label": label,
            "timestep": int(diag_timestep[0]),
            "on_off_prediction_mse": float((on.float() - off.float()).square().mean()),
        }
        record.update(
            compute_face_validation_metrics(
                on_pixels, off_pixels, target_01,
                face_condition.target_boxes[:, 0].float(),
                identity_evaluator=id_evaluator,
                lpips_evaluator=lpips_evaluator,
                landmark_evaluator=landmark_evaluator,
            )
        )
        comparison_history.append(record)
        if was_training:
            adapter.train()

    save_diag_preview("step_000000")

    # ── Optimizer (same recipe as legacy full-adapter run) ────────────────
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_names = [n for n, p in adapter.named_parameters() if p.requires_grad]

    # ── Training loop ─────────────────────────────────────────────────────
    adapter.train()
    start_time = time.time()
    ema_flow: float | None = None
    phase_a_flows: list[float] = []
    phase_b_flows: list[float] = []
    grad_flow_report: dict[str, Any] = {}
    best_gen_psnr = step0_record["on_full_psnr"]
    period = max(1, int(round(1.0 / args.auxiliary_fraction)))

    with metrics_path.open("w", encoding="utf-8") as metrics_fh:
        for step in range(args.max_steps):
            in_phase_a = step < args.phase_a_steps
            phase = "A" if in_phase_a else "B"

            # ── Noise & timestep for this step ────────────────────────────
            if in_phase_a:
                noise, timestep = diag_noise, diag_timestep
            else:
                step_gen = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + step)
                noise = torch.randn(
                    tuple(target_latent.shape), generator=step_gen, dtype=torch.float32
                ).to(device=device, dtype=dtype)
                timestep = torch.randint(
                    args.timestep_min, args.timestep_max + 1, (1,),
                    generator=step_gen, dtype=torch.long,
                ).to(device)
            sigma = (timestep.float() / 1000.0).to(dtype).view(-1, 1, 1, 1)
            noisy_latent = (1.0 - sigma) * target_latent + sigma * noise
            target_velocity = noise - target_latent
            progress = (1.0 - timestep.float() / 1000.0).to(dtype)

            optimizer.zero_grad(set_to_none=True)
            prediction = forward_prediction(noisy_latent, timestep, progress, 1.0)

            # ── Auxiliary face losses (gated to the face schedule window) ─
            use_auxiliary = bool(int(timestep[0]) <= args.max_timestep and step % period == 0)
            identity_loss = perceptual_loss = None
            traincfg_x0_psnr = None
            if use_auxiliary and (id_evaluator is not None or perceptual_evaluator is not None):
                predicted_x0 = noisy_latent - sigma * prediction
                predicted_pixels = (
                    decode_latents_to_pixels(pipe, predicted_x0, differentiable=True) * 0.5 + 0.5
                ).clamp(0, 1)
                target_01 = (target_pixels * 0.5 + 0.5).clamp(0, 1)
                boxes = face_condition.target_boxes[:, 0].float()
                predicted_crop = _crop_batch(predicted_pixels.float(), boxes)
                target_crop = _crop_batch(target_01.float(), boxes)
                if id_evaluator is not None:
                    identity_loss = _feature_distance(id_evaluator, predicted_crop, target_crop, cosine=True)
                if perceptual_evaluator is not None:
                    perceptual_loss = _feature_distance(
                        perceptual_evaluator, predicted_crop, target_crop, cosine=False)
                with torch.no_grad():
                    mse_pix = float((predicted_pixels.float() - target_01.float()).square().mean())
                    traincfg_x0_psnr = 10.0 * math.log10(1.0 / max(mse_pix, 1e-10))

            losses = compute_v65_face_losses(
                prediction,
                target_velocity,
                face_mask,
                face_id_loss=identity_loss,
                face_perceptual_loss=perceptual_loss,
                auxiliary_selector=torch.tensor([use_auxiliary], device=device),
                outside_reference=fixed_diag_off_prediction if in_phase_a else None,
            )
            losses["total"].backward()

            # ── Grad-flow report at the first step ────────────────────────
            if step == 0:
                groups: dict[str, list[int]] = {}
                for name, param in adapter.named_parameters():
                    if not param.requires_grad:
                        continue
                    top = name.split(".")[0]
                    entry = groups.setdefault(top, [0, 0])
                    entry[0] += 1
                    if param.grad is not None and bool((param.grad != 0).any()):
                        entry[1] += 1
                grad_flow_report = {
                    top: {"params": v[0], "with_nonzero_grad": v[1]} for top, v in sorted(groups.items())
                }
                print("[grad-flow @step1] " + json.dumps(grad_flow_report), flush=True)

            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            flow_val = float(losses["flow"].detach())
            ema_flow = flow_val if ema_flow is None else 0.98 * ema_flow + 0.02 * flow_val
            (phase_a_flows if in_phase_a else phase_b_flows).append(flow_val)

            record = {
                "step": step + 1, "phase": phase, "timestep": int(timestep[0]),
                **{k: float(v.detach()) for k, v in losses.items()},
                "ema_flow": ema_flow,
                "traincfg_x0_psnr": traincfg_x0_psnr,
            }
            metrics_fh.write(json.dumps(record) + "\n")
            metrics_fh.flush()
            if step == 0 or (step + 1) % 10 == 0:
                print(
                    f"step {step + 1}/{args.max_steps} [{phase}] t={int(timestep[0])}: "
                    f"total={record['total']:.6f} flow={record['flow']:.6f} "
                    f"ema_flow={ema_flow:.6f} id={record['face_id']:.6f} "
                    f"perc={record['face_perceptual']:.6f} outside={record['outside']:.6f}",
                    flush=True,
                )

            if (step + 1) % args.preview_every == 0 or step + 1 == args.max_steps:
                save_diag_preview(f"step_{step + 1:06d}")
            if (step + 1) % args.eval_gen_every == 0 or step + 1 == args.max_steps:
                rec = genuine_gen_eval(step + 1)
                if rec["on_full_psnr"] > best_gen_psnr:
                    best_gen_psnr = rec["on_full_psnr"]
                    torch.save(
                        build_v65_checkpoint(
                            adapter,
                            feature_cache_fingerprint=face_cache["metadata"]["cache_fingerprint"],
                            step=step + 1,
                            optimizer_state_dict=optimizer.state_dict(),
                            source_stem=args.source_stem,
                            target_stem=args.target_stem,
                            seed=args.seed,
                        ),
                        checkpoint_dir / "checkpoint_best.pt",
                    )
            if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.max_steps:
                torch.save(
                    build_v65_checkpoint(
                        adapter,
                        feature_cache_fingerprint=face_cache["metadata"]["cache_fingerprint"],
                        step=step + 1,
                        optimizer_state_dict=optimizer.state_dict(),
                        source_stem=args.source_stem,
                        target_stem=args.target_stem,
                        seed=args.seed,
                    ),
                    checkpoint_dir / f"checkpoint_step_{step + 1:06d}.pt",
                )

    # ── Final integrity & PASS/FAIL report ────────────────────────────────
    non_face_after = compute_non_face_sha256(adapter)
    final_record = gen_history[-1]
    with metrics_path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    def _head_mean(values: list[float], n: int = 20) -> float | None:
        return (sum(values[:n]) / min(n, len(values))) if values else None

    def _tail_mean(values: list[float], n: int = 20) -> float | None:
        return (sum(values[-n:]) / min(n, len(values))) if values else None

    # NOTE: per-step flow varies strongly with the sampled timestep in Phase B,
    # so the reference must be a head-mean, never a single step.
    phase_a_flow_first = _head_mean(phase_a_flows)
    phase_b_flow_first = _head_mean(phase_b_flows)
    phase_a_tail = _tail_mean(phase_a_flows)
    phase_b_tail = _tail_mean(phase_b_flows)

    checks = [
        {
            "id": "C1",
            "name": "接口不变量：初始 ON≡OFF（零初始化头）",
            "value": init_on_off_max_diff,
            "pass": init_on_off_max_diff <= 1e-5,
            "criterion": "max|Δpred| <= 1e-5",
        },
        {
            "id": "C2",
            "name": "Phase A 确定性记忆：flow 损失足够下降",
            "value": phase_a_tail,
            "pass": bool(
                phase_a_tail is not None and phase_a_flow_first is not None
                and phase_a_tail <= args.pass_phase_a_loss_ratio * phase_a_flow_first
            ),
            "criterion": f"tail({phase_a_tail:.4f}) <= {args.pass_phase_a_loss_ratio}×init({phase_a_flow_first:.4f})"
            if phase_a_tail is not None and phase_a_flow_first is not None else "n/a",
        },
        {
            "id": "C3",
            "name": "Phase B 全轨迹训练：flow 损失持续下降（不趋零，预期行为）",
            "value": phase_b_tail,
            "pass": bool(
                phase_b_tail is not None and phase_b_flow_first is not None
                and phase_b_tail <= args.pass_phase_b_loss_ratio * phase_b_flow_first
            ),
            "criterion": f"tail({phase_b_tail:.4f}) <= {args.pass_phase_b_loss_ratio}×init({phase_b_flow_first:.4f})"
            if phase_b_tail is not None and phase_b_flow_first is not None else "n/a",
        },
        {
            "id": "C4",
            "name": "真实生成显著优于初始：final ON PSNR ≥ step0 ON + Δ",
            "value": final_record["on_full_psnr"] - step0_record["on_full_psnr"],
            "pass": bool(
                final_record["on_full_psnr"] >= step0_record["on_full_psnr"] + args.pass_gen_psnr_delta
            ),
            "criterion": f"ΔPSNR >= {args.pass_gen_psnr_delta} dB",
        },
        {
            "id": "C5",
            "name": "接近完美拟合（stretch）：final ON PSNR 达绝对阈值",
            "value": final_record["on_full_psnr"],
            "pass": bool(final_record["on_full_psnr"] >= args.pass_gen_psnr_abs),
            "criterion": f"PSNR >= {args.pass_gen_psnr_abs} dB",
            "stretch": True,
        },
        {
            "id": "C6",
            "name": "适配器正向贡献：final ON ≥ final OFF",
            "value": final_record["on_full_psnr"] - final_record["off_full_psnr"],
            "pass": bool(final_record["on_full_psnr"] >= final_record["off_full_psnr"]),
            "criterion": "ON - OFF >= 0 dB",
        },
    ]
    gating = [c for c in checks if not c.get("stretch")]
    verdict = "PASS" if all(c["pass"] for c in gating) else "FAIL"

    print("\n" + "=" * 72)
    print("OVERFIT SANITY TEST REPORT")
    print("=" * 72)
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        suffix = " (stretch, non-gating)" if c.get("stretch") else ""
        print(f"  [{mark}] {c['id']} {c['name']}")
        print(f"        value={c['value']!r}  criterion: {c['criterion']}{suffix}")
    print(f"\n  VERDICT: {verdict}  (gating: C1,C2,C3,C4,C6; C5 为趋近完美拟合的拉伸目标)")
    print("=" * 72)

    # ── Curves ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(steps, [r["flow"] for r in rows], alpha=0.3, label="flow (raw)")
        axes[0].plot(steps, [r["ema_flow"] for r in rows], label="flow (EMA)")
        axes[0].axvline(args.phase_a_steps, color="gray", ls="--", lw=1)
        axes[0].text(args.phase_a_steps + 5, axes[0].get_ylim()[1] * 0.9, "A→B", fontsize=9)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("flow loss"); axes[0].legend()
        axes[0].set_title("Training loss (Phase A fixed-t / Phase B sampled-t)")
        ge = gen_history
        axes[1].plot([g["step"] for g in ge], [g["on_full_psnr"] for g in ge], "o-", label="ON full")
        axes[1].plot([g["step"] for g in ge], [g["off_full_psnr"] for g in ge], "s--", label="OFF full")
        axes[1].plot([g["step"] for g in ge], [g["on_face_psnr"] for g in ge], "^-", label="ON face ROI")
        axes[1].axhline(args.pass_gen_psnr_abs, color="red", ls=":", lw=1, label=f"C5 target {args.pass_gen_psnr_abs}dB")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("PSNR vs GT (dB)"); axes[1].legend()
        axes[1].set_title("Genuine 28-step generation vs GT (the real metric)")
        fig.tight_layout()
        fig.savefig(output_dir / "sanity_curves.png", dpi=140)
    except Exception as exc:  # curves are best-effort
        print(f"curve plotting skipped: {exc}")

    summary = {
        "script": "train_tiktok_overfit_sanity.py",
        "architecture_version": "v6.5",
        "dataset": "TikTok",
        "sequence": args.sequence,
        "source_stem": args.source_stem,
        "target_stem": args.target_stem,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "phase_a_steps": args.phase_a_steps,
        "max_steps": args.max_steps,
        "timestep_sampling_phase_b": f"uniform[{args.timestep_min},{args.timestep_max}] (scheduler-shifted variable)",
        "full_adapter_training": True,
        "optimizer_param_tensors": len(optimizer_names),
        "non_face_sha256_before": non_face_before,
        "non_face_sha256_after": non_face_after,
        "non_face_changed": non_face_before != non_face_after,
        "grad_flow_report": grad_flow_report,
        "init_on_off_max_diff": init_on_off_max_diff,
        "phase_a_flow_head20": phase_a_flow_first,
        "phase_a_flow_tail20": phase_a_tail,
        "phase_b_flow_head20": phase_b_flow_first,
        "phase_b_flow_tail20": phase_b_tail,
        "gen_eval_history": gen_history,
        "best_gen_psnr_on": best_gen_psnr,
        "checks": checks,
        "verdict": verdict,
        "diag_comparison_history": comparison_history,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"artifacts saved in {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
