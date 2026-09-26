#!/usr/bin/env python3
"""V6.5 independent Face Adapter single-sample TikTok training entry point.

Mirrors the iPER V6.5 Face training pipeline
(:pymod:`train_iper_v65_face`) with TikTok-specific data loading via
:class:`TikTokV65FaceOverfitLoader`.

The light-weight helpers in this file are intentionally importable without
Diffusers, InsightFace, or a GPU.  Heavy training dependencies are imported
only inside :func:`run_training`, with actionable errors when unavailable.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared V6.5 Face training helpers (importable without GPU / Diffusers)
# ---------------------------------------------------------------------------
from train_iper_v65_face import (
    FACE_PARAMETER_PREFIXES,
    FP32MasterAdamW,
    build_face_optimizer,
    checkpoint_float_dtype,
    compute_face_validation_metrics,
    compute_non_face_sha256,
    compute_v65_face_losses,
    decode_latents_to_pixels,
    make_fixed_training_state,
    rasterize_box_mask,
    select_face_trainable_parameters,
    update_face_optimizer_lrs,
)

# Private helpers from the iPER script needed in the training loop.
from train_iper_v65_face import (
    _crop_batch,
    _feature_distance,
    _load_torchscript_evaluator,
)

# TikTok face-cache loader (same V6.5 schema as iPER).
from scripts.precompute_tiktok_face_features_v65 import load_v65_face_cache


# ───────────────────────────────────────────────────────────────────────────
# Condition caching (replaces deleted ``train_iper_v6_native.get_cached_conditions``)
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_cached_conditions(
    pipe: Any,
    batch: dict[str, Any],
    prompt: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieves or precomputes VLM semantic features and reference latents.

    Caches per source image stem to avoid repetitive VLM execution.
    Supports arbitrary batch sizes (B >= 1) with per-sample caching.
    """
    os_cache_dir = Path(cache_dir)
    os_cache_dir.mkdir(parents=True, exist_ok=True)

    appearances = batch.get("appearance", ["default"])
    source_stems = batch.get("source_stem", ["frame_0"])
    b_count = len(appearances)
    ref_list = []
    seq_list = []
    pool_list = []

    for b in range(b_count):
        app = appearances[b]
        src_stem = source_stems[b]
        cache_path = os_cache_dir / f"{app}_{src_stem}.pt"

        loaded = False
        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, map_location=device, weights_only=False)
                ref_list.append(cached_data["reference_latent"].to(device, dtype=dtype))
                seq_list.append(cached_data["sequence"].to(device, dtype=dtype))
                pool_list.append(cached_data["pooled"].to(device, dtype=dtype))
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            source_pixels = batch["src_image"][b : b + 1].to(device, dtype=dtype)
            if (
                hasattr(pipe, "connector_module")
                and hasattr(pipe, "llm")
                and hasattr(pipe, "get_semantic_features_dynamic")
                and getattr(pipe, "connector_module", None) is not None
            ):
                image_embeds, image_grid = pipe.get_semantic_features_dynamic([source_pixels[0]])
                text_inputs = pipe.prepare_image2image_prompts(
                    [prompt],
                    num_refs=[1],
                    ref_lens=[len(image_embeds[0])],
                )
                text_inputs.update(
                    image_embeds=torch.cat(image_embeds),
                    image_grid_thw=image_grid,
                )
                queries = pipe.connector_module.meta_queries[None]
                forward_inputs = pipe.prepare_forward_input(query_embeds=queries, **text_inputs)

                llm_output = pipe.llm(
                    **forward_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                )
                hidden_states = llm_output.hidden_states
                merged = torch.cat(
                    [hidden_states[index] for index in range(len(hidden_states) - 2, 0, -6)],
                    dim=-1,
                )
                pooled_b, sequence_b = pipe.connector_module.llm2dit(merged)
                ref_latent_b = pipe.pixels_to_latents(source_pixels)
            elif hasattr(pipe, "encode_prompt"):
                res = pipe.encode_prompt(prompt=prompt)
                sequence_b = res[0]
                pooled_b = res[2] if len(res) >= 4 else res[1]
                ref_latent_b = pipe.pixels_to_latents(source_pixels)
            else:
                raise RuntimeError(
                    "DeepGen pipeline does not expose get_semantic_features_dynamic or encode_prompt; "
                    "cannot encode text embeddings for training"
                )

            torch.save(
                {
                    "reference_latent": ref_latent_b.cpu().to(torch.bfloat16),
                    "sequence": sequence_b.cpu().to(torch.bfloat16),
                    "pooled": pooled_b.cpu().to(torch.bfloat16),
                },
                cache_path,
            )
            ref_list.append(ref_latent_b.to(device, dtype=dtype))
            seq_list.append(sequence_b.to(device, dtype=dtype))
            pool_list.append(pooled_b.to(device, dtype=dtype))

    reference_latent = torch.cat(ref_list, dim=0)
    sequence = torch.cat(seq_list, dim=0)
    pooled = torch.cat(pool_list, dim=0)
    return reference_latent, sequence, pooled


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V6.5 independent Face Adapter on TikTok"
    )
    parser.add_argument("--stage", choices=("overfit",), default="overfit")

    # ── TikTok dataset paths ──────────────────────────────────────────────
    parser.add_argument("--sequence", required=True, help="TikTok sequence ID, e.g. 00001")
    parser.add_argument("--source-stem", required=True, help="Source frame stem, e.g. 0014")
    parser.add_argument("--target-stem", required=True, help="Target frame stem, e.g. 0074")
    parser.add_argument(
        "--condition-override",
        help="Audited single-target pose/part .pt; leaves v6_conditions.pt untouched",
    )
    parser.add_argument(
        "--raw-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
        help="Root path to raw TikTok images and masks",
    )
    parser.add_argument(
        "--assets-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
        help="Root path to TikTok 3D assets (v6_conditions.pt, v65_face_features.pt, etc.)",
    )

    # ── Model and checkpoint ──────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
    )
    parser.add_argument(
        "--v64-checkpoint",
        default=(
            "/home/shangguanrz/project/pic-edit/experiments/"
            "iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt"
        ),
    )

    # ── Training hyper-parameters ─────────────────────────────────────────
    parser.add_argument("--output-dir", default="experiments/tiktok_v65_face_overfit")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-timestep", type=int, default=650)
    parser.add_argument(
        "--fixed-timestep",
        type=int,
        default=150,
        help="Low-noise overfit timestep; 150 keeps the late texture path active.",
    )
    parser.add_argument("--auxiliary-fraction", type=float, default=0.25)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--prompt",
        default="Change the person pose to match the target pose. Preserve identity, clothing, and background.",
    )

    # ── Face evaluators ───────────────────────────────────────────────────
    parser.add_argument(
        "--face-id-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts",
    )
    parser.add_argument(
        "--face-perceptual-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/perceptual_encoder.ts",
    )
    parser.add_argument(
        "--face-lpips-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/lpips.ts",
    )
    parser.add_argument(
        "--face-landmark-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/landmark_encoder.ts",
    )
    parser.add_argument("--disable-face-id-loss", action="store_true")
    parser.add_argument("--disable-face-perceptual-loss", action="store_true")
    parser.add_argument("--disable-face-lpips-metric", action="store_true")
    parser.add_argument("--disable-face-landmark-metric", action="store_true")

    # ── Logging ───────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--preview-every", type=int, default=250)

    return parser.parse_args(argv)


# ───────────────────────────────────────────────────────────────────────────
# Small utilities
# ───────────────────────────────────────────────────────────────────────────

def _dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _save_preview(image: torch.Tensor, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for previews") from exc
    pixels = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((pixels * 255.0 + 0.5).astype("uint8")).save(path)


# ───────────────────────────────────────────────────────────────────────────
# Training entry point
# ───────────────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    """Run single-sample overfit training of the V6.5 Face Adapter on TikTok."""

    if args.max_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("max_steps must be positive and warmup_steps non-negative")
    if not 0.0 < args.auxiliary_fraction <= 1.0:
        raise ValueError("auxiliary_fraction must be in (0,1]")

    # ── Lazy heavy imports ────────────────────────────────────────────────
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError("V6.5 training requires the project Diffusers environment") from exc
    try:
        from src.pose_control.v6.checkpoint import build_v65_checkpoint, load_v64_checkpoint
        from src.pose_control.v6.conditions import AdapterIdentityCondition
        from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
        from src.data.tiktok_dataset import TikTokV65FaceOverfitLoader
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "the V6.5 face/checkpoint integration is unavailable; "
            "update the adapter before training"
        ) from exc

    # ── Device / dtype / seed ─────────────────────────────────────────────
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── Output directories ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    condition_cache_dir = Path(args.cache_dir or output_dir / "condition_cache")
    condition_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load TikTok face cache (for fingerprint & frame validation) ──────
    face_cache_path = Path(args.assets_root) / args.sequence / "v65_face_features.pt"
    face_cache = load_v65_face_cache(face_cache_path)
    stems = face_cache["frame_stems"]
    stem_to_idx = face_cache["stem_to_idx"]
    if args.source_stem not in stem_to_idx:
        raise KeyError(f"source stem {args.source_stem!r} absent from v65_face_features.pt")
    if args.target_stem not in stem_to_idx:
        raise KeyError(f"target stem {args.target_stem!r} absent from v65_face_features.pt")
    source_index = stem_to_idx[args.source_stem]
    target_index = stem_to_idx[args.target_stem]
    if not (face_cache["valid"][source_index] and face_cache["valid"][target_index]):
        raise RuntimeError(
            "source and target frames must both have valid face features in v65_face_features.pt"
        )

    # ── Load TikTok data ──────────────────────────────────────────────────
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
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    pipe.to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(
        pipe, condition_use_depth=False
    ).to(device=device)

    # ── Load V6.4 checkpoint & freeze ─────────────────────────────────────
    if not args.v64_checkpoint:
        raise RuntimeError("--v64-checkpoint is required to migrate trained Body/Interaction/Hand weights")
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

    # ── Optimizer ─────────────────────────────────────────────────────────
    phase = "warmup" if args.warmup_steps > 0 else "full"
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=args.weight_decay)
    optimizer_names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    if phase != "full":
        update_face_optimizer_lrs(optimizer, step=0, warmup_steps=args.warmup_steps)

    # ── Face & hand conditions (from TikTokV65OverfitInputs) ──────────────
    face_condition = overfit_inputs.face_condition.to(device=device, dtype=dtype)
    face_references = overfit_inputs.face_references.to(device=device, dtype=dtype)
    detail_condition = overfit_inputs.hand_detail_condition.to(device=device, dtype=dtype)
    detail_references = overfit_inputs.hand_detail_references.to(device=device, dtype=dtype)

    # ── Geometry conditions ───────────────────────────────────────────────
    normal = batch["normal"].to(device, dtype=dtype)
    pose = batch["pose_heatmap"].to(device, dtype=dtype)
    part = batch["part_onehot"].to(device, dtype=dtype)
    smplx_global = batch["smplx_global"].to(device, dtype=dtype)
    human_mask = batch["human_mask"].to(device, dtype=dtype)
    task_id = batch["task_id"].to(device, dtype=torch.long)
    target_pixels = batch["tgt_image"].to(device, dtype=dtype)
    target_latent = pipe.pixels_to_latents(target_pixels).detach()
    latent_height, latent_width = target_latent.shape[-2:]

    # ── Encode & cache pipeline conditions ────────────────────────────────
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

    # ── Face mask for outside-region protection loss ──────────────────────
    face_mask = rasterize_box_mask(
        face_condition.target_boxes[:, 0].float(), latent_height, latent_width
    ).to(device=device, dtype=dtype)

    # ── Fixed training state (deterministic noise & timestep) ─────────────
    fixed = make_fixed_training_state(
        target_latent.shape,
        seed=args.seed,
        max_timestep=args.max_timestep,
        fixed_timestep=args.fixed_timestep,
        device=device,
        dtype=dtype,
    )
    noise = fixed["noise"]
    timestep = fixed["timestep"]
    sigma = (timestep.float() / 1000.0).to(dtype).view(-1, 1, 1, 1)
    noisy_latent = (1.0 - sigma) * target_latent + sigma * noise
    target_velocity = noise - target_latent
    progress = (1.0 - timestep.float() / 1000.0).to(dtype)

    # ── Face evaluators ───────────────────────────────────────────────────
    id_evaluator = None
    if not args.disable_face_id_loss:
        id_evaluator = _load_torchscript_evaluator(
            args.face_id_evaluator, label="face_id", device=device
        )
    perceptual_evaluator = None
    if not args.disable_face_perceptual_loss:
        perceptual_evaluator = _load_torchscript_evaluator(
            args.face_perceptual_evaluator, label="face_perceptual", device=device
        )
    lpips_evaluator = None
    if not args.disable_face_lpips_metric:
        lpips_evaluator = _load_torchscript_evaluator(
            args.face_lpips_evaluator, label="face_lpips", device=device
        )
    landmark_evaluator = None
    if not args.disable_face_landmark_metric:
        landmark_evaluator = _load_torchscript_evaluator(
            args.face_landmark_evaluator, label="face_landmark", device=device
        )

    # ── Prediction helper ─────────────────────────────────────────────────
    metrics: list[dict[str, Any]] = []

    def predict_with_face_strength(face_strength: float) -> torch.Tensor:
        """Use the fixed sample/noise/timestep for fair Face OFF/ON comparison."""
        geometry_bundle = adapter.condition_injector(
            normal_a=normal,
            pose_heatmap_a=pose,
            part_onehot_a=part,
            smplx_global_a=smplx_global,
            human_mask_a=human_mask,
            task_id=task_id,
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

    # ── Fixed OFF/ON comparison helper ────────────────────────────────────
    comparison_history: list[dict[str, Any]] = []
    adapter.eval()
    with torch.no_grad():
        fixed_face_off_prediction = predict_with_face_strength(0.0).detach()

    @torch.no_grad()
    def save_fixed_comparison(label: str) -> None:
        """Persist fair OFF/ON previews and face metrics with identical stochastic state."""
        was_training = adapter.training
        adapter.eval()
        off = fixed_face_off_prediction
        on = predict_with_face_strength(1.0)
        on_x0 = noisy_latent - sigma * on
        off_x0 = noisy_latent - sigma * off
        on_pixels = (decode_latents_to_pixels(pipe, on_x0) * 0.5 + 0.5).clamp(0, 1)
        off_pixels = (decode_latents_to_pixels(pipe, off_x0) * 0.5 + 0.5).clamp(0, 1)
        _save_preview(on_pixels, preview_dir / f"{label}_face_on.png")
        _save_preview(off_pixels, preview_dir / f"{label}_face_off.png")
        outside = 1.0 - face_mask.float()
        on_error = (on.float() - target_velocity.float()).square()
        off_error = (off.float() - target_velocity.float()).square()
        record = {
            "label": label,
            "seed": args.seed,
            "timestep": int(timestep[0]),
            "face_on_flow_mse": float(on_error.mean()),
            "face_off_flow_mse": float(off_error.mean()),
            "on_off_prediction_mse": float((on.float() - off.float()).square().mean()),
            "outside_on_off_mse": float(
                ((on.float() - off.float()).square() * outside).sum()
                / (outside.sum() * on.shape[1]).clamp_min(1.0)
            ),
        }
        target_01 = (target_pixels.float() * 0.5 + 0.5).clamp(0, 1)
        record.update(
            compute_face_validation_metrics(
                on_pixels,
                off_pixels,
                target_01,
                face_condition.target_boxes[:, 0].float(),
                identity_evaluator=id_evaluator,
                lpips_evaluator=lpips_evaluator,
                landmark_evaluator=landmark_evaluator,
            )
        )
        comparison_history.append(record)
        if was_training:
            adapter.train()

    save_fixed_comparison("step_000000")

    # ── Training loop ─────────────────────────────────────────────────────
    adapter.train()
    transition_done = phase == "full"
    start_time = time.time()
    for step in range(args.max_steps):
        if not transition_done and step >= args.warmup_steps:
            trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=args.weight_decay)
            optimizer_names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
            transition_done = True
        if phase != "full":
            update_face_optimizer_lrs(
                optimizer, step=step, warmup_steps=args.warmup_steps
            )
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_with_face_strength(1.0)

        period = max(1, int(round(1.0 / args.auxiliary_fraction)))
        use_auxiliary = bool(int(timestep[0]) <= args.max_timestep and step % period == 0)
        identity_loss = perceptual_loss = None
        predicted_pixels = None
        if use_auxiliary and (id_evaluator is not None or perceptual_evaluator is not None):
            predicted_x0 = noisy_latent - sigma * prediction
            predicted_pixels = (
                decode_latents_to_pixels(
                    pipe,
                    predicted_x0,
                    differentiable=True,
                )
                * 0.5
                + 0.5
            ).clamp(0, 1)
            target_01 = (target_pixels * 0.5 + 0.5).clamp(0, 1)
            boxes = face_condition.target_boxes[:, 0].float()
            predicted_crop = _crop_batch(predicted_pixels.float(), boxes)
            target_crop = _crop_batch(target_01.float(), boxes)
            if id_evaluator is not None:
                identity_loss = _feature_distance(
                    id_evaluator, predicted_crop, target_crop, cosine=True
                )
            if perceptual_evaluator is not None:
                perceptual_loss = _feature_distance(
                    perceptual_evaluator, predicted_crop, target_crop, cosine=False
                )
        losses = compute_v65_face_losses(
            prediction,
            target_velocity,
            face_mask,
            face_id_loss=identity_loss,
            face_perceptual_loss=perceptual_loss,
            auxiliary_selector=torch.tensor([use_auxiliary], device=device),
            outside_reference=fixed_face_off_prediction,
        )
        losses["total"].backward()
        trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        record = {"step": step + 1, **{key: float(value.detach()) for key, value in losses.items()}}
        metrics.append(record)
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"step {step + 1}/{args.max_steps}: total={record['total']:.6f}, "
                f"flow={record['flow']:.6f}, id={record['face_id']:.6f}, "
                f"perc={record['face_perceptual']:.6f}, outside={record['outside']:.6f}",
                flush=True,
            )
        if (step + 1) % args.preview_every == 0 or step + 1 == args.max_steps:
            save_fixed_comparison(f"step_{step + 1:06d}")
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.max_steps:
            checkpoint_out = build_v65_checkpoint(
                adapter,
                feature_cache_fingerprint=face_cache["metadata"]["cache_fingerprint"],
                step=step + 1,
                optimizer_state_dict=optimizer.state_dict(),
                source_stem=args.source_stem,
                target_stem=args.target_stem,
                seed=args.seed,
            )
            torch.save(
                checkpoint_out,
                checkpoint_dir / f"checkpoint_step_{step + 1:06d}.pt",
            )

    # ── Final integrity check ─────────────────────────────────────────────
    non_face_after = compute_non_face_sha256(adapter)
    if phase != "full" and non_face_before != non_face_after:
        raise RuntimeError(
            f"frozen Body/Interaction/Hand/DeepGen parameters changed: "
            f"{non_face_before} != {non_face_after}"
        )

    summary = {
        "architecture_version": "v6.5",
        "dataset": "TikTok",
        "sequence": args.sequence,
        "source_stem": args.source_stem,
        "target_stem": args.target_stem,
        "seed": args.seed,
        "fixed_timestep": int(timestep[0]),
        "cache_fingerprint": face_cache["metadata"]["cache_fingerprint"],
        "non_face_sha256_before": non_face_before,
        "non_face_sha256_after": non_face_after,
        "full_adapter_training": phase == "full",
        "optimizer_groups": optimizer_names,
        "discarded_v64_optimizer_state": discarded_old_optimizer,
        "fixed_off_on_comparisons": comparison_history,
        "elapsed_seconds": time.time() - start_time,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"training complete; outputs saved in {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
