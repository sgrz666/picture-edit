#!/usr/bin/env python3
"""V6.5 independent Face Adapter single-sample iPER training entry point.

The light-weight helpers in this file are intentionally importable without
Diffusers, InsightFace, or a GPU. Heavy training dependencies are imported only
inside :func:`run_training`, with actionable errors when unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.precompute_iper_face_features_v65 import load_v65_face_cache


FACE_PARAMETER_PREFIXES = (
    "face_preparer.",
    "control_interface.face_adapter.",
    "control_interface.strength_controller.face_log_group_scale",
)


def resolve_frame_selector(cache: Mapping[str, Any], selector: int | str) -> int:
    """Resolve either a cache index or a stem, rejecting silent clamping."""

    frame_count = len(cache["frame_stems"])
    if isinstance(selector, str) and selector in cache["stem_to_idx"]:
        return int(cache["stem_to_idx"][selector])
    try:
        index = int(selector)
    except (TypeError, ValueError) as exc:
        raise KeyError(f"unknown frame stem: {selector}") from exc
    if index < 0 or index >= frame_count:
        if isinstance(selector, str) and not selector.lstrip("+-").isdigit():
            raise KeyError(f"unknown frame stem: {selector}")
        raise IndexError(f"frame index {index} is outside [0,{frame_count - 1}]")
    return index


def build_face_inputs(
    cache: Mapping[str, Any],
    *,
    source_frame: int | str,
    target_frame: int | str,
):
    """Assemble one iPER pair as [B=1,P=2,R=1] V6.5 face inputs."""

    # Delayed only to keep this training helper independent while the V6.5 face
    # package is developed/tested in isolation.
    from src.pose_control.v6.face.conditions import (
        FaceFineCondition,
        FaceReferenceFeatures,
    )

    source_index = resolve_frame_selector(cache, source_frame)
    target_index = resolve_frame_selector(cache, target_frame)
    slot_valid = bool(cache["valid"][source_index] and cache["valid"][target_index])

    landmarks = torch.zeros(1, 2, 72, 3, dtype=torch.float32)
    jaw = torch.zeros(1, 2, 3, dtype=torch.float32)
    expression = torch.zeros(1, 2, 10, dtype=torch.float32)
    source_boxes = torch.zeros(1, 2, 4, dtype=torch.float32)
    target_boxes = torch.zeros(1, 2, 4, dtype=torch.float32)
    face_valid = torch.zeros(1, 2, dtype=torch.bool)
    arcface = torch.zeros(1, 2, 1, 512, dtype=torch.float16)
    dino = torch.zeros(1, 2, 1, 256, 1536, dtype=torch.float16)
    reference_valid = torch.zeros(1, 2, 1, dtype=torch.bool)
    if slot_valid:
        landmarks[0, 0] = cache["face_landmarks"][target_index].float()
        jaw[0, 0] = cache["jaw_pose"][target_index].float()
        expression[0, 0] = cache["expression"][target_index].float()
        source_boxes[0, 0] = cache["face_boxes"][source_index].float()
        target_boxes[0, 0] = cache["face_boxes"][target_index].float()
        face_valid[0, 0] = True
        arcface[0, 0, 0] = cache["arcface"][source_index]
        dino[0, 0, 0] = cache["dino_patches"][source_index]
        reference_valid[0, 0, 0] = True

    condition = FaceFineCondition(
        landmarks=landmarks,
        jaw_pose=jaw,
        expression=expression,
        source_boxes=source_boxes,
        target_boxes=target_boxes,
        face_valid=face_valid,
        source_indices=torch.tensor([[0, 1]], dtype=torch.long),
    ).validate()
    references = FaceReferenceFeatures(
        arcface=arcface,
        dino_patches=dino,
        reference_valid=reference_valid,
    ).validate()
    return condition, references


def _is_face_parameter(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FACE_PARAMETER_PREFIXES)


def _is_warmup_parameter(name: str) -> bool:
    return (
        name.startswith("control_interface.face_adapter.")
        and ".zero_heads." in name
    ) or name == "control_interface.strength_controller.face_log_group_scale"


def select_face_trainable_parameters(
    model: nn.Module,
    *,
    phase: str,
) -> list[tuple[str, nn.Parameter]]:
    """Freeze everything, then expose either Face zero-head warmup or all Face."""

    if phase not in {"warmup", "full"}:
        raise ValueError("phase must be 'warmup' or 'full'")
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        enabled = _is_warmup_parameter(name) if phase == "warmup" else _is_face_parameter(name)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError(
            "no V6.5 Face parameters were found; build the adapter with its face branch enabled"
        )
    return selected


def _optimizer_group_for_name(name: str) -> int:
    if _is_warmup_parameter(name):
        return 0
    if name.startswith("face_preparer."):
        return 1
    if name.startswith("control_interface.face_adapter."):
        return 2
    raise ValueError(f"unclassified Face parameter: {name}")


def build_face_optimizer(
    model: nn.Module,
    *,
    phase: str,
    weight_decay: float = 0.01,
) -> tuple[torch.optim.Optimizer, dict[str, list[str]]]:
    selected = select_face_trainable_parameters(model, phase=phase)
    grouped_parameters: list[list[nn.Parameter]] = [[], [], []]
    grouped_names = {
        "zero_heads_and_gate": [],
        "face_encoders": [],
        "face_attention_and_adapters": [],
    }
    labels = list(grouped_names)
    for name, parameter in selected:
        group = _optimizer_group_for_name(name)
        grouped_parameters[group].append(parameter)
        grouped_names[labels[group]].append(name)
    assigned = sum(len(values) for values in grouped_names.values())
    if assigned != len(selected) or len(set().union(*map(set, grouped_names.values()))) != len(selected):
        raise AssertionError("Face optimizer groups must cover each trainable parameter exactly once")
    optimizer = torch.optim.AdamW(
        [
            {"params": grouped_parameters[0], "lr": 1e-4, "name": labels[0]},
            {"params": grouped_parameters[1], "lr": 0.0, "name": labels[1]},
            {"params": grouped_parameters[2], "lr": 0.0, "name": labels[2]},
        ],
        weight_decay=weight_decay,
    )
    return optimizer, grouped_names


def update_face_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    warmup_steps: int = 1000,
) -> None:
    if len(optimizer.param_groups) != 3:
        raise ValueError("V6.5 Face optimizer must contain exactly three groups")
    rates = (1e-4, 0.0, 0.0) if step < warmup_steps else (1e-4, 5e-5, 1e-5)
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate


def _masked_mean(value: torch.Tensor, selector: torch.Tensor | None) -> torch.Tensor:
    if value.ndim == 0:
        value = value.reshape(1)
    if selector is None:
        return value.mean()
    selector = selector.to(device=value.device, dtype=torch.bool).reshape(-1)
    if value.shape[0] != selector.shape[0]:
        raise ValueError("auxiliary loss and selector batch dimensions must match")
    if not selector.any():
        return value.sum() * 0.0
    return value[selector].mean()


def compute_v65_face_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    face_mask: torch.Tensor,
    *,
    face_id_loss: torch.Tensor | None = None,
    face_perceptual_loss: torch.Tensor | None = None,
    auxiliary_selector: torch.Tensor | None = None,
    outside_reference: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute L_flow + .10 L_id + .05 L_perc + .02 L_outside."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if face_mask.ndim != 4 or face_mask.shape[0] != prediction.shape[0]:
        raise ValueError("face_mask must have shape [B,1,H,W]")
    mask = F.interpolate(
        face_mask.float(), size=prediction.shape[-2:], mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    squared = (prediction.float() - target.float()).square()
    outside = 1.0 - mask
    if outside_reference is not None and outside_reference.shape != prediction.shape:
        raise ValueError("outside_reference must match prediction shape")
    outside_error = squared if outside_reference is None else (
        prediction.float() - outside_reference.float()
    ).square()
    outside_loss = (outside_error * outside).sum() / (
        outside.sum() * prediction.shape[1]
    ).clamp_min(1.0)
    flow_loss = squared.mean()
    zero = flow_loss * 0.0
    identity = zero if face_id_loss is None else _masked_mean(face_id_loss, auxiliary_selector)
    perceptual = zero if face_perceptual_loss is None else _masked_mean(
        face_perceptual_loss, auxiliary_selector
    )
    total = flow_loss + 0.10 * identity + 0.05 * perceptual + 0.02 * outside_loss
    return {
        "total": total,
        "flow": flow_loss,
        "face_id": identity,
        "face_perceptual": perceptual,
        "outside": outside_loss,
    }


def make_fixed_training_state(
    latent_shape: Sequence[int],
    *,
    seed: int,
    max_timestep: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create the one immutable noise/timestep pair used by overfit mode."""

    if len(latent_shape) != 4 or int(latent_shape[0]) <= 0:
        raise ValueError("latent_shape must be [B,C,H,W]")
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(tuple(latent_shape), generator=generator, dtype=torch.float32)
    timestep = torch.randint(
        0, max_timestep + 1, (int(latent_shape[0]),), generator=generator, dtype=torch.long
    )
    return {
        "noise": noise.to(device=device, dtype=dtype),
        "timestep": timestep.to(device=device),
    }


def rasterize_box_mask(
    boxes: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [B,4]")
    yy = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
    xx = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
    batch = boxes.shape[0]
    return (
        (xx[None, None, None, :] >= boxes[:, 0].view(batch, 1, 1, 1))
        & (xx[None, None, None, :] <= boxes[:, 2].view(batch, 1, 1, 1))
        & (yy[None, None, :, None] >= boxes[:, 1].view(batch, 1, 1, 1))
        & (yy[None, None, :, None] <= boxes[:, 3].view(batch, 1, 1, 1))
    ).to(boxes.dtype)


def compute_non_face_sha256(model: nn.Module) -> str:
    """Hash all frozen architecture state independent of changing train phases."""

    hasher = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if _is_face_parameter(name):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _crop_batch(images: torch.Tensor, boxes: torch.Tensor, size: int = 112) -> torch.Tensor:
    """Differentiably crop one normalized face box per batch with grid_sample."""

    batch = images.shape[0]
    axis = torch.linspace(0.0, 1.0, size, device=images.device, dtype=images.dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grids = []
    for index in range(batch):
        x0, y0, x1, y1 = boxes[index]
        x = x0 + grid_x * (x1 - x0)
        y = y0 + grid_y * (y1 - y0)
        grids.append(torch.stack((x * 2.0 - 1.0, y * 2.0 - 1.0), dim=-1))
    return F.grid_sample(images, torch.stack(grids), mode="bilinear", align_corners=True)


def _feature_distance(
    evaluator: nn.Module, predicted_crop: torch.Tensor, target_crop: torch.Tensor, *, cosine: bool
) -> torch.Tensor:
    predicted_features = evaluator(predicted_crop.float())
    with torch.no_grad():
        target_features = evaluator(target_crop.float())
    if isinstance(predicted_features, (tuple, list)):
        predicted_features = predicted_features[-1]
        target_features = target_features[-1]
    predicted_features = predicted_features.flatten(1)
    target_features = target_features.flatten(1)
    if cosine:
        return 1.0 - F.cosine_similarity(predicted_features, target_features, dim=-1)
    return (predicted_features - target_features).square().mean(dim=-1)


def _load_torchscript_evaluator(path: str | None, *, label: str, device: torch.device) -> nn.Module:
    if not path:
        raise RuntimeError(
            f"{label} evaluator is required. Supply its local --{label.replace('_', '-')}-evaluator "
            f"path, or explicitly disable that loss."
        )
    evaluator_path = Path(path)
    if not evaluator_path.exists():
        raise FileNotFoundError(f"{label} evaluator not found: {evaluator_path}")
    evaluator = torch.jit.load(str(evaluator_path), map_location=device).eval()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    return evaluator


def decode_latents_to_pixels(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode DeepGen latents to [-1,1], supporting custom and Diffusers VAEs."""

    if hasattr(pipe, "latents_to_pixels"):
        return pipe.latents_to_pixels(latents)
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "decode"):
        raise RuntimeError("DeepGen pipeline exposes neither latents_to_pixels nor vae.decode")
    config = getattr(vae, "config", None)
    scaling = float(getattr(config, "scaling_factor", 1.0))
    shift = float(getattr(config, "shift_factor", 0.0) or 0.0)
    decoded = vae.decode(latents / scaling + shift, return_dict=False)
    return decoded[0] if isinstance(decoded, (tuple, list)) else decoded.sample


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V6.5 independent iPER Face Adapter")
    parser.add_argument("--stage", choices=("overfit", "face"), default="overfit")
    parser.add_argument("--appearance", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-index")
    source.add_argument("--source-stem")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-index")
    target.add_argument("--target-stem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
    )
    parser.add_argument(
        "--sampled-root",
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
    )
    parser.add_argument(
        "--assets-root",
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
    )
    parser.add_argument(
        "--v64-checkpoint",
        default=(
            "/home/shangguanrz/project/pic-edit/experiments/"
            "iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt"
        ),
    )
    parser.add_argument("--output-dir", default="experiments/iper_v65_face_overfit")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-timestep", type=int, default=650)
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
    parser.add_argument(
        "--face-id-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/identity_encoder.ts",
    )
    parser.add_argument(
        "--face-perceptual-evaluator",
        default="/home/shangguanrz/project/pic-edit/models/face_eval/perceptual_encoder.ts",
    )
    parser.add_argument("--disable-face-id-loss", action="store_true")
    parser.add_argument("--disable-face-perceptual-loss", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--preview-every", type=int, default=250)
    return parser.parse_args(argv)


def _dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _frame_roles(sampled_app: Path) -> dict[str, str]:
    path = sampled_app / "smplestx" / "frame_map.json"
    if not path.exists():
        raise FileNotFoundError(f"iPER frame map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        frames = json.load(handle).get("frames", [])
    return {str(frame["stem"]): str(frame.get("role", "target")) for frame in frames}


def _save_preview(image: torch.Tensor, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for previews") from exc
    pixels = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((pixels * 255.0 + 0.5).astype("uint8")).save(path)


def run_training(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("max_steps must be positive and warmup_steps non-negative")
    if not 0.0 < args.auxiliary_fraction <= 1.0:
        raise ValueError("auxiliary_fraction must be in (0,1]")
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError("V6.5 training requires the project Diffusers environment") from exc
    try:
        from src.data.iper_detail_dataset import IPERDetailDataset
        from src.pose_control.v6.checkpoint import (
            build_v65_checkpoint,
            freeze_for_face_training,
            load_v64_checkpoint,
        )
        from src.pose_control.v6.conditions import AdapterIdentityCondition
        from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
        from train_iper_v6_native import get_cached_conditions
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "the V6.5 face/checkpoint integration is unavailable; update the adapter before training"
        ) from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    condition_cache_dir = Path(args.cache_dir or output_dir / "condition_cache")
    condition_cache_dir.mkdir(parents=True, exist_ok=True)

    face_cache_path = Path(args.assets_root) / args.appearance / "v65_face_features.pt"
    face_cache = load_v65_face_cache(face_cache_path)
    source_selector = args.source_stem if args.source_stem is not None else args.source_index
    target_selector = args.target_stem if args.target_stem is not None else args.target_index
    source_index = resolve_frame_selector(face_cache, source_selector)
    target_index = resolve_frame_selector(face_cache, target_selector)
    source_stem = face_cache["frame_stems"][source_index]
    target_stem = face_cache["frame_stems"][target_index]
    if not (face_cache["valid"][source_index] and face_cache["valid"][target_index]):
        raise RuntimeError("source and target frames must both be valid in v65_face_features.pt")
    roles = _frame_roles(Path(args.sampled_root) / args.appearance)

    dataset = IPERDetailDataset(
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        single_sample={
            "appearance": args.appearance,
            "source": {"stem": source_stem, "role": roles[source_stem]},
            "target": {"stem": target_stem, "role": roles[target_stem]},
        },
        resolution=args.resolution,
        augment=False,
    )
    batch = dataset[0]
    # Add the batch dimension expected by the existing iPER training helpers.
    batch = {
        key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
        for key, value in batch.items()
    }

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    pipe.to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(
        pipe, condition_use_depth=False
    ).to(device=device, dtype=dtype)
    if not args.v64_checkpoint:
        raise RuntimeError("--v64-checkpoint is required to migrate trained Body/Interaction/Hand weights")
    checkpoint = torch.load(args.v64_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("V6.4 checkpoint must be a mapping")
    checkpoint_for_migration = dict(checkpoint)
    discarded_old_optimizer = checkpoint_for_migration.pop("optimizer_state_dict", None) is not None
    if discarded_old_optimizer:
        print("discarding incompatible V6.4 optimizer_state_dict before V6.5 migration")
    load_v64_checkpoint(adapter, checkpoint_for_migration)
    # Assert that the central helper agrees with our strict prefix-based selector.
    freeze_for_face_training(adapter)
    non_face_before = compute_non_face_sha256(adapter)

    phase = "warmup" if args.warmup_steps > 0 else "full"
    optimizer, optimizer_names = build_face_optimizer(
        adapter, phase=phase, weight_decay=args.weight_decay
    )
    update_face_optimizer_lrs(optimizer, step=0, warmup_steps=args.warmup_steps)

    face_condition, face_references = build_face_inputs(
        face_cache, source_frame=source_index, target_frame=target_index
    )
    face_condition = face_condition.to(device=device, dtype=dtype)
    face_references = face_references.to(device=device, dtype=dtype)
    detail_condition, detail_references = dataset.get_detail_condition_and_reference(
        args.appearance, source_stem, target_stem
    )
    detail_condition = detail_condition.to(device=device, dtype=dtype)
    detail_references = detail_references.to(device=device, dtype=dtype)

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
    geometry_bundle = adapter.condition_injector(
        normal_a=normal,
        pose_heatmap_a=pose,
        part_onehot_a=part,
        smplx_global_a=smplx_global,
        human_mask_a=human_mask,
        task_id=task_id,
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
    fixed = make_fixed_training_state(
        target_latent.shape,
        seed=args.seed,
        max_timestep=args.max_timestep,
        device=device,
        dtype=dtype,
    )
    noise = fixed["noise"]
    timestep = fixed["timestep"]
    sigma = (timestep.float() / 1000.0).to(dtype).view(-1, 1, 1, 1)
    noisy_latent = (1.0 - sigma) * target_latent + sigma * noise
    target_velocity = noise - target_latent
    progress = (1.0 - timestep.float() / 1000.0).to(dtype)

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

    metrics: list[dict[str, Any]] = []

    def predict_with_face_strength(face_strength: float) -> torch.Tensor:
        """Use the fixed sample/noise/timestep for fair Face OFF/ON comparison."""

        prepared = adapter.prepare_conditioning(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            target_latent_hw=(latent_height, latent_width),
            detail_condition=detail_condition,
            detail_references=detail_references,
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

    comparison_history: list[dict[str, Any]] = []
    adapter.eval()
    with torch.no_grad():
        # Body/Interaction/Hand and the DiT are frozen, so this fixed Face-OFF
        # prediction is the exact outside-region reference for every step.
        fixed_face_off_prediction = predict_with_face_strength(0.0).detach()

    @torch.no_grad()
    def save_fixed_comparison(label: str) -> None:
        """Persist fair OFF/ON previews and flow metrics with identical stochastic state."""

        was_training = adapter.training
        adapter.eval()
        off = fixed_face_off_prediction
        # Recompute both passes in eval mode. Reusing the train-mode ON tensor
        # could introduce dropout differences and invalidate the comparison.
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
        comparison_history.append(
            {
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
        )
        if was_training:
            adapter.train()

    save_fixed_comparison("step_000000")
    adapter.train()
    transition_done = phase == "full"
    start_time = time.time()
    for step in range(args.max_steps):
        if not transition_done and step >= args.warmup_steps:
            optimizer, optimizer_names = build_face_optimizer(
                adapter, phase="full", weight_decay=args.weight_decay
            )
            transition_done = True
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
            predicted_pixels = (decode_latents_to_pixels(pipe, predicted_x0) * 0.5 + 0.5).clamp(0, 1)
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
                source_stem=source_stem,
                target_stem=target_stem,
                seed=args.seed,
            )
            torch.save(
                checkpoint_out,
                checkpoint_dir / f"checkpoint_step_{step + 1:06d}.pt",
            )

    non_face_after = compute_non_face_sha256(adapter)
    if non_face_before != non_face_after:
        raise RuntimeError(
            f"frozen Body/Interaction/Hand/DeepGen parameters changed: {non_face_before} != {non_face_after}"
        )
    summary = {
        "architecture_version": "v6.5",
        "appearance": args.appearance,
        "source_stem": source_stem,
        "target_stem": target_stem,
        "seed": args.seed,
        "fixed_timestep": int(timestep[0]),
        "cache_fingerprint": face_cache["metadata"]["cache_fingerprint"],
        "non_face_sha256_before": non_face_before,
        "non_face_sha256_after": non_face_after,
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
