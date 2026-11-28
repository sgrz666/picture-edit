"""Qualified output-side objectives for frozen RGB heads and hand-region comparisons."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


LEGACY_RGB_OBJECTIVES = ("joint", "part", "contact", "anatomy")
POSE_OBJECTIVES = {"pose_2d": 2, "pose_3d": 3}
BINARY_OBJECTIVES = {"contact", "anatomy", "occlusion", "collision"}
RGB_OBJECTIVES = (*LEGACY_RGB_OBJECTIVES, *POSE_OBJECTIVES, "occlusion", "collision")
IMAGE_OBJECTIVES = ("appearance", "perceptual")
MIN_HAND_PIXELS = 16


def objective_weights(options):
    weights = {}
    for option in options:
        name, separator, raw_weight = option.partition("=")
        if not separator or name not in ("flow", *RGB_OBJECTIVES, *IMAGE_OBJECTIVES) or name in weights:
            raise ValueError(f"Expected a unique known objective weight NAME=VALUE, got {option!r}")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"Objective {name} requires a numeric weight") from error
        if not math.isfinite(weight) or weight < 0 or (name == "flow" and weight == 0):
            raise ValueError(f"Invalid objective weight for {name}")
        weights[name] = weight
    weights.setdefault("flow", 1.)
    return weights


def pose_output_labels(condition):
    """Build fixed-slot 2D/3D labels with independent visibility/geometry qualification."""
    height, width = condition.canvas.canvas_hw
    xy = condition.joints[..., :2].unsqueeze(0) / condition.joints.new_tensor([width, height])
    seen = condition.active_joints()[None, ..., None].expand_as(xy)
    labels = {"pose_2d": (xy, seen)}
    if condition.joint_3d_valid is not None:
        # A held-out/cropped point remains useful to geometry, but cannot label rendered RGB.
        projected = condition.projected_joints()
        valid = condition.joint_3d_valid & condition.joint_3d_valid[:, :1] & projected & projected[:, :1]
        valid[:, 0] = False  # The root is analytically zero and supplies no articulation evidence.
        relative = (condition.joints_3d - condition.joints_3d[:, :1]).unsqueeze(0)
        labels["pose_3d"] = (relative, valid[None, ..., None].expand_as(relative))
    return labels


def eligible_output_labels(labels):
    if not isinstance(labels, dict):
        raise ValueError("RGB labels must contain named (target, valid) tensor pairs")
    eligible = []
    for name, pair in labels.items():
        if name not in RGB_OBJECTIVES or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid RGB supervision objective {name!r}")
        target, valid = pair
        if (not torch.is_tensor(target) or not torch.is_tensor(valid) or valid.dtype != torch.bool
                or valid.shape != target.shape):
            raise ValueError(f"RGB {name} supervision requires same-shaped targets and boolean validity")
        if name in POSE_OBJECTIVES:
            if (target.shape != (1, 4, 21, POSE_OBJECTIVES[name]) or not target.is_floating_point()
                    or not torch.equal(valid, valid[..., :1].expand_as(valid))):
                raise ValueError(f"RGB {name} labels require complete fixed-slot joint coordinates")
        known = target[valid]
        if not torch.isfinite(known).all():
            raise ValueError(f"Eligible RGB {name} labels must be finite")
        if name in BINARY_OBJECTIVES and not ((known >= 0) & (known <= 1)).all():
            raise ValueError("Binary RGB labels must be in [0,1]; mask out unknown entries")
        if name == "pose_2d" and not ((known >= 0) & (known <= 1)).all():
            raise ValueError("Pose 2D labels must use normalized canvas coordinates")
        if name == "part" and (known.is_floating_point() or (known < 0).any()):
            raise ValueError("Part labels must be nonnegative integer classes")
        if known.numel():
            eligible.append(name)
    return tuple(eligible)


def output_supervision(predictions, labels):
    """Apply only independently qualified labels to differentiable frozen-head predictions."""
    eligible_output_labels(labels)
    if not isinstance(predictions, dict):
        raise ValueError("Frozen RGB evaluator must return a dictionary of named tensors")
    terms = {}
    for name, (target, valid) in labels.items():
        prediction = predictions.get(name)
        if not torch.is_tensor(prediction) or not prediction.is_floating_point():
            raise ValueError(f"Frozen RGB evaluator must supply floating predictions for {name}")
        prediction = prediction.float()
        valid, target = valid.to(prediction.device), target.detach().to(prediction.device)
        if name == "part":
            prediction = prediction.movedim(1, -1)
            if prediction.shape[:-1] != target.shape:
                raise ValueError("Part logits must have a channel axis over the label canvas")
        elif prediction.shape != target.shape:
            raise ValueError(f"RGB {name} prediction and label shapes disagree")
        values, target = prediction[valid], target[valid]
        if not torch.isfinite(values).all():
            raise ValueError(f"RGB {name} evaluator produced a non-finite qualified prediction")
        if not values.numel():
            terms[name] = values.sum()
        elif name in ("joint", *POSE_OBJECTIVES):
            beta = .01 if name in POSE_OBJECTIVES else 1.
            terms[name] = F.smooth_l1_loss(values, target.float(), beta=beta)
        elif name == "part":
            if (target >= values.shape[-1]).any():
                raise ValueError("Qualified part class is outside the evaluator's output range")
            terms[name] = F.cross_entropy(values, target.long())
        else:
            terms[name] = F.binary_cross_entropy_with_logits(values, target.float())
    return terms


def check_output_gradients(clean, losses, *, adapter_parameters=()):
    """Measure each enabled output's first finite, nonzero path to latent and Adapter."""
    parameters = tuple(adapter_parameters)
    metrics = {}
    for name, loss in losses.items():
        gradients = (torch.autograd.grad(loss, (clean, *parameters), retain_graph=True, allow_unused=True)
                     if loss.requires_grad else (None,))
        clean_gradient, *adapter_gradients = gradients
        magnitude = 0. if clean_gradient is None else clean_gradient.detach().float().norm().item()
        if not math.isfinite(magnitude) or magnitude == 0:
            raise ValueError(f"RGB {name} objective must supply a finite nonzero gradient to the clean latent")
        metrics[name] = {"clean": magnitude}
        if parameters:
            squares = [gradient.detach().float().norm().square()
                       for gradient in adapter_gradients if gradient is not None]
            magnitude = torch.stack(squares).sum().sqrt().item() if squares else 0.
            if not math.isfinite(magnitude) or magnitude == 0:
                raise ValueError(f"RGB {name} objective must supply a finite nonzero Adapter gradient")
            metrics[name]["adapter"] = magnitude
    return metrics


@dataclass(frozen=True)
class ImageObjectiveData:
    appearance: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    perceptual: tuple[torch.Tensor, ...] = ()


def image_objective_data(pair, names=IMAGE_OBJECTIVES):
    """Qualify foreground crops and same-slot, same-part source/target color references."""
    if not any(name in IMAGE_OBJECTIVES for name in names):
        return ImageObjectiveData()
    condition = pair.condition
    canvas = condition.canvas.valid_mask(dtype=torch.bool)[0]
    hands = (pair.regions.hands >= .5) & canvas
    perceptual = tuple(mask for mask in hands if int(mask.sum()) >= MIN_HAND_PIXELS) if "perceptual" in names else ()
    source_parts = getattr(condition, "appearance_parts", None)
    source_mask = getattr(condition, "appearance_mask", None)
    appearance = []
    if ("appearance" in names and source_parts is not None and source_mask is not None
            and condition.appearance is not None and condition.geometry_maps is not None and hands.shape[0] == 4):
        target_valid = condition.map_valid[:, 4] & hands
        target_valid &= target_valid.sum(0) == 1  # Do not supervise ambiguous overlaps.
        normalized = condition.geometry_maps[:, 4] * 15
        canonical = normalized.round()
        target_valid &= (normalized - canonical).abs() < 1e-3
        for slot in condition.appearance_valid.nonzero().flatten().tolist():
            for part in range(16):
                in_source = source_mask[slot] & (source_parts[slot] == part)
                if int(in_source.sum()) < MIN_HAND_PIXELS:
                    continue
                in_target = target_valid[slot] & (canonical[slot] == part)
                if int(in_target.sum()) >= MIN_HAND_PIXELS:
                    source_mean = condition.appearance[slot, :, in_source].detach().float().mean(-1)
                    appearance.append((in_target, source_mean))
    return ImageObjectiveData(tuple(appearance), perceptual)


def _masked_crop(rgb, mask):
    y, x = mask.nonzero(as_tuple=True)
    top, left, bottom, right = int(y.min()), int(x.min()), int(y.max()) + 1, int(x.max()) + 1
    crop = rgb[..., top:bottom, left:right] * mask[top:bottom, left:right].to(rgb.device)
    height, width = crop.shape[-2:]
    size = max(height, width)
    offset_x, offset_y = (size - width) // 2, (size - height) // 2
    return F.pad(crop, (offset_x, size - width - offset_x, offset_y, size - height - offset_y))


def image_output_supervision(decoded, target, data, names, feature_extractor=None):
    terms = {}
    if "appearance" in names:
        terms["appearance"] = torch.stack([
            F.smooth_l1_loss(decoded[0, :, mask.to(decoded.device)].mean(-1), source.to(decoded.device), beta=.05)
            for mask, source in data.appearance
        ]).mean()
    if "perceptual" in names:
        if feature_extractor is None:
            raise ValueError("The perceptual objective requires a locked frozen feature model")
        reference = torch.from_numpy(np.array(target.convert("RGB"), copy=True)).permute(2, 0, 1)[None]
        reference = reference.to(device=decoded.device, dtype=decoded.dtype) / 255
        distance = []
        for mask in data.perceptual:
            with torch.no_grad():
                expected = feature_extractor(_masked_crop(reference, mask))
            actual = feature_extractor(_masked_crop(decoded, mask))
            if (actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
                raise ValueError("Perceptual model must provide finite, aligned feature maps")
            distance.append(F.mse_loss(actual.float(), expected.float()))
        terms["perceptual"] = torch.stack(distance).mean()
    return terms


def load_perceptual_features(root, *, device):
    """Load pinned local DINO weights and preserve autograd from features to decoded RGB."""
    from transformers import AutoConfig, AutoModel

    root = Path(root).resolve(strict=True)
    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(root, **options)
    if config.model_type not in ("dinov2", "dinov2_with_registers", "dinov3_vit"):
        raise ValueError("The perceptual feature model must be a pinned DINO ViT")
    size = 224
    patch = config.patch_size
    patch_hw = (patch, patch) if isinstance(patch, int) else tuple(patch)
    if len(patch_hw) != 2 or any(step <= 0 or size % step for step in patch_hw):
        raise ValueError("DINO patch size must divide the fixed 224-pixel hand crop")
    processor = json.loads((root / "preprocessor_config.json").read_text(encoding="utf-8"))
    mean = torch.tensor(processor["image_mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor["image_std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid pinned DINO image normalization")
    model = AutoModel.from_pretrained(root, config=config, use_safetensors=True, **options)
    model.eval().requires_grad_(False).to(device)

    def features(rgb):
        resized = F.interpolate(rgb.float(), (size, size), mode="bilinear", align_corners=False, antialias=True)
        tokens = model(pixel_values=(resized - mean) / std).last_hidden_state.float()
        return F.layer_norm(tokens, (tokens.shape[-1],))

    return features
