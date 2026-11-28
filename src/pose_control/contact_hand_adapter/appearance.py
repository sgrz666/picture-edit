"""Source-only hand appearance with semantic pooling and optional frozen DINOv3."""

from dataclasses import replace
import json
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from src.pose_edit.common import fingerprint, sha256_file

from .conditions import APPEARANCE_DINO_DIM, dino_crop_fingerprint, verify_appearance_sources


APPEARANCE_SIZE = 256
_PARTS = 16
_COLOR_BINS = 8
_COLOR_STATS = 3 * (2 + _COLOR_BINS)


class AppearanceEncoding(NamedTuple):
    tokens: torch.Tensor  # [S,16,width], one semantic token or shared global fallback per part
    part_valid: torch.Tensor  # [S,16], source evidence for the corresponding anatomical part


def masked_rgb_stats(crops, mask=None):
    """Color moments and histograms without pixel positions."""
    pixels = crops.float().flatten(2)
    if mask is None:
        weights = pixels.new_ones(pixels.shape[0], 1, pixels.shape[-1])
    else:
        weights = mask.flatten(1)[:, None].to(pixels)
    samples = torch.where(weights.bool(), pixels, 0.)
    count = weights.sum(-1).clamp_min(1)
    mean = samples.sum(-1) / count
    variance = (samples.square().sum(-1) / count - mean.square()).clamp_min(0)
    bins = (samples.clamp(0, 1) * _COLOR_BINS).long().clamp_max(_COLOR_BINS - 1)
    histogram = pixels.new_zeros(pixels.shape[0], 3, _COLOR_BINS)
    histogram.scatter_add_(2, bins, weights.expand_as(pixels))
    histogram = (histogram / count[:, :, None]).flatten(1)
    return torch.cat((mean, variance, histogram), dim=-1)


class AppearanceEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.global_color = nn.Sequential(nn.Linear(_COLOR_STATS, width), nn.SiLU(), nn.Linear(width, width))
        self.global_dino = nn.Sequential(nn.LayerNorm(2 * APPEARANCE_DINO_DIM),
                                         nn.Linear(2 * APPEARANCE_DINO_DIM, width, bias=False))
        self.pyramid = nn.ModuleList((
            nn.Sequential(nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU()),
            nn.Sequential(nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU()),
        ))
        self.local_part = nn.Sequential(nn.Linear(32 + 64 + 96, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, crops, parts=None, global_descriptors=None, *, mask=None, global_valid=None):
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("Appearance crops must be [S,3,H,W]")
        shape = (len(crops), *crops.shape[-2:])
        if parts is not None and (parts.shape != shape or parts.dtype != torch.long):
            raise ValueError("Source anatomical parts must be a long [S,H,W] tensor")
        if mask is not None and (mask.shape != shape or mask.dtype != torch.bool):
            raise ValueError("Source appearance mask must be a bool [S,H,W] tensor")
        part_pixels = None if parts is None else (parts >= 0) & (parts < _PARTS)
        if mask is None and part_pixels is not None:
            mask = torch.where(part_pixels.flatten(1).any(1)[:, None, None], part_pixels, True)
        dtype = self.global_color[0].weight.dtype
        global_token = self.global_color(masked_rgb_stats(crops, mask).to(dtype=dtype))
        if global_descriptors is not None:
            if global_descriptors.shape != (len(crops), 2, APPEARANCE_DINO_DIM):
                raise ValueError("DINO appearances require separate CLS and pooled patch descriptors")
            if global_valid is None:
                global_valid = torch.ones(len(crops), dtype=torch.bool, device=crops.device)
            if global_valid.shape != (len(crops),) or global_valid.dtype != torch.bool:
                raise ValueError("DINO descriptor validity must be a bool [S] tensor")
            safe = torch.where(global_valid[:, None, None], global_descriptors, 0.)
            dino = self.global_dino(safe.flatten(1).to(dtype=dtype))
            global_token = global_token + dino * global_valid[:, None]
        tokens = global_token[:, None].expand(-1, _PARTS, -1)
        valid = torch.zeros(len(crops), _PARTS, dtype=torch.bool, device=crops.device)
        if parts is None:
            return AppearanceEncoding(tokens, valid)
        labels = torch.arange(_PARTS, device=crops.device)[None, :, None, None]
        semantic = (parts[:, None] == labels) & mask[:, None]
        valid = semantic.flatten(2).any(-1)
        slots = valid.any(1)
        if not slots.any():
            return AppearanceEncoding(tokens, valid)
        feature = torch.where(mask[slots, None], crops[slots], 0.).to(dtype=dtype)
        semantic = semantic[slots].float()
        pooled = []
        for layer in self.pyramid:
            feature = layer(feature)
            coverage = F.adaptive_avg_pool2d(semantic, feature.shape[-2:]).flatten(2)
            mean = (coverage @ feature.float().flatten(2).transpose(1, 2)) / coverage.sum(-1, keepdim=True).clamp_min(1e-6)
            pooled.append(mean.to(dtype=dtype))
        local = self.local_part(torch.cat(pooled, dim=-1)) * valid[slots, :, None]
        tokens = tokens.clone()
        tokens[slots] = tokens[slots] + local
        return AppearanceEncoding(tokens, valid)


def dino_model_fingerprint(root):
    """Hash the exact local Hugging Face DINOv3 config, preprocessing and weights."""
    root = Path(root).resolve(strict=True)
    names = ("config.json", "preprocessor_config.json")
    entries = [root / name for name in names]
    weights = sorted(root.glob("model*.safetensors"))
    if not weights or any(not path.is_file() for path in entries):
        raise ValueError("Expected local DINOv3 config, preprocessing config and safetensors weights")
    entries += weights + sorted(root.glob("model*.safetensors.index.json"))
    return fingerprint([(path.name, sha256_file(path)) for path in entries])


class FrozenDinoAppearance:
    """Optional off-model cache extractor for a pinned local DINOv3 ViT."""

    def __init__(self, root, *, expected_sha, device="cpu"):
        from transformers import AutoConfig, AutoModel

        root = Path(root).resolve(strict=True)
        self.model_sha = dino_model_fingerprint(root)
        if self.model_sha != expected_sha:
            raise ValueError("Local DINOv3 model SHA256 differs from the pinned fingerprint")
        config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        if config.model_type != "dinov3_vit":
            raise ValueError("Appearance cache requires a DINOv3 ViT with CLS and patch tokens")
        self.model = AutoModel.from_pretrained(root, config=config, local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True).eval().to(device)
        self.model.requires_grad_(False)
        params = json.loads((root / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.mean = torch.as_tensor(params["image_mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.as_tensor(params["image_std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid local DINOv3 image normalization")
        patch_size = config.patch_size
        patch_hw = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(patch_hw) != 2 or any(step <= 0 or APPEARANCE_SIZE % step for step in patch_hw):
            raise ValueError("DINOv3 patch size must partition the 256-pixel source crop")
        self.patch_grid = tuple(APPEARANCE_SIZE // step for step in patch_hw)
        self.prefix = 1 + config.num_register_tokens

    @torch.no_grad()
    def __call__(self, crops, mask):
        crops = crops.to(self.mean)
        mask = mask[:, None].to(self.mean)
        normalized = ((crops - self.mean) / self.std) * mask
        target = (APPEARANCE_SIZE, APPEARANCE_SIZE)
        if crops.shape[-2:] != target:
            normalized = F.interpolate(normalized, size=target, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=target, mode="nearest")
        pixels = (normalized * mask).to(next(self.model.parameters()).dtype)
        hidden = self.model(pixel_values=pixels).last_hidden_state.float()
        patches = hidden[:, self.prefix:]
        coverage = F.adaptive_avg_pool2d(mask, self.patch_grid).flatten(2)
        if patches.shape[1] != coverage.shape[-1]:
            raise ValueError("DINOv3 returned a patch grid inconsistent with the pinned config")
        patch_mean = (coverage @ patches).squeeze(1) / coverage.sum(-1).clamp_min(1e-6)
        features = torch.stack((hidden[:, 0], patch_mean), dim=1)
        reduced = F.adaptive_avg_pool1d(features.flatten(0, 1)[:, None], APPEARANCE_DINO_DIM).squeeze(1)
        return F.normalize(reduced, dim=-1).view(len(crops), 2, APPEARANCE_DINO_DIM)


def attach_dino_appearance(condition, extractor, source_hashes):
    """Bind cached descriptors to source image, source crop, mask and model hashes."""
    condition.validate()
    verify_appearance_sources(condition, source_hashes)
    if condition.appearance is None or condition.appearance_mask is None or not condition.appearance_valid.any():
        raise ValueError("DINOv3 caching requires eligible source RGB crops with an explicit foreground mask")
    slots = condition.appearance_valid.nonzero().flatten()
    descriptors = extractor(condition.appearance[slots], condition.appearance_mask[slots])
    if descriptors.shape != (len(slots), 2, APPEARANCE_DINO_DIM) or not torch.isfinite(descriptors).all():
        raise ValueError("DINOv3 extractor returned invalid CLS/patch descriptors")
    cached = condition.appearance.new_zeros(4, 2, APPEARANCE_DINO_DIM, dtype=torch.float32)
    cached[slots] = descriptors.to(cached)
    digests = [None] * 4
    for slot in slots.tolist():
        digests[slot] = dino_crop_fingerprint(condition.appearance_sources[slot], extractor.model_sha,
                                              condition.appearance[slot], condition.appearance_mask[slot], cached[slot])
    return replace(condition, appearance_dino=cached, appearance_dino_valid=condition.appearance_valid.clone(),
                   appearance_dino_model_sha=extractor.model_sha, appearance_dino_crop_sha=tuple(digests)).validate()
