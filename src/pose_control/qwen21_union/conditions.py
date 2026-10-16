"""官方 4-token image slot、129 通道控制和有效内容区契约。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def expand_image_slots(img_mask, img_shapes):
    """Expand VLM image slots fourfold, retaining text/padding positions.

    ``img_shapes`` describes each reference followed by the target. This helper
    validates cached layouts; live scope masking uses official metadata instead.
    """
    if img_mask.ndim != 2 or img_mask.shape[0] != 1 or img_mask.dtype != torch.bool:
        raise ValueError("img_mask must be a single-row boolean VLM slot mask")
    tokens = [math.prod(shape) for shape in img_shapes]
    if not tokens or any(n <= 0 or n % 4 for n in tokens):
        raise ValueError("image token counts must be positive multiples of four")
    repeats = torch.where(img_mask[0], 4, 1)
    image_mask = torch.repeat_interleave(img_mask[0], repeats)
    if int(image_mask.sum()) != sum(tokens):
        raise ValueError("image slots and img_shapes have different token counts")
    positions = image_mask.nonzero(as_tuple=True)[0]
    target_mask = torch.zeros_like(image_mask)
    target_mask[positions[-tokens[-1]:]] = True
    if not bool(target_mask[-tokens[-1]:].all()):
        raise ValueError("target image token slots must be appended last")
    return image_mask, target_mask


def build_control_context(reference_latents, control_latents):
    """Official pure-pose recipe: references=0, target=[control64,zeros65]."""
    if reference_latents.ndim != 3 or control_latents.ndim != 3:
        raise ValueError("latents must have shape [B,N,64]")
    if reference_latents.shape[0] != control_latents.shape[0] or control_latents.shape[-1] != 64:
        raise ValueError("reference/control batch or latent channels mismatch")
    if reference_latents.shape[-1] != 64:
        raise ValueError("reference latent channels must be 64")
    b, nt, _ = control_latents.shape
    target = torch.cat([control_latents, control_latents.new_zeros(b, nt, 65)], dim=-1)
    return torch.cat([target.new_zeros(b, reference_latents.shape[1], 129), target], dim=1)


def valid_target_tokens(target_hw, resized_wh, offset_xy, vae_scale_factor=16):
    """Area coverage of letterbox content, including fractional boundary tiles."""
    height, width = map(int, target_hw)
    resized_w, resized_h = map(int, resized_wh)
    left, top = map(int, offset_xy)
    if min(height, width, resized_w, resized_h) <= 0 or min(left, top) < 0:
        raise ValueError("invalid letterbox geometry")
    if left + resized_w > width or top + resized_h > height:
        raise ValueError("letterbox content exceeds target canvas")
    if height % vae_scale_factor or width % vae_scale_factor:
        raise ValueError("target dimensions must be divisible by VAE scale factor")
    mask = torch.zeros(1, 1, height, width, dtype=torch.float32)
    mask[:, :, top:top + resized_h, left:left + resized_w] = 1
    return F.interpolate(mask, size=(height // vae_scale_factor, width // vae_scale_factor),
                         mode="area").flatten(1)
