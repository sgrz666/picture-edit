"""Is attention-level padding masking actually reachable? Proven by doing it, not by reading.

transformer_qwenimage21 builds `joint_key_valid` as all-True for every image position and only
overwrites the text slots, so canvas padding is always an attendable key. The same tensor is
handed to each block as `key_valid` — the argument our own native-body mixin already intercepts.

This script flips the padding positions of that argument to False from outside the backbone and
re-runs the perturbation probe. If the valid-token shift collapses, the plan's "attention 级屏蔽"
is implementable without editing diffusers; if it does not, the claim is wrong and we say so.
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/shangguanrz/project/pic-edit")

from PIL import Image

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.execution import MODEL
from src.pose_control.qwen21_full.qwen_runtime import (
    PROMPT, Qwen21IPERCache, transformer_inputs)
from diffusers import QwenImage21Pipeline

OUT = Path("/tmp/padmask_proof")
OUT.mkdir(parents=True, exist_ok=True)
spec = native_pad((400, 225))
token_h, token_w = spec.token_hw
height, width = spec.original_hw
tokens = token_h * token_w

invalid = torch.tensor([row * token_w + col
                        for row in range(token_h) for col in range(token_w)
                        if (min(col * 16 + 16, width) - col * 16) *
                           (min(row * 16 + 16, height) - row * 16) <= 0], device="cuda")
valid = torch.tensor([i for i in range(tokens) if i not in set(invalid.tolist())], device="cuda")


def canvas_tensor():
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    band = (rows + cols * 0.5) % 256.0
    rgb = torch.from_numpy(np.dstack([band, 255.0 - band, np.roll(band, 31, axis=1)])).permute(2, 0, 1)
    return (spec.stack(rgb) / 127.5 - 1.0).unsqueeze(0).float()


pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to("cuda")
cache = Qwen21IPERCache(OUT / "cache", resolution="native", canvas_policy="native")
batch = {"src_image": canvas_tensor(), "tgt_image": canvas_tensor()}
pair = cache.pair(pipe, batch, torch.device("cuda"), prompts=[PROMPT])
sigma = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)

from src.pose_control.qwen21_full.padding_attention import (
    install_padding_key_mask, padding_key_masked)

install_padding_key_mask(pipe.transformer)


def run(perturb_padding: bool, *, mask: bool):
    latents = pair["target_latent"].clone()
    if perturb_padding:
        generator = torch.Generator(device="cuda").manual_seed(3)
        noise = torch.randn((1, len(invalid), latents.shape[-1]), generator=generator,
                            device="cuda", dtype=latents.dtype)
        latents[0, invalid] = noise * (float(latents.float().abs().mean()) * 10.0)
    inputs = transformer_inputs(pair, latents, sigma)
    context = padding_key_masked(pipe.transformer, spec, allow_semantic_change=True) if mask else \
        nullcontext()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), context:
        return pipe.transformer(**inputs)[0][:, -tokens:]


with torch.no_grad():
    clean = run(False, mask=False)
    leaked = run(True, mask=False)
    masked_clean = run(False, mask=True)
    masked_leaked = run(True, mask=True)

signal = float(clean[0, valid].float().abs().mean())
shift = lambda a, b: float((a[0, valid].float() - b[0, valid].float()).abs().max())
print(f"grid {(token_h, token_w)}: {len(valid)} valid, {len(invalid)} padding tokens")
print(f"valid-token signal RMS: {signal:.6f}")
print(f"WITHOUT mask: padding perturbation shifts valid outputs by {shift(clean, leaked):.6f} "
      f"({shift(clean, leaked) / signal:.2%} of signal)")
print(f"WITH    mask: padding perturbation shifts valid outputs by {shift(masked_clean, masked_leaked):.6f} "
      f"({shift(masked_clean, masked_leaked) / signal:.2%} of signal)")
print(f"(sanity) mask itself changes valid outputs by {shift(clean, masked_clean):.6f} "
      f"({shift(clean, masked_clean) / signal:.2%}) — should be small but nonzero")
ratio = shift(masked_clean, masked_leaked) / max(shift(clean, leaked), 1e-9)
print(f"VERDICT: {'reachable — key_valid override suppresses the leak' if ratio < 0.05 else 'partial' if ratio < 0.5 else 'NOT reachable this way'} "
      f"(residual leak ratio {ratio:.4f})")
