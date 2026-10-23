"""Does padding reach valid tokens through Qwen's attention? Measured the right way.

An earlier probe took the gradient of an *unmasked* loss and reported a 1.33 "leak". That proved
nothing about attention: an unmasked loss touches every token by construction. The training loss
does mask (valid_body_velocity_loss with the canvas-valid tokens), so the question the plan asks
is narrower and testable directly:

    perturb only the fully-invalid latent tokens; do the outputs at VALID token positions move?

If they do not, padding cannot influence the result and no attention mask is needed. If they do,
the amount is reported so the decision is made on a number rather than a worry.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/shangguanrz/project/pic-edit")

from PIL import Image

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.execution import MODEL
from src.pose_control.qwen21_full.qwen_runtime import (
    PROMPT, Qwen21IPERCache, dataset_image_to_pil, transformer_inputs)

OUT = Path("/tmp/padding_probe")
OUT.mkdir(parents=True, exist_ok=True)

# 400x225 (H,W) -> canvas 416x256 -> grid 26x16. The 16px right band and 16px bottom band
# are entirely padding, so the probe has genuinely invalid tokens to perturb.
spec = native_pad((400, 225))
token_h, token_w = spec.token_hw
height, width = spec.original_hw


def token_classes():
    invalid, valid = [], []
    for row in range(token_h):
        for col in range(token_w):
            x0, y0 = col * 16, row * 16
            area = (min(x0 + 16, width) - x0) * (min(y0 + 16, height) - y0)
            (invalid if area <= 0 else valid).append(row * token_w + col)
    return torch.tensor(valid, device="cuda"), torch.tensor(invalid, device="cuda")


def canvas_tensor(kind="zero"):
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    band = (rows + cols * 0.5) % 256.0
    rgb = torch.from_numpy(np.dstack([band, 255.0 - band, np.roll(band, 31, axis=1)])).permute(2, 0, 1)
    if kind == "zero":
        canvas = spec.stack(rgb) / 127.5 - 1.0
    else:
        # Same pixels, but the margin is replicated from the edge instead of constant black.
        padded = torch.nn.functional.pad(rgb.unsqueeze(0), (0, spec.canvas_hw[1] - width,
                                                             0, spec.canvas_hw[0] - height),
                                         mode="replicate")
        canvas = padded[0] / 127.5 - 1.0
    return canvas.unsqueeze(0).float()


from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to("cuda")
cache = Qwen21IPERCache(OUT / "cache", resolution="native", canvas_policy="native")
batch = {"src_image": canvas_tensor("zero"), "tgt_image": canvas_tensor("zero")}
pair = cache.pair(pipe, batch, torch.device("cuda"), prompts=[PROMPT])
valid, invalid = token_classes()
print(f"canvas {spec.canvas_hw} grid {(token_h, token_w)} tokens {token_h * token_w}: "
      f"{len(valid)} valid, {len(invalid)} fully padding", flush=True)

sigma = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
inputs = transformer_inputs(pair, pair["target_latent"].clone(), sigma)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    baseline = pipe.transformer(**inputs)[0][:, -pair["target_latent"].shape[1]:]

# Same tokens, but the padding positions now carry a large perturbation.
perturbed_latent = pair["target_latent"].clone()
generator = torch.Generator(device="cuda").manual_seed(3)
noise = torch.randn((1, len(invalid), perturbed_latent.shape[-1]), generator=generator,
                    device="cuda", dtype=perturbed_latent.dtype)
magnitude = float(perturbed_latent.float().abs().mean())
perturbed_latent[0, invalid] = noise * (magnitude * 10.0)
inputs_p = transformer_inputs(pair, perturbed_latent, sigma)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    after = pipe.transformer(**inputs_p)[0][:, -pair["target_latent"].shape[1]:]

base_valid = baseline[0, valid].float()
moved_valid = after[0, valid].float()
delta = (moved_valid - base_valid).abs()
signal = base_valid.abs().mean()
print(f"padding perturbation = 10x mean |latent| (mean |latent| = {magnitude:.4f})")
print(f"valid-token output shift: max {float(delta.max()):.6f}  mean {float(delta.mean()):.6f}  "
      f"relative to signal {float(delta.max() / signal):.4%}")
print(f"valid-token output RMS before/after: "
      f"{float(base_valid.square().mean().sqrt()):.6f} / {float(moved_valid.square().mean().sqrt()):.6f}")
shift_only_padding = (after[0, invalid].float() - baseline[0, invalid].float()).abs().mean()
print(f"(control) padding-token shift itself: {float(shift_only_padding):.6f}")
verdict = ("no measurable influence: padding does not reach valid tokens, so no attention mask is needed"
           if float(delta.max()) < 1e-4 else
           "padding DOES influence valid-token outputs; attention-level masking is required")
print(f"VERDICT: {verdict}")

# ---- the decision-relevant case: does the padding's CONTENT matter in practice? ----------
# Real padding latents are a near-constant, not noise, so the practical question is how much
# the result moves when only the margin differs (black vs edge-replicated) with identical pixels.
edge_batch = {"src_image": canvas_tensor("edge"), "tgt_image": canvas_tensor("edge")}
pair_edge = cache.pair(pipe, edge_batch, torch.device("cuda"), prompts=[PROMPT])
edge_inputs = transformer_inputs(pair_edge, pair_edge["target_latent"].clone(), sigma)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    edge_out = pipe.transformer(**edge_inputs)[0][:, -pair["target_latent"].shape[1]:]
edge_delta = (edge_out[0, valid].float() - baseline[0, valid].float()).abs()
signal = float(base_valid.abs().mean())
print(f"\nrealistic case, identical image pixels, margin black -> edge-replicated:")
print(f"  valid-token shift: max {float(edge_delta.max()):.6f}  mean {float(edge_delta.mean()):.6f}  "
      f"relative to signal {float(edge_delta.max()) / signal:.4%}")
print(f"  (same-batch repeat check) padding-token shift: "
      f"{float((edge_out[0, invalid].float() - baseline[0, invalid].float()).abs().mean()):.6f}")
practical = float(edge_delta.max()) / signal
print(f"PRACTICAL VERDICT: {'negligible — the mechanism exists but padding content does not move the result' if practical < 0.01 else 'material — padding content changes valid outputs by ' + format(practical, '.2%') + ' of signal, so attention-level masking is worth the semantics change'}")
