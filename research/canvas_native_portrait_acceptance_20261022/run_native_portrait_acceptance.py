#!/usr/bin/env python3
"""Real Qwen VAE + DiT acceptance for the aspect-preserving canvas (plan §8, GPU part).

CPU tests can prove the contract; they cannot prove that a portrait survives the actual
VAE, diffusers' 32-px latent unpack, the DiT and a training backward pass. This probe does,
on real TikTok footage, and measures the one question the plan left open: do padding tokens
stay out of Qwen's attention, or do they still receive gradient?

Run: /home/shangguanrz/nihaochen/envs/qwen21/bin/python run_native_portrait_acceptance.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/shangguanrz/project/pic-edit")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImage21Pipeline

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.execution import MODEL, Qwen21IPERCache
from src.pose_control.qwen21_full.qwen_runtime import PROMPT, transformer_inputs

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
TIKTOK_FRAME = (ROOT / "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset/00012/images/0264.png")
REPORT: dict = {"steps": [], "errors": []}


def note(name: str, payload: dict) -> None:
    REPORT["steps"].append({"step": name, **payload})
    print(f"[{name}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def to_batch_tensor(canvas: torch.Tensor) -> torch.Tensor:
    """Dataset-normalised [1,3,H,W]: portrait_pixels() already returns CHW, so no permute here."""
    return (canvas / 127.5 - 1.0).unsqueeze(0).float()


def portrait_pixels(spec) -> torch.Tensor:
    """Deterministic gradient body so pixel positions map to token positions analytically."""
    height, width = spec.original_hw
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    band = (rows + cols * 0.5) % 256.0
    rgb = np.dstack([band, 255.0 - band, np.roll(band, 32, axis=1)])
    return spec.stack(torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), fill=0.0)


def bucket_gradient(gradient, spec):
    """Split target-token gradient norms by how much of the 16x16 block is real image."""
    height, width = spec.original_hw
    token_h, token_w = spec.token_hw
    per_token = gradient[0].detach().float().square().sum(-1).sqrt()
    buckets: dict[str, list] = {"fully_invalid": [], "partial": [], "valid": []}
    for row in range(token_h):
        for col in range(token_w):
            x0, y0 = col * 16, row * 16
            area = (min(x0 + 16, width) - x0) * (min(y0 + 16, height) - y0)
            value = float(per_token[row * token_w + col])
            key = "fully_invalid" if area <= 0 else "partial" if area < 256 else "valid"
            buckets[key].append(value)
    mean = lambda values: float(np.mean(values)) if values else None
    valid_mean, invalid_mean = mean(buckets["valid"]), mean(buckets["fully_invalid"])
    return {"fully_invalid_tokens": len(buckets["fully_invalid"]),
            "partial_tokens": len(buckets["partial"]), "valid_tokens": len(buckets["valid"]),
            "grad_norm_fully_invalid": invalid_mean, "grad_norm_partial": mean(buckets["partial"]),
            "grad_norm_valid": valid_mean,
            "invalid_over_valid": (invalid_mean / valid_mean) if invalid_mean and valid_mean else None}


def main() -> None:
    started = time.time()
    if not TIKTOK_FRAME.is_file():
        note("input", {"fatal": f"missing real frame {TIKTOK_FRAME}"})
        raise SystemExit(1)

    real = Image.open(TIKTOK_FRAME).convert("RGB")
    note("input", {"real_frame_wh": list(real.size)})

    # Canvas for the real 604x1080 frame, and for the plan's 750x1333 -> 768x1344 example.
    real_spec = native_pad((real.height, real.width))
    odd_spec = native_pad((1333, 750))
    note("canvas", {"real": {"original_hw": list(real_spec.original_hw),
                             "canvas_hw": list(real_spec.canvas_hw),
                             "token_hw": list(real_spec.token_hw),
                             "token_count": real_spec.token_count},
                    "odd": {"original_hw": list(odd_spec.original_hw),
                            "canvas_hw": list(odd_spec.canvas_hw),
                            "token_hw": list(odd_spec.token_hw),
                            "token_count": odd_spec.token_count},
                    "square_reference": {"token_hw": [64, 64], "token_count": 4096}})

    cache = Qwen21IPERCache(OUT / "cache", resolution="native", canvas_policy="native")
    note("cache", {"preprocessing": cache.preprocessing, "canvas_policy": cache.canvas_policy,
                   "root": str(OUT / "cache")})

    print("loading pipeline...", flush=True)
    load_started = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to("cuda")
    note("pipeline", {"load_seconds": round(time.time() - load_started, 1),
                      "model": str(MODEL)})

    # ---- A. the real VAE must produce a rectangular token grid -----------------------
    source_tensor = to_batch_tensor(portrait_pixels(real_spec))
    target_tensor = to_batch_tensor(portrait_pixels(odd_spec))
    batch = {"src_image": source_tensor, "tgt_image": target_tensor}
    pair = cache.pair(pipe, batch, torch.device("cuda"), prompts=[PROMPT])
    note("vae_pair", {
        "source_canvas": list(source_tensor.shape[-2:]), "target_canvas": list(target_tensor.shape[-2:]),
        "source_token_hw": list(pair["source_token_hw"]), "target_token_hw": list(pair["target_token_hw"]),
        "source_tokens": int(pair["source_latent"].shape[1]),
        "target_tokens": int(pair["target_latent"].shape[1]),
        "latent_dim": int(pair["target_latent"].shape[-1]),
        "rectangular": pair["target_token_hw"][0] != pair["target_token_hw"][1],
        "not_square_64": tuple(pair["target_token_hw"]) != (64, 64)})

    # Build every cache entry while the text encoder is still on GPU; stage C parks it on CPU.
    small_spec = native_pad((400, 225))
    small = to_batch_tensor(portrait_pixels(small_spec))
    small_pair = cache.pair(pipe, {"src_image": small, "tgt_image": small},
                            torch.device("cuda"), prompts=[PROMPT])
    note("small_pair", {"canvas_hw": list(small_spec.canvas_hw),
                        "token_hw": list(small_spec.token_hw),
                        "token_count": int(small_pair["target_latent"].shape[1])})

    # ---- B. decode back: the 32-px unpack path must return the canvas, not a square ----
    # Non-fatal: a mismatch here is recorded, but must not hide the DiT and sampling results.
    try:
        import inspect
        with torch.no_grad():
            unpacked = pipe._unpack_latents(pair["target_latent"], odd_spec.canvas_hw[0],
                                            odd_spec.canvas_hw[1], pipe.vae_scale_factor)
            decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
        note("vae_decode", {"unpack_signature": str(inspect.signature(pipe._unpack_latents)),
                            "unpacked_shape": list(unpacked.shape),
                            "decoded_shape": list(decoded.shape),
                            "expected_canvas": list(odd_spec.canvas_hw),
                            "decoded_matches_canvas": list(decoded.shape[-2:]) == list(odd_spec.canvas_hw)})
    except Exception as error:
        note("vae_decode", {"failed": f"{type(error).__name__}: {str(error)[:300]}"})
        REPORT["errors"].append(str(error)[:300])

    # ---- C. DiT forward on the rectangular grid, then backward on a smaller one -------
    # A fp32 uncheckpointed backward through the whole 20B transformer at 4032 tokens does not
    # fit in 83 GiB, so the full-size case proves the grid is accepted and the small portrait
    # case carries the gradient/padding measurement. The text encoder is parked on CPU first.
    try:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        sigma = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
        inputs = transformer_inputs(pair, pair["target_latent"].clone(), sigma)
        shapes = [[list(s) for s in shape] for shape in inputs["img_shapes"][0]]
        forward_started = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = pipe.transformer(**inputs)[0]
        note("dit_forward", {
            "forward_seconds": round(time.time() - forward_started, 1),
            "img_shapes": shapes,
            "hidden_shape": list(hidden.shape),
            "target_tokens": int(pair["target_latent"].shape[1]),
            "token_layout_ok": int(hidden.shape[1]) == int(pair["source_latent"].shape[1]
                                                           + pair["target_latent"].shape[1])})
    except Exception as error:
        import traceback
        note("dit_forward", {"failed": f"{type(error).__name__}: {str(error)[:200]}",
                             "traceback": traceback.format_exc(limit=6).strip().splitlines()[-4:]})

    try:
        noisy = small_pair["target_latent"].clone().float().requires_grad_(True)
        small_inputs = transformer_inputs(small_pair, noisy.to(small_pair["prompt_embeds"].dtype),
                                          torch.full((1,), 0.5, device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            small_hidden = pipe.transformer(**small_inputs)[0]
        tokens = int(small_pair["target_latent"].shape[1])
        small_loss = small_hidden[:, -tokens:].float().square().mean()
        small_loss.backward()
        gradient = noisy.grad
        note("dit_backward", {
            "canvas_hw": list(small_spec.canvas_hw), "token_hw": list(small_spec.token_hw),
            "token_count": tokens, "loss": float(small_loss),
            "grad_shape": list(gradient.shape), "grad_finite": bool(torch.isfinite(gradient).all())})
        note("padding_participation", {**bucket_gradient(gradient, small_spec),
                                       "reading": "SUPERSEDED for the attention question: this is the "
                                                  "gradient of an UNMASKED loss, so a nonzero value only "
                                                  "proves the loss included those tokens. See "
                                                  "probe_padding_attention.py, which perturbs padding and "
                                                  "measures valid-token outputs instead"})
    except Exception as error:
        note("dit_backward", {"failed": f"{type(error).__name__}: {str(error)[:200]}"})
        REPORT["errors"].append(str(error)[:200])

    # ---- D. short real sample at the portrait size ------------------------------------
    try:
        sample_started = time.time()
        pipe.text_encoder.to("cuda")  # stage C parked it on CPU to fit the backward graph
        torch.cuda.empty_cache()
        with torch.no_grad():
            image = pipe(prompt=PROMPT, image=[real], width=odd_spec.canvas_hw[1],
                         height=odd_spec.canvas_hw[0], num_inference_steps=2).images[0]
        path = OUT / "portrait_768x1344.png"
        image.save(path)
        unpadded = odd_spec.unpad(image)
        unpadded.save(OUT / "portrait_750x1333_unpadded.png")
        note("sample", {"seconds": round(time.time() - sample_started, 1),
                        "output_wh": list(image.size), "unpadded_wh": list(unpadded.size),
                        "requested_wh": [odd_spec.canvas_hw[1], odd_spec.canvas_hw[0]],
                        "aspect_kept": abs((unpadded.size[0] / unpadded.size[1]) - (750 / 1333)) < 1e-6,
                        "path": str(path)})
    except Exception as error:  # the sampling API differs from the reference call
        note("sample", {"failed": f"{type(error).__name__}: {str(error)[:300]}"})
        REPORT["errors"].append(str(error)[:300])

    REPORT["total_seconds"] = round(time.time() - started, 1)
    (OUT / "results.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE in {REPORT['total_seconds']}s -> {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
