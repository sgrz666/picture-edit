#!/usr/bin/env python3
"""Objective evaluation of the TikTok single-sample V6.5 Face experiment.

Read-only: consumes only existing PNG artefacts plus the four TorchScript
face evaluators.  No generative model is loaded, so the GPU is not touched
and no training artefact is modified.

The experiment is ``tiktok_v65_face_overfit_00001`` (TikTok sequence 00001,
source frame 0014 -> target frame 0074) and its generative counterpart
``tiktok_v65_pure_inference``.

Two metric families are reported:

* **Reproduction** replicating the training-script definitions exactly
  (FaceSim / LPIPS / landmark NME against the *target* frame crop).  These
  must line up with ``metrics.json``; a mismatch would indicate an
  evaluation bug rather than a modelling result.
* **Extension** metrics the training script does not report: pixel-space
  PSNR/SSIM (ROI and full image) and, crucially, FaceSim against the
  *source* frame crop, which is the correct reference for an
  identity-preserving edit.

Usage::

    python scripts/eval_tiktok_single_sample_objective.py \
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_iper_v65_face import (
    _crop_batch,
    _feature_distance,
    _landmark_nme,
    _load_torchscript_evaluator,
    _pairwise_distance,
)

DEFAULT_FEATURES = (
    "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
    "TikTok_3d_assets_native/00001/v65_face_features.pt"
)
DEFAULT_FRAMES = (
    "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
    "TikTok_dataset/TikTok_dataset/00001/images"
)
DEFAULT_PREVIEWS = (
    "/home/shangguanrz/project/pic-edit/experiments/"
    "tiktok_v65_face_overfit_00001/previews"
)
DEFAULT_PURE = (
    "/home/shangguanrz/project/pic-edit/experiments/tiktok_v65_pure_inference"
)
DEFAULT_EVAL = "/home/shangguanrz/project/pic-edit/models/face_eval"


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def load_rgb(path: Path, size: int = 512) -> torch.Tensor:
    """Load a PNG as a [1,3,size,size] float tensor in [0,1] (bilinear)."""
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None].contiguous()


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = F.mse_loss(reference.float(), candidate.float()).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Mean SSIM on [B,3,H,W] float tensors in [0,1] (skimage, RGB)."""
    from skimage.metrics import structural_similarity

    a = reference.detach().cpu().numpy()
    b = candidate.detach().cpu().numpy()
    values = [
        structural_similarity(
            a[i].transpose(1, 2, 0),
            b[i].transpose(1, 2, 0),
            channel_axis=2,
            data_range=1.0,
        )
        for i in range(a.shape[0])
    ]
    return float(np.mean(values))


def crop_face(image: torch.Tensor, box: torch.Tensor, size: int = 112) -> torch.Tensor:
    """Reuse the project crop so the reproduction metrics share one definition."""
    return _crop_batch(image.float(), box.float(), size)


def arccos_similarity(evaluator, a: torch.Tensor, b: torch.Tensor) -> float:
    distance = _feature_distance(evaluator, a, b, cosine=True)
    return float((1.0 - distance).mean().cpu())


# ---------------------------------------------------------------------------
# evaluation cores
# ---------------------------------------------------------------------------
def evaluate_pair(
    evaluators: dict[str, Any],
    on_pixels: torch.Tensor,
    off_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    source_pixels: torch.Tensor,
    target_box: torch.Tensor,
    source_box: torch.Tensor,
) -> dict[str, float]:
    """All metrics for one ON/OFF pair against the same target reference."""
    identity = evaluators["identity"]
    lpips = evaluators["lpips"]
    landmark = evaluators["landmark"]

    on_crop = crop_face(on_pixels, target_box)
    off_crop = crop_face(off_pixels, target_box)
    tgt_crop = crop_face(target_pixels, target_box)
    # Source crop uses the *source* frame box: the source face sits elsewhere.
    src_crop = crop_face(source_pixels, source_box)

    record: dict[str, float] = {}

    # ---- reproduction family (against target frame, as the trainer does) ----
    record["repro_face_on_similarity"] = arccos_similarity(identity, on_crop, tgt_crop)
    record["repro_face_off_similarity"] = arccos_similarity(identity, off_crop, tgt_crop)
    record["repro_face_on_lpips"] = float(
        _pairwise_distance(lpips, on_crop, tgt_crop).mean().cpu()
    )
    record["repro_face_off_lpips"] = float(
        _pairwise_distance(lpips, off_crop, tgt_crop).mean().cpu()
    )
    record["repro_face_on_landmark_nme"] = float(
        _landmark_nme(landmark, on_crop, tgt_crop).mean().cpu()
    )
    record["repro_face_off_landmark_nme"] = float(
        _landmark_nme(landmark, off_crop, tgt_crop).mean().cpu()
    )

    # ---- extension family: identity preservation against the SOURCE face ----
    record["ext_face_on_similarity_to_source"] = arccos_similarity(identity, on_crop, src_crop)
    record["ext_face_off_similarity_to_source"] = arccos_similarity(identity, off_crop, src_crop)

    # ---- extension family: pixel fidelity, ROI and full frame ----
    record["ext_roi_psnr_on"] = psnr(tgt_crop, on_crop)
    record["ext_roi_psnr_off"] = psnr(tgt_crop, off_crop)
    record["ext_roi_ssim_on"] = ssim(tgt_crop, on_crop)
    record["ext_roi_ssim_off"] = ssim(tgt_crop, off_crop)
    record["ext_full_psnr_on"] = psnr(target_pixels, on_pixels)
    record["ext_full_psnr_off"] = psnr(target_pixels, off_pixels)
    record["ext_full_ssim_on"] = ssim(target_pixels, on_pixels)
    record["ext_full_ssim_off"] = ssim(target_pixels, off_pixels)

    # ---- raw ON/OFF divergence, inside and outside the face ROI ----
    full_mask = torch.ones_like(on_pixels)
    face_mask = rasterize_box(on_pixels, target_box)
    outside = full_mask - face_mask
    diff = (on_pixels.float() - off_pixels.float()).square()
    record["ext_on_off_roi_mse"] = float(
        (diff * face_mask).sum() / face_mask.sum().clamp_min(1.0)
    )
    record["ext_on_off_outside_mse"] = float(
        (diff * outside).sum() / outside.sum().clamp_min(1.0)
    )
    return record


def rasterize_box(pixels: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Binary mask of the normalized box, broadcast to the full image."""
    b, _, h, w = pixels.shape
    y = (torch.arange(h, dtype=torch.float32, device=pixels.device) + 0.5) / h
    x = (torch.arange(w, dtype=torch.float32, device=pixels.device) + 0.5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x0, y0, x1, y1 = box[0].to(pixels.device)
    mask = ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).float()
    return mask[None, None].expand(b, 1, h, w)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--previews", default=DEFAULT_PREVIEWS)
    parser.add_argument("--pure-dir", default=DEFAULT_PURE)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", default="experiments/tiktok_objective_eval")
    parser.add_argument("--source-stem", default="0014")
    parser.add_argument("--target-stem", default="0074")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    features = torch.load(args.features, map_location="cpu", weights_only=False)
    boxes = features["face_boxes"].float()
    identity_cache = features["arcface"].float()
    stem_to_idx = features["stem_to_idx"]
    source_index = stem_to_idx[args.source_stem]
    target_index = stem_to_idx[args.target_stem]
    source_box = boxes[source_index][None]
    target_box = boxes[target_index][None]

    frames = Path(args.frames)
    source_pixels = load_rgb(frames / f"{args.source_stem}.png", args.resolution).to(device)
    target_pixels = load_rgb(frames / f"{args.target_stem}.png", args.resolution).to(device)

    evaluators = {
        name: _load_torchscript_evaluator(
            str(Path(args.eval_root) / filename), label=name, device=device
        )
        for name, filename in (
            ("identity", "identity_encoder.ts"),
            ("lpips", "lpips.ts"),
            ("landmark", "landmark_encoder.ts"),
        )
    }

    report: dict[str, Any] = {
        "experiment": "tiktok_v65_face_overfit_00001",
        "sequence": "00001",
        "source_stem": args.source_stem,
        "target_stem": args.target_stem,
        "resolution": args.resolution,
        "source_box": [round(v, 5) for v in source_box[0].tolist()],
        "target_box": [round(v, 5) for v in target_box[0].tolist()],
    }

    # --- identity sanity check: are source and target the same person? ---
    seq_similarities = F.cosine_similarity(
        identity_cache[:, None, :], identity_cache[None, :, :], dim=-1
    )
    off_diagonal = seq_similarities[~torch.eye(len(identity_cache), dtype=torch.bool)]
    reference = float(
        F.cosine_similarity(identity_cache[source_index], identity_cache[target_index], dim=-1)
    )
    report["identity_sanity"] = {
        "source_vs_target_cosine": round(reference, 5),
        "sequence_mean_cosine": round(float(off_diagonal.mean()), 5),
        "sequence_p05": round(float(off_diagonal.quantile(0.05)), 5),
        "sequence_p95": round(float(off_diagonal.quantile(0.95)), 5),
        "sequence_min": round(float(off_diagonal.min()), 5),
        "sequence_max": round(float(off_diagonal.max()), 5),
    }

    # --- static references for context ---
    report["references"] = {
        "source_vs_target_roi_psnr": psnr(
            crop_face(target_pixels, target_box), crop_face(source_pixels, target_box)
        ),
        "source_vs_target_full_psnr": psnr(target_pixels, source_pixels),
    }

    # --- training-script snapshots (single-step x0 estimates) ---
    snapshots: list[dict[str, Any]] = []
    previews = sorted(Path(args.previews).glob("step_*_face_on.png"))
    for on_path in previews:
        label = on_path.name.replace("_face_on.png", "")
        off_path = on_path.with_name(f"{label}_face_off.png")
        if not off_path.exists():
            continue
        on_pixels = load_rgb(on_path, args.resolution).to(device)
        off_pixels = load_rgb(off_path, args.resolution).to(device)
        record = {"label": label}
        record.update(
            evaluate_pair(
                evaluators,
                on_pixels,
                off_pixels,
                target_pixels,
                source_pixels,
                target_box,
                source_box,
            )
        )
        snapshots.append(record)
        print(
            f"{label}: FaceSim(on/off vs tgt)="
            f"{record['repro_face_on_similarity']:.5f}/{record['repro_face_off_similarity']:.5f} "
            f"LPIPS={record['repro_face_on_lpips']:.5f}/{record['repro_face_off_lpips']:.5f} "
            f"vsSrc={record['ext_face_on_similarity_to_source']:.5f}",
            flush=True,
        )
    report["snapshots"] = snapshots

    # --- genuine generative inference (28-step sampling from pure noise) ---
    pure = Path(args.pure_dir)
    generative: dict[str, Any] = {}
    gen_on = pure / "pure_inference_face_on.png"
    gen_off = pure / "pure_inference_face_off.png"
    if gen_on.exists() and gen_off.exists():
        on_pixels = load_rgb(gen_on, args.resolution).to(device)
        off_pixels = load_rgb(gen_off, args.resolution).to(device)
        generative.update(
            evaluate_pair(
                evaluators,
                on_pixels,
                off_pixels,
                target_pixels,
                source_pixels,
                target_box,
                source_box,
            )
        )
        # For a from-noise synthesis the target GT is not a reconstruction
        # target, so add the source-referenced identity reading explicitly.
        generative["note"] = (
            "生成图为 28 步从纯噪声采样，与 target 帧不构成重建关系；"
            "PSNR/SSIM/LPIPS 仅作参考，身份相似度为主要证据。"
        )
    report["generative"] = generative

    path = output / "objective_eval.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
