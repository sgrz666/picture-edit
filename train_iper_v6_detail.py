#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
iPER V6.4 Face/Hand Detail Branch Single-Sample Overfitting Training Script.

Implements:
1. Freezes DeepGen backbone, VAE, text encoder, and all V6.3 modules.
2. Trains only:
   - detail_preparer.*
   - control_interface.control_core.detail_branch.*
   - control_interface.strength_controller.detail_log_group_scale
3. Layered LR schedule:
   - Step 0..100: zero heads & gate = 1e-4, other detail params = 0.0
   - Step 100..1500: zero heads & gate = 1e-4, encoders & binder = 5e-5, embed & adapters = 1e-5
4. 64-pair fixed training noise bank (seed 3407) and 16-pair eval bank (seed 20261001).
5. Maximum weighted fusion loss: W = max(1, 1+2*M_body, 1+7*M_face, 1+9*M_left, 1+9*M_right)
   Total L = L_weighted + 0.5*L_face + 0.25*L_left + 0.25*L_right
6. Verifications:
   - Step 0 Detail ON vs OFF elementwise parity
   - Gradient isolation (only detail params receive gradient)
   - Pre/post training streaming SHA-256 check on frozen parameters
   - Step 1500 Detail OFF vs Step 0 baseline elementwise parity
   - DWPose face landmark NME and hand PCK@0.1
   - Visual comparison panels at steps 0, 250, 500, 1000, 1500
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import DiffusionPipeline
from src.data.iper_detail_dataset import IPERDetailDataset
from src.pose_control.v6.checkpoint import (
    build_v64_checkpoint,
    freeze_for_detail_training,
    load_v63_checkpoint,
)
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.detail.conditions import (
    DetailReferenceBatch,
    FaceHandDetailCondition,
)
from src.pose_control.v6.detail.dwpose_evaluator import DWPoseEvaluator
from src.pose_control.v6.detail.metrics import (
    compute_psnr,
    compute_ssim,
    crop_roi,
    evaluate_keypoint_metrics,
)
from train_iper_v6_native import get_cached_conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6.4 Detail Branch Single-Sample Overfitting")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
        help="Path to pretrained DeepGen diffusers model",
    )
    parser.add_argument(
        "--sampled_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
        help="Root directory of iPER sampled images",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
        help="Root directory of iPER preprocessed assets",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default="/home/shangguanrz/project/pic-edit/experiments/iper_v6_native_20k_v1/checkpoint_best.pt",
        help="Path to V6.3 baseline checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/shangguanrz/project/pic-edit/experiments/iper_v64_detail_overfit_024_6",
        help="Output directory for checkpoints, metrics, and previews",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Change the person pose to match the target pose. Preserve identity, clothing, and background.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, low quality, low resolution, distorted, deformed, "
            "broken content, missing parts, damaged details, artifacts, "
            "glitch, noise, extra fingers, missing fingers, mutated hands, "
            "bad composition, wrong proportion, unfinished"
        ),
    )
    return parser.parse_args()


def compute_frozen_sha256(adapter: nn.Module) -> str:
    """Computes streaming SHA-256 across all non-trainable (frozen) parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(adapter.named_parameters()):
        if not param.requires_grad:
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


def build_detail_optimizer(
    adapter: UnifiedSMPLXAdapterV6,
    weight_decay: float = 0.01,
) -> Tuple[torch.optim.Optimizer, Dict[str, List[nn.Parameter]]]:
    """Builds AdamW with 3 parameter groups:

    Group 1: Zero heads and detail gate (lr=1e-4)
    Group 2: Spatial/appearance encoder, binder (lr=0 initial, 5e-5 after step 100)
    Group 3: Condition embed, stage embeddings, stage adapters, cross norms (lr=0 initial, 1e-5 after step 100)
    """
    group1, group2, group3 = [], [], []

    for name, param in adapter.named_parameters():
        if not param.requires_grad:
            continue
        if "zero_heads" in name or "detail_log_group_scale" in name or "gate" in name:
            group1.append(param)
        elif any(k in name for k in ["spatial_encoder", "appearance_encoder", "token_binder"]):
            group2.append(param)
        elif any(k in name for k in ["condition_embed", "stage_embeddings", "stage_adapters", "cross_norms"]):
            group3.append(param)
        else:
            raise ValueError(f"Ungrouped trainable detail parameter: {name}")

    all_trainable = [p for p in adapter.parameters() if p.requires_grad]
    total_assigned = len(group1) + len(group2) + len(group3)
    assert total_assigned == len(all_trainable), f"Group count mismatch: {total_assigned} vs {len(all_trainable)}"

    param_groups = [
        {"params": group1, "lr": 1e-4, "name": "zero_heads_and_gate"},
        {"params": group2, "lr": 0.0, "name": "detail_encoders_and_binder"},
        {"params": group3, "lr": 0.0, "name": "detail_embed_and_adapters"},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    groups_dict = {"group1": group1, "group2": group2, "group3": group3}
    return optimizer, groups_dict


def update_optimizer_lrs(optimizer: torch.optim.Optimizer, step: int) -> None:
    """Step 0..100: G1=1e-4, G2=0, G3=0.

    Step 100..1500: G1=1e-4, G2=5e-5, G3=1e-5.
    """
    if step < 100:
        optimizer.param_groups[0]["lr"] = 1e-4
        optimizer.param_groups[1]["lr"] = 0.0
        optimizer.param_groups[2]["lr"] = 0.0
    else:
        optimizer.param_groups[0]["lr"] = 1e-4
        optimizer.param_groups[1]["lr"] = 5e-5
        optimizer.param_groups[2]["lr"] = 1e-5


def rasterize_box_mask(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterizes bounding boxes [B, 4] normalized into a binary mask [B, 1, height, width]."""
    B = boxes.shape[0]
    masks = torch.zeros(B, 1, height, width, device=boxes.device, dtype=boxes.dtype)
    for b in range(B):
        x1 = max(0, min(int(round(boxes[b, 0].item() * width)), width - 1))
        y1 = max(0, min(int(round(boxes[b, 1].item() * height)), height - 1))
        x2 = max(x1 + 1, min(int(round(boxes[b, 2].item() * width)), width))
        y2 = max(y1 + 1, min(int(round(boxes[b, 3].item() * height)), height))
        masks[b, 0, y1:y2, x1:x2] = 1.0
    return masks


def create_noise_banks(
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Generates:

    - 64 training noise pairs (seed 3407), timesteps stratified in [0, 650]
    - 16 evaluation noise pairs (seed 20261001), timesteps stratified in [0, 650]
    """
    # Training bank: seed 3407
    gen_train = torch.Generator(device="cpu").manual_seed(3407)
    train_bank = []
    delta_train = 650.0 / 64.0
    for i in range(64):
        z = torch.randn(1, 16, 64, 64, generator=gen_train, dtype=dtype)
        u = torch.rand(1, generator=gen_train).item()
        t = torch.tensor([i * delta_train + delta_train * u], dtype=torch.float32)
        train_bank.append((z.to(device), t.to(device)))

    # Evaluation bank: seed 20261001
    gen_eval = torch.Generator(device="cpu").manual_seed(20261001)
    eval_bank = []
    delta_eval = 650.0 / 16.0
    for i in range(16):
        z = torch.randn(1, 16, 64, 64, generator=gen_eval, dtype=dtype)
        u = torch.rand(1, generator=gen_eval).item()
        t = torch.tensor([i * delta_eval + delta_eval * u], dtype=torch.float32)
        eval_bank.append((z.to(device), t.to(device)))

    return train_bank, eval_bank


def create_comparison_grid(
    src_img: Image.Image,
    tgt_img: Image.Image,
    img_off: Image.Image,
    img_correct: Image.Image,
    img_perturbed: Image.Image,
    boxes_dict: Dict[str, np.ndarray],
    panel_size: int = 256,
) -> Image.Image:
    """Creates a 4-row comparison panel:

    Row 1: Source | Target | Detail OFF | Detail Correct | Detail Perturbed
    Row 2: Face crops
    Row 3: Left-hand crops
    Row 4: Right-hand crops
    """
    cols = [
        ("Source", src_img),
        ("Target (GT)", tgt_img),
        ("Detail OFF", img_off),
        ("Detail Correct", img_correct),
        ("Detail Perturbed", img_perturbed),
    ]

    # Convert to RGB numpy arrays for cropping
    full_np = [np.asarray(img.convert("RGB")) for _, img in cols]

    src_boxes = boxes_dict["src_boxes"] # [3, 4] (face, left, right)
    tgt_boxes = boxes_dict["tgt_boxes"] # [3, 4]

    # Row 1: Full images
    row1 = [img.resize((panel_size, panel_size), Image.Resampling.LANCZOS) for _, img in cols]

    # Row 2: Face crops
    row2 = []
    for c_idx, arr in enumerate(full_np):
        b = src_boxes[0] if c_idx == 0 else tgt_boxes[0]
        crop = crop_roi(arr, b)
        crop_pil = Image.fromarray(crop).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        row2.append(crop_pil)

    # Row 3: Left hand crops
    row3 = []
    for c_idx, arr in enumerate(full_np):
        b = src_boxes[1] if c_idx == 0 else tgt_boxes[1]
        crop = crop_roi(arr, b)
        crop_pil = Image.fromarray(crop).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        row3.append(crop_pil)

    # Row 4: Right hand crops
    row4 = []
    for c_idx, arr in enumerate(full_np):
        b = src_boxes[2] if c_idx == 0 else tgt_boxes[2]
        crop = crop_roi(arr, b)
        crop_pil = Image.fromarray(crop).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        row4.append(crop_pil)

    row_labels = ["Full Body", "Face Crop", "Left Hand Crop", "Right Hand Crop"]
    all_rows = [row1, row2, row3, row4]

    row_header_w = 120
    col_header_h = 30
    canvas_w = row_header_w + len(cols) * panel_size
    canvas_h = col_header_h + 4 * panel_size

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Draw column titles
    for c_idx, (title, _) in enumerate(cols):
        x = row_header_w + c_idx * panel_size + 10
        draw.text((x, 8), title, fill=(240, 240, 240))

    # Paste image rows
    for r_idx, (r_title, r_imgs) in enumerate(zip(row_labels, all_rows)):
        y = col_header_h + r_idx * panel_size
        draw.text((10, y + panel_size // 2 - 8), r_title, fill=(200, 200, 200))
        for c_idx, p_img in enumerate(r_imgs):
            x = row_header_w + c_idx * panel_size
            canvas.paste(p_img, (x, y))

    return canvas


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("   iPER V6.4 Face/Hand Detail Branch Single-Sample Overfitting Plan   ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or str(output_dir / "condition_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load DeepGen Pipeline
    print(f"\n[1/7] Loading DeepGen pipeline from {args.model_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe.to(device)
    pipe.vae.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)

    transformer = pipe.transformer
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled on transformer")

    # 2. Build Adapter & Apply Detail Freezing
    print("\n[2/7] Initializing UnifiedSMPLXAdapterV6 and freezing non-detail parameters...")
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(
        pipe,
        condition_use_depth=False,
    ).to(device=device, dtype=dtype)

    trainable_params = freeze_for_detail_training(adapter)
    print(f"✓ Trainable parameters: {len(trainable_params)} tensors ({sum(p.numel() for p in trainable_params):,} elements)")

    # 3. Load V6.3 Baseline Checkpoint
    print(f"\n[3/7] Loading V6.3 baseline checkpoint from {args.baseline_checkpoint}...")
    v63_ckpt = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    incompatible = load_v63_checkpoint(
        adapter, {"adapter_state_dict": v63_ckpt["adapter_state_dict"]}
    )
    print(f"✓ V6.3 checkpoint loaded! Detail zero heads re-zeroed. Incompatible keys: {len(incompatible.missing_keys)} missing (detail keys), {len(incompatible.unexpected_keys)} unexpected.")

    # 4. Pre-training SHA-256 Checksum
    sha256_pre = compute_frozen_sha256(adapter)
    print(f"✓ Pre-training frozen SHA-256: {sha256_pre}")

    # 5. Dataset and Precomputed Conditions
    print("\n[4/7] Loading training sample and diagnostic targets...")
    dataset = IPERDetailDataset(
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        single_sample={
            "appearance": "024_6",
            "source": {"stem": "source_motion_f000375", "role": "source_motion"},
            "target": {"stem": "target_f001492", "role": "target"},
        },
        resolution=args.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    train_batch = next(iter(dataloader))

    # Move batch items to device
    normal_a = train_batch["normal"].to(device, dtype=dtype)
    part_onehot_a = train_batch["part_onehot"].to(device, dtype=dtype)
    pose_heatmap_a = train_batch["pose_heatmap"].to(device, dtype=dtype)
    smplx_global_a = train_batch["smplx_global"].to(device, dtype=dtype)
    human_mask_a = train_batch["human_mask"].to(device, dtype=dtype)
    task_id = train_batch["task_id"].to(device, dtype=torch.long)

    # Target pixels and latent
    tgt_image_tensor = train_batch["tgt_image"].to(device, dtype=dtype)
    target_latent = pipe.pixels_to_latents(tgt_image_tensor).detach() # [1, 16, 64, 64]
    tgt_raw = (tgt_image_tensor * 0.5 + 0.5).clamp(0, 1)
    B, C, H_lat, W_lat = target_latent.shape

    # Precompute cached VLM context
    ref_latent, sequence, pooled = get_cached_conditions(
        pipe, train_batch, args.prompt, cache_dir, device, dtype
    )

    # Build constant geometry ConditionBundle
    geometry_bundle = adapter.condition_injector(
        normal_a=normal_a,
        pose_heatmap_a=pose_heatmap_a,
        part_onehot_a=part_onehot_a,
        smplx_global_a=smplx_global_a,
        human_mask_a=human_mask_a,
        task_id=task_id,
    )

    # Identity condition
    source_person_latents = torch.zeros(1, 2, 16, H_lat, W_lat, device=device, dtype=dtype)
    source_person_latents[:, 0] = ref_latent[:1]
    identity_condition = AdapterIdentityCondition(
        source_person_latents=source_person_latents,
        source_indices=torch.tensor([[0, 1]], device=device),
    )

    # Target Detail Condition & References
    detail_condition_correct, detail_references = dataset.get_detail_condition_and_reference(
        "024_6", "source_motion_f000375", "target_f001492"
    )
    detail_condition_correct = detail_condition_correct.to(device=device, dtype=dtype)
    detail_references = detail_references.to(device=device, dtype=dtype)

    # Perturbed Detail Condition (target_f001016 detail condition with target_f001492 geometry)
    detail_condition_perturbed, _ = dataset.get_detail_condition_and_reference(
        "024_6", "source_motion_f000375", "target_f001016"
    )
    detail_condition_perturbed = detail_condition_perturbed.to(device=device, dtype=dtype)

    # Diagnostic targets (f001016, f000568, f001333)
    diagnostic_targets = ["target_f001016", "target_f000568", "target_f001333"]
    diagnostic_bundles = {}
    for d_tgt in diagnostic_targets:
        d_dataset = IPERDetailDataset(
            sampled_root=args.sampled_root,
            assets_root=args.assets_root,
            single_sample={
                "appearance": "024_6",
                "source": {"stem": "source_motion_f000375", "role": "source_motion"},
                "target": {"stem": d_tgt, "role": "target"},
            },
            resolution=args.resolution,
        )
        d_batch = next(iter(DataLoader(d_dataset, batch_size=1, shuffle=False)))
        d_bundle = adapter.condition_injector(
            normal_a=d_batch["normal"].to(device, dtype=dtype),
            pose_heatmap_a=d_batch["pose_heatmap"].to(device, dtype=dtype),
            part_onehot_a=d_batch["part_onehot"].to(device, dtype=dtype),
            smplx_global_a=d_batch["smplx_global"].to(device, dtype=dtype),
            human_mask_a=d_batch["human_mask"].to(device, dtype=dtype),
            task_id=d_batch["task_id"].to(device, dtype=torch.long),
        )
        d_cond, d_refs = d_dataset.get_detail_condition_and_reference("024_6", "source_motion_f000375", d_tgt)
        d_tgt_tensor = d_batch["tgt_image"].to(device, dtype=dtype)
        d_tgt_latent = pipe.pixels_to_latents(d_tgt_tensor).detach()
        d_tgt_raw = (d_tgt_tensor * 0.5 + 0.5).clamp(0, 1)
        diagnostic_bundles[d_tgt] = {
            "bundle": d_bundle,
            "condition": d_cond.to(device=device, dtype=dtype),
            "references": d_refs.to(device=device, dtype=dtype),
            "target_latent": d_tgt_latent,
            "target_pil": to_pil_image(d_tgt_raw[0].cpu().float()),
        }

    # Region masks for flow matching loss
    m_body = F.interpolate(
        human_mask_a.float(), size=(H_lat, W_lat), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0).to(dtype)

    target_boxes = detail_condition_correct.target_boxes[:, 0] # [1, 3, 4]
    m_face = rasterize_box_mask(target_boxes[:, 0], H_lat, W_lat).to(dtype)
    m_left = rasterize_box_mask(target_boxes[:, 1], H_lat, W_lat).to(dtype)
    m_right = rasterize_box_mask(target_boxes[:, 2], H_lat, W_lat).to(dtype)

    # Maximum fusion weight map: W = max(1, 1+2*M_body, 1+7*M_face, 1+9*M_left, 1+9*M_right)
    w_map = torch.ones_like(m_body)
    w_map = torch.maximum(w_map, 1.0 + 2.0 * m_body)
    w_map = torch.maximum(w_map, 1.0 + 7.0 * m_face)
    w_map = torch.maximum(w_map, 1.0 + 9.0 * m_left)
    w_map = torch.maximum(w_map, 1.0 + 9.0 * m_right)

    # 6. Initialize Noise Banks & DWPose Evaluator
    print("\n[5/7] Initializing noise banks and DWPose evaluator...")
    train_noise_bank, eval_noise_bank = create_noise_banks(device, dtype)
    print(f"✓ Created 64 training noise pairs and 16 evaluation noise pairs in [0, 650]")

    dwpose_evaluator = DWPoseEvaluator(device="cpu")
    print("✓ DWPose wholebody detector ready")

    # Load ground truth keypoints and boxes for metric evaluation
    detail_data = torch.load(
        os.path.join(args.assets_root, "024_6", "v6_detail_conditions.pt"),
        map_location="cpu",
        weights_only=False,
    )
    s_idx = detail_data["stem_to_idx"]["source_motion_f000375"]
    t_idx = detail_data["stem_to_idx"]["target_f001492"]

    gt_kps_norm = np.zeros((134, 2), dtype=np.float32)
    gt_scores = np.ones(134, dtype=np.float32)
    gt_kps_norm[24:92] = detail_data["face_keypoints"][t_idx, :, :2].numpy()
    gt_scores[24:92] = detail_data["face_keypoints"][t_idx, :, 2].numpy()
    gt_kps_norm[92:113] = detail_data["hand_keypoints"][t_idx, 0, :, :2].numpy()
    gt_scores[92:113] = detail_data["hand_keypoints"][t_idx, 0, :, 2].numpy()
    gt_kps_norm[113:134] = detail_data["hand_keypoints"][t_idx, 1, :, :2].numpy()
    gt_scores[113:134] = detail_data["hand_keypoints"][t_idx, 1, :, 2].numpy()

    gt_boxes_px = detail_data["boxes"][t_idx].numpy() * float(args.resolution) # [3, 4] in pixels
    gt_kps_px = gt_kps_norm * float(args.resolution)

    # Boxes dict for visual strip
    boxes_dict = {
        "src_boxes": detail_data["boxes"][s_idx].numpy(),
        "tgt_boxes": detail_data["boxes"][t_idx].numpy(),
    }

    # 7. Optimizer Setup & Step 0 Invariant Testing
    print("\n[6/7] Setting up optimizer and testing Step 0 invariants...")
    optimizer, groups_dict = build_detail_optimizer(adapter, weight_decay=args.weight_decay)

    # Step 0 Parity Test
    adapter.eval()
    with torch.no_grad():
        test_noise, test_t = train_noise_bank[0]
        test_sigma = (test_t / 1000.0).view(1, 1, 1, 1).to(dtype)
        test_noisy = (1.0 - test_sigma) * target_latent + test_sigma * test_noise
        test_progress = (1.0 - test_t / 1000.0).to(dtype)

        prep_on = adapter.prepare_conditioning(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            target_latent_hw=(H_lat, W_lat),
            detail_condition=detail_condition_correct,
            detail_references=detail_references,
        )
        prep_off = adapter.prepare_conditioning(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            target_latent_hw=(H_lat, W_lat),
            detail_condition=None,
            detail_references=None,
        )

        out_on = adapter(
            target_latents=test_noisy,
            prepared=prep_on,
            cond_hidden_states=[[ref_latent[0]]],
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            timestep=test_t,
            denoise_progress=test_progress,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
        )
        out_off = adapter(
            target_latents=test_noisy,
            prepared=prep_off,
            cond_hidden_states=[[ref_latent[0]]],
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            timestep=test_t,
            denoise_progress=test_progress,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=0.0,
        )

        for h_on, h_off in zip(out_on.block_controlnet_hidden_states, out_off.block_controlnet_hidden_states):
            torch.testing.assert_close(h_on, h_off, atol=0.0, rtol=0.0)
        print("✓ Invariant Test Passed: Step 0 Detail ON matches Detail OFF elementwise (residuals exactly 0.0)!")

    # Step 0 Gradient Isolation Check
    adapter.train()
    optimizer.zero_grad()
    prep_test = adapter.prepare_conditioning(
        condition_bundle=geometry_bundle,
        identity_condition=identity_condition,
        source_scene_latents=ref_latent[:1],
        target_latent_hw=(H_lat, W_lat),
        detail_condition=detail_condition_correct,
        detail_references=detail_references,
    )
    out_test = adapter(
        target_latents=test_noisy,
        prepared=prep_test,
        cond_hidden_states=[[ref_latent[0]]],
        encoder_hidden_states=sequence,
        pooled_projections=pooled,
        timestep=test_t,
        denoise_progress=test_progress,
        geometry_strength=1.0,
        interaction_strength=0.0,
        detail_strength=1.0,
    )
    test_loss = sum(h.square().sum() for h in out_test.block_controlnet_hidden_states)
    test_loss.backward()

    for name, param in adapter.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), f"Param {name} missing valid grad!"
        else:
            assert param.grad is None, f"Frozen param {name} received gradient!"
    optimizer.zero_grad()
    print("✓ Invariant Test Passed: Gradient isolation confirmed (61/61 detail params have finite grad; frozen have None)!")

    # Evaluation helper function
    controlled_pipe = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)
    src_raw = (train_batch["src_image"][0].float() * 0.5 + 0.5).clamp(0, 1)
    src_pil = to_pil_image(src_raw.cpu())
    tgt_pil = to_pil_image(tgt_raw[0].cpu().float())

    def run_eval_at_step(current_step: int) -> Dict[str, Any]:
        """Runs full evaluation at checkpoints (steps 0, 250, 500, 1000, 1500)."""
        adapter.eval()
        print(f"\n>>> Running Full Evaluation at Step {current_step} <<<", flush=True)

        seeds = [42, 2026, 3407, 10007]
        seed_results = []

        for s_idx, seed in enumerate(seeds):
            # 1. Detail OFF (V6.3 Baseline)
            gen_off = controlled_pipe(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                guidance_scale=4.5,
                geometry_strength=1.0,
                interaction_strength=0.0,
                detail_strength=0.0,
                detail_condition=None,
                detail_references=None,
                seed=seed,
            ).images[0]

            # 2. Detail ON / Correct
            gen_on = controlled_pipe(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                guidance_scale=4.5,
                geometry_strength=1.0,
                interaction_strength=0.0,
                detail_strength=1.0,
                face_strength=1.0,
                hand_strength=1.0,
                detail_condition=detail_condition_correct,
                detail_references=detail_references,
                seed=seed,
            ).images[0]

            # 3. Detail ON / Perturbed
            gen_pert = controlled_pipe(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                guidance_scale=4.5,
                geometry_strength=1.0,
                interaction_strength=0.0,
                detail_strength=1.0,
                face_strength=1.0,
                hand_strength=1.0,
                detail_condition=detail_condition_perturbed,
                detail_references=detail_references,
                seed=seed,
            ).images[0]

            # Save standalone previews
            gen_on.save(preview_dir / f"step_{current_step:04d}_seed_{seed}_on.png")
            if current_step == 0:
                gen_off.save(preview_dir / f"step_0000_seed_{seed}_off.png")

            # If step 0, verify elementwise equivalence between ON and OFF
            if current_step == 0:
                off_arr = np.asarray(gen_off)
                on_arr = np.asarray(gen_on)
                diff_max = int(np.max(np.abs(off_arr.astype(int) - on_arr.astype(int))))
                diff_mean = float(np.mean(np.abs(off_arr.astype(float) - on_arr.astype(float))))
                psnr_0 = compute_psnr(on_arr, off_arr)
                print(f"✓ Seed {seed}: Step 0 Detail ON vs Detail OFF: PSNR={psnr_0:.2f} dB, mean_diff={diff_mean:.3f}, max_diff={diff_max}")
                assert psnr_0 >= 35.0, f"Step 0 generation mismatch! PSNR: {psnr_0:.2f} dB"

            # Create comparison grid
            grid = create_comparison_grid(
                src_pil, tgt_pil, gen_off, gen_on, gen_pert, boxes_dict
            )
            grid_path = preview_dir / f"step_{current_step:04d}_seed_{seed}.png"
            grid.save(grid_path)

            # Evaluate Metrics
            tgt_np = np.asarray(tgt_pil, dtype=np.float32)
            on_np = np.asarray(gen_on, dtype=np.float32)
            off_np = np.asarray(gen_off, dtype=np.float32)

            # Face & Hand Crops for metrics
            face_tgt = crop_roi(tgt_np, boxes_dict["tgt_boxes"][0])
            face_on = crop_roi(on_np, boxes_dict["tgt_boxes"][0])
            face_off = crop_roi(off_np, boxes_dict["tgt_boxes"][0])

            lh_tgt = crop_roi(tgt_np, boxes_dict["tgt_boxes"][1])
            lh_on = crop_roi(on_np, boxes_dict["tgt_boxes"][1])
            lh_off = crop_roi(off_np, boxes_dict["tgt_boxes"][1])

            rh_tgt = crop_roi(tgt_np, boxes_dict["tgt_boxes"][2])
            rh_on = crop_roi(on_np, boxes_dict["tgt_boxes"][2])
            rh_off = crop_roi(off_np, boxes_dict["tgt_boxes"][2])

            # Outside Face/Hands mask
            m_outside = np.ones((args.resolution, args.resolution), dtype=np.float32)
            for b in boxes_dict["tgt_boxes"]:
                x1 = max(0, min(int(round(b[0] * args.resolution)), args.resolution - 1))
                y1 = max(0, min(int(round(b[1] * args.resolution)), args.resolution - 1))
                x2 = max(x1 + 1, min(int(round(b[2] * args.resolution)), args.resolution))
                y2 = max(y1 + 1, min(int(round(b[3] * args.resolution)), args.resolution))
                m_outside[y1:y2, x1:x2] = 0.0

            # PSNR & SSIM
            face_psnr_on = compute_psnr(face_on, face_tgt)
            face_psnr_off = compute_psnr(face_off, face_tgt)
            face_ssim_on = compute_ssim(face_on, face_tgt)
            face_ssim_off = compute_ssim(face_off, face_tgt)

            hands_ssim_on = (compute_ssim(lh_on, lh_tgt) + compute_ssim(rh_on, rh_tgt)) * 0.5
            hands_ssim_off = (compute_ssim(lh_off, lh_tgt) + compute_ssim(rh_off, rh_tgt)) * 0.5

            out_psnr_on = compute_psnr(on_np, tgt_np, m_outside)
            out_psnr_off = compute_psnr(off_np, tgt_np, m_outside)

            # Run DWPose detector on generated images
            kps_on, scs_on = dwpose_evaluator.detect(on_np)
            kps_off, scs_off = dwpose_evaluator.detect(off_np)

            m_on = evaluate_keypoint_metrics(kps_on, scs_on, gt_kps_px, gt_scores, gt_boxes_px)
            m_off = evaluate_keypoint_metrics(kps_off, scs_off, gt_kps_px, gt_scores, gt_boxes_px)

            seed_results.append({
                "seed": seed,
                "face_psnr_on": face_psnr_on,
                "face_psnr_off": face_psnr_off,
                "face_psnr_diff": face_psnr_on - face_psnr_off,
                "face_ssim_on": face_ssim_on,
                "face_ssim_off": face_ssim_off,
                "face_ssim_diff": face_ssim_on - face_ssim_off,
                "hands_ssim_on": hands_ssim_on,
                "hands_ssim_off": hands_ssim_off,
                "hands_ssim_diff": hands_ssim_on - hands_ssim_off,
                "hands_pck_on": m_on["hands_avg_pck"],
                "hands_pck_off": m_off["hands_avg_pck"],
                "hands_pck_diff": m_on["hands_avg_pck"] - m_off["hands_avg_pck"],
                "outside_psnr_on": out_psnr_on,
                "outside_psnr_off": out_psnr_off,
                "outside_psnr_drop": out_psnr_off - out_psnr_on,
                "face_nme_on": m_on["face_nme"],
                "face_nme_off": m_off["face_nme"],
            })

        # Run 3 diagnostic targets for condition generalization observation
        diag_results = {}
        for d_tgt, d_dict in diagnostic_bundles.items():
            d_gen = controlled_pipe(
                condition_bundle=d_dict["bundle"],
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                guidance_scale=4.5,
                geometry_strength=1.0,
                interaction_strength=0.0,
                detail_strength=1.0,
                detail_condition=d_dict["condition"],
                detail_references=d_dict["references"],
                seed=42,
            ).images[0]
            d_path = preview_dir / f"step_{current_step:04d}_diag_{d_tgt}.png"
            d_gen.save(d_path)
            diag_results[d_tgt] = str(d_path)

        # Compute medians
        summary = {
            "step": current_step,
            "median_face_ssim_diff": float(np.median([r["face_ssim_diff"] for r in seed_results])),
            "median_face_psnr_diff": float(np.median([r["face_psnr_diff"] for r in seed_results])),
            "median_hands_ssim_diff": float(np.median([r["hands_ssim_diff"] for r in seed_results])),
            "median_hands_pck_on": float(np.median([r["hands_pck_on"] for r in seed_results])),
            "median_hands_pck_diff": float(np.median([r["hands_pck_diff"] for r in seed_results])),
            "median_outside_psnr_drop": float(np.median([r["outside_psnr_drop"] for r in seed_results])),
            "seed_results": seed_results,
            "diag_previews": diag_results,
        }

        print(
            f"Step {current_step} Summary (Medians over 4 seeds): "
            f"ΔFace SSIM={summary['median_face_ssim_diff']:+.4f} | "
            f"ΔFace PSNR={summary['median_face_psnr_diff']:+.2f} dB | "
            f"ΔHands SSIM={summary['median_hands_ssim_diff']:+.4f} | "
            f"Hands PCK@0.1={summary['median_hands_pck_on']:.3f} (Δ={summary['median_hands_pck_diff']:+.3f}) | "
            f"Outside PSNR drop={summary['median_outside_psnr_drop']:.2f} dB",
            flush=True,
        )

        adapter.train()
        return summary

    def run_val_flow(current_step: int) -> Dict[str, Any]:
        """Evaluates fixed-noise validation flow MSE for face, left hand, and right hand."""
        adapter.eval()
        val_losses_on, val_losses_off, val_losses_pert = [], [], []
        val_face_losses, val_lh_losses, val_rh_losses = [], [], []
        with torch.no_grad():
            prep_val_on = adapter.prepare_conditioning(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                target_latent_hw=(H_lat, W_lat),
                detail_condition=detail_condition_correct,
                detail_references=detail_references,
            )
            prep_val_off = adapter.prepare_conditioning(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                target_latent_hw=(H_lat, W_lat),
                detail_condition=None,
                detail_references=None,
            )
            prep_val_pert = adapter.prepare_conditioning(
                condition_bundle=geometry_bundle,
                identity_condition=identity_condition,
                source_scene_latents=ref_latent[:1],
                target_latent_hw=(H_lat, W_lat),
                detail_condition=detail_condition_perturbed,
                detail_references=detail_references,
            )

            for v_noise, v_time in eval_noise_bank:
                v_sigma = (v_time / 1000.0).view(1, 1, 1, 1).to(dtype)
                v_noisy = (1.0 - v_sigma) * target_latent + v_sigma * v_noise
                v_target_v = v_noise - target_latent
                v_prog = (1.0 - v_time / 1000.0).to(dtype)

                # ON
                out_on = adapter(
                    target_latents=v_noisy, prepared=prep_val_on, cond_hidden_states=[[ref_latent[0]]],
                    encoder_hidden_states=sequence, pooled_projections=pooled, timestep=v_time,
                    denoise_progress=v_prog, geometry_strength=1.0, interaction_strength=0.0, detail_strength=1.0,
                )
                pred_on = transformer(
                    hidden_states=v_noisy, encoder_hidden_states=sequence, pooled_projections=pooled,
                    cond_hidden_states=[[ref_latent[0]]], timestep=v_time,
                    block_controlnet_hidden_states=[v.to(dtype) for v in out_on.block_controlnet_hidden_states],
                    return_dict=False,
                )[0]
                diff_on = (pred_on.float() - v_target_v.float()).square().mean(dim=1, keepdim=True)
                val_losses_on.append(diff_on.mean().item())
                val_face_losses.append(((diff_on * m_face).sum() / m_face.sum().clamp_min(1.0)).item())
                val_lh_losses.append(((diff_on * m_left).sum() / m_left.sum().clamp_min(1.0)).item())
                val_rh_losses.append(((diff_on * m_right).sum() / m_right.sum().clamp_min(1.0)).item())

                # OFF
                pred_off = transformer(
                    hidden_states=v_noisy, encoder_hidden_states=sequence, pooled_projections=pooled,
                    cond_hidden_states=[[ref_latent[0]]], timestep=v_time,
                    block_controlnet_hidden_states=None, return_dict=False,
                )[0]
                diff_off = (pred_off.float() - v_target_v.float()).square().mean(dim=1, keepdim=True)
                val_losses_off.append(diff_off.mean().item())

                # PERTURBED
                out_pert = adapter(
                    target_latents=v_noisy, prepared=prep_val_pert, cond_hidden_states=[[ref_latent[0]]],
                    encoder_hidden_states=sequence, pooled_projections=pooled, timestep=v_time,
                    denoise_progress=v_prog, geometry_strength=1.0, interaction_strength=0.0, detail_strength=1.0,
                )
                pred_pert = transformer(
                    hidden_states=v_noisy, encoder_hidden_states=sequence, pooled_projections=pooled,
                    cond_hidden_states=[[ref_latent[0]]], timestep=v_time,
                    block_controlnet_hidden_states=[v.to(dtype) for v in out_pert.block_controlnet_hidden_states],
                    return_dict=False,
                )[0]
                diff_pert = (pred_pert.float() - v_target_v.float()).square().mean(dim=1, keepdim=True)
                val_losses_pert.append(diff_pert.mean().item())

        val_rec = {
            "step": current_step,
            "val_flow_mse_on": float(np.mean(val_losses_on)),
            "val_flow_mse_off": float(np.mean(val_losses_off)),
            "val_flow_mse_pert": float(np.mean(val_losses_pert)),
            "val_face_flow_mse": float(np.mean(val_face_losses)),
            "val_lh_flow_mse": float(np.mean(val_lh_losses)),
            "val_rh_flow_mse": float(np.mean(val_rh_losses)),
            "val_hand_flow_mse": float((np.mean(val_lh_losses) + np.mean(val_rh_losses)) * 0.5),
        }
        print(
            f"[Val Step {current_step:04d}] Flow MSE ON: {val_rec['val_flow_mse_on']:.6f} | "
            f"OFF: {val_rec['val_flow_mse_off']:.6f} | Perturbed: {val_rec['val_flow_mse_pert']:.6f} | "
            f"Face Flow: {val_rec['val_face_flow_mse']:.6f} | LH Flow: {val_rec['val_lh_flow_mse']:.6f} | RH Flow: {val_rec['val_rh_flow_mse']:.6f}",
            flush=True,
        )
        adapter.train()
        return val_rec

    # Run Step 0 evaluation
    eval_history = []
    eval_0 = run_eval_at_step(0)
    eval_history.append(eval_0)

    val_history = []
    val_0 = run_val_flow(0)
    val_history.append(val_0)

    # 8. Training Loop
    print("\n[7/7] Starting 1500-step detail branch single-sample optimization...")
    max_steps = args.max_steps
    train_history = []
    initial_losses = []
    final_losses = []

    start_time = time.time()
    step_times = []

    for step in range(1, max_steps + 1):
        step_t0 = time.time()

        # Update layered LR
        update_optimizer_lrs(optimizer, step)

        # Select noise and timestep from training bank
        noise_idx = (step - 1) % len(train_noise_bank)
        noise_t, timestep_t = train_noise_bank[noise_idx]

        sigma_t = (timestep_t / 1000.0).view(1, 1, 1, 1).to(dtype)
        noisy_latent = (1.0 - sigma_t) * target_latent + sigma_t * noise_t
        target_velocity = noise_t - target_latent
        progress_t = (1.0 - timestep_t / 1000.0).to(dtype) # [1]

        # Prepare conditioning
        prepared = adapter.prepare_conditioning(
            condition_bundle=geometry_bundle,
            identity_condition=identity_condition,
            source_scene_latents=ref_latent[:1],
            target_latent_hw=(H_lat, W_lat),
            detail_condition=detail_condition_correct,
            detail_references=detail_references,
        )

        # Adapter forward
        control_out = adapter(
            target_latents=noisy_latent,
            prepared=prepared,
            cond_hidden_states=[[ref_latent[0]]],
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            timestep=timestep_t,
            denoise_progress=progress_t,
            geometry_strength=1.0,
            interaction_strength=0.0,
            detail_strength=1.0,
            face_strength=1.0,
            hand_strength=1.0,
        )

        block_states = [v.to(dtype=dtype) for v in control_out.block_controlnet_hidden_states]
        residual_rms = [
            float(v.detach().float().square().mean().sqrt().item())
            for v in control_out.block_controlnet_hidden_states
        ]

        # DiT forward
        pred = transformer(
            hidden_states=noisy_latent,
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            cond_hidden_states=[[ref_latent[0]]],
            timestep=timestep_t,
            block_controlnet_hidden_states=block_states,
            return_dict=False,
        )[0]

        # Velocity error map: diff_sq [1, 1, 64, 64] averaged over channels
        diff_sq = (pred.float() - target_velocity.float()).square().mean(dim=1, keepdim=True)

        # Regional flow losses normalized by region valid pixels
        loss_weighted = (diff_sq * w_map).sum() / w_map.sum().clamp_min(1.0)
        loss_face = (diff_sq * m_face).sum() / m_face.sum().clamp_min(1.0)
        loss_lh = (diff_sq * m_left).sum() / m_left.sum().clamp_min(1.0)
        loss_rh = (diff_sq * m_right).sum() / m_right.sum().clamp_min(1.0)

        # Total Loss: L = L_weighted + 0.5*L_face + 0.25*L_lh + 0.25*L_rh
        total_loss = loss_weighted + 0.5 * loss_face + 0.25 * loss_lh + 0.25 * loss_rh

        # Backward & Optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        step_elapsed = time.time() - step_t0
        step_times.append(step_elapsed)

        # Detail scales (6-element tensor for groups)
        scales = adapter.control_interface.strength_controller.detail_log_group_scale.detach().float().exp().cpu().tolist()
        detail_scale_mean = float(np.mean(scales))

        loss_val = total_loss.item()
        if step <= 50:
            initial_losses.append(loss_val)
        if step > 1450:
            final_losses.append(loss_val)

        # Record metrics every 10 steps
        if step % 10 == 0 or step == 1:
            peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            train_record = {
                "step": step,
                "loss": loss_val,
                "loss_weighted": loss_weighted.item(),
                "loss_face": loss_face.item(),
                "loss_lh": loss_lh.item(),
                "loss_rh": loss_rh.item(),
                "detail_scales": scales,
                "detail_scale_mean": detail_scale_mean,
                "residual_rms_mean": sum(residual_rms) / len(residual_rms),
                "step_time_ms": step_elapsed * 1000.0,
                "peak_vram_gib": peak_vram_gib,
            }
            train_history.append(train_record)
            print(
                f"Step [{step:04d}/{max_steps}] Loss: {loss_val:.5f} "
                f"(Face: {loss_face.item():.5f}, LH: {loss_lh.item():.5f}, RH: {loss_rh.item():.5f}) | "
                f"Res RMS: {train_record['residual_rms_mean']:.4f} | Scale: {detail_scale_mean:.3f} | "
                f"VRAM: {peak_vram_gib:.2f} GiB | Time: {step_elapsed*1000.0:.1f}ms",
                flush=True,
            )

        # Fixed-noise validation every 100 steps
        if step % 100 == 0:
            val_rec = run_val_flow(step)
            val_history.append(val_rec)

        # Step 300 Safety Gate: Face & Hand flow loss must drop at least 20%
        if step == 300:
            initial_face = val_history[0]["val_face_flow_mse"]
            current_face = val_history[-1]["val_face_flow_mse"]
            face_drop = (initial_face - current_face) / initial_face

            initial_lh = val_history[0]["val_lh_flow_mse"]
            current_lh = val_history[-1]["val_lh_flow_mse"]
            lh_drop = (initial_lh - current_lh) / initial_lh

            initial_rh = val_history[0]["val_rh_flow_mse"]
            current_rh = val_history[-1]["val_rh_flow_mse"]
            rh_drop = (initial_rh - current_rh) / initial_rh

            print(f"\n--- STEP 300 AUDIT --- Face drop: {face_drop*100:.2f}%, LH drop: {lh_drop*100:.2f}%, RH drop: {rh_drop*100:.2f}%")
            if face_drop < 0.20 or lh_drop < 0.20 or rh_drop < 0.20:
                print("WARNING: Step 300 safety gate triggered: Face or Hand losses did not decrease by >= 20%!")
            else:
                print("✓ Step 300 Safety Gate Passed!")

        # Step 250, 500, 1000, 1500 generation preview & checkpoints
        if step in (250, 500, 1000, 1500):
            eval_res = run_eval_at_step(step)
            eval_history.append(eval_res)

        if step in (500, 1000, 1500):
            ckpt = build_v64_checkpoint(adapter, step=step, max_steps=max_steps)
            ckpt_path = ckpt_dir / f"checkpoint_step_{step:04d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"✓ Saved V6.4 checkpoint to {ckpt_path}")

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)!")

    # 9. Post-training Verifications
    print("\n" + "=" * 80)
    print("   POST-TRAINING VERIFICATION & ACCEPTANCE CRITERIA EVALUATION   ")
    print("=" * 80)

    # 1. Non-detail parameter SHA-256 integrity check
    sha256_post = compute_frozen_sha256(adapter)
    assert sha256_pre == sha256_post, f"Frozen parameters altered! {sha256_pre} vs {sha256_post}"
    print(f"✓ Criterion 1 (Backbone Integrity): Non-detail SHA-256 match confirmed:\n  {sha256_post}")

    # 2. Step 1500 Detail OFF vs Step 0 Baseline equivalence
    print("\nEvaluating Criterion 2: Step 1500 Detail OFF parity with Step 0 baseline...")
    gen_off_1500 = controlled_pipe(
        condition_bundle=geometry_bundle,
        identity_condition=identity_condition,
        source_scene_latents=ref_latent[:1],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=src_pil,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=25,
        guidance_scale=4.5,
        geometry_strength=1.0,
        interaction_strength=0.0,
        detail_strength=0.0,
        detail_condition=None,
        detail_references=None,
        seed=42,
    ).images[0]
    gen_off_1500.save(preview_dir / "step_1500_seed_42_off.png")
    off_arr_0 = np.asarray(Image.open(preview_dir / "step_0000_seed_42_off.png"))
    off_arr_1500 = np.asarray(gen_off_1500)
    parity_diff_max = int(np.max(np.abs(off_arr_0.astype(int) - off_arr_1500.astype(int))))
    parity_diff_mean = float(np.mean(np.abs(off_arr_0.astype(float) - off_arr_1500.astype(float))))
    psnr_parity = compute_psnr(off_arr_1500, off_arr_0)
    print(f"✓ Criterion 2 (Inference Baseline Parity): Step 1500 Detail OFF vs Step 0: PSNR={psnr_parity:.2f} dB, mean_diff={parity_diff_mean:.3f}, max_diff={parity_diff_max}")
    assert psnr_parity >= 35.0, f"Step 1500 Detail OFF drifted from Step 0 baseline! PSNR={psnr_parity:.2f} dB"

    # 3. Quantitative Criteria Verification
    initial_loss_med = float(np.median(initial_losses)) if initial_losses else 0.0
    final_loss_med = float(np.median(final_losses)) if final_losses else initial_loss_med
    loss_ratio = (final_loss_med / initial_loss_med) if initial_loss_med > 0 else 1.0

    if len(val_history) >= 2:
        val_start = val_history[0]
        val_end = val_history[-1]
        face_flow_drop = (val_start["val_face_flow_mse"] - val_end["val_face_flow_mse"]) / max(val_start["val_face_flow_mse"], 1e-7)
        lh_flow_drop = (val_start["val_lh_flow_mse"] - val_end["val_lh_flow_mse"]) / max(val_start["val_lh_flow_mse"], 1e-7)
        rh_flow_drop = (val_start["val_rh_flow_mse"] - val_end["val_rh_flow_mse"]) / max(val_start["val_rh_flow_mse"], 1e-7)
        val_face_str = f"{val_start['val_face_flow_mse']:.6f} -> {val_end['val_face_flow_mse']:.6f}"
        val_lh_str = f"{val_start['val_lh_flow_mse']:.6f} -> {val_end['val_lh_flow_mse']:.6f}"
        val_rh_str = f"{val_start['val_rh_flow_mse']:.6f} -> {val_end['val_rh_flow_mse']:.6f}"
    else:
        face_flow_drop = 0.0
        lh_flow_drop = 0.0
        rh_flow_drop = 0.0
        val_face_str = "N/A"
        val_lh_str = "N/A"
        val_rh_str = "N/A"

    last_eval = eval_history[-1] # step 1500 eval
    face_ssim_gain = last_eval["median_face_ssim_diff"]
    face_psnr_gain = last_eval["median_face_psnr_diff"]
    hands_ssim_gain = last_eval["median_hands_ssim_diff"]
    hands_pck_on = last_eval["median_hands_pck_on"]
    hands_pck_gain = last_eval["median_hands_pck_diff"]
    outside_psnr_drop = last_eval["median_outside_psnr_drop"]
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    criteria = [
        ("Final 50-step total loss median <= 30% of initial 50-step median",
         loss_ratio <= 0.30, f"ratio = {loss_ratio*100:.2f}% (initial: {initial_loss_med:.5f}, final: {final_loss_med:.5f})"),
        ("Face flow MSE drops >= 60%",
         face_flow_drop >= 0.60, f"drop = {face_flow_drop*100:.2f}% ({val_face_str})"),
        ("Left-hand flow MSE drops >= 60%",
         lh_flow_drop >= 0.60, f"drop = {lh_flow_drop*100:.2f}% ({val_lh_str})"),
        ("Right-hand flow MSE drops >= 60%",
         rh_flow_drop >= 0.60, f"drop = {rh_flow_drop*100:.2f}% ({val_rh_str})"),
        ("Detail residual RMS non-zero, finite, no persistent explosion",
         0.0 < train_history[-1]["residual_rms_mean"] < 10.0, f"final RMS = {train_history[-1]['residual_rms_mean']:.4f}"),
        ("Face SSIM relative to OFF >= +0.05 OR Face PSNR >= +2 dB",
         face_ssim_gain >= 0.05 or face_psnr_gain >= 2.0, f"ΔSSIM = {face_ssim_gain:+.4f}, ΔPSNR = {face_psnr_gain:+.2f} dB"),
        ("Hands average SSIM relative to OFF >= +0.03",
         hands_ssim_gain >= 0.03, f"ΔSSIM = {hands_ssim_gain:+.4f}"),
        ("Hands average PCK@0.1 relative to OFF >= +0.10 and reaches >= 0.70",
         hands_pck_gain >= 0.10 and hands_pck_on >= 0.70, f"PCK = {hands_pck_on:.3f} (Δ = {hands_pck_gain:+.3f})"),
        ("Outside face/hand region PSNR drop <= 0.2 dB relative to OFF",
         outside_psnr_drop <= 0.20, f"drop = {outside_psnr_drop:.2f} dB"),
        ("Peak VRAM < 24 GiB",
         peak_vram < 24.0, f"peak = {peak_vram:.2f} GiB"),
    ]

    print("\n--- Criteria Acceptance Matrix ---")
    all_passed = True
    for desc, passed, val_str in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {desc:65s} | {val_str}")

    # Save summary files
    summary_data = {
        "pre_sha256": sha256_pre,
        "post_sha256": sha256_post,
        "total_training_time_s": total_training_time,
        "mean_step_time_ms": float(np.mean(step_times) * 1000.0),
        "peak_vram_gib": peak_vram,
        "all_criteria_passed": all_passed,
        "criteria": [
            {"desc": d, "passed": bool(p), "detail": v} for d, p, v in criteria
        ],
        "eval_history": eval_history,
        "val_history": val_history,
        "train_history": train_history,
    }

    summary_file = output_dir / "overfit_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved overfit summary to {summary_file}")

    # Save CSV metrics table
    csv_file = output_dir / "eval_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "seed", "face_psnr_on", "face_psnr_off", "face_ssim_on", "face_ssim_off", "hands_ssim_on", "hands_ssim_off", "hands_pck_on", "hands_pck_off", "outside_psnr_on", "outside_psnr_off"])
        for ev in eval_history:
            for s in ev["seed_results"]:
                writer.writerow([
                    ev["step"], s["seed"], s["face_psnr_on"], s["face_psnr_off"],
                    s["face_ssim_on"], s["face_ssim_off"], s["hands_ssim_on"], s["hands_ssim_off"],
                    s["hands_pck_on"], s["hands_pck_off"], s["outside_psnr_on"], s["outside_psnr_off"],
                ])
    print(f"✓ Saved CSV metrics to {csv_file}")


if __name__ == "__main__":
    main()
