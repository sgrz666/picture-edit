#!/usr/bin/env python3
"""Native V6.3 Training Script for iPER Pose Editing (Scheme A: Single-Person Geometry).

Implements Scheme A:
- Trains: Geometry branch, Appearance identity binding, Dynamic control core, Zero heads.
- Freezes: Interaction branch, Contact relation reasoning, Person-B modules, DeepGen backbone.
- Features:
  1. Mask-weighted flow matching velocity loss (foreground human weighting).
  2. Layered learning rates across 4 parameter groups:
     - Zero Heads & group gates: 1e-4
     - Condition Injector & Reasoner: 5e-5
     - Reasoner Bridge & Appearance Token Binder: 5e-5
     - Shared Control Core Block: 1e-5
  3. Bounded checkpoint rotation (best, last, and at most 2 rolling checkpoints).
  4. Condition caching with disk management to save VRAM and avoid repeated VLM computation.
  5. Built-in smoke test mode (--smoke_test) and small-sample overfit mode (--overfit_samples).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Project root configuration
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from diffusers import DiffusionPipeline
from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.interface import PreparedControlConditioning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Unified SMPL-X Adapter V6 on iPER (Scheme A)")
    
    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
        help="Path to pretrained DeepGen diffusers model",
    )
    parser.add_argument(
        "--train_pairs",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits/dev_train_pairs.jsonl",
        help="Path to dev-train pairs jsonl (69 appearances)",
    )
    parser.add_argument(
        "--val_pairs",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits/dev_val_pairs.jsonl",
        help="Path to dev-val pairs jsonl (10 appearances)",
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
        help="Root directory of iPER preprocessed assets (normals, parts, conditions)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for precomputed VLM/connector condition cache. Defaults to output_dir/condition_cache",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/shangguanrz/project/pic-edit/experiments/iper_v6_native_finetune_v1",
        help="Output directory for checkpoints and logs",
    )

    # Training Hyperparameters
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--lr_zero_heads", type=float, default=1e-4, help="Learning rate for Zero Heads & gates")
    parser.add_argument("--lr_condition", type=float, default=5e-5, help="Learning rate for Condition Injector & Reasoner")
    parser.add_argument("--lr_bridge", type=float, default=5e-5, help="Learning rate for Bridge & Appearance Binder")
    parser.add_argument("--lr_control_block", type=float, default=1e-5, help="Learning rate for Shared Control Core block")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient clipping norm")
    parser.add_argument("--human_loss_weight", type=float, default=3.0, help="Loss weight on human foreground mask")
    parser.add_argument("--warmup_heads_steps", type=int, default=0, help="Steps to train only zero-heads before unfreezing other trainable groups")
    parser.add_argument("--max_train_steps", type=int, default=25000, help="Total training steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--val_every", type=int, default=500, help="Validation interval")
    parser.add_argument("--val_samples", type=int, default=20, help="Number of validation samples per check")
    parser.add_argument("--max_rolling_checkpoints", type=int, default=2, help="Maximum number of rolling step checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default="Change the person pose to match the target pose. Preserve identity, clothing, and background.")

    # V6 Architecture Options
    parser.add_argument("--condition_use_depth", action="store_true", default=False, help="Whether to encode scene depth in spatial encoder")
    parser.add_argument("--condition_backend", type=str, default="native", choices=["native", "champ"], help="Condition backend")
    
    # Modes & Visualization
    parser.add_argument("--smoke_test", action="store_true", help="Run 3-step smoke test and verify gradient flow and checkpointing")
    parser.add_argument("--overfit_samples", type=int, default=0, help="If > 0, overfit on a fixed small subset of N samples")
    parser.add_argument("--eval_generate_samples", type=int, default=0, help="Number of visual preview samples to render during validation (0 to disable)")
    parser.add_argument("--sample_inference_steps", type=int, default=25, help="Number of diffusion inference steps for sample generation")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from (or 'last')")
    parser.add_argument("--max_micro_steps", type=int, default=None, help="If set, stop after this many micro-steps (batch samples)")
    parser.add_argument("--save_micro_steps", type=int, default=None, help="If set, save checkpoint every N micro-steps")
    parser.add_argument("--val_micro_steps", type=int, default=None, help="If set, run validation every N micro-steps")
    parser.add_argument("--no_shuffle", action="store_true", default=False, help="If set, do not shuffle training dataloader")

    return parser.parse_args()


# ==============================================================================
# 1. Parameter Freezing & Layered Optimizer Grouping (Scheme A)
# ==============================================================================

def configure_scheme_a_parameters(adapter: UnifiedSMPLXAdapterV6) -> None:
    """Configures parameter freezing strictly following Scheme A:
    
    - Freezes Interaction, Contact relation reasoning, and Person-B modules.
    - Enables gradients for Geometry branch, Appearance identity binding, and Dynamic Control Core.
    """
    # 1. Condition Injector: freeze dual-person / contact modules
    adapter.condition_injector.relative_encoder.requires_grad_(False)
    adapter.condition_injector.contact_raster_encoder.requires_grad_(False)
    adapter.condition_injector.contact_relation_encoder.requires_grad_(False)

    # 2. Reasoner: freeze cross-person, contact, and dual feature fusion
    adapter.reasoner.cross_person_reasoner.requires_grad_(False)
    adapter.reasoner.contact_reasoner.requires_grad_(False)
    adapter.reasoner.dual_fusion.requires_grad_(False)

    # 3. Reasoner Bridge: freeze interaction projections
    adapter.condition_bridge.interaction_projection.requires_grad_(False)
    adapter.condition_bridge.interaction_high_downsample.requires_grad_(False)

    # 4. Control Core: freeze interaction branch entirely
    adapter.control_core.interaction_branch.requires_grad_(False)

    # 5. Strength Controller: freeze interaction log group scales
    adapter.control_interface.strength_controller.interaction_log_group_scale.requires_grad_(False)

    # V6.3 Scheme A remains a legacy geometry-only recipe. V6.4 detail tuning
    # uses freeze_for_detail_training instead and is intentionally not mixed in.
    if hasattr(adapter, "detail_preparer"):
        adapter.detail_preparer.requires_grad_(False)
        adapter.control_core.detail_branch.requires_grad_(False)
        adapter.control_interface.strength_controller.detail_log_group_scale.requires_grad_(False)


def build_scheme_a_optimizer(
    adapter: UnifiedSMPLXAdapterV6,
    lr_zero_heads: float = 1e-4,
    lr_condition: float = 5e-5,
    lr_bridge: float = 5e-5,
    lr_control_block: float = 1e-5,
    weight_decay: float = 1e-2,
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """Constructs AdamW optimizer with layered learning rates for Scheme A.
    
    Groups:
    1. Zero Heads & group gates: lr_zero_heads (1e-4)
    2. Condition Injector & Reasoner: lr_condition (5e-5)
    3. Bridge & Appearance Token Binder: lr_bridge (5e-5)
    4. Shared Control Core block: lr_control_block (1e-5)
    
    Ensures every trainable parameter is assigned to exactly one group.
    """
    # Group 1: Zero Heads & group gates
    params_group_1: List[nn.Parameter] = []
    params_group_1.extend(p for p in adapter.control_core.geometry_zero_heads.parameters() if p.requires_grad)
    if adapter.control_interface.strength_controller.geometry_log_group_scale.requires_grad:
        params_group_1.append(adapter.control_interface.strength_controller.geometry_log_group_scale)

    # Group 2: Condition Injector & Reasoner (Geometry)
    params_group_2: List[nn.Parameter] = []
    params_group_2.extend(p for p in adapter.condition_injector.spatial_encoder.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.condition_injector.global_encoder.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.condition_injector.task_encoder.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.reasoner.person_reasoner.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.reasoner.role_embedding.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.reasoner.single_geometry_block.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.reasoner.single_high_projection.parameters() if p.requires_grad)
    params_group_2.extend(p for p in adapter.reasoner.single_high_block.parameters() if p.requires_grad)

    # Group 3: Bridge & Appearance Token Binder
    params_group_3: List[nn.Parameter] = []
    params_group_3.extend(p for p in adapter.condition_bridge.parameters() if p.requires_grad)
    params_group_3.extend(p for p in adapter.geometry_token_projection.parameters() if p.requires_grad)
    params_group_3.extend(p for p in adapter.appearance_token_encoder.parameters() if p.requires_grad)
    params_group_3.extend(p for p in adapter.person_token_binder.parameters() if p.requires_grad)

    # Group 4: Shared Control Core Block & Geometry Condition Embed
    params_group_4: List[nn.Parameter] = []
    params_group_4.extend(p for p in adapter.control_core.shared_block.parameters() if p.requires_grad)
    params_group_4.extend(p for p in adapter.control_core.pos_embed.parameters() if p.requires_grad)
    params_group_4.extend(p for p in adapter.control_core.time_text_embed.parameters() if p.requires_grad)
    params_group_4.extend(p for p in adapter.control_core.context_embedder.parameters() if p.requires_grad)
    params_group_4.extend(p for p in adapter.control_core.geometry_branch.condition_embed.parameters() if p.requires_grad)
    if adapter.control_core.geometry_branch.stage_embeddings.requires_grad:
        params_group_4.append(adapter.control_core.geometry_branch.stage_embeddings)
    params_group_4.extend(p for p in adapter.control_core.geometry_branch.stage_adapters.parameters() if p.requires_grad)

    # Verify partition integrity
    all_grouped_ids = set()
    for grp_idx, grp_params in enumerate([params_group_1, params_group_2, params_group_3, params_group_4], 1):
        for p in grp_params:
            pid = id(p)
            assert pid not in all_grouped_ids, f"Parameter duplicated in optimizer group {grp_idx}"
            all_grouped_ids.add(pid)

    all_trainable_ids = {id(p) for p in adapter.parameters() if p.requires_grad}
    missing_ids = all_trainable_ids - all_grouped_ids
    assert len(missing_ids) == 0, f"Found {len(missing_ids)} trainable parameters not assigned to any group!"

    param_groups = [
        {"params": params_group_1, "lr": lr_zero_heads, "name": "zero_heads_and_gates"},
        {"params": params_group_2, "lr": lr_condition, "name": "condition_injector_reasoner"},
        {"params": params_group_3, "lr": lr_bridge, "name": "bridge_and_appearance_binder"},
        {"params": params_group_4, "lr": lr_control_block, "name": "shared_control_core_block"},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    meta = {
        "group_1_count": sum(p.numel() for p in params_group_1),
        "group_2_count": sum(p.numel() for p in params_group_2),
        "group_3_count": sum(p.numel() for p in params_group_3),
        "group_4_count": sum(p.numel() for p in params_group_4),
        "total_trainable": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in adapter.parameters()),
    }
    return optimizer, meta


# ==============================================================================
# 2. Checkpoint Management with Strict Rolling Retention
# ==============================================================================

class CheckpointManager:
    """Manages model checkpoints maintaining only best, last, and rolling checkpoints.
    
    Prevents storage exhaustion by automatically pruning older rolling checkpoints.
    """

    def __init__(self, output_dir: str | Path, max_rolling: int = 2) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rolling = max_rolling
        self.rolling_checkpoints: List[Path] = []
        self.best_loss = float("inf")
        self._discover_existing()

    def _discover_existing(self) -> None:
        """Discovers existing rolling checkpoints and sorts by step."""
        existing = []
        for p in self.output_dir.glob("checkpoint_step_*.pt"):
            try:
                step = int(p.stem.split("_")[-1])
                existing.append((step, p))
            except ValueError:
                continue
        existing.sort(key=lambda x: x[0])
        self.rolling_checkpoints = [p for _, p in existing]
        # Prune if already exceeding max_rolling
        while len(self.rolling_checkpoints) > self.max_rolling:
            old = self.rolling_checkpoints.pop(0)
            if old.exists():
                try:
                    old.unlink()
                except OSError:
                    pass

        # Recover historical best validation loss if checkpoint_best.pt exists
        best_path = self.output_dir / "checkpoint_best.pt"
        if best_path.is_file():
            try:
                best_data = torch.load(best_path, map_location="cpu", weights_only=False)
                if best_data.get("val_loss") is not None:
                    self.best_loss = float(best_data["val_loss"])
            except Exception:
                pass

    def save(
        self,
        step: int,
        adapter: nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: Optional[float] = None,
        is_best: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves rolling checkpoint, updates last checkpoint, and optionally updates best."""
        payload = {
            "step": step,
            "adapter_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "metadata": extra_meta or {},
        }
        
        # 1. Save rolling checkpoint
        rolling_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(payload, rolling_path)
        self.rolling_checkpoints.append(rolling_path)

        # Prune older rolling checkpoints
        while len(self.rolling_checkpoints) > self.max_rolling:
            oldest = self.rolling_checkpoints.pop(0)
            if oldest.exists() and oldest != rolling_path:
                try:
                    oldest.unlink()
                except OSError:
                    pass

        # 2. Always update checkpoint_last.pt
        last_path = self.output_dir / "checkpoint_last.pt"
        torch.save(payload, last_path)

        # 3. If best, save checkpoint_best.pt
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(payload, best_path)
            if val_loss is not None:
                self.best_loss = float(val_loss)

        return rolling_path

    def load(
        self,
        checkpoint_path: str | Path,
        adapter: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Loads weights and optimizer state from checkpoint."""
        p = Path(checkpoint_path)
        if not p.is_file():
            if str(checkpoint_path) == "last":
                p = self.output_dir / "checkpoint_last.pt"
            elif str(checkpoint_path) == "best":
                p = self.output_dir / "checkpoint_best.pt"
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = torch.load(p, map_location="cpu", weights_only=False)
        adapter.load_state_dict(data["adapter_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in data:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        return data


# ==============================================================================
# 3. Condition Caching & Source Image Preparation
# ==============================================================================

def get_cached_conditions(
    pipe: DiffusionPipeline,
    batch: Dict[str, Any],
    prompt: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieves or precomputes VLM semantic features and reference latents.
    
    Caches per source image stem to avoid repetitive VLM execution.
    Supports arbitrary batch sizes (B >= 1) with per-sample caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    B = len(batch["appearance"])
    ref_list = []
    seq_list = []
    pool_list = []

    for b in range(B):
        appearance = batch["appearance"][b]
        src_stem = batch["source_stem"][b]
        cache_path = os.path.join(cache_dir, f"{appearance}_{src_stem}.pt")

        loaded = False
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, map_location=device, weights_only=False)
                ref_list.append(cached_data["reference_latent"].to(device, dtype=dtype))
                seq_list.append(cached_data["sequence"].to(device, dtype=dtype))
                pool_list.append(cached_data["pooled"].to(device, dtype=dtype))
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            with torch.no_grad():
                source_pixels = batch["src_image"][b : b + 1].to(device, dtype=dtype)
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

                # Save to disk in bfloat16 to conserve space (~1 MB per file)
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


# ==============================================================================
# 4. Single-Person Forward Pass & Mask-Weighted Loss
# ==============================================================================

def compute_single_person_step(
    adapter: UnifiedSMPLXAdapterV6,
    transformer: nn.Module,
    pipe: DiffusionPipeline,
    batch: Dict[str, Any],
    prompt: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    human_loss_weight: float = 3.0,
    condition_use_depth: bool = False,
    disable_adapter: bool = False,
    perturb_conditions: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Performs full forward pass and computes mask-weighted flow matching loss.
    
    Args:
        adapter: Unified SMPL-X Adapter V6 model
        transformer: DeepGen SD3Transformer2DModel backbone (frozen)
        pipe: Base DeepGen pipeline for VAE / token encoding
        batch: Dataset sample batch
        prompt: Task prompt
        cache_dir: Condition cache directory
        device: Torch device
        dtype: Computation dtype (bfloat16)
        human_loss_weight: Loss weight on human pixels (default 3.0)
        condition_use_depth: Whether depth stem is used in adapter
        disable_adapter: If True, bypasses adapter (for validation baseline)
        perturb_conditions: If True, perturbs condition maps (for validation sensitivity check)
        
    Returns:
        loss: Scaled scalar loss tensor
        metrics: Dictionary of step metrics
    """
    reference_latent, sequence, pooled = get_cached_conditions(
        pipe, batch, prompt, cache_dir, device, dtype
    )
    B = reference_latent.shape[0]

    # Target pixels and latent
    target_pixels = batch["tgt_image"].to(device, dtype=dtype)
    with torch.no_grad():
        target_latent = pipe.pixels_to_latents(target_pixels)

    # Flow matching schedule
    noise = torch.randn_like(target_latent)
    timesteps = torch.rand(B, device=device) * 1000.0
    sigma = (timesteps / 1000.0).view(B, 1, 1, 1).to(dtype)
    noisy_latent = (1.0 - sigma) * target_latent + sigma * noise
    target_velocity = noise - target_latent

    # Progress in [0, 1] aligned with inference: 0.0 at pure noise (t=1000), 1.0 at clean latent (t=0)
    denoise_progress = float((1.0 - timesteps.mean() / 1000.0).clamp(0.0, 1.0).item())

    # Build condition maps
    normal_a = batch["normal"].to(device, dtype=dtype)
    part_onehot_a = batch["part_onehot"].to(device, dtype=dtype)
    pose_heatmap_a = batch["pose_heatmap"].to(device, dtype=dtype)
    smplx_global_a = batch["smplx_global"].to(device, dtype=dtype)
    human_mask_a = batch["human_mask"].to(device, dtype=dtype)
    task_id = batch["task_id"].to(device, dtype=torch.long)
    depth_a = batch["depth"].to(device, dtype=dtype) if condition_use_depth else None

    if perturb_conditions:
        # Scramble conditions by rolling channels and shifting global vector
        normal_a = torch.roll(normal_a, shifts=1, dims=1)
        part_onehot_a = torch.roll(part_onehot_a, shifts=3, dims=1)
        pose_heatmap_a = torch.roll(pose_heatmap_a, shifts=5, dims=1)
        smplx_global_a = torch.roll(smplx_global_a, shifts=3, dims=-1)

    block_controlnet_states = None
    residual_rms = [0.0] * 6

    if not disable_adapter:
        # 1. Condition Injector
        bundle = adapter.condition_injector(
            normal_a=normal_a,
            pose_heatmap_a=pose_heatmap_a,
            part_onehot_a=part_onehot_a,
            smplx_global_a=smplx_global_a,
            human_mask_a=human_mask_a,
            task_id=task_id,
            depth_a=depth_a,
        )

        # 2. Identity Condition (Person A = reference_latent, Person B = unused zeros)
        source_person_latents = torch.zeros(
            B, 2, 16, reference_latent.shape[-2], reference_latent.shape[-1],
            device=device, dtype=dtype,
        )
        source_person_latents[:, 0] = reference_latent
        source_indices = torch.tensor([[0, 1]], device=device).expand(B, -1)
        identity = AdapterIdentityCondition(
            source_person_latents=source_person_latents,
            source_indices=source_indices,
        )

        # 3. Prepare conditioning (Bridging, Appearance Binding)
        prepared = adapter.prepare_conditioning(
            condition_bundle=bundle,
            identity_condition=identity,
            source_scene_latents=reference_latent,
            target_latent_hw=(target_latent.shape[-2], target_latent.shape[-1]),
        )

        # 4. Adapter forward
        control_out = adapter(
            target_latents=noisy_latent,
            prepared=prepared,
            cond_hidden_states=[[reference_latent[b]] for b in range(B)],
            encoder_hidden_states=sequence,
            pooled_projections=pooled,
            timestep=timesteps,
            denoise_progress=denoise_progress,
            geometry_strength=1.0,
            interaction_strength=0.0,
        )
        block_controlnet_states = [v.to(dtype=dtype) for v in control_out.block_controlnet_hidden_states]
        residual_rms = [
            float(v.detach().float().square().mean().sqrt().item())
            for v in control_out.block_controlnet_hidden_states
        ]

    # DeepGen DiT forward
    pred = transformer(
        hidden_states=noisy_latent,
        encoder_hidden_states=sequence,
        pooled_projections=pooled,
        cond_hidden_states=[[reference_latent[b]] for b in range(B)],
        timestep=timesteps,
        block_controlnet_hidden_states=block_controlnet_states,
        return_dict=False,
    )[0]

    # Mask-weighted flow matching loss
    mask_lat = F.interpolate(
        batch["human_mask"].to(device, dtype=torch.float32),
        size=target_latent.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    
    diff_sq = (pred.float() - target_velocity.float()) ** 2
    weights = (1.0 + (human_loss_weight - 1.0) * mask_lat).expand_as(diff_sq)

    loss = (diff_sq * weights).sum() / weights.sum()

    # Breakdown metrics
    with torch.no_grad():
        raw_mse = diff_sq.mean().item()
        human_bool = (mask_lat > 0.5).expand_as(diff_sq)
        loss_human = diff_sq[human_bool].mean().item() if human_bool.any() else 0.0
        loss_bg = diff_sq[~human_bool].mean().item() if (~human_bool).any() else 0.0

    metrics = {
        "loss": loss.item(),
        "raw_mse": raw_mse,
        "loss_human": loss_human,
        "loss_bg": loss_bg,
        "residual_rms_mean": sum(residual_rms) / len(residual_rms) if residual_rms else 0.0,
        "residual_rms": residual_rms,
    }

    return loss, metrics


# ==============================================================================
# 5. Validation Protocol
# ==============================================================================

def validate(
    args: argparse.Namespace,
    pipe: DiffusionPipeline,
    adapter: UnifiedSMPLXAdapterV6,
    transformer: nn.Module,
    val_dataloader: DataLoader,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
) -> Dict[str, float]:
    """Evaluates validation loss comparing:
    1. Adapter ON (with accurate conditions)
    2. Adapter OFF (uncontrolled baseline)
    3. Adapter ON + Perturbed conditions (sensitivity check)
    """
    adapter.eval()
    losses_on, losses_off, losses_mismatch = [], [], []
    raw_mses_on, raw_mses_off, raw_mses_mismatch = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= args.val_samples:
                break

            # 1. Adapter ON
            loss_on, m_on = compute_single_person_step(
                adapter=adapter,
                transformer=transformer,
                pipe=pipe,
                batch=batch,
                prompt=args.prompt,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                human_loss_weight=args.human_loss_weight,
                condition_use_depth=args.condition_use_depth,
                disable_adapter=False,
                perturb_conditions=False,
            )
            losses_on.append(loss_on.item())
            raw_mses_on.append(m_on.get("raw_mse", loss_on.item()))

            # 2. Adapter OFF
            loss_off, m_off = compute_single_person_step(
                adapter=adapter,
                transformer=transformer,
                pipe=pipe,
                batch=batch,
                prompt=args.prompt,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                human_loss_weight=args.human_loss_weight,
                condition_use_depth=args.condition_use_depth,
                disable_adapter=True,
                perturb_conditions=False,
            )
            losses_off.append(loss_off.item())
            raw_mses_off.append(m_off.get("raw_mse", loss_off.item()))

            # 3. Adapter ON + Perturbed
            loss_mis, m_mis = compute_single_person_step(
                adapter=adapter,
                transformer=transformer,
                pipe=pipe,
                batch=batch,
                prompt=args.prompt,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                human_loss_weight=args.human_loss_weight,
                condition_use_depth=args.condition_use_depth,
                disable_adapter=False,
                perturb_conditions=True,
            )
            losses_mismatch.append(loss_mis.item())
            raw_mses_mismatch.append(m_mis.get("raw_mse", loss_mis.item()))

    adapter.train()
    mean_on = sum(losses_on) / max(len(losses_on), 1)
    mean_off = sum(losses_off) / max(len(losses_off), 1)
    mean_mis = sum(losses_mismatch) / max(len(losses_mismatch), 1)

    mean_raw_on = sum(raw_mses_on) / max(len(raw_mses_on), 1)
    mean_raw_off = sum(raw_mses_off) / max(len(raw_mses_off), 1)
    mean_raw_mis = sum(raw_mses_mismatch) / max(len(raw_mses_mismatch), 1)

    print(f"--- Running validation at step {step} ---", flush=True)
    print(
        f"Validation Loss: {mean_raw_on:.6f} (Weighted: {mean_on:.6f}) | "
        f"OFF: {mean_raw_off:.6f} | "
        f"Perturbed: {mean_raw_mis:.6f}",
        flush=True,
    )

    # Optional: Render visual preview samples using ControlledDeepGenPipeline
    if getattr(args, "eval_generate_samples", 0) > 0:
        render_validation_previews(
            args=args,
            pipe=pipe,
            adapter=adapter,
            val_dataloader=val_dataloader,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            step=step,
        )

    return {
        "val_loss_on": mean_on,
        "val_loss_off": mean_off,
        "val_loss_perturbed": mean_mis,
        "val_raw_mse_on": mean_raw_on,
        "val_raw_mse_off": mean_raw_off,
        "val_raw_mse_perturbed": mean_raw_mis,
    }


def render_validation_previews(
    args: argparse.Namespace,
    pipe: DiffusionPipeline,
    adapter: UnifiedSMPLXAdapterV6,
    val_dataloader: DataLoader,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
) -> None:
    """Renders side-by-side visual comparisons:
    [Source | Ground Truth | Adapter OFF | Adapter ON (Correct) | Adapter ON (Perturbed)]
    """
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_pil_image
    from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline

    preview_dir = Path(args.output_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)
    adapter.eval()

    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            if count >= args.eval_generate_samples:
                break
            count += 1

            normal_a = batch["normal"][:1].to(device, dtype=dtype)
            part_onehot_a = batch["part_onehot"][:1].to(device, dtype=dtype)
            pose_heatmap_a = batch["pose_heatmap"][:1].to(device, dtype=dtype)
            smplx_global_a = batch["smplx_global"][:1].to(device, dtype=dtype)
            human_mask_a = batch["human_mask"][:1].to(device, dtype=dtype)
            task_id = batch["task_id"][:1].to(device, dtype=torch.long)
            depth_a = batch["depth"][:1].to(device, dtype=dtype) if args.condition_use_depth else None

            bundle_correct = adapter.condition_injector(
                normal_a=normal_a,
                pose_heatmap_a=pose_heatmap_a,
                part_onehot_a=part_onehot_a,
                smplx_global_a=smplx_global_a,
                human_mask_a=human_mask_a,
                task_id=task_id,
                depth_a=depth_a,
            )

            bundle_perturbed = adapter.condition_injector(
                normal_a=torch.roll(normal_a, shifts=1, dims=1),
                pose_heatmap_a=torch.roll(pose_heatmap_a, shifts=5, dims=1),
                part_onehot_a=torch.roll(part_onehot_a, shifts=3, dims=1),
                smplx_global_a=torch.roll(smplx_global_a, shifts=3, dims=-1),
                human_mask_a=human_mask_a,
                task_id=task_id,
                depth_a=depth_a,
            )

            ref_latent, _, _ = get_cached_conditions(pipe, batch, args.prompt, cache_dir, device, dtype)
            src_person_latents = torch.zeros(
                1, 2, 16, ref_latent.shape[-2], ref_latent.shape[-1], device=device, dtype=dtype
            )
            src_person_latents[:, 0] = ref_latent[:1]
            identity = AdapterIdentityCondition(
                source_person_latents=src_person_latents,
                source_indices=torch.tensor([[0, 1]], device=device),
            )

            src_raw = (batch["src_image"][0].float() * 0.5 + 0.5).clamp(0, 1)
            tgt_raw = (batch["tgt_image"][0].float() * 0.5 + 0.5).clamp(0, 1)
            src_pil = to_pil_image(src_raw.cpu())
            tgt_pil = to_pil_image(tgt_raw.cpu())

            # 1. Adapter OFF (strength = 0.0)
            img_off = controlled(
                condition_bundle=bundle_correct,
                identity_condition=identity,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.sample_inference_steps,
                geometry_strength=0.0,
                interaction_strength=0.0,
            ).images[0]

            # 2. Adapter ON (Correct condition)
            img_on = controlled(
                condition_bundle=bundle_correct,
                identity_condition=identity,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.sample_inference_steps,
                geometry_strength=1.0,
                interaction_strength=0.0,
            ).images[0]

            # 3. Adapter ON (Perturbed condition)
            img_pert = controlled(
                condition_bundle=bundle_perturbed,
                identity_condition=identity,
                source_scene_latents=ref_latent[:1],
                prompt=args.prompt,
                image=src_pil,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.sample_inference_steps,
                geometry_strength=1.0,
                interaction_strength=0.0,
            ).images[0]

            # Build comparison strip
            panels = [src_pil, tgt_pil, img_off, img_on, img_pert]
            labels = ["Source", "Ground Truth", "Adapter OFF", "Adapter ON (Correct)", "Adapter ON (Perturbed)"]
            thumb_size = 256
            canvas = Image.new("RGB", (thumb_size * len(panels), thumb_size + 30), color=(25, 25, 25))
            draw = ImageDraw.Draw(canvas)
            for p_idx, (p_img, p_lbl) in enumerate(zip(panels, labels)):
                resized_p = p_img.resize((thumb_size, thumb_size))
                canvas.paste(resized_p, (p_idx * thumb_size, 30))
                draw.text((p_idx * thumb_size + 6, 6), p_lbl, fill=(240, 240, 240))

            out_f = preview_dir / f"step_{step:06d}_sample_{count}.png"
            canvas.save(out_f)
            print(f"✓ Saved visual preview to: {out_f}", flush=True)

    adapter.train()


# ==============================================================================
# 6. Main Training Loop & Smoke Test
# ==============================================================================

def main() -> None:
    args = parse_args()
    print("==================================================================", flush=True)
    print("   Unified SMPL-X Adapter V6.3 Training (Scheme A: Single Person)  ", flush=True)
    print("==================================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition cache directory: check args.cache_dir, fallback to existing or output_dir
    cache_dir = args.cache_dir
    if cache_dir is None:
        default_shared_cache = "/home/shangguanrz/project/pic-edit/experiments/iper_adapter_finetune_v1/condition_cache"
        if os.path.isdir(default_shared_cache):
            cache_dir = default_shared_cache
            print(f"✓ Reusing existing condition cache: {cache_dir}", flush=True)
        else:
            cache_dir = str(output_dir / "condition_cache")
            print(f"✓ Using local condition cache: {cache_dir}", flush=True)
    else:
        print(f"✓ Explicit condition cache: {cache_dir}", flush=True)

    # 1. Pipeline & Adapter Setup
    print(f"Loading pretrained DeepGen from {args.model_path}...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe.to(device)
    pipe.vae.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe._load_extras(attn_implementation="sdpa")

    # Freeze pipeline components
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    transformer = pipe.transformer
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled on DeepGen transformer", flush=True)

    # Build V6.3 Adapter from DeepGen
    print("Instantiating UnifiedSMPLXAdapterV6 from DeepGen...", flush=True)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(
        pipe,
        condition_use_depth=args.condition_use_depth,
        normal_backend=args.condition_backend,
        depth_backend=args.condition_backend,
    ).to(device=device, dtype=dtype)
    adapter.train()

    # Apply Scheme A parameter freezing
    configure_scheme_a_parameters(adapter)

    # Build layered optimizer
    optimizer, param_meta = build_scheme_a_optimizer(
        adapter,
        lr_zero_heads=args.lr_zero_heads,
        lr_condition=args.lr_condition,
        lr_bridge=args.lr_bridge,
        lr_control_block=args.lr_control_block,
        weight_decay=args.weight_decay,
    )

    print(f"✓ Adapter Total Parameters:     {param_meta['total_params'] / 1e6:.2f} M", flush=True)
    print(f"✓ Scheme A Trainable Parameters: {param_meta['total_trainable'] / 1e6:.2f} M", flush=True)
    print(f"  - Group 1 (Zero Heads & gates):  {param_meta['group_1_count'] / 1e6:.4f} M (lr={args.lr_zero_heads})", flush=True)
    print(f"  - Group 2 (Injector & Reasoner): {param_meta['group_2_count'] / 1e6:.2f} M (lr={args.lr_condition})", flush=True)
    print(f"  - Group 3 (Bridge & Appearance): {param_meta['group_3_count'] / 1e6:.2f} M (lr={args.lr_bridge})", flush=True)
    print(f"  - Group 4 (Shared Control Core): {param_meta['group_4_count'] / 1e6:.2f} M (lr={args.lr_control_block})", flush=True)

    # Checkpoint Manager
    ckpt_mgr = CheckpointManager(output_dir, max_rolling=args.max_rolling_checkpoints)

    # 2. Data Loading
    print(f"Loading dev-train pairs from {args.train_pairs}...", flush=True)
    train_dataset = IPERPoseDataset(
        pairs_jsonl=args.train_pairs,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        augment=False,
    )
    print(f"Loading dev-val pairs from {args.val_pairs}...", flush=True)
    val_dataset = IPERPoseDataset(
        pairs_jsonl=args.val_pairs,
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        resolution=args.resolution,
        augment=False,
    )

    if args.overfit_samples > 0:
        indices = list(range(min(args.overfit_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = Subset(train_dataset, indices)
        print(f"✓ Overfit mode: restricted train and val datasets to first {len(train_dataset)} samples", flush=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=2,
        drop_last=not args.no_shuffle,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    print(f"✓ Train dataset pairs: {len(train_dataset)}, Val dataset pairs: {len(val_dataset)}", flush=True)

    # Resume handling
    global_step = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}...", flush=True)
        ckpt_data = ckpt_mgr.load(args.resume_from, adapter, optimizer)
        global_step = ckpt_data.get("step", 0)
        print(f"✓ Resumed at global step {global_step}", flush=True)

    # 3. Smoke Test Mode
    if args.smoke_test:
        print("\n--- RUNNING BACKPROPAGATION SMOKE TEST (Phase 1) ---", flush=True)
        smoke_steps = 3
        optimizer.zero_grad()
        for batch in train_dataloader:
            t0 = time.time()
            loss, metrics = compute_single_person_step(
                adapter=adapter,
                transformer=transformer,
                pipe=pipe,
                batch=batch,
                prompt=args.prompt,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                human_loss_weight=args.human_loss_weight,
                condition_use_depth=args.condition_use_depth,
            )
            loss.backward()
            
            # Check gradients
            missing_grad_names = [name for name, p in adapter.named_parameters() if p.requires_grad and p.grad is None]
            if missing_grad_names:
                print(f"DEBUG: {len(missing_grad_names)} parameters have grad is None:")
                for n in missing_grad_names[:20]:
                    print(f"  {n}")
            trainable_grads = [p.grad for p in adapter.parameters() if p.requires_grad]
            assert all(g is not None for g in trainable_grads), f"{len(missing_grad_names)} trainable parameters did not receive gradients: {missing_grad_names[:5]}"
            assert all(torch.isfinite(g).all() for g in trainable_grads), "NaN or Inf detected in gradients!"
            
            # Check frozen parameters have NO gradients
            frozen_grads = [p.grad for p in adapter.parameters() if not p.requires_grad]
            assert all(g is None for g in frozen_grads), "Frozen parameter erroneously received gradient!"

            # Explicitly verify the 12 Residual Heads contract
            geo_heads = adapter.control_core.geometry_zero_heads
            int_heads = adapter.control_core.interaction_zero_heads
            assert len(geo_heads) == 6, f"Expected 6 geometry heads, got {len(geo_heads)}"
            assert len(int_heads) == 6, f"Expected 6 interaction heads, got {len(int_heads)}"
            for idx, gh in enumerate(geo_heads):
                for p in gh.parameters():
                    assert p.requires_grad, f"Geometry head {idx} parameter must be trainable"
                    assert p.grad is not None, f"Geometry head {idx} parameter missing gradient"
            for idx, ih in enumerate(int_heads):
                for p in ih.parameters():
                    assert not p.requires_grad, f"Interaction head {idx} parameter must be frozen"
                    assert p.grad is None, f"Interaction head {idx} parameter erroneously received gradient"

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad], args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            step_ms = (time.time() - t0) * 1000
            alloc_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            print(
                f"Smoke Step {global_step}/{smoke_steps} | Loss: {metrics['loss']:.4f} "
                f"(Human: {metrics['loss_human']:.4f}, BG: {metrics['loss_bg']:.4f}) | "
                f"Grad Norm: {grad_norm:.4f} | RMS: {metrics['residual_rms_mean']:.6f} | "
                f"VRAM Alloc: {alloc_gib:.2f} GiB | Time: {step_ms:.1f}ms",
                flush=True,
            )
            if global_step >= smoke_steps:
                break

        print("✓ Verified 12 Residual Heads: 6 Geometry Heads active with finite gradients, 6 Interaction Heads frozen with None gradients", flush=True)

        # Verify Checkpoint Save & Restore
        test_ckpt = ckpt_mgr.save(
            step=global_step,
            adapter=adapter,
            optimizer=optimizer,
            val_loss=metrics["loss"],
            is_best=True,
        )
        assert test_ckpt.exists(), f"Smoke checkpoint not found at {test_ckpt}"
        ckpt_mgr.load(test_ckpt, adapter, optimizer)
        print(f"✓ Checkpoint rotation and restoration verified: {test_ckpt}", flush=True)
        print("=== SMOKE TEST PASSED SUCCESSFULLY ===", flush=True)
        return

    # 4. Main Training Loop
    use_micro_budget = (args.max_micro_steps is not None)
    target_micro_steps = args.max_micro_steps if use_micro_budget else (args.max_train_steps * args.gradient_accumulation_steps)
    val_interval_micro = args.val_micro_steps if args.val_micro_steps is not None else (args.val_every * args.gradient_accumulation_steps)
    save_interval_micro = args.save_micro_steps if args.save_micro_steps is not None else (args.save_steps * args.gradient_accumulation_steps)

    if use_micro_budget:
        print(f"\nStarting Scheme A Training up to {args.max_micro_steps} micro-steps ({math.ceil(args.max_micro_steps / args.gradient_accumulation_steps)} optimizer steps)...", flush=True)
    else:
        print(f"\nStarting Scheme A Training up to {args.max_train_steps} optimizer steps...", flush=True)

    t_train_start = time.time()
    optimizer.zero_grad()

    micro_step = 0
    accum_count = 0
    running_loss = 0.0
    running_raw_mse = 0.0
    running_loss_human = 0.0
    running_loss_bg = 0.0
    latest_metrics = {}
    epoch = 0

    should_stop = False
    while not should_stop:
        epoch += 1
        for batch in train_dataloader:
            if use_micro_budget and micro_step >= args.max_micro_steps:
                should_stop = True
                break
            if not use_micro_budget and global_step >= args.max_train_steps:
                should_stop = True
                break

            t_step_0 = time.time()

            # Warmup logic for zero-heads if configured
            if args.warmup_heads_steps > 0:
                current_cmp_step = micro_step if use_micro_budget else global_step
                if current_cmp_step < args.warmup_heads_steps:
                    for grp in optimizer.param_groups[1:]:
                        grp["lr"] = 0.0
                else:
                    if optimizer.param_groups[1]["lr"] == 0.0:
                        optimizer.param_groups[1]["lr"] = args.lr_condition
                        optimizer.param_groups[2]["lr"] = args.lr_bridge
                        optimizer.param_groups[3]["lr"] = args.lr_control_block
                        print(f"✓ Step {current_cmp_step}: Warmup complete. Restored condition, reasoner, and control core learning rates.", flush=True)

            loss, metrics = compute_single_person_step(
                adapter=adapter,
                transformer=transformer,
                pipe=pipe,
                batch=batch,
                prompt=args.prompt,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
                human_loss_weight=args.human_loss_weight,
                condition_use_depth=args.condition_use_depth,
            )

            loss_scaled = loss / args.gradient_accumulation_steps
            loss_scaled.backward()

            micro_step += 1
            accum_count += 1
            running_loss += metrics["loss"]
            running_raw_mse += metrics.get("raw_mse", metrics["loss"])
            running_loss_human += metrics["loss_human"]
            running_loss_bg += metrics["loss_bg"]
            latest_metrics = metrics

            grad_norm = 0.0
            # Optimizer step on accumulation boundary
            if accum_count % args.gradient_accumulation_steps == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in adapter.parameters() if p.requires_grad], args.max_grad_norm
                ).item())
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_duration = time.time() - t_step_0

            # Logging
            if use_micro_budget:
                if micro_step % args.log_steps == 0 or micro_step == 1:
                    alloc = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                    resv = torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                    print(
                        f"Step {micro_step:4d}/{args.max_micro_steps} | "
                        f"Loss: {metrics['loss']:.4f} (Raw MSE: {metrics.get('raw_mse', metrics['loss']):.4f}) | "
                        f"Speed: {step_duration*1000:.1f}ms/step | "
                        f"VRAM Alloc: {alloc:.2f} GiB, Resv: {resv:.2f} GiB",
                        flush=True,
                    )
            else:
                if accum_count % args.gradient_accumulation_steps == 0 and (global_step % args.log_steps == 0 or global_step == 1):
                    avg_loss = running_loss / args.gradient_accumulation_steps
                    avg_human = running_loss_human / args.gradient_accumulation_steps
                    avg_bg = running_loss_bg / args.gradient_accumulation_steps
                    running_loss = 0.0
                    running_raw_mse = 0.0
                    running_loss_human = 0.0
                    running_loss_bg = 0.0
                    alloc = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                    resv = torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                    print(
                        f"Step {global_step:5d}/{args.max_train_steps} (Epoch {epoch}) | "
                        f"Loss: {avg_loss:.4f} (Human: {avg_human:.4f}, BG: {avg_bg:.4f}) | "
                        f"GradNorm: {grad_norm:.3f} | RMS: {latest_metrics['residual_rms_mean']:.6f} | "
                        f"Speed: {step_duration*1000:.1f}ms/micro-step | "
                        f"VRAM Alloc: {alloc:.2f} GiB, Resv: {resv:.2f} GiB",
                        flush=True,
                    )

            # Validation condition
            trigger_val = False
            val_step_id = micro_step if use_micro_budget else global_step
            if use_micro_budget:
                if val_interval_micro > 0 and (micro_step % val_interval_micro == 0 or micro_step == args.max_micro_steps):
                    trigger_val = True
            else:
                if accum_count % args.gradient_accumulation_steps == 0 and (global_step % args.val_every == 0):
                    trigger_val = True

            val_loss = None
            if trigger_val:
                val_metrics = validate(
                    args=args,
                    pipe=pipe,
                    adapter=adapter,
                    transformer=transformer,
                    val_dataloader=val_dataloader,
                    cache_dir=cache_dir,
                    device=device,
                    dtype=dtype,
                    step=val_step_id,
                )
                val_loss = val_metrics["val_loss_on"]

            # Checkpointing condition
            trigger_save = False
            save_step_id = micro_step if use_micro_budget else global_step
            if use_micro_budget:
                if save_interval_micro > 0 and (micro_step % save_interval_micro == 0 or micro_step == args.max_micro_steps):
                    trigger_save = True
            else:
                if accum_count % args.gradient_accumulation_steps == 0 and (
                    global_step % args.save_steps == 0 or global_step == args.max_train_steps
                ):
                    trigger_save = True

            if trigger_save:
                is_best = False
                if val_loss is not None and val_loss < ckpt_mgr.best_loss:
                    ckpt_mgr.best_loss = val_loss
                    is_best = True

                # 1. Rolling & best checkpoint via CheckpointManager
                saved_p = ckpt_mgr.save(
                    step=save_step_id,
                    adapter=adapter,
                    optimizer=optimizer,
                    val_loss=val_loss,
                    is_best=is_best,
                    extra_meta={"metrics": latest_metrics, "args": vars(args), "epoch": epoch, "micro_step": micro_step, "global_step": global_step},
                )

                # 2. Also always save adapter_step_{save_step_id}.pt for 1:1 multi-stage evaluation compatibility
                direct_adapter_ckpt = output_dir / f"adapter_step_{save_step_id}.pt"
                torch.save(
                    {
                        "step": save_step_id,
                        "micro_step": micro_step,
                        "global_step": global_step,
                        "adapter_state_dict": adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "val_loss": val_loss,
                    },
                    direct_adapter_ckpt,
                )
                print(f"✓ Checkpoint saved to {direct_adapter_ckpt} (best={is_best})", flush=True)

            if use_micro_budget and micro_step >= args.max_micro_steps:
                should_stop = True
                break

    total_time = time.time() - t_train_start
    if use_micro_budget:
        print(f"\n=== TRAINING COMPLETE: {micro_step} micro-steps ({global_step} optimizer steps) in {total_time:.2f}s ===", flush=True)
    else:
        print(f"\n=== TRAINING COMPLETE: {global_step} optimizer steps completed in {total_time/3600:.2f}h ===", flush=True)


if __name__ == "__main__":
    main()
