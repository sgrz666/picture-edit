#!/usr/bin/env python3
"""Real DeepGen structural smoke test for Unified SMPL-X Adapter V6.

This script intentionally uses synthetic conditions and never creates an
optimizer, loads a training checkpoint, or saves weights. Zero-initialized
heads mean the injected and non-injected DeepGen outputs must be identical.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_control.v6.conditions import TaskSpec, UnifiedAdapterCondition
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "dual"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-deepgen-equivalence",
        action="store_true",
        help="Run adapter-only checks without the two frozen-backbone forwards.",
    )
    return parser.parse_args()


def load_deepgen_transformer(model_path: Path, device: torch.device):
    module_path = model_path / "deepgen_pipeline.py"
    spec = importlib.util.spec_from_file_location("deepgen_v6_smoke_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DeepGen runtime from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transformer = module.SD3Transformer2DModel.from_pretrained(
        str(model_path),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return transformer.to(device).eval()


def make_condition(mode: str, image_size: int, latent_size: int, device: torch.device):
    people = 1 if mode == "single" else 2
    person_valid = torch.zeros(1, 2, dtype=torch.bool, device=device)
    person_valid[:, :people] = True
    modality_valid = person_valid[..., None].expand(-1, -1, 5).clone()
    silhouette = torch.zeros(1, 2, 1, image_size, image_size, device=device)
    silhouette[:, 0, :, :, : image_size // 2 + 4] = 1
    if people == 2:
        silhouette[:, 1, :, :, image_size // 2 - 4 :] = 1
    contact_maps = torch.zeros(1, 2, image_size, image_size, device=device)
    if people == 2:
        center = image_size // 2
        contact_maps[:, :, center - 3 : center + 3, center - 3 : center + 3] = 1
    contact_pairs = torch.zeros(1, 4, 4, device=device)
    contact_pairs[:, 0] = torch.tensor([0, 7, 1, 10], device=device)
    contact_valid = torch.zeros(1, 4, dtype=torch.bool, device=device)
    contact_valid[:, 0] = people == 2
    return UnifiedAdapterCondition(
        normal=torch.randn(1, 2, 3, image_size, image_size, device=device),
        depth=torch.rand(1, 2, 1, image_size, image_size, device=device),
        skeleton=torch.rand(1, 2, 3, image_size, image_size, device=device),
        silhouette=silhouette,
        person_valid=person_valid,
        source_person_latents=torch.randn(1, 2, 16, latent_size, latent_size, device=device),
        smpl_pose6d=torch.randn(1, 2, 52, 6, device=device),
        source_betas=torch.randn(1, 2, 10, device=device),
        camera=torch.randn(1, 2, 7, device=device),
        task_spec=TaskSpec(
            interaction_type=torch.tensor([1 if people == 2 else 0], device=device),
            declared_num_people=torch.tensor([people], device=device),
            prompts=("two people shake hands" if people == 2 else "one person raises an arm",),
        ),
        part_ids=torch.randint(0, 24, (1, 2, image_size, image_size), device=device),
        modality_valid=modality_valid,
        source_indices=torch.tensor([[0, 1]], device=device),
        contact_maps=contact_maps,
        contact_pairs=contact_pairs,
        contact_valid=contact_valid,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.latent_size % 2:
        raise ValueError("latent-size must be divisible by DeepGen patch size 2")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    transformer = load_deepgen_transformer(args.model_path, device)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen(transformer).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    condition = make_condition(args.mode, args.image_size, args.latent_size, device)
    target = torch.randn(1, 16, args.latent_size, args.latent_size, device=device, dtype=torch.bfloat16)
    source = torch.randn_like(target)
    references = [[source[0]]]
    text = torch.randn(1, 12, transformer.config.joint_attention_dim, device=device, dtype=torch.bfloat16)
    pooled = torch.randn(1, transformer.config.pooled_projection_dim, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500], device=device)

    adapter_kwargs = dict(
        target_latents=target,
        condition=condition,
        source_scene_latents=source,
        cond_hidden_states=references,
        encoder_hidden_states=text,
        pooled_projections=pooled,
        timestep=timestep,
    )
    with torch.inference_mode():
        prepared = adapter.prepare_conditioning(condition.to(device=device, dtype=torch.bfloat16))
        residuals = adapter(**adapter_kwargs)
        max_residual = max(float(value.abs().max().item()) for value in residuals)
        max_equivalence_error = None
        if not args.skip_deepgen_equivalence:
            transformer_kwargs = dict(
                hidden_states=target,
                cond_hidden_states=references,
                encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )
            without_adapter = transformer(**transformer_kwargs)[0]
            with_adapter = transformer(
                **transformer_kwargs,
                block_controlnet_hidden_states=residuals,
            )[0]
            max_equivalence_error = float((without_adapter - with_adapter).abs().max().item())

    if len(residuals) != 6 or max_residual != 0.0:
        raise RuntimeError("zero-initialized adapter did not emit six exact-zero residuals")
    if max_equivalence_error is not None and max_equivalence_error > 1e-6:
        raise RuntimeError(f"DeepGen zero-injection error is {max_equivalence_error}, expected <= 1e-6")
    report = {
        "status": "PASS",
        "mode": args.mode,
        "route_num_people": prepared.route.num_people.tolist(),
        "adapter_trainable_parameters": adapter.trainable_parameter_count,
        "residual_shapes": [list(value.shape) for value in residuals],
        "max_zero_residual": max_residual,
        "max_deepgen_equivalence_error": max_equivalence_error,
        "backbone_frozen": all(not parameter.requires_grad for parameter in transformer.parameters()),
        "training_started": False,
    }
    output = args.output or ROOT / "outputs" / f"adapter_v6_smoke_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
