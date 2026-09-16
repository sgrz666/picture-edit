#!/usr/bin/env python3
"""Real DeepGen structural smoke test for Unified SMPL-X Adapter V6.

Synthetic prepared conditions exercise the ConditionBundle boundary. The
script never estimates SMPL-X, trains, creates an optimizer, or saves weights.
Zero-initialized heads must leave the frozen DeepGen result unchanged.
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

from src.pose_control.v6.conditions import AdapterIdentityCondition, ContactRelationBatch, TaskType
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "dual"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--condition-backend", choices=("native", "champ"), default="native")
    parser.add_argument("--use-depth", action="store_true")
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


def default_report_name(mode: str, condition_backend: str, use_depth: bool) -> str:
    """Keep optional-backend smoke evidence separate from the native baseline."""

    suffix = mode
    if condition_backend != "native":
        suffix = f"{suffix}_{condition_backend}"
    elif use_depth:
        suffix = f"{suffix}_depth"
    return f"adapter_v6_smoke_{suffix}.json"


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


def _person_maps(
    side: str,
    image_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    mask = torch.zeros(1, 1, image_size, image_size, device=device, dtype=dtype)
    if side == "a":
        mask[..., 4:-4, 2 : image_size // 2 + 4] = 1
    else:
        mask[..., 4:-4, image_size // 2 - 4 : -2] = 1
    normal = (torch.rand(1, 3, image_size, image_size, device=device, dtype=dtype) * 2 - 1) * mask
    pose = torch.zeros(1, 25, image_size, image_size, device=device, dtype=dtype)
    pose[:, 0, image_size // 2, image_size // 2] = 1
    part = torch.zeros(1, 14, image_size, image_size, device=device, dtype=dtype)
    part[:, 1] = mask[:, 0]
    return {
        f"normal_{side}": normal,
        f"pose_heatmap_{side}": pose,
        f"part_onehot_{side}": part,
        f"smplx_global_{side}": torch.randn(1, 26, device=device, dtype=dtype),
        f"human_mask_{side}": mask,
        f"depth_{side}": torch.rand(1, 1, image_size, image_size, device=device, dtype=dtype),
    }


def make_conditions(
    adapter: UnifiedSMPLXAdapterV6,
    mode: str,
    image_size: int,
    latent_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    raw = _person_maps("a", image_size, device=device, dtype=dtype)
    raw["task_id"] = torch.tensor([int(TaskType.SINGLE)], device=device)
    if mode == "dual":
        raw.update(_person_maps("b", image_size, device=device, dtype=dtype))
        raw["task_id"] = torch.tensor([int(TaskType.HANDSHAKE)], device=device)
        raw["relative_geometry"] = torch.randn(1, 11, device=device, dtype=dtype)
        contact_raster = torch.zeros(1, 2, image_size, image_size, device=device, dtype=dtype)
        center = image_size // 2
        contact_raster[..., center - 3 : center + 3, center - 3 : center + 3] = 1
        raw["contact_raster"] = contact_raster
        valid = torch.zeros(1, 8, dtype=torch.bool, device=device)
        valid[:, 0] = True
        raw["contact_relations"] = ContactRelationBatch(
            src_person=torch.zeros(1, 8, dtype=torch.long, device=device),
            src_part=torch.ones(1, 8, dtype=torch.long, device=device) * 4,
            dst_person=torch.ones(1, 8, dtype=torch.long, device=device),
            dst_part=torch.ones(1, 8, dtype=torch.long, device=device) * 7,
            contact_type=torch.ones(1, 8, dtype=torch.long, device=device),
            distance=torch.zeros(1, 8, 1, device=device, dtype=dtype),
            valid_mask=valid,
        )
    bundle = adapter.condition_injector(**raw)
    identity = AdapterIdentityCondition(
        source_person_latents=torch.randn(
            1, 2, 16, latent_size, latent_size, device=device, dtype=dtype
        ),
        source_indices=torch.tensor([[0, 1]], device=device),
    )
    return bundle, identity


def parameter_report(adapter: UnifiedSMPLXAdapterV6) -> dict[str, int]:
    modules = {
        "condition_injector": adapter.condition_injector,
        "reasoner": adapter.reasoner,
        "reasoner_control_bridge": adapter.condition_bridge,
        "appearance_and_binding": torch.nn.ModuleList(
            [
                adapter.geometry_token_projection,
                adapter.appearance_token_encoder,
                adapter.person_token_binder,
            ]
        ),
        "dynamic_control_core": adapter.control_core,
        "control_strength": adapter.control_interface.strength_controller,
    }
    return {
        name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        for name, module in modules.items()
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.image_size % 8:
        raise ValueError("image-size must be divisible by condition stride 8")
    if args.latent_size % 2:
        raise ValueError("latent-size must be divisible by DeepGen patch size 2")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    transformer = load_deepgen_transformer(args.model_path, device)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen(
        transformer,
        condition_use_depth=args.use_depth,
        normal_backend=args.condition_backend,
        depth_backend=args.condition_backend,
    ).to(device=device, dtype=dtype).eval()
    condition_bundle, identity_condition = make_conditions(
        adapter, args.mode, args.image_size, args.latent_size, device, dtype
    )
    target = torch.randn(1, 16, args.latent_size, args.latent_size, device=device, dtype=dtype)
    source = torch.randn_like(target)
    references = [[source[0]]]
    text = torch.randn(1, 12, transformer.config.joint_attention_dim, device=device, dtype=dtype)
    pooled = torch.randn(1, transformer.config.pooled_projection_dim, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)
    with torch.inference_mode():
        prepared = adapter.prepare_conditioning(
            condition_bundle,
            identity_condition,
            source,
            target.shape[-2:],
        )
        adapter_kwargs = dict(
            target_latents=target,
            prepared=prepared,
            cond_hidden_states=references,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            denoise_progress=0.0,
        )
        control_output = adapter(**adapter_kwargs)
        residuals = control_output.block_controlnet_hidden_states
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
        raise RuntimeError(
            f"DeepGen zero-injection error is {max_equivalence_error}, expected <= 1e-6"
        )
    report = {
        "status": "PASS",
        "mode": args.mode,
        "condition_backend": args.condition_backend,
        "use_depth": args.use_depth,
        "route_num_people": prepared.person_count.tolist(),
        "adapter_trainable_parameters": adapter.trainable_parameter_count,
        "parameters_by_module": parameter_report(adapter),
        "residual_shapes": [list(value.shape) for value in residuals],
        "control_diagnostics": control_output.diagnostics,
        "max_zero_residual": max_residual,
        "max_deepgen_equivalence_error": max_equivalence_error,
        "backbone_frozen": all(not parameter.requires_grad for parameter in transformer.parameters()),
        "training_started": False,
    }
    output = args.output or ROOT / "outputs" / default_report_name(
        args.mode, args.condition_backend, args.use_depth
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
