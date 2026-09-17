#!/usr/bin/env python3
"""Real DeepGen smoke for the V6.3 dynamic control interface.

The script uses synthetic prepared SMPL-X conditions and local model weights.
It performs no training, optimizer creation, checkpoint save, or download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_adapter_reasoner import make_bundle
from scripts.smoke_adapter_v6 import load_deepgen_transformer, parameter_report
from src.pose_control.v6.conditions import AdapterIdentityCondition
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "dual", "mixed"), required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
        ),
    )
    parser.add_argument("--geometry-strength", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.image_size % 16:
        raise ValueError("image-size must be divisible by 16")
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    transformer = load_deepgen_transformer(args.model_path, device)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen(transformer).to(
        device=device, dtype=dtype
    ).eval()
    bundle = make_bundle(
        adapter.condition_injector,
        args.mode,
        args.image_size,
        device,
        dtype,
    )
    batch_size = bundle.batch_size
    latent_size = args.image_size // 8
    identity = AdapterIdentityCondition(
        source_person_latents=torch.randn(
            batch_size,
            2,
            16,
            latent_size,
            latent_size,
            device=device,
            dtype=dtype,
        ),
        source_indices=torch.tensor(
            [[0, 1]] * batch_size, device=device, dtype=torch.long
        ),
    )
    target = torch.randn(
        batch_size,
        16,
        latent_size,
        latent_size,
        device=device,
        dtype=dtype,
    )
    source = torch.randn_like(target)
    references = [[source[index]] for index in range(batch_size)]
    text = torch.randn(
        batch_size,
        12,
        transformer.config.joint_attention_dim,
        device=device,
        dtype=dtype,
    )
    pooled = torch.randn(
        batch_size,
        transformer.config.pooled_projection_dim,
        device=device,
        dtype=dtype,
    )

    diagnostics = []
    max_residual = 0.0
    max_equivalence_error = 0.0
    with torch.inference_mode():
        prepared = adapter.prepare_conditioning(
            bundle, identity, source, target.shape[-2:]
        )
        for step_index in range(args.steps):
            progress = step_index / max(args.steps - 1, 1)
            timestep = torch.full(
                (batch_size,),
                1000 - step_index * max(1, 900 // args.steps),
                device=device,
            )
            output = adapter(
                target_latents=target,
                prepared=prepared,
                cond_hidden_states=references,
                encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=timestep,
                denoise_progress=progress,
                geometry_strength=args.geometry_strength,
                interaction_strength=args.interaction_strength,
            )
            residuals = output.block_controlnet_hidden_states
            max_residual = max(
                max_residual,
                max(float(value.abs().max().item()) for value in residuals),
            )
            transformer_kwargs = dict(
                hidden_states=target,
                cond_hidden_states=references,
                encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )
            baseline = transformer(**transformer_kwargs)[0]
            controlled = transformer(
                **transformer_kwargs,
                block_controlnet_hidden_states=residuals,
            )[0]
            max_equivalence_error = max(
                max_equivalence_error,
                float((baseline - controlled).abs().max().item()),
            )
            diagnostics.append(output.diagnostics)

    if max_residual != 0.0:
        raise RuntimeError("zero-initialized dynamic interface emitted nonzero residuals")
    if max_equivalence_error > 1e-6:
        raise RuntimeError(
            f"DeepGen zero-injection error is {max_equivalence_error}, expected <= 1e-6"
        )
    if len(diagnostics) != args.steps:
        raise RuntimeError("dynamic control call count does not match denoising steps")

    report = {
        "status": "PASS",
        "mode": args.mode,
        "image_size": args.image_size,
        "steps": args.steps,
        "device": str(device),
        "dtype": str(dtype),
        "person_count": prepared.person_count.tolist(),
        "adapter_trainable_parameters": adapter.trainable_parameter_count,
        "parameters_by_module": parameter_report(adapter),
        "dynamic_control_calls": len(diagnostics),
        "denoise_progress": [item["denoise_progress"] for item in diagnostics],
        "residual_shapes": [
            list(value.shape) for value in output.block_controlnet_hidden_states
        ],
        "max_zero_residual": max_residual,
        "max_deepgen_equivalence_error": max_equivalence_error,
        "backbone_frozen": all(
            not parameter.requires_grad for parameter in transformer.parameters()
        ),
        "training_started": False,
    }
    output_path = args.output or (
        ROOT / "outputs" / f"deepgen_control_interface_smoke_{args.mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
