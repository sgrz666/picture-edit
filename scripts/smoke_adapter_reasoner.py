#!/usr/bin/env python3
"""GPU/CPU structural smoke for the standalone V6.2 Adapter reasoner.

This uses synthetic, already-prepared SMPL-X conditions. It does not train,
create an optimizer, save weights, estimate bodies, or download checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_control.v6.condition_injector import SMPLXConditionInjector
from src.pose_control.v6.conditions import ContactRelationBatch, TaskType
from src.pose_control.v6.reasoning import AdapterReasoningConfig, SMPLXAdapterReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "dual", "mixed"), required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_report_name(mode: str) -> str:
    return f"adapter_reasoner_smoke_{mode}.json"


def _person_maps(
    side: str,
    batch_size: int,
    image_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    mask = torch.zeros(
        batch_size, 1, image_size, image_size, device=device, dtype=dtype
    )
    if side == "a":
        mask[..., image_size // 8 : -image_size // 8, image_size // 12 : image_size // 2 + image_size // 12] = 1
    else:
        mask[..., image_size // 8 : -image_size // 8, image_size // 2 - image_size // 12 : -image_size // 12] = 1
    normal = (
        torch.rand(batch_size, 3, image_size, image_size, device=device, dtype=dtype) * 2 - 1
    ) * mask
    pose = torch.zeros(
        batch_size, 25, image_size, image_size, device=device, dtype=dtype
    )
    pose[:, 0, image_size // 2, image_size // 2] = 1
    part = torch.zeros(
        batch_size, 14, image_size, image_size, device=device, dtype=dtype
    )
    part[:, 1] = mask[:, 0]
    return {
        f"normal_{side}": normal,
        f"pose_heatmap_{side}": pose,
        f"part_onehot_{side}": part,
        f"smplx_global_{side}": torch.randn(
            batch_size, 26, device=device, dtype=dtype
        ),
        f"human_mask_{side}": mask,
    }


def make_bundle(
    injector: SMPLXConditionInjector,
    mode: str,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    counts = (1, 2) if mode == "mixed" else ((1,) if mode == "single" else (2,))
    batch_size = len(counts)
    raw = _person_maps("a", batch_size, image_size, device=device, dtype=dtype)
    raw["task_id"] = torch.tensor(
        [int(TaskType.SINGLE) if count == 1 else int(TaskType.HANDSHAKE) for count in counts],
        dtype=torch.long,
        device=device,
    )
    if max(counts) == 2:
        raw.update(_person_maps("b", batch_size, image_size, device=device, dtype=dtype))
        person_b_valid = torch.tensor(
            [count == 2 for count in counts], dtype=torch.bool, device=device
        )
        raw["person_b_valid"] = person_b_valid
        raw["relative_geometry"] = torch.randn(
            batch_size, 11, device=device, dtype=dtype
        )
        raster = torch.zeros(
            batch_size, 2, image_size, image_size, device=device, dtype=dtype
        )
        center = image_size // 2
        radius = max(1, image_size // 32)
        raster[person_b_valid, :, center - radius : center + radius, center - radius : center + radius] = 1
        raw["contact_raster"] = raster
        relation_valid = torch.zeros(
            batch_size, 8, dtype=torch.bool, device=device
        )
        relation_valid[person_b_valid, 0] = True
        raw["contact_relations"] = ContactRelationBatch(
            src_person=torch.zeros(batch_size, 8, dtype=torch.long, device=device),
            src_part=torch.full(
                (batch_size, 8), 4, dtype=torch.long, device=device
            ),
            dst_person=torch.ones(batch_size, 8, dtype=torch.long, device=device),
            dst_part=torch.full(
                (batch_size, 8), 7, dtype=torch.long, device=device
            ),
            contact_type=torch.ones(
                batch_size, 8, dtype=torch.long, device=device
            ),
            distance=torch.zeros(batch_size, 8, 1, device=device, dtype=dtype),
            valid_mask=relation_valid,
        )
    return injector(**raw)


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_report(
    injector: SMPLXConditionInjector, reasoner: SMPLXAdapterReasoner
) -> dict[str, int]:
    single_modules = (
        reasoner.single_geometry_block,
        reasoner.single_high_projection,
        reasoner.single_high_block,
        reasoner.role_embedding,
    )
    report = {
        "condition_injector": _parameter_count(injector),
        "person_geometry_reasoner": _parameter_count(reasoner.person_reasoner),
        "cross_person_reasoner": _parameter_count(reasoner.cross_person_reasoner),
        "contact_reasoner": _parameter_count(reasoner.contact_reasoner),
        "dual_fusion": _parameter_count(reasoner.dual_fusion),
        "single_path_and_roles": sum(_parameter_count(module) for module in single_modules),
        "reasoner_total": _parameter_count(reasoner),
    }
    report["condition_and_reasoner_total"] = (
        report["condition_injector"] + report["reasoner_total"]
    )
    return report


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.image_size % 16:
        raise ValueError("image-size must be divisible by 16")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    injector = SMPLXConditionInjector().to(device=device, dtype=dtype).eval()
    reasoner = SMPLXAdapterReasoner(AdapterReasoningConfig()).to(
        device=device, dtype=dtype
    ).eval()
    bundle = make_bundle(injector, args.mode, args.image_size, device, dtype)
    with torch.inference_mode():
        state = reasoner(bundle).validate()

    condition_size = args.image_size // 8
    reasoning_size = condition_size // 2
    expected_batch = 2 if args.mode == "mixed" else 1
    expected_counts = [1, 2] if args.mode == "mixed" else [1 if args.mode == "single" else 2]
    if state.geometry_feature.shape != (
        expected_batch,
        512,
        reasoning_size,
        reasoning_size,
    ):
        raise RuntimeError("unexpected reasoning feature shape")
    if state.geometry_highres.shape[-2:] != (condition_size, condition_size):
        raise RuntimeError("unexpected high-resolution reasoning feature shape")
    if state.person_count.tolist() != expected_counts:
        raise RuntimeError("reasoner route does not match requested smoke mode")
    if not all(torch.isfinite(value).all() for value in (
        state.geometry_feature,
        state.geometry_highres,
        state.interaction_feature,
        state.interaction_highres,
        state.person_tokens,
    )):
        raise RuntimeError("reasoner emitted a non-finite value")

    report = {
        "status": "PASS",
        "mode": args.mode,
        "image_size": args.image_size,
        "device": str(device),
        "dtype": str(dtype),
        "person_count": state.person_count.tolist(),
        "geometry_feature_shape": list(state.geometry_feature.shape),
        "geometry_highres_shape": list(state.geometry_highres.shape),
        "interaction_feature_shape": list(state.interaction_feature.shape),
        "interaction_highres_shape": list(state.interaction_highres.shape),
        "parameters_by_module": parameter_report(injector, reasoner),
        "training_started": False,
    }
    output = args.output or ROOT / "outputs" / default_report_name(args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
