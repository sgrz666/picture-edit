"""CPU-only Full frontend migration audit; never modifies source checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _conditions(torch, width, dual):
    generator = torch.Generator().manual_seed(20261022)
    inputs = {
        "normal_a": torch.rand(1, 3, 32, width, generator=generator) * 2 - 1,
        "pose_heatmap_a": torch.rand(1, 25, 32, width, generator=generator),
        "part_onehot_a": torch.zeros(1, 14, 32, width),
        "smplx_global_a": torch.randn(1, 26, generator=generator),
        "human_mask_a": torch.ones(1, 1, 32, width),
        "task_id": torch.tensor([int(dual)]),
    }
    inputs["part_onehot_a"][:, 1] = 1
    if dual:
        for name in ("normal", "pose_heatmap", "part_onehot", "smplx_global", "human_mask"):
            inputs[name + "_b"] = inputs[name + "_a"].clone()
        inputs["normal_b"].neg_()
    return inputs


def audit_checkpoint(checkpoint, *, reasoning_config=None, threads=2):
    import torch
    from src.pose_control.qwen21_full.condition_bridge import ReasonerControlBridge
    from src.pose_control.qwen21_full.condition_injector import SMPLXConditionInjector
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig, SMPLXAdapterReasoner

    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    path = Path(checkpoint).resolve(strict=True)
    before = path.stat()
    fingerprint = _sha256(path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        config = reasoning_config or AdapterReasoningConfig()
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        native = Qwen21NativeGeometry.from_full_checkpoint(payload, legacy=True, reasoning_config=config).eval()
        state_dict = payload["adapter_state_dict"]

        def restore(module, prefix):
            module.load_state_dict({name[len(prefix):]: value for name, value in state_dict.items() if name.startswith(prefix)}, strict=True)
            return module.eval()

        # Independent legacy components, including the old learned projections.
        # Neither the legacy control core nor a Qwen model is instantiated.
        injector = restore(SMPLXConditionInjector(), "legacy.condition_injector.")
        reasoner = restore(SMPLXAdapterReasoner(config), "legacy.reasoner.")
        projection_channels = state_dict["legacy.condition_bridge.geometry_projection.weight"].shape[0]
        bridge = restore(ReasonerControlBridge(reasoning_dim=config.hidden_dim, geometry_channels=projection_channels), "legacy.condition_bridge.")
        cases = []
        with torch.no_grad():
            for width, dual in ((32, False), (64, False), (64, True)):
                inputs = _conditions(torch, width, dual)
                old_state = reasoner(injector(**inputs))
                captured = {}
                handles = []
                for name in ("geometry", "interaction"):
                    def capture(_module, args, key=name):
                        captured[key] = args[0]
                    handles.append(getattr(bridge, name + "_projection").register_forward_pre_hook(capture))
                try:
                    old_output = bridge(old_state)
                finally:
                    for handle in handles:
                        handle.remove()
                new_output = native(native.condition_injector(**inputs), target_token_hw=(2, width // 16))
                # Restore the old NCHW layout to avoid comparing different CPU
                # Conv2d kernels. Also compare the full unprojected features.
                geometry_map = new_output.geometry.transpose(1, 2).reshape_as(old_state.geometry_feature).contiguous()
                interaction_map = new_output.interaction.transpose(1, 2).reshape_as(old_state.interaction_feature).contiguous()
                geometry_fused_diff = (geometry_map - captured["geometry"]).abs().max().item()
                interaction_fused_diff = (interaction_map - captured["interaction"]).abs().max().item()
                geometry_diff = (bridge.geometry_projection(geometry_map) - old_output.geometry_scene).abs().max().item()
                interaction_diff = (bridge.interaction_projection(interaction_map) - old_output.interaction_scene).abs().max().item()
                finite = bool(torch.isfinite(new_output.geometry).all() and torch.isfinite(new_output.interaction).all())
                cases.append({"name": f"{'dual' if dual else 'single'}_32x{width}",
                              "shape": list(new_output.geometry.shape),
                              "geometry_max_abs": geometry_diff, "interaction_max_abs": interaction_diff,
                              "geometry_fused_max_abs": geometry_fused_diff, "interaction_fused_max_abs": interaction_fused_diff,
                              "finite": finite, "passed": finite and geometry_diff == 0 and interaction_diff == 0
                              and geometry_fused_diff == 0 and interaction_fused_diff == 0})
        after = path.stat()
        unchanged = (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if not unchanged:
            raise RuntimeError("Source checkpoint changed during the audit; results are not stable")
        source_root = ROOT / "src/pose_control/qwen21_full"
        return {
            "schema": "qwen21_native_geometry_audit_v1", "passed": all(case["passed"] for case in cases),
            "checkpoint": str(path), "checkpoint_sha256": fingerprint, "checkpoint_bytes": before.st_size,
            "checkpoint_stat_unchanged": unchanged, "code_root": str(ROOT),
            "source_sha256": {file.relative_to(ROOT).as_posix(): _sha256(file) for file in sorted(source_root.rglob("*.py"))},
            "torch_version": torch.__version__, "device": "cpu", "threads": threads,
            "inheritance": native.inheritance_report, "cases": cases,
            "limits": "Synthetic conditioning only; no image generation, training, Qwen/Union model, or quality claim.",
        }
    finally:
        torch.set_num_threads(old_threads)


def write_report(report, output):
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file; parent must already exist")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new filename")
    report = audit_checkpoint(args.checkpoint, threads=args.threads)
    if args.output is not None:
        write_report(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
