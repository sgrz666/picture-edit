"""CPU audit of native Full Body control on real 32-layer Qwen block structure.

Uses a small random Qwen, not the pretrained 7B model. An optional existing Full
checkpoint supplies the real trained geometry frontend. No weights are saved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def audit_native_body(*, full_checkpoint=None, threads=2):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures, Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_integration import install_native_body
    from scripts.audit.audit_qwen21_native_geometry import _conditions, _sha256

    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be positive")
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(20261022)
        checkpoint_record = None
        if full_checkpoint is None:
            geometry_dim = 32
            feature_a = torch.randn(1, 4, geometry_dim)
            feature_b = torch.randn(1, 4, geometry_dim)
        else:
            checkpoint = Path(full_checkpoint).resolve(strict=True)
            stat = checkpoint.stat()
            fingerprint = _sha256(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            frontend = Qwen21NativeGeometry.from_full_checkpoint(payload, legacy=True).eval().requires_grad_(False)
            raw_a = _conditions(torch, 32, False)
            raw_b = {key: value.clone() for key, value in raw_a.items()}
            raw_b["normal_a"].neg_()
            raw_b["pose_heatmap_a"] = raw_b["pose_heatmap_a"].flip(-1)
            with torch.no_grad():
                feature_a = frontend(frontend.condition_injector(**raw_a), target_token_hw=(2, 2)).geometry
                feature_b = frontend(frontend.condition_injector(**raw_b), target_token_hw=(2, 2)).geometry
            after = checkpoint.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError("Full checkpoint changed during the audit")
            geometry_dim = feature_a.shape[-1]
            checkpoint_record = {"path": str(checkpoint), "sha256": fingerprint,
                                 "stat_unchanged": True, "inheritance": frontend.inheritance_report}

        def feature(value, *, gradients=False):
            return NativeGeometryFeatures(value.detach().clone().requires_grad_(gradients),
                                          torch.zeros_like(value), (2, 2), torch.tensor([[True, False]]))

        def inputs(references, length, sigma):
            mask = torch.zeros(1, length + 1, dtype=torch.bool)
            mask[:, :references] = True
            mask[:, -1] = True
            return dict(hidden_states=torch.randn(1, 4 * (references + 1), 64),
                        encoder_hidden_states=torch.randn(1, length, 8),
                        encoder_hidden_states_mask=torch.ones(1, length, dtype=torch.bool),
                        timestep=torch.tensor([sigma]), img_shapes=[[(1, 2, 2)] * (references + 1)],
                        img_mask=mask, return_dict=False)

        model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=32,
                  num_attention_heads=2, attention_head_dim=8, context_in_dim=8,
                  axes_dims_rope=(2, 2, 4), mlp_ratio=2).eval()
        checked = copy.deepcopy(model)
        original_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        data_a, data_b = inputs(1, 3, .5), inputs(2, 5, .2)
        with torch.no_grad():
            baseline = model(**data_a)[0]
        adapter = Qwen21NativeBody(NativeBodyConfig(hidden_dim=16, width=8, geometry_dim=geometry_dim))
        install = install_native_body(model, adapter)
        with torch.no_grad():
            zero = model(**data_a, native_geometry=feature(feature_a))[0]
            disabled = model(**data_a, native_enabled=False)[0]
        zero_diff = (zero - baseline).abs().max().item()
        disabled_diff = (disabled - baseline).abs().max().item()
        with torch.no_grad():
            for head in adapter.output_heads:
                torch.nn.init.normal_(head.weight, std=.04)
        install_native_body(checked, copy.deepcopy(adapter))
        checked.enable_gradient_checkpointing()
        checkpoint_function = checked._gradient_checkpointing_func
        a_ref, b_ref = feature(feature_a, gradients=True), feature(feature_b, gradients=True)
        a_check, b_check = feature(feature_a, gradients=True), feature(feature_b, gradients=True)
        outputs_ref = [model(**data_a, native_geometry=a_ref)[0], model(**data_b, native_geometry=b_ref)[0]]
        outputs_check = [checked(**data_a, native_geometry=a_check)[0], checked(**data_b, native_geometry=b_check)[0]]
        replay_diff = max((a - b).abs().max().item() for a, b in zip(outputs_ref, outputs_check))
        prefix_diff = (outputs_ref[0][:, :-4] - baseline[:, :-4]).abs().max().item()
        target_change = (outputs_ref[0][:, -4:] - baseline[:, -4:]).abs().max().item()
        sum(output[:, -4:].square().mean() for output in outputs_ref).backward()
        sum(output[:, -4:].square().mean() for output in outputs_check).backward()
        gradient_max, gradient_match = 0., True
        for (_, expected), (_, actual) in zip(model.native_body.named_parameters(), checked.native_body.named_parameters()):
            if expected.grad is None or actual.grad is None:
                gradient_match &= expected.grad is None and actual.grad is None
            else:
                gradient_max = max(gradient_max, (actual.grad - expected.grad).abs().max().item())
                gradient_match &= torch.allclose(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
        geometry_gradients = all(value.geometry.grad is not None and value.geometry.grad.abs().sum() > 0
                                 for value in (a_ref, b_ref, a_check, b_check))
        gradient_match &= torch.allclose(a_check.geometry.grad, a_ref.geometry.grad, rtol=1e-5, atol=1e-7)
        gradient_match &= torch.allclose(b_check.geometry.grad, b_ref.geometry.grad, rtol=1e-5, atol=1e-7)
        heads_active = all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0 for head in checked.native_body.output_heads)
        after_weights = dict(model.named_parameters())
        base_frozen = all(not after_weights[name].requires_grad and after_weights[name].grad is None for name in original_weights)
        base_unchanged = all(torch.equal(after_weights[name].detach(), value) for name, value in original_weights.items())
        helper_restored = checked._gradient_checkpointing_func is checkpoint_function and checked._native_body_execution.get() is None
        with torch.device("meta"):
            production_adapter = Qwen21NativeBody()
        production_count = sum(p.numel() for p in production_adapter.parameters())
        passed = (zero_diff == 0 and disabled_diff == 0 and prefix_diff == 0 and replay_diff == 0
                  and target_change > 0 and gradient_match and geometry_gradients and heads_active
                  and base_frozen and base_unchanged and helper_restored)
        installed_source = Path(inspect.getfile(QwenImage21Transformer2DModel))
        return {"schema": "qwen21_native_body_audit_v1", "passed": bool(passed),
                "mode": "small-random-native-qwen-structure", "num_qwen_layers": 32,
                "hidden_dim": 16, "geometry_dim": geometry_dim, "device": "cpu", "threads": threads,
                "torch_version": torch.__version__, "full_frontend": checkpoint_record,
                "zero_max_abs": zero_diff, "disabled_max_abs": disabled_diff, "prefix_max_abs": prefix_diff,
                "target_max_change": target_change, "checkpoint_forward_max_abs": replay_diff,
                "checkpoint_gradient_max_abs": gradient_max, "checkpoint_gradients_match": bool(gradient_match),
                "geometry_receives_gradients": bool(geometry_gradients), "all_output_heads_have_gradients": bool(heads_active),
                "base_parameters_frozen": base_frozen, "base_weights_unchanged": base_unchanged,
                "checkpoint_context_restored": helper_restored, "installation": install,
                "production_adapter_parameter_count": production_count,
                "source_sha256": {name: _sha256(ROOT / name) for name in (
                    "src/pose_control/qwen21_full/native_geometry.py", "src/pose_control/qwen21_full/native_body.py",
                    "src/pose_control/qwen21_full/native_integration.py", "scripts/audit/audit_qwen21_native_body.py")},
                "qwen_source": {"path": str(installed_source), "sha256": _sha256(installed_source)},
                "limits": "Engineering validation only; no pretrained 7B sampling, optimizer update, or image-quality claim."}
    finally:
        torch.set_num_threads(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file; refuses overwrite")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new filename")
    report = audit_native_body(full_checkpoint=args.full_checkpoint, threads=args.threads)
    if args.output is not None:
        from scripts.audit.audit_qwen21_native_geometry import write_report
        write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
