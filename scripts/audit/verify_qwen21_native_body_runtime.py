"""Bounded CUDA forward/backward audit using existing Qwen7B and Full weights.

Synthetic latent/text/geometry inputs; no image sampling, optimizer update,
downloads or checkpoint writes. JSON output is exclusive, never overwritten.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify_runtime(*, base_model, full_checkpoint, device="cuda", resolution=512, min_free_gib=40.):
    import torch
    if torch.device(device).type != "cuda":
        raise ValueError("This bounded runtime verification requires CUDA")
    if type(resolution) is not int or resolution < 32 or resolution % 32:
        raise ValueError("resolution must be positive and divisible by 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU structure verification uses audit_qwen21_native_body.py")
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_integration import bind_native_body, install_native_body
    from scripts.audit.audit_qwen21_native_geometry import _sha256, _conditions

    device = torch.device(device)
    free, total = torch.cuda.mem_get_info(device)
    if min_free_gib <= 0 or free < min_free_gib * 1024**3:
        raise RuntimeError(f"Not enough free memory for bounded verification: {free / 1024**3:.2f} GiB")
    started = time.perf_counter()
    torch.manual_seed(20261022)
    torch.cuda.reset_peak_memory_stats(device)
    base = Path(base_model).resolve(strict=True)
    checkpoint = Path(full_checkpoint).resolve(strict=True)
    checkpoint_stat = checkpoint.stat()
    checkpoint_sha = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    frontend = Qwen21NativeGeometry.from_full_checkpoint(payload).eval().requires_grad_(False).to(device)
    # The audit helper's width is variable; make its height match the requested
    # canvas without changing any trained preprocessing convention.
    raw = _conditions(torch, resolution, False)
    if resolution != 32:
        for key in ("normal_a", "pose_heatmap_a", "part_onehot_a", "human_mask_a"):
            raw[key] = raw[key].repeat(1, 1, resolution // 32, 1)
    raw = {key: value.to(device) for key, value in raw.items()}
    grid = resolution // 16
    tokens = grid * grid
    with torch.no_grad():
        geometry = frontend(frontend.condition_injector(**raw), target_token_hw=(grid, grid))
    model = QwenImage21Transformer2DModel.from_pretrained(
        str(base), subfolder="transformer", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    base_parameters = dict(model.named_parameters())
    versions = {name: parameter._version for name, parameter in base_parameters.items()}
    text_length = tokens // 4 + 4
    img_mask = torch.zeros(1, text_length + tokens // 4, device=device, dtype=torch.bool)
    img_mask[:, 2:2 + tokens // 4] = True
    img_mask[:, text_length:] = True
    inputs = dict(
        hidden_states=torch.randn(1, tokens * 2, 64, device=device, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, text_length, 4096, device=device, dtype=torch.bfloat16),
        encoder_hidden_states_mask=torch.ones(1, text_length, device=device, dtype=torch.bool),
        timestep=torch.tensor([.5], device=device), img_shapes=[[(1, grid, grid)] * 2],
        img_mask=img_mask, return_dict=False,
    )
    with torch.no_grad():
        baseline = model(**inputs)[0].detach()
    adapter = Qwen21NativeBody().to(device)
    installation = install_native_body(model, adapter)
    with torch.no_grad(), bind_native_body(model, geometry):
        zero = model(**inputs)[0]
    zero_diff = (zero.float() - baseline.float()).abs().max().item()
    prefix_diff = (zero[:, :-tokens].float() - baseline[:, :-tokens].float()).abs().max().item()
    model.enable_gradient_checkpointing()
    # A single gradient probe, with fresh zero heads. The input is synthetic and
    # this scalar is not a training/quality metric. No optimizer is constructed.
    with bind_native_body(model, geometry):
        prediction = model(**inputs)[0][:, -tokens:]
    prediction.float().square().mean().backward()
    gradients = [None if head.weight.grad is None else float(head.weight.grad.float().norm()) for head in adapter.output_heads]
    active = all(value is not None and value > 0 for value in gradients)
    finite = bool(torch.isfinite(prediction).all())
    frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in base_parameters.values())
    unchanged = all(parameter._version == versions[name] for name, parameter in base_parameters.items())
    after = checkpoint.stat()
    checkpoint_unchanged = ((checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ctime_ns)
                            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns))
    clean = model._native_body_execution.get() is None and model._native_body_binding.get() is None
    torch.cuda.synchronize(device)
    return {
        "schema": "qwen21_native_body_runtime_v1", "passed": bool(zero_diff == 0 and prefix_diff == 0
            and active and finite and frozen and unchanged and checkpoint_unchanged and clean),
        "base_model": str(base), "base_config_sha256": _sha256(base / "transformer/config.json"),
        "full_checkpoint": str(checkpoint), "full_checkpoint_sha256": checkpoint_sha,
        "checkpoint_stat_unchanged": checkpoint_unchanged, "resolution": resolution,
        "target_tokens": tokens, "qwen_layers": len(model.transformer_blocks),
        "qwen_hidden_dim": model.inner_dim, "dtype": "bfloat16 base / float32 adapter",
        "zero_max_abs": zero_diff, "prefix_max_abs": prefix_diff,
        "all_output_heads_have_gradients": active, "output_head_gradient_norms": gradients,
        "base_parameters_frozen": frozen, "base_parameter_versions_unchanged": unchanged,
        "finite_prediction": finite, "execution_context_clean": clean,
        "frontend_inheritance": frontend.inheritance_report, "installation": installation,
        "gpu": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "free_gib_before": free / 1024**3, "total_gib": total / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "seconds": time.perf_counter() - started,
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "src/pose_control/qwen21_full/native_geometry.py", "src/pose_control/qwen21_full/native_body.py",
            "src/pose_control/qwen21_full/native_integration.py", "scripts/audit/verify_qwen21_native_body_runtime.py")},
        "qwen_source_sha256": _sha256(Path(inspect.getfile(QwenImage21Transformer2DModel))),
        "limits": "Synthetic inputs; forward/backward engineering probe only. No image quality, trained control, optimizer update or speedup claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--full-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--min-free-gib", type=float, default=40.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new filename")
    report = verify_runtime(base_model=args.base_model, full_checkpoint=args.full_checkpoint,
                            device=args.device, resolution=args.resolution, min_free_gib=args.min_free_gib)
    if args.output is not None:
        from scripts.audit.audit_qwen21_native_geometry import write_report
        write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
