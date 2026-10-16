"""方案 B 工程门禁：有界显存探针、真实更新恢复、overfit6 可学习性。"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import TrainingCursor, capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from .training import (balanced_sample_order, configure_determinism, flow_loss, make_lr_scheduler, make_optimizer,
                       make_training_schedule, run_training_updates, TrainingResourceError)
from .weights import (EXPECTED_LORA_PARAMETERS, frozen_parameter_digest, load_lora_state_dict,
                      lora_state_dict)


def _clean(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _progress(stage, event):
    clean = {key: value for key, value in event.items() if key not in ("cursor", "trajectory")}
    print(json.dumps({"stage": stage, **_clean(clean)}, ensure_ascii=False, allow_nan=False), flush=True)


def _logs(run_dir, prefix):
    path = Path(run_dir) / f"{prefix}.jsonl"
    with path.open("x", encoding="utf-8"):
        pass
    def callback(event):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_clean(event), ensure_ascii=False, allow_nan=False) + "\n")
        _progress(prefix, event)
    return callback


def overfit_sample_order(workspace):
    from .runner import read_jsonl
    root = Path(workspace) / "data"
    conditions = read_jsonl(root / "overfit6_conditions.jsonl")
    supervision = read_jsonl(root / "overfit6_supervision.jsonl")
    ids = [record["sample_id"] for record in conditions]
    if len(ids) != 6 or len(set(ids)) != 6 or Counter(row["dataset"] for row in conditions) != {"iper": 2, "tiktok": 2, "hi4d": 2}:
        raise ValueError("overfit6 requires exactly two unique samples per training dataset")
    if any(record.get("split") != "train" for record in conditions):
        raise ValueError("overfit6 must use train conditions only")
    if len(supervision) != 6 or {record["sample_id"] for record in supervision} != set(ids):
        raise ValueError("overfit6 supervision must match its six condition ids exactly")
    return balanced_sample_order({dataset: [row["sample_id"] for row in conditions if row["dataset"] == dataset]
                                  for dataset in ("iper", "tiktok", "hi4d")})


def _gate_metadata(metadata, order, kind, accumulation):
    result = dict(metadata)
    trajectory = {"order": order, "train_seed": 3407, "sampler_seed": 20261015,
                  "accumulation_steps": accumulation, "kind": kind,
                  "lr": 1e-4, "warmup_updates": 32}
    result.update(run_kind=kind, training_protocol=trajectory,
                  trajectory_sha256=hashlib.sha256(json.dumps(trajectory, sort_keys=True).encode()).hexdigest())
    return result


def evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, *, device="cuda"):
    if len(sample_order) != 6 or len({sample_id for _, sample_id in sample_order}) != 6:
        raise ValueError("Fixed loss gate requires six distinct samples")
    was_training = model.training
    model.eval()
    losses, schedules = [], {}
    device = torch.device(device)
    try:
        with torch.no_grad():
            for dataset, sample_id in sample_order:
                condition, supervision = load_sample(sample_id)
                target = supervision["target_latent"]
                generator = torch.Generator(device=device).manual_seed(43)
                noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
                tokens = target.shape[1]
                if tokens not in schedules:
                    schedules[tokens] = make_training_schedule(scheduler, tokens)
                for index in (250, 500, 750):
                    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                    with autocast:
                        loss = flow_loss(model, condition, supervision, index, noise, schedules[tokens], device)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite fixed evaluation flow loss")
                    losses.append({"dataset": dataset, "sample_id": sample_id, "index": index,
                                   "noise_seed": 43, "loss": float(loss)})
    finally:
        model.train(was_training)
    return {"mean_loss": sum(item["loss"] for item in losses) / 18, "losses": losses}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _state_signature(value):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:{item.stride()}".encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())
    add(value)
    return digest.hexdigest()


def _start_state(model, optimizer, lr):
    return {"lora_sha256": _state_signature(lora_state_dict(model)),
            "optimizer_sha256": _state_signature(optimizer.state_dict()),
            "lr_scheduler_sha256": _state_signature(lr.state_dict()),
            "rng_sha256": _state_signature(capture_rng_state()),
            "optimizer_state_entries": len(optimizer.state),
            "lr": [group["lr"] for group in optimizer.param_groups]}


def state_comparison_report(left, right, path="state", *, atol=1e-6, rtol=1e-5):
    """比较全部 CPU 状态后返回报告，不在第一个不匹配的 tensor 提前终止。"""
    report = {"passed": True, "atol": atol, "rtol": rtol, "compared_tensors": 0,
              "total_elements": 0, "mismatched_tensors": 0, "mismatched_elements": 0,
              "max_abs_error": 0., "max_rel_error": 0., "worst_tensor": None,
              "relative_denominator_floor": 1e-30, "tensor_differences": [], "structure_differences": []}
    def structure(where, reason):
        report["passed"] = False
        report["structure_differences"].append({"path": where, "reason": reason})
    def visit(a, b, where):
        if isinstance(a, torch.Tensor):
            report["compared_tensors"] += 1
            report["total_elements"] += a.numel()
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                report["mismatched_tensors"] += 1
                structure(where, "tensor shape/dtype/type differs")
                return
            aa, bb = a.detach().cpu().double(), b.detach().cpu().double()
            if not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
                report["mismatched_tensors"] += 1
                structure(where, "nonfinite tensor values")
                return
            difference = (aa - bb).abs()
            maximum = float(difference.max()) if difference.numel() else 0.
            relative = float((difference / bb.abs().clamp_min(1e-30)).max()) if difference.numel() else 0.
            mismatches = int((~torch.isclose(aa, bb, atol=atol, rtol=rtol)).sum())
            if maximum > report["max_abs_error"]:
                report["max_abs_error"], report["worst_tensor"] = maximum, where
            report["max_rel_error"] = max(report["max_rel_error"], relative)
            if mismatches:
                report["passed"] = False
                report["mismatched_tensors"] += 1
                report["mismatched_elements"] += mismatches
                report["tensor_differences"].append({"path": where, "shape": list(a.shape),
                    "max_abs_error": maximum, "max_rel_error": relative, "mismatched_elements": mismatches})
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                structure(where, "dictionary keys/type differ")
            if isinstance(b, dict):
                for key in a.keys() & b.keys():
                    visit(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, (tuple, list)):
            if not isinstance(b, (tuple, list)) or len(a) != len(b):
                structure(where, "sequence length/type differs")
            if isinstance(b, (tuple, list)):
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{where}.{index}")
        elif a != b:
            structure(where, f"scalar differs: {a!r} != {b!r}")
    visit(left, right, path)
    return report


def check_resume_equivalence(model, sample_order, load_sample, scheduler, run_dir, metadata,
                             *, device="cuda", expected_parameters=EXPECTED_LORA_PARAMETERS):
    """同一已加载模型顺序重置，避免两个完整基座同时驻留 GPU。"""
    run_dir = Path(run_dir)
    initial = lora_state_dict(model)
    rng = capture_rng_state()
    starts = {}
    metadata = _gate_metadata(metadata, sample_order, "smoke_resume", 3)
    def fresh():
        load_lora_state_dict(model, initial)
        optimizer = make_optimizer(model, expected_parameters=expected_parameters)
        return optimizer, make_lr_scheduler(optimizer)
    optimizer, lr = fresh()
    restore_rng_state(rng)
    starts["continuous"] = _start_state(model, optimizer, lr)
    _write(run_dir / "resume_start_continuous.json", starts["continuous"])
    continuous = run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
        sample_order=sample_order, load_sample=load_sample, updates=4, repeat_samples=True,
        device=device, expected_parameters=expected_parameters, on_update=_logs(run_dir, "continuous_updates"))
    full_lora, full_optimizer, full_lr = lora_state_dict(model), _cpu_tree(optimizer.state_dict()), _cpu_tree(lr.state_dict())
    save_checkpoint(run_dir / "continuous_checkpoint_000004", model, optimizer, lr,
                    metadata=metadata, cursor=continuous["cursor"])
    optimizer, lr = fresh()
    gc.collect()
    restore_rng_state(rng)
    starts["split_first"] = _start_state(model, optimizer, lr)
    _write(run_dir / "resume_start_split_first.json", starts["split_first"])
    if starts["continuous"] != starts["split_first"]:
        _write(run_dir / "resume_start_states.json", starts)
        raise AssertionError("resume comparison fresh-start LoRA/optimizer/LR/RNG fingerprints differ")
    first = run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
        sample_order=sample_order, load_sample=load_sample, updates=2, repeat_samples=True,
        device=device, expected_parameters=expected_parameters, on_update=_logs(run_dir, "split_first_updates"))
    starts["saved_step2"] = _start_state(model, optimizer, lr)
    saved = save_checkpoint(run_dir / "resume_checkpoint_000002", model, optimizer, lr,
                             metadata=metadata, cursor=first["cursor"])
    optimizer, lr = fresh()
    restored_cursor = load_checkpoint(saved, model, optimizer, lr, expected_metadata=metadata)
    starts["restored_step2"] = _start_state(model, optimizer, lr)
    _write(run_dir / "resume_start_states.json", starts)
    if starts["saved_step2"] != starts["restored_step2"]:
        raise AssertionError("resume checkpoint restored LoRA/optimizer/LR/RNG fingerprints differ")
    second = run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
        sample_order=sample_order, load_sample=load_sample, updates=4, cursor=restored_cursor, repeat_samples=True,
        device=device, expected_parameters=expected_parameters, on_update=_logs(run_dir, "split_resumed_updates"))
    final_checkpoint = save_checkpoint(run_dir / "resumed_checkpoint_000004", model, optimizer, lr,
                                       metadata=metadata, cursor=second["cursor"])
    comparisons = {"lora": state_comparison_report(full_lora, lora_state_dict(model), "lora"),
                  "optimizer": state_comparison_report(full_optimizer, _cpu_tree(optimizer.state_dict()), "optimizer"),
                  "lr_scheduler": state_comparison_report(full_lr, _cpu_tree(lr.state_dict()), "lr_scheduler")}
    comparisons["passed"] = all(item["passed"] for item in comparisons.values())
    comparisons["resumed_checkpoint"] = str(final_checkpoint)
    _write(run_dir / "resume_state_comparison.json", comparisons)
    if not comparisons["passed"]:
        raise AssertionError("resume equivalence failed at unchanged atol=1e-6 rtol=1e-5; "
                             f"all tensor errors retained in {run_dir / 'resume_state_comparison.json'}")
    max_lora_error = comparisons["lora"]["max_abs_error"]
    optimizer_error = comparisons["optimizer"]["max_abs_error"]
    fields = ("dataset", "sample_id", "update", "microstep", "index", "noise_seed", "timestep_seed", "sigma", "timestep")
    full_trajectory = [{key: item[key] for key in fields} for item in continuous["trajectory"]]
    split_trajectory = [{key: item[key] for key in fields} for item in first["trajectory"] + second["trajectory"]]
    if full_trajectory != split_trajectory:
        raise ValueError("Continuous/resumed sample, noise or timestep trajectory differs")
    result = {"passed": True, "continuous_updates": 4, "resumed_updates": [2, 2], "microsteps": 12,
              "max_lora_abs_error": max_lora_error, "max_optimizer_abs_error": optimizer_error,
              "atol": 1e-6, "rtol": 1e-5, "resume_checkpoint": str(saved),
              "final_resumed_checkpoint": str(final_checkpoint),
              "trajectory_sha256": hashlib.sha256(json.dumps(full_trajectory, sort_keys=True).encode()).hexdigest()}
    _write(run_dir / "resume_equivalence.json", result)
    return result


def train_overfit_cached(model, sample_order, load_sample, scheduler, run_dir, metadata, *,
                         updates=200, device="cuda", expected_parameters=EXPECTED_LORA_PARAMETERS):
    run_dir = Path(run_dir)
    metadata = _gate_metadata(metadata, sample_order, "overfit6", 1)
    before = evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, device=device)
    _write(run_dir / "fixed_loss_step0.json", before)
    optimizer = make_optimizer(model, expected_parameters=expected_parameters)
    lr = make_lr_scheduler(optimizer)
    save_checkpoint(run_dir / "checkpoint_000000", model, optimizer, lr, metadata=metadata,
                    cursor=TrainingCursor(accumulation_steps=1))
    result = run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
        sample_order=sample_order, load_sample=load_sample, updates=updates, accumulation_steps=1,
        repeat_samples=True, device=device, expected_parameters=expected_parameters,
        on_microstep=_logs(run_dir, "overfit_microsteps"), on_update=_logs(run_dir, "overfit_updates"))
    checkpoint = save_checkpoint(run_dir / f"checkpoint_{updates:06d}", model, optimizer, lr,
                                 metadata=metadata, cursor=result["cursor"])
    after = evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, device=device)
    _write(run_dir / f"fixed_loss_step{updates}.json", after)
    relative_drop = 1. - after["mean_loss"] / before["mean_loss"] if before["mean_loss"] > 0 else 0.
    return {"gate_pass": relative_drop >= .2, "loss_before": before["mean_loss"], "loss_after": after["mean_loss"],
            "relative_loss_drop": relative_drop, "required_relative_loss_drop": .2, "evaluated_positions": 18,
            "optimizer_step": result["cursor"].optimizer_step, "last_checkpoint": str(checkpoint),
            "elapsed_seconds": result["elapsed_seconds"], "max_memory_allocated": result["max_memory_allocated"],
            "max_memory_reserved": result["max_memory_reserved"], "metadata": metadata}


def _cuda_gate():
    deterministic = configure_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke/overfit require real CUDA; CPU contract tests do not satisfy these gates")
    matrix = torch.ones(4, 4, device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    torch.cuda.synchronize()
    if not torch.isfinite(product).all() or not torch.equal(product, torch.full_like(product, 4)):
        raise RuntimeError("Real CUDA/BF16 matrix computation failed")
    return {"device": torch.cuda.get_device_name(), "bf16_computation_passed": True,
            "determinism": deterministic,
            "total_memory": torch.cuda.get_device_properties("cuda").total_memory}


def run_smoke_gates(workspace, run_dir):
    from .runner import initialize_training, load_cached_sample
    run_dir = Path(run_dir)
    device = _cuda_gate()
    order = overfit_sample_order(workspace)
    state = initialize_training(workspace, "joint", run_dir)
    model = state["model"]
    load_sample = lambda sample_id: load_cached_sample(workspace, sample_id)
    frozen_before = frozen_parameter_digest(model)
    initial = lora_state_dict(model)
    try:
        probe = run_training_updates(model, optimizer=state["optimizer"], lr_scheduler=state["lr_scheduler"],
            scheduler=state["scheduler"], sample_order=order, load_sample=load_sample, updates=10,
            accumulation_steps=1, repeat_samples=True, on_microstep=_logs(run_dir, "probe_microsteps"))
        _write(run_dir / "probe.json", probe)
        load_lora_state_dict(model, initial)
        state["optimizer"], state["lr_scheduler"] = None, None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        resume = check_resume_equivalence(model, order, load_sample, state["scheduler"], run_dir, state["metadata"])
        unchanged = frozen_parameter_digest(model) == frozen_before
        if not unchanged:
            raise RuntimeError("Frozen base/Union checksum changed during smoke training")
        return {"gate_pass": True, "cuda": device, "probe_microsteps": 10,
                "probe_seconds": probe["elapsed_seconds"], "max_memory_allocated": probe["max_memory_allocated"],
                "max_memory_reserved": probe["max_memory_reserved"], "resume_equivalence": resume,
                "frozen_unchanged": unchanged, "frozen_sha256": frozen_before, "metadata": state["metadata"]}
    except Exception as error:
        details = error.report if isinstance(error, TrainingResourceError) else {}
        _write(run_dir / "smoke_failure.json", {"error": repr(error), "gate_pass": False, **details})
        raise


def overfit_gate(workspace, run_dir, updates=200):
    if updates != 200:
        raise ValueError("The locked overfit6 gate requires exactly 200 updates")
    from .runner import initialize_training, load_cached_sample
    _cuda_gate()
    order = overfit_sample_order(workspace)
    state = initialize_training(workspace, "joint", run_dir)
    state["optimizer"], state["lr_scheduler"] = None, None
    model = state["model"]
    frozen_before = frozen_parameter_digest(model)
    try:
        result = train_overfit_cached(model, order, lambda sid: load_cached_sample(workspace, sid),
            state["scheduler"], run_dir, state["metadata"], updates=updates)
        result["frozen_unchanged"] = frozen_parameter_digest(model) == frozen_before
        if not result["frozen_unchanged"]:
            raise RuntimeError("Frozen base/Union checksum changed during overfit6")
        return result
    except Exception as error:
        details = error.report if isinstance(error, TrainingResourceError) else {}
        _write(Path(run_dir) / "overfit_failure.json", {"error": repr(error), "gate_pass": False, **details})
        raise
