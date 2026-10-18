"""Pro 独立 pilot：固定18 global MSE、200步门禁与可恢复更新边界。"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import torch

from . import checkpoint, training
from .ablations import active_gradient_report, tensor_inventory
from .cache import file_sha256, ordinary_cpu, stable_hash

PROFILES = {"overfit": 200, "ablate": 25, "smoke": 2, "smoke_exact": 4}


def clean(value):
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, *, device="cuda"):
    """每样本新建 seed43 Generator，BF16噪声；global与区域加权总loss分列。"""
    if len(sample_order) != 6 or len({sid for _, sid in sample_order}) != 6:
        raise ValueError("Fixed18 evaluation requires six distinct samples")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    entries, schedules = [], {}
    try:
        with torch.no_grad():
            for dataset, sid in sample_order:
                condition, supervision, pack = load_sample(sid)
                target = supervision["target_latent"]
                noise = torch.randn(target.shape, device=device, dtype=torch.bfloat16,
                    generator=torch.Generator(device=device).manual_seed(43))
                tokens = target.shape[1]
                if tokens not in schedules:
                    schedules[tokens] = training.make_training_schedule(scheduler, tokens, steps=1000)
                for index in (250, 500, 750):
                    with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
                        total, metrics = training.flow_loss(model, condition, supervision, index, noise,
                            schedules[tokens], device, task_condition=pack, return_metrics=True)
                    global_mse = metrics["losses"]["global"]
                    if not math.isfinite(global_mse) or not torch.isfinite(total):
                        raise ValueError("Nonfinite fixed18 loss")
                    entries.append({"dataset": dataset, "sample_id": sid, "index": index,
                        "global_content_mse": global_mse, "regional_total": float(total),
                        "losses": metrics["losses"], "coverage": metrics["coverage"]})
    finally:
        model.train(was_training)
    return {"schema": "qwen21_pro_fixed18_global_v1", "noise_seed": 43,
        "noise_dtype": "torch.bfloat16", "indices": [250, 500, 750], "schedule_steps": 1000,
        "global_content_mse": sum(e["global_content_mse"] for e in entries) / 18,
        "regional_total": sum(e["regional_total"] for e in entries) / 18, "entries": entries}


def overfit_gate(*, completed_updates, planned_updates, device, initial_global, final_global,
                 fixed_evaluations, frozen_unchanged, cursor_complete, source_verified, stopped_early):
    finite = all(isinstance(v, (float, int)) and math.isfinite(v) for v in (initial_global, final_global))
    drop = (1 - final_global / initial_global) if finite and initial_global > 0 else None
    checks = {"completed200": completed_updates == planned_updates == 200,
        "cuda": torch.device(device).type == "cuda", "fixed18": fixed_evaluations == 18,
        "frozen_unchanged": bool(frozen_unchanged), "cursor_complete": bool(cursor_complete),
        "source_verified": bool(source_verified), "not_stopped": not stopped_early,
        "global_drop_at_least20pct": drop is not None and drop >= .2 - 1e-12}
    return {"passed": all(checks.values()), "checks": checks, "relative_global_drop": drop,
            "threshold": .2, "metric": "global_content_mse", "quality_gate": "overfit200"}


def engineering_gate(*, complete, device, frozen_unchanged, prefix_unchanged, source_verified,
                     zero_equivalence, profile, exact_resume, arm, branch_maxima):
    disabled = ({"appearance", "identity"} if arm == "no_appearance" else
                {"contacts"} if arm == "no_contact" else set())
    expected = set(branch_maxima) - disabled
    branch_checks = {key: math.isfinite(branch_maxima[key]) and branch_maxima[key] > 0 for key in expected}
    disabled_checks = {key: branch_maxima[key] == 0 for key in disabled}
    checks = {"complete": bool(complete), "cuda": torch.device(device).type == "cuda",
        "frozen_unchanged": bool(frozen_unchanged), "prefix_unchanged": bool(prefix_unchanged),
        "source_verified": bool(source_verified), "zero_equivalence": bool(zero_equivalence),
        "exact_resume": profile != "smoke_exact" or exact_resume is True,
        "active_branches": all(branch_checks.values()), "disabled_branches": all(disabled_checks.values())}
    return {"passed": all(checks.values()), "checks": checks, "branch_checks": branch_checks,
        "disabled_branch_checks": disabled_checks, "expected_active_branches": sorted(expected),
        "disabled_branches": sorted(disabled)}


def _equal(left, right):
    import numpy as np
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def probe_zero_equivalence(model, sample, *, device="cuda"):
    condition, supervision, pack = sample
    noise = training.training_draw(train_seed=3407, update=0, microstep=0, sample_id="zero_equivalence_probe",
        target_shape=supervision["target_latent"].shape, device=device, dtype=torch.bfloat16)["noise"]
    frozen = next((p for p in model.parameters() if not p.requires_grad and p.is_floating_point()), None)
    dtype = frozen.dtype if frozen is not None else noise.dtype
    inputs = training.transformer_inputs(condition, noise, torch.tensor([.5]), device=device, dtype=dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else torch.no_grad():
        base = ordinary_cpu(model(**inputs, task_condition=pack, task_enabled=False, return_dict=False)[0])
        initial = ordinary_cpu(model(**inputs, task_condition=pack, return_dict=False)[0])
    if not torch.equal(base, initial):
        raise ValueError("Pro zero task branch differs from frozen published Union")
    return inputs, base


def verify_reference_prefix(model, inputs, pack, base):
    device = inputs["hidden_states"].device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        actual = ordinary_cpu(model(**inputs, task_condition=pack, return_dict=False)[0])
    count = pack.target_hw[0] * pack.target_hw[1]
    if not torch.equal(base[:, :-count], actual[:, :-count]):
        raise ValueError("Trained Pro task modified causal reference prefix")
    return True


def run_pilot(model, *, scheduler, load_sample, sample_order, metadata, run_dir, profile="overfit",
              device="cuda", resume=None, max_memory_bytes=None, stop_requested=None,
              require_verified_conditions=True, sample_validator=None):
    """受控overfit/消融均accum1；正式1536例训练另设协议，不从本门禁启动。"""
    if profile not in PROFILES:
        raise ValueError("Unknown Pro pilot profile")
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    updates = PROFILES[profile]
    accumulation = 3 if profile.startswith("smoke") else 1
    optimizer = training.make_optimizer(model, control_lora_targets=metadata["control_lora_targets"])
    lr = training.make_lr_scheduler(optimizer, warmup_updates=32)
    frozen = training.frozen_parameter_digest(model)
    identities = {n: id(p) for n, p in model.named_parameters()}
    initial_state = tensor_inventory(model.task_adapter)
    first_sample = load_sample(sample_order[0][1])
    probe_inputs, probe_base = probe_zero_equivalence(model, first_sample, device=device)
    initial = evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, device=device)
    write_json(run / "fixed_eval_000000.json", initial)
    cursor = checkpoint.TrainingCursor(accumulation_steps=accumulation)
    if resume is not None:
        cursor = checkpoint.load_checkpoint(resume, model, optimizer, lr, expected_metadata=metadata)
    write_json(run / "initialization.json", {"tensor_inventory": initial_state, "seed": 3407,
        "baseline_global": initial["global_content_mse"], "resume": str(resume) if resume else None})
    saved = set()
    def save(at):
        if at.optimizer_step not in saved:
            checkpoint.save_checkpoint(run / f"checkpoint_{at.optimizer_step:06d}", model, optimizer, lr,
                metadata=metadata, cursor=at)
            saved.add(at.optimizer_step)
    save(cursor)
    logs = {key: run / f"{key}.jsonl" for key in ("microsteps", "updates", "resume_verification")}
    for path in logs.values():
        with path.open("x", encoding="utf-8"):
            pass
    active_names, active_sizes = set(), {}
    prefixes = {"heads": "output_heads.", "geometry": "geometry_encoder.projections.",
                "contacts": "geometry_encoder.contacts.", "appearance": "banks.source_", "identity": "banks.identity_projection."}
    branch_maxima = {key: 0. for key in prefixes}
    def append(kind, value):
        with logs[kind].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + "\n")
    def micro(event):
        report = active_gradient_report(model)
        active_names.update(report["active_names"])
        for n, p in model.named_parameters():
            if n in active_names:
                active_sizes[n] = p.numel()
        norms = {key: sum(float(p.grad.detach().double().square().sum())
            for name, p in model.task_adapter.named_parameters() if name.startswith(prefix) and p.grad is not None) ** .5
            for key, prefix in prefixes.items()}
        for key, norm in norms.items():
            branch_maxima[key] = max(branch_maxima[key], norm)
        append("microsteps", {**event, **report, "branch_gradient_norms": norms})
    evaluations = {0: initial}
    def update(event):
        append("updates", event)
        step = event["optimizer_step"]
        if step % 25 == 0 or step == updates or (profile == "smoke_exact" and step == 2):
            save(event["cursor"])
        if step in {25, 200}:
            evaluations[step] = evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, device=device)
            write_json(run / f"fixed_eval_{step:06d}.json", evaluations[step])
    stop = stop_requested or (lambda: (run / "STOP").exists())
    result = training.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
        sample_order=sample_order, load_sample=load_sample, updates=updates, device=device, train_seed=3407,
        accumulation_steps=accumulation, cursor=cursor, repeat_samples=True, max_memory_bytes=max_memory_bytes,
        control_lora_targets=metadata["control_lora_targets"], on_microstep=micro, on_update=update,
        stop_requested=stop, require_verified_conditions=require_verified_conditions, sample_validator=sample_validator)
    cursor = result["cursor"]
    save(cursor)
    exact = None
    if profile == "smoke_exact" and cursor.optimizer_step == 4 and resume is None:
        reference = ordinary_cpu({"task": model.task_adapter.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr.state_dict()})
        rng = checkpoint.capture_rng_state()
        restored = checkpoint.load_checkpoint(run / "checkpoint_000002", model, optimizer, lr, expected_metadata=metadata)
        tail = training.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
            sample_order=sample_order, load_sample=load_sample, updates=4, device=device, train_seed=3407,
            accumulation_steps=accumulation, cursor=restored, repeat_samples=True, max_memory_bytes=max_memory_bytes,
            control_lora_targets=metadata["control_lora_targets"], on_microstep=lambda e: append("resume_verification", e),
            require_verified_conditions=require_verified_conditions, sample_validator=sample_validator)
        current = {"task": model.task_adapter.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr.state_dict()}
        exact = (_equal(reference, current) and _equal(rng, checkpoint.capture_rng_state())
                 and result["trajectory"][2 * accumulation:] == tail["trajectory"] and cursor == tail["cursor"]
                 and identities == {n: id(p) for n, p in model.named_parameters()})
        if not exact:
            raise ValueError("Pro 4=2+2 exact resume differs")
    frozen_unchanged = frozen == training.frozen_parameter_digest(model)
    if not frozen_unchanged:
        raise ValueError("Published frozen parameters changed during Pro pilot")
    prefix_unchanged = verify_reference_prefix(model, probe_inputs, first_sample[2], probe_base)
    final = evaluations.get(cursor.optimizer_step)
    if final is None:
        final = evaluate_fixed_overfit_loss(model, sample_order, load_sample, scheduler, device=device)
        write_json(run / f"fixed_eval_final_{cursor.optimizer_step:06d}.json", final)
    gate = overfit_gate(completed_updates=cursor.optimizer_step, planned_updates=updates, device=device,
        initial_global=initial["global_content_mse"], final_global=final["global_content_mse"],
        fixed_evaluations=len(final["entries"]), frozen_unchanged=frozen_unchanged,
        cursor_complete=cursor.validate() is cursor, source_verified=require_verified_conditions,
        stopped_early=result["stopped_early"])
    complete = cursor.optimizer_step == updates
    engineering = engineering_gate(complete=complete, device=device, frozen_unchanged=frozen_unchanged,
        prefix_unchanged=prefix_unchanged, source_verified=require_verified_conditions, zero_equivalence=True,
        profile=profile, exact_resume=exact, arm=metadata.get("arm", "full4"), branch_maxima=branch_maxima)
    passed = gate["passed"] and engineering["passed"]
    final_path = (run / f"checkpoint_{cursor.optimizer_step:06d}").resolve()
    final_manifest = json.loads((final_path / "checkpoint.json").read_text(encoding="utf-8"))
    if (final_manifest["cursor"] != asdict(cursor) or final_manifest["metadata"] != clean(metadata)
            or file_sha256(final_path / "state.pt") != final_manifest["state_sha256"]):
        raise ValueError("Final pilot checkpoint cursor/metadata/state SHA256 differs")
    selected_checkpoint = {"path": str(final_path), "cursor": final_manifest["cursor"],
        "metadata": final_manifest["metadata"], "state_sha256": final_manifest["state_sha256"],
        "manifest_sha256": file_sha256(final_path / "checkpoint.json"),
        "selection_role": "selected_stage1" if passed else "engineering_probe"}
    report = {**gate, "passed": passed, "overfit_metric_gate": gate, "engineering_gate": engineering,
        "final_checkpoint": selected_checkpoint, "profile": profile, "status": "complete" if complete else "incomplete",
        "cursor": asdict(cursor), "engineering_only": profile != "overfit", "exact_resume": exact,
        "engineering_passed": engineering["passed"],
        "zero_equivalence": True, "prefix_unchanged": prefix_unchanged, "branch_gradient_maxima": branch_maxima,
        **{key: engineering[key] for key in ("expected_active_branches", "disabled_branches", "branch_checks", "disabled_branch_checks")},
        "frozen_unchanged": frozen_unchanged, "frozen_sha256": frozen,
        "configured_trainable": training.audit_trainable_parameters(model, control_lora_targets=metadata["control_lora_targets"]),
        "observed_active_names": sorted(active_names), "observed_active_parameters": sum(active_sizes.values()),
        "initial_global": initial["global_content_mse"], "final_global": final["global_content_mse"],
        "max_memory_allocated": result["max_memory_allocated"], "max_memory_reserved": result["max_memory_reserved"]}
    write_json(run / "gate.json", report)
    return report
