"""只读核对V4 main200与四个独立25步arm；CUDA噪声SHA须显式同backend重放。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

V4_SIGNATURE = "1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e"
ARMS = ("full4", "no_dense", "no_appearance", "no_contact", "full8")
DRAW_FIELDS = ("sample_id", "update", "microstep", "noise_seed", "timestep_seed", "index")
COMMON_METADATA = ("cache_signature", "data_signature", "source_signature", "code_signature", "task_signature",
    "trajectory_signature", "memory_mode", "training_device", "precision", "noise_backend", "noise_dtype",
    "frozen_dtype", "condition_kind", "train_seed", "sampler_seed", "warmup_updates", "initialization_seed", "control_lora_targets")


def _render_tool():
    path = Path(__file__).with_name("render_pilot.py")
    spec = importlib.util.spec_from_file_location("private_pilot_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_code(code_root):
    return _render_tool().load_frozen_code(code_root)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path):
    return _render_tool().file_sha256(path)


def _product(shape):
    return math.prod(shape)


def _draw(row):
    return {key: row[key] for key in DRAW_FIELDS}


def _cursor(value, step):
    fields = ("optimizer_step", "microstep", "next_sample_position", "noise_draw_cursor", "timestep_draw_cursor",
              "loss_log_cursor", "gradient_log_cursor", "accumulation_steps")
    if (set(value) != set(fields) or any(type(value[key]) is not int for key in fields)
            or value["accumulation_steps"] != 1 or any(value[key] != step for key in fields if key != "accumulation_steps")):
        raise ValueError("Pilot cursor does not prove a complete accumulation1 update boundary")


def _eval(path):
    value = _read(path)
    if (value.get("schema") != "qwen21_pro_fixed18_global_v1" or value.get("noise_seed") != 43
            or value.get("noise_dtype") != "torch.bfloat16" or value.get("indices") != [250, 500, 750]
            or value.get("schedule_steps") != 1000 or len(value.get("entries", [])) != 18):
        raise ValueError(f"Fixed18 protocol differs: {path}")
    entries = value["entries"]
    if len({(e["sample_id"], e["index"]) for e in entries}) != 18 or len({e["sample_id"] for e in entries}) != 6:
        raise ValueError(f"Fixed18 sample/index coverage differs: {path}")
    for row in entries:
        numbers = [row["global_content_mse"], row["regional_total"], *row["losses"].values(), *row["coverage"].values()]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in numbers):
            raise ValueError(f"Nonfinite/negative fixed18 metric: {path}")
        if row["losses"].get("global") != row["global_content_mse"]:
            raise ValueError(f"Fixed18 entry global_content_mse differs from losses.global: {path}")
    for field in ("global_content_mse", "regional_total"):
        if not math.isclose(value[field], sum(e[field] for e in entries) / 18, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Fixed18 aggregate metric differs from entries: {path}")
    return value


def _coverage(evaluation):
    return [(e["sample_id"], e["index"], e["coverage"]) for e in evaluation["entries"]]


def _zero_hash(item):
    if item["dtype"] != "torch.float32":
        raise ValueError("Initial task tensors must be FP32")
    digest = hashlib.sha256(str((tuple(item["shape"]), item["dtype"])).encode())
    remaining = _product(item["shape"]) * 4
    while remaining:
        size = min(remaining, 1 << 20)
        digest.update(bytes(size)); remaining -= size
    return digest.hexdigest()


def _active_elements(names, inventory):
    if len(set(names)) != len(names):
        raise ValueError("Duplicate configured/active parameter name")
    result = 0
    for name in names:
        if not name.startswith("task_adapter.") or name.removeprefix("task_adapter.") not in inventory or name == "task_adapter.gate":
            raise ValueError(f"Unknown/frozen parameter in active coverage: {name}")
        result += _product(inventory[name.removeprefix("task_adapter.")]["shape"])
    return result


def _engineering(gate, arm):
    if (gate.get("engineering_passed") is not True or gate.get("zero_equivalence") is not True
            or gate.get("frozen_unchanged") is not True or gate.get("prefix_unchanged") is not True):
        raise ValueError(f"V4 engineering zero/frozen/prefix gate failed: {arm}")
    engineering = gate.get("engineering_gate", {})
    wanted = {"complete", "cuda", "frozen_unchanged", "prefix_unchanged", "source_verified",
              "zero_equivalence", "exact_resume", "active_branches", "disabled_branches"}
    if (engineering.get("passed") is not True or set(engineering.get("checks", {})) != wanted
            or any(value is not True for value in engineering["checks"].values())):
        raise ValueError(f"V4 engineering checks failed or are incomplete: {arm}")
    disabled = {"appearance", "identity"} if arm == "no_appearance" else {"contacts"} if arm == "no_contact" else set()
    groups = {"heads", "geometry", "contacts", "appearance", "identity"}
    active = groups - disabled
    norms = gate.get("branch_gradient_maxima", {})
    if (set(norms) != groups or any(not math.isfinite(norms[key]) or norms[key] <= 0 for key in active)
            or any(norms[key] != 0 for key in disabled)):
        raise ValueError(f"V4 branch gradient evidence failed: {arm}")
    for name, expected in (("branch_checks", {key: True for key in active}),
                           ("disabled_branch_checks", {key: True for key in disabled}),
                           ("expected_active_branches", sorted(active)), ("disabled_branches", sorted(disabled))):
        if gate.get(name) != expected or engineering.get(name) != expected:
            raise ValueError(f"V4 engineering branch check differs: {arm}: {name}")


def _gate_equal(saved, expected):
    if type(expected) is bool:
        return saved is expected
    if isinstance(expected, dict):
        return isinstance(saved, dict) and saved.keys() == expected.keys() and all(_gate_equal(saved[key], value) for key, value in expected.items())
    if isinstance(expected, (int, float)):
        return type(saved) in (int, float) and math.isfinite(saved) and math.isclose(saved, expected, rel_tol=1e-12, abs_tol=1e-12)
    return saved == expected


def _validate_metric_gate(gate, initial, final, *, updates, device):
    # This is the frozen pure metric gate, not a second implementation of its
    # threshold or completed200/device rules. No CUDA kernel is executed here.
    from src.pose_control.qwen21_pro.gates import overfit_gate
    from src.pose_control.qwen21_pro.runner import source_snapshot
    if source_snapshot()["signature"] != V4_SIGNATURE:
        raise ValueError("Metric gate implementation is outside frozen V4 source bytes")
    expected = overfit_gate(completed_updates=updates, planned_updates=updates, device=device,
        initial_global=initial["global_content_mse"], final_global=final["global_content_mse"],
        fixed_evaluations=len(final["entries"]), frozen_unchanged=gate["frozen_unchanged"], cursor_complete=True,
        source_verified=gate["engineering_gate"]["checks"]["source_verified"], stopped_early=False)
    if not _gate_equal(gate.get("overfit_metric_gate"), expected):
        raise ValueError("Nested overfit metric gate differs from actual fixed18 global losses")
    for key, value in expected.items():
        if key != "passed" and not _gate_equal(gate.get(key), value):
            raise ValueError(f"Top overfit metric gate differs from actual fixed18: {key}")
    if (not _gate_equal(gate.get("initial_global"), initial["global_content_mse"])
            or not _gate_equal(gate.get("final_global"), final["global_content_mse"])):
        raise ValueError("Gate initial/final global loss differs from actual fixed18")
    if gate.get("passed") is not (expected["passed"] and gate["engineering_gate"]["passed"]):
        raise ValueError("Top passed gate differs from actual overfit metric AND engineering gate")


def _complete_run(path, arm):
    path = Path(path)
    expected = 200 if arm == "full4" else 25
    gate = _read(path / "gate.json")
    protocol, initialization = _read(path / "training_protocol.json"), _read(path / "initialization.json")
    metadata = protocol["metadata"]
    _engineering(gate, arm)
    if (gate.get("status") != "complete" or gate["cursor"]["optimizer_step"] != expected
            or protocol["updates"] != expected or protocol["accumulation_steps"] != 1
            or metadata["profile"] != ("overfit" if arm == "full4" else "ablate") or metadata["arm"] != arm):
        raise ValueError(f"Incomplete or wrong training budget/profile: {arm}")
    _cursor(gate["cursor"], expected)
    wanted_optimizer = {"kind": "AdamW", "lr": 1e-4, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01}
    if (protocol["optimizer"] != wanted_optimizer or protocol["gradient_clip"] != 1.
            or metadata["train_seed"] != 3407 or metadata["sampler_seed"] != 20261015
            or metadata["warmup_updates"] != 32 or metadata["initialization_seed"] != 3407
            or metadata["control_lora_targets"] != []):
        raise ValueError("Pilot optimizer/seed/warmup/LoRA protocol differs from registration")
    if any(metadata["config"].get(key) != value for key, value in
           {"hidden_dim": 4096, "width": 512, "heads": 8, "source_dim": 1600}.items()):
        raise ValueError("Pilot effect1600 dimensions differ from registration")
    if (len(protocol["order"]) != 6 or len({row[1] for row in protocol["order"]}) != 6
            or [row[0] for row in protocol["order"]] != ["iper", "tiktok", "hi4d"] * 2):
        raise ValueError("Pilot sample order is not the exact balanced six-example protocol")
    selected = gate["final_checkpoint"]
    final_path = path / f"checkpoint_{expected:06d}"
    final_manifest = _read(final_path / "checkpoint.json")
    if (Path(selected["path"]).resolve() != final_path.resolve() or selected["cursor"] != gate["cursor"]
            or final_manifest["cursor"] != gate["cursor"] or final_manifest["metadata"] != metadata
            or selected["metadata"] != metadata or selected["state_sha256"] != final_manifest["state_sha256"]
            or _sha(final_path / "state.pt") != selected["state_sha256"]
            or _sha(final_path / "checkpoint.json") != selected["manifest_sha256"]):
        raise ValueError("Final checkpoint actual cursor/metadata/state proof differs from gate")
    if gate.get("frozen_sha256") != final_manifest.get("frozen_sha256") or not gate.get("frozen_sha256"):
        raise ValueError("Final checkpoint frozen parameter proof differs from engineering gate")
    if protocol["native_inputs"] != "locked_legacy_union" or metadata["condition_kind"] != "legacy":
        raise ValueError("Controlled pilot cannot substitute current native inputs")
    if metadata["code_signature"] != V4_SIGNATURE or _read(path / "run.json")["code_signature"] != V4_SIGNATURE:
        raise ValueError("Actual training code signature differs from frozen V4")
    if initialization["seed"] != 3407 or initialization.get("resume"):
        raise ValueError("Summary needs the complete original initialization/trajectory, not a missing resumed prefix")
    inventory = initialization["tensor_inventory"]
    configured = gate["configured_trainable"]
    if (set(configured["trainable_names"]) != {f"task_adapter.{key}" for key in inventory if key != "gate"}
            or _active_elements(configured["trainable_names"], inventory) != configured["trainable_parameters"]):
        raise ValueError("Configured parameter element inventory differs")
    micros, updates = _lines(path / "microsteps.jsonl"), _lines(path / "updates.jsonl")
    if len(micros) != expected or len(updates) != expected:
        raise ValueError("Training logs do not cover every complete update")
    observed = set()
    for i, (micro, update) in enumerate(zip(micros, updates)):
        dataset, sid = protocol["order"][i % len(protocol["order"])]
        if (micro["update"] != i or micro["microstep"] != 0 or micro["sample_id"] != sid or micro["dataset"] != dataset
                or micro["position"] != i or update["optimizer_step"] != i + 1
                or len(update["trajectory"]) != 1 or _draw(update["trajectory"][0]) != _draw(micro)):
            raise ValueError(f"Sample/noise/timestep draw or update log differs: {arm}, step{i}")
        _cursor(update["cursor"], i + 1)
        if (micro["configured_names"] != configured["trainable_names"]
                or micro["configured_parameters"] != configured["trainable_parameters"]):
            raise ValueError("Microstep configured parameter inventory differs from gate")
        if not set(micro["active_names"]) <= set(configured["trainable_names"]):
            raise ValueError("Active tensor is not configured trainable")
        if _active_elements(micro["active_names"], inventory) != micro["active_parameters"]:
            raise ValueError("Microstep active tensor element coverage differs")
        observed.update(micro["active_names"])
    if (set(gate["observed_active_names"]) != observed
            or _active_elements(sorted(observed), inventory) != gate["observed_active_parameters"]):
        raise ValueError("Observed active tensor element coverage differs from gate")
    first_active = sorted({name for row in micros[:25] for name in row["active_names"]})
    initial, step25 = _eval(path / "fixed_eval_000000.json"), _eval(path / "fixed_eval_000025.json")
    expected_eval_order = [(dataset, sid, index) for dataset, sid in protocol["order"] for index in (250, 500, 750)]
    if any([(e["dataset"], e["sample_id"], e["index"]) for e in ev["entries"]] != expected_eval_order
           for ev in (initial, step25)):
        raise ValueError("Fixed18 evaluation IDs/order differ from locked sample trajectory")
    if _coverage(initial) != _coverage(step25):
        raise ValueError("Regional supervision coverage changed during training")
    step200 = _eval(path / "fixed_eval_000200.json") if arm == "full4" else None
    if step200 is not None and ([(e["dataset"], e["sample_id"], e["index"]) for e in step200["entries"]] != expected_eval_order
                                or _coverage(step200) != _coverage(initial)):
        raise ValueError("Main200 fixed18 IDs or regional supervision coverage changed")
    _validate_metric_gate(gate, initial, step200 if step200 is not None else step25,
                          updates=expected, device=metadata["training_device"])
    return {"path": str(path.resolve()), "gate": gate, "protocol": protocol, "initialization": initialization,
        "inventory": inventory, "micros": micros, "initial": initial, "step25": step25,
        "step200": step200,
        "configured_parameters": configured["trainable_parameters"], "active_tensor_names": first_active,
        "active_tensor_elements": _active_elements(first_active, inventory)}


def build_summary(run_paths, *, replay_noise=None):
    if set(run_paths) != set(ARMS):
        raise ValueError("Explicit main/full4 plus four retrained arm run paths are required")
    pending = []
    for arm in ARMS:
        path = Path(run_paths[arm])
        expected = 200 if arm == "full4" else 25
        if not (path / "gate.json").is_file():
            pending.append({"arm": arm, "reason": "gate.json missing", "path": str(path)})
            continue
        gate = _read(path / "gate.json")
        if gate.get("status") != "complete" or gate.get("cursor", {}).get("optimizer_step") != expected:
            pending.append({"arm": arm, "reason": "updates incomplete", "cursor": gate.get("cursor"), "path": str(path)})
    if pending:
        return {"status": "pending", "protocol_matched": False, "pending": pending, "quality_claim": False}
    runs = {arm: _complete_run(run_paths[arm], arm) for arm in ARMS}
    main = runs["full4"]
    reference = main["protocol"]
    meta = reference["metadata"]
    common_config = {k: v for k, v in meta["config"].items() if k != "stage_indices"}
    draw_sequence = [_draw(row) for row in main["micros"][:25]]
    for arm, run in runs.items():
        protocol, current = run["protocol"], run["protocol"]["metadata"]
        if any(current[key] != meta[key] for key in COMMON_METADATA):
            raise ValueError(f"Common cache/data/code/runtime/device/precision/noise protocol differs: {arm}")
        if run["gate"]["frozen_sha256"] != main["gate"]["frozen_sha256"]:
            raise ValueError(f"Frozen published weights differ across arms: {arm}")
        if ({k: v for k, v in current["config"].items() if k != "stage_indices"} != common_config
                or current["config"]["stage_indices"] != (list(range(0, 16, 2)) if arm == "full8" else [0, 4, 8, 12])):
            raise ValueError(f"Adapter configuration differs from registered arm: {arm}")
        for key in ("order", "optimizer", "gradient_clip", "native_inputs"):
            if protocol[key] != reference[key]:
                raise ValueError(f"Common training protocol differs: {arm}: {key}")
        if protocol["memory_budget"]["budget_bytes"] != reference["memory_budget"]["budget_bytes"]:
            raise ValueError(f"Fixed memory budget differs: {arm}")
        if [_draw(row) for row in run["micros"][:25]] != draw_sequence:
            raise ValueError(f"First25 recorded sample/noise/timestep sequence differs: {arm}")
        if any(run["inventory"].get(name) != value for name, value in main["inventory"].items()):
            raise ValueError(f"Shared initial tensor SHA/shape/dtype differs: {arm}")
        extras = set(run["inventory"]) - set(main["inventory"])
        expected_extras = {f"output_heads.{i}.{suffix}" for i in (2, 6, 10, 14) for suffix in ("weight", "bias")} if arm == "full8" else set()
        if extras != expected_extras or any(run["inventory"][key]["sha256"] != _zero_hash(run["inventory"][key]) for key in extras):
            raise ValueError(f"Extra full8 heads are not exactly the registered zero initialization: {arm}")
        if run["initial"] != main["initial"] or _coverage(run["step25"]) != _coverage(main["step25"]):
            raise ValueError(f"Initial fixed18 loss or unchanged regional supervision coverage differs: {arm}")
    proof = replay_noise(main["micros"][:25], meta, _read(Path(main["path"]) / "run.json")["config"]) if replay_noise else None
    if proof is not None:
        if (proof.get("hash_source") != "replayed_not_captured" or not proof.get("runtime_verified")
                or not proof.get("target_binding_verified") or proof.get("device") != meta["training_device"]
                or proof.get("noise_dtype") != meta["noise_dtype"] or len(proof.get("draws", [])) != 25):
            raise ValueError("Noise SHA replay lacks the actual runtime/backend/target binding proof")
        for logged, replayed in zip(draw_sequence, proof["draws"]):
            if (_draw(replayed) != logged or len(replayed.get("noise_sha256", "")) != 64
                    or replayed.get("dtype") != meta["noise_dtype"] or len(replayed.get("shape", [])) != 3
                    or replayed["shape"][0] != 1 or replayed["shape"][-1] != 64):
                raise ValueError("Noise replay draw seed/index/shape/dtype/SHA differs from logged stream")
    rows = []
    for arm, run in runs.items():
        last = run["step25"]
        keys = last["entries"][0]["losses"]
        rows.append({"arm": arm, "run": run["path"], "checkpoint_step": 25,
            "actual_config": run["protocol"]["metadata"]["config"], "training_protocol": run["protocol"],
            "fixed18_initial": run["initial"], "fixed18_step25": last,
            "initial_global": run["initial"]["global_content_mse"], "global_step25": last["global_content_mse"],
            "global_relative_drop": 1 - last["global_content_mse"] / run["initial"]["global_content_mse"],
            "regional_total_step25": last["regional_total"],
            "regional_losses_step25": {k: sum(e["losses"][k] for e in last["entries"]) / 18 for k in keys},
            "regional_coverage": {k: sum(e["coverage"][k] for e in last["entries"]) / 18 for k in last["entries"][0]["coverage"]},
            "configured_parameters": run["configured_parameters"], "active_tensor_elements": run["active_tensor_elements"],
            "active_tensor_names": run["active_tensor_names"], "gate": run["gate"]})
    final = main["step200"]
    return {"schema": "pro_private_pilot_summary_v1", "status": "complete" if proof else "pending_noise_proof",
        "protocol_matched": True, "source_inputs": "locked_legacy_B", "source_snapshot_signature": V4_SIGNATURE,
        "common_protocol": {"metadata": {key: meta[key] for key in COMMON_METADATA}, "adapter_common_config": common_config,
            "order": reference["order"], "accumulation_steps": reference["accumulation_steps"], "optimizer": reference["optimizer"],
            "gradient_clip": reference["gradient_clip"], "native_inputs": reference["native_inputs"],
            "fixed_memory_budget_bytes": reference["memory_budget"]["budget_bytes"]},
        "ablations_step25": rows, "main200": {"initial_global": main["initial"]["global_content_mse"],
            "final_global": final["global_content_mse"], "global_relative_drop": 1 - final["global_content_mse"] / main["initial"]["global_content_mse"],
            "fixed18_step0": main["initial"], "fixed18_step200": final,
            "regional_total_initial": main["initial"]["regional_total"], "regional_total_final": final["regional_total"],
            "regional_losses_initial": {key: sum(e["losses"][key] for e in main["initial"]["entries"]) / 18 for key in final["entries"][0]["losses"]},
            "regional_losses_final": {key: sum(e["losses"][key] for e in final["entries"]) / 18 for key in final["entries"][0]["losses"]},
            "regional_coverage": {key: sum(e["coverage"][key] for e in final["entries"]) / 18 for key in final["entries"][0]["coverage"]},
            "gate": main["gate"]}, "noise_proof": proof, "quality_claim": False,
        "legacy_B200": {"relative_global_drop": .177868, "threshold": .2, "passed": False,
                        "source": "此前已登记的旧B200失败边界，本工具不改写其门禁"},
        "limitations": ["训练日志未捕获noise tensor SHA；本报告SHA来自同backend的确定性重放（replayed_not_captured）。",
            "active_tensor_elements是至少出现一个非零梯度的张量所覆盖元素量，不是非零scalar梯度计数。"]}


def replay_draws(frozen, logs, metadata, shape_by_id, *, device):
    torch = frozen.torch
    if torch.device(device) != torch.device(metadata["training_device"]):
        raise ValueError("Replay device/backend differs from actual training metadata")
    dtype = getattr(torch, metadata["noise_dtype"].removeprefix("torch."))
    frozen_dtype = getattr(torch, metadata["frozen_dtype"].removeprefix("torch."))
    protocol = frozen.dataflow.training_execution_protocol(device, frozen_dtype=frozen_dtype, noise_dtype=dtype)
    if any(protocol[key] != metadata[key] for key in protocol):
        raise ValueError("Replay precision/noise backend differs from actual training")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA replay unavailable; CPU substitution is forbidden")
    values = []
    for row in logs:
        shape = tuple(shape_by_id[row["sample_id"]])
        if len(shape) != 3 or shape[0] != 1 or shape[-1] != 64 or any(v <= 0 for v in shape):
            raise ValueError("Replay requires actual bound target grid [1,N,64]")
        draw = frozen.training.training_draw(train_seed=metadata["train_seed"], update=row["update"], microstep=row["microstep"],
            sample_id=row["sample_id"], target_shape=shape, schedule_steps=1000, device=device, dtype=dtype)
        if _draw(draw) != _draw(row):
            raise ValueError("Deterministic replay seed/index differs from recorded training draw")
        values.append({**_draw(draw), "shape": list(shape), "dtype": str(dtype),
                       "noise_sha256": frozen.cache.tensor_sha256(draw["noise"])})
    return {"hash_source": "replayed_not_captured", "device": str(torch.device(device)), "noise_dtype": str(dtype),
            "draws": values}


def _bound_replay(frozen, logs, metadata, config, *, device):
    from types import SimpleNamespace
    args = SimpleNamespace(**config)
    if getattr(args, "kind", "legacy") != "legacy" or getattr(args, "cache_run", None):
        raise ValueError("Noise proof requires the original bound legacy inputs")
    args.sample_id, args.arm = [], "full4"
    rows = frozen.dataflow.records(args)
    cfg = frozen.dataflow.configuration(args)
    samples, _, signature = frozen.dataflow._legacy_samples(args, rows, cfg, with_supervision=False)
    if signature != metadata["source_signature"] or cfg.to_dict() != metadata["config"]:
        raise ValueError("Noise proof actual source/config binding differs from main training")
    feature = Path(args.feature_run)
    if _sha(feature / "report.json") != metadata["cache_signature"]:
        raise ValueError("Noise proof feature report changed")
    data_paths = {"conditions": args.conditions or str(Path(args.legacy_workspace) / "data/overfit6_conditions.jsonl"),
                  "supervision": args.supervision or str(Path(args.legacy_workspace) / "data/overfit6_supervision.jsonl")}
    if frozen.cache.stable_hash(frozen.cache.asset_hashes(data_paths)) != metadata["data_signature"]:
        raise ValueError("Noise proof locked condition/supervision manifests changed")
    shapes = {sid: [1, pack.target_hw[0] * pack.target_hw[1], 64] for sid, (_, _, pack) in samples.items()}
    proof = replay_draws(frozen, logs, metadata, shapes, device=device)
    runtime = frozen.union_runner.model_signature()["runtime"]
    if frozen.torch.device(device).type == "cuda":
        actual_total = frozen.torch.cuda.get_device_properties(device).total_memory
        if any(row["total_memory"] != actual_total for row in logs):
            raise ValueError("Noise replay GPU total memory differs from logged training device")
        proof["cuda_device"] = {"name": frozen.torch.cuda.get_device_name(device), "total_memory": actual_total}
    proof.update(runtime=runtime, runtime_verified=True, target_binding_verified=True,
        condition_bindings={sid: dict(condition.attestation) for sid, (condition, _, _) in samples.items()},
        runtime_training_proof="V4 frozen _legacy_context checks actual runtime against the same pinned legacy producer before every run")
    return proof


def markdown(report):
    if report["status"] == "pending":
        return "# Pro pilot待完成\n\n" + "\n".join(f"- {p['arm']}：{p['reason']}" for p in report["pending"]) + "\n"
    main = report["main200"]
    lines = ["# Pro受控六例pilot与25步消融", "", f"状态：{report['status']}。输入：locked_legacy_B。",
        f"主200步global MSE：{main['initial_global']:.8f} → {main['final_global']:.8f}，下降{main['global_relative_drop']:.4%}；原门禁passed={main['gate']['passed']}。",
        "旧B200步下降17.7868%，仍未通过20%门槛。", "",
        "| arm | step | global MSE | global下降 | regional总loss | 配置参数元素 | 活跃张量覆盖元素 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["ablations_step25"]:
        lines.append(f"| {row['arm']} | 25 | {row['global_step25']:.8f} | {row['global_relative_drop']:.4%} | {row['regional_total_step25']:.8f} | {row['configured_parameters']} | {row['active_tensor_elements']} |")
    lines += ["", "full4的25步取自主200步run；其余为独立重训。分项loss、coverage与张量名见JSON。", "",
        "共同protocol、每arm实际config及全部18个sample×index的loss/coverage保留在report.json；不以聚合值替代原始明细。", "",
        f"主200步regional总loss：{main['regional_total_initial']:.8f} → {main['regional_total_final']:.8f}。", "",
        "| 主200分项 | step0 loss | step200 loss | coverage |", "|---|---:|---:|---:|"]
    for key, loss in main["regional_losses_final"].items():
        lines.append(f"| {key} | {main['regional_losses_initial'][key]:.8f} | {loss:.8f} | {main['regional_coverage'].get(key, 0):.4f} |")
    lines += ["", "| arm | 实际融合stage |", "|---|---|"]
    for row in report["ablations_step25"]:
        lines.append(f"| {row['arm']} | {row['actual_config']['stage_indices']} |")
    lines += ["",
        "训练时未捕获noise张量SHA；SHA仅来自同backend确定性重放，标记replayed_not_captured。缺少重放证明时状态为pending_noise_proof。",
        "活跃张量覆盖元素量不是非零scalar梯度计数；它统计至少出现一个非零梯度的张量的完整元素量。", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1")
    parser.add_argument("--legacy-workspace", default="/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1")
    parser.add_argument("--main-run", required=True)
    for name in ARMS[1:]:
        parser.add_argument(f"--{name.replace('_', '-')}-run", required=True)
    parser.add_argument("--noise-replay-device", choices=("cuda", "cpu"), help="explicit same training backend; no fallback")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    frozen = load_frozen_code(args.code_root)
    run = frozen.runner.allocate_run(args.output_root, args.name, legacy_workspace=args.legacy_workspace)
    helper = _render_tool()
    helper.write_json(run / "run.json", {"argv": vars(args), "actual_src_root": str(frozen.code_root),
        "source_snapshot_signature": frozen.signature, "script_sha256": _sha(__file__),
        "bootstrap_script_sha256": _sha(Path(__file__).with_name("render_pilot.py")), "quality_claim": False})
    started = time.monotonic()
    paths = {"full4": args.main_run, **{arm: getattr(args, f"{arm}_run") for arm in ARMS[1:]}}
    try:
        callback = (lambda logs, meta, config: _bound_replay(frozen, logs, meta, config, device=args.noise_replay_device)) if args.noise_replay_device else None
        report = build_summary(paths, replay_noise=callback)
        report["seconds"] = time.monotonic() - started
        helper.write_json(run / "report.json", report)
        with (run / "report.zh-CN.md").open("x", encoding="utf-8") as stream:
            stream.write(markdown(report))
        print(json.dumps({"status": report["status"], "run_dir": str(run)}, ensure_ascii=False))
        return 0 if report["status"] == "complete" else 2
    except Exception as error:
        helper.write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
