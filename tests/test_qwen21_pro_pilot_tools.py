"""实验私有工具的CPU契约；JSON fixture不代表GPU训练或生成结果。"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "experiments/qwen21_pro_adapter_v1/scripts"
V4 = "1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e"
ARMS = ("full4", "no_dense", "no_appearance", "no_contact", "full8")


def tool(name):
    path = TOOLS / f"{name}.py"
    assert path.is_file(), f"Private pilot tool missing: {path.name}"
    spec = importlib.util.spec_from_file_location(f"private_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pilot_runs(tmp_path):
    from src.pose_control.qwen21_pro.gates import overfit_gate
    order = [[d, f"{d}{i}"] for i in range(2) for d in ("iper", "tiktok", "hi4d")]
    paths = {}
    coverage = {"global": 1., "face": .2, "hands": .1}
    def cursor(step):
        return {"optimizer_step": step, "microstep": step, "next_sample_position": step, "noise_draw_cursor": step,
            "timestep_draw_cursor": step, "loss_log_cursor": step, "gradient_log_cursor": step, "accumulation_steps": 1}
    def evaluation(value):
        entries = [{"dataset": dataset, "sample_id": sid, "index": index,
                    "global_content_mse": value, "regional_total": value * 2,
                    "losses": {"global": value, "face": value * 1.1, "hands": value * 1.2},
                    "coverage": coverage} for dataset, sid in order for index in (250, 500, 750)]
        return {"schema": "qwen21_pro_fixed18_global_v1", "noise_seed": 43, "noise_dtype": "torch.bfloat16",
            "indices": [250, 500, 750], "schedule_steps": 1000, "entries": entries,
            "global_content_mse": value, "regional_total": value * 2}
    for arm in ARMS:
        run = tmp_path / arm
        run.mkdir()
        count = 200 if arm == "full4" else 25
        stages = list(range(0, 16, 2)) if arm == "full8" else [0, 4, 8, 12]
        cfg = {"hidden_dim": 4096, "width": 512, "heads": 8, "source_dim": 1600, "stage_indices": stages}
        metadata = {"config": cfg, "arm": arm, "profile": "overfit" if arm == "full4" else "ablate",
            "cache_signature": "cache", "data_signature": "data", "source_signature": "source",
            "code_signature": V4, "task_signature": "regional_default", "trajectory_signature": "shared_stream",
            "memory_mode": "resident", "training_device": "cuda", "frozen_dtype": "torch.bfloat16",
            "noise_dtype": "torch.bfloat16", "precision": "cuda_torch.bfloat16_frozen_bfloat16_autocast_fp32_trainables",
            "noise_backend": "torch.randn_cuda_torch.bfloat16_generator",
            "condition_kind": "legacy", "train_seed": 3407, "sampler_seed": 20261015,
            "warmup_updates": 32, "initialization_seed": 3407, "control_lora_targets": []}
        protocol = {"metadata": metadata, "order": order, "updates": count, "accumulation_steps": 1,
            "optimizer": {"kind": "AdamW", "lr": .0001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01},
            "gradient_clip": 1., "native_inputs": "locked_legacy_union", "memory_budget": {"budget_bytes": 32 * 1024**3}}
        inventory = {"query_projection.weight": {"shape": [2, 2], "dtype": "torch.float32", "sha256": "a" * 64},
                     "gate": {"shape": [], "dtype": "torch.float32", "sha256": "g" * 64}}
        for stage in stages:
            for suffix, shape in (("weight", [2, 2]), ("bias", [2])):
                header = str((tuple(shape), "torch.float32")).encode()
                elements = 1
                for size in shape:
                    elements *= size
                inventory[f"output_heads.{stage}.{suffix}"] = {"shape": shape, "dtype": "torch.float32",
                    "sha256": hashlib.sha256(header + bytes(elements * 4)).hexdigest()}
        names = [f"task_adapter.{n}" for n in inventory if n != "gate"]
        active = ["task_adapter.query_projection.weight", "task_adapter.output_heads.0.weight"]
        total = 4 + 6 * len(stages)
        micro, updates = [], []
        for i in range(count):
            dataset, sid = order[i % 6]
            draw = {"dataset": dataset, "sample_id": sid, "update": i, "microstep": 0, "index": i % 1000,
                    "noise_seed": i + 100, "timestep_seed": i + 200, "position": i,
                    "sigma": .5, "timestep": 500., "active_names": active, "active_parameters": 8,
                    "configured_names": names, "configured_parameters": total, "total_memory": 89_171_099_648}
            micro.append(draw)
            updates.append({"optimizer_step": i + 1, "cursor": cursor(i + 1), "loss": 1. / (i + 1), "trajectory": [draw]})
        (run / "microsteps.jsonl").write_text("\n".join(map(json.dumps, micro)) + "\n")
        (run / "updates.jsonl").write_text("\n".join(map(json.dumps, updates)) + "\n")
        checkpoint = run / f"checkpoint_{count:06d}"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(f"JSON contract fixture {arm}".encode())
        state_sha = hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest()
        write(checkpoint / "checkpoint.json", {"metadata": metadata, "cursor": cursor(count), "state_sha256": state_sha,
                                              "frozen_sha256": "f" * 64})
        selected = {"path": str(checkpoint.resolve()), "metadata": metadata, "cursor": cursor(count),
                    "state_sha256": state_sha, "manifest_sha256": hashlib.sha256((checkpoint / "checkpoint.json").read_bytes()).hexdigest()}
        gate = {"status": "complete", "profile": metadata["profile"], "cursor": cursor(count), "final_checkpoint": selected,
            "passed": False, "engineering_passed": True,
            "configured_trainable": {"trainable_names": names, "trainable_parameters": total},
            "observed_active_names": active, "observed_active_parameters": 8,
            "overfit_metric_gate": {"passed": False, "relative_global_drop": .15 if arm == "full4" else .05}}
        disabled = {"appearance", "identity"} if arm == "no_appearance" else {"contacts"} if arm == "no_contact" else set()
        norms = {name: 0. if name in disabled else 1. for name in ("heads", "geometry", "contacts", "appearance", "identity")}
        active_branches = sorted(set(norms) - disabled)
        branch_checks = {key: True for key in active_branches}
        disabled_checks = {key: True for key in disabled}
        gate.update(zero_equivalence=True, frozen_unchanged=True, prefix_unchanged=True, frozen_sha256="f" * 64,
            branch_gradient_maxima=norms, expected_active_branches=active_branches, disabled_branches=sorted(disabled),
            branch_checks=branch_checks, disabled_branch_checks=disabled_checks,
            engineering_gate={"passed": True, "checks": {key: True for key in ("complete", "cuda", "frozen_unchanged",
                "prefix_unchanged", "source_verified", "zero_equivalence", "exact_resume", "active_branches", "disabled_branches")},
                "expected_active_branches": active_branches, "disabled_branches": sorted(disabled),
                "branch_checks": branch_checks, "disabled_branch_checks": disabled_checks})
        final_global = .85 if arm == "full4" else .95
        metric_gate = overfit_gate(completed_updates=count, planned_updates=count, device="cuda", initial_global=1.,
            final_global=final_global, fixed_evaluations=18, frozen_unchanged=True, cursor_complete=True,
            source_verified=True, stopped_early=False)
        gate.update(**{key: value for key, value in metric_gate.items() if key != "passed"})
        gate.update(initial_global=1., final_global=final_global, overfit_metric_gate=metric_gate)
        write(run / "gate.json", gate)
        write(run / "run.json", {"code_signature": V4, "config": {"feature_run": "fixture", "conditions": "fixture"}})
        write(run / "training_protocol.json", protocol)
        write(run / "initialization.json", {"seed": 3407, "tensor_inventory": inventory, "resume": None})
        write(run / "fixed_eval_000000.json", evaluation(1.))
        write(run / "fixed_eval_000025.json", evaluation(.95))
        if arm == "full4":
            write(run / "fixed_eval_000200.json", evaluation(.85))
        paths[arm] = run
    return paths


def replay_fixture(logs, metadata, config):
    return {"hash_source": "replayed_not_captured", "runtime_verified": True, "target_binding_verified": True,
        "device": metadata["training_device"], "noise_dtype": metadata["noise_dtype"], "runtime": {"test_fixture": True},
        "draws": [{**{k: row[k] for k in ("sample_id", "update", "microstep", "noise_seed", "timestep_seed", "index")},
                   "shape": [1, 1024, 64], "dtype": metadata["noise_dtype"], "noise_sha256": sha(row["noise_seed"])} for row in logs]}


def test_private_tools_reject_missing_frozen_code_root(tmp_path):
    for name in ("render_pilot", "summarize_pilot"):
        with pytest.raises(ValueError, match="code.root|frozen|源码"):
            tool(name).load_frozen_code(tmp_path)


def test_summary_keeps_failed_main_and_reports_five_step25_arms(tmp_path):
    summary = tool("summarize_pilot")
    result = summary.build_summary(pilot_runs(tmp_path), replay_noise=replay_fixture)
    assert result["status"] == "complete" and result["protocol_matched"]
    assert len(result["ablations_step25"]) == 5
    assert result["ablations_step25"][0]["arm"] == "full4"
    assert result["main200"]["global_relative_drop"] == pytest.approx(.15)
    assert result["main200"]["gate"]["passed"] is False
    assert result["legacy_B200"]["passed"] is False
    assert result["noise_proof"]["hash_source"] == "replayed_not_captured"
    assert result["ablations_step25"][0]["active_tensor_elements"] == 8
    assert "非零scalar" in summary.markdown(result)


@pytest.mark.parametrize("mutation", ["engineering", "zero", "frozen", "prefix", "active_branch", "disabled_branch", "gradient_zero"])
def test_summary_rejects_any_v4_engineering_failure(tmp_path, mutation):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    path = paths["no_contact"] / "gate.json"
    value = json.loads(path.read_text())
    if mutation == "engineering": value["engineering_passed"] = False
    elif mutation == "zero": value["zero_equivalence"] = False
    elif mutation == "frozen": value["frozen_unchanged"] = False
    elif mutation == "prefix": value["prefix_unchanged"] = False
    elif mutation == "active_branch": value["branch_checks"]["heads"] = False
    elif mutation == "disabled_branch": value["disabled_branch_checks"]["contacts"] = False
    else: value["branch_gradient_maxima"]["heads"] = 0.
    write(path, value)
    with pytest.raises(ValueError, match="engineering|branch|gradient|工程|frozen"):
        summary.build_summary(paths, replay_noise=replay_fixture)


def test_summary_preserves_actual_configs_and_complete_regional_fixed18_evidence(tmp_path):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    report = summary.build_summary(paths, replay_noise=replay_fixture)
    assert report["common_protocol"]["optimizer"]["lr"] == .0001
    assert report["common_protocol"]["metadata"]["code_signature"] == V4
    for row in report["ablations_step25"]:
        expected = json.loads((paths[row["arm"]] / "training_protocol.json").read_text())
        assert row["actual_config"] == expected["metadata"]["config"]
        assert row["training_protocol"] == expected
        assert len(row["fixed18_initial"]["entries"]) == len(row["fixed18_step25"]["entries"]) == 18
        assert "face" in row["fixed18_step25"]["entries"][0]["losses"]
        assert "hands" in row["fixed18_step25"]["entries"][0]["coverage"]
    assert len(report["main200"]["fixed18_step0"]["entries"]) == 18
    assert len(report["main200"]["fixed18_step200"]["entries"]) == 18
    assert report["main200"]["regional_total_final"] == pytest.approx(1.7)


@pytest.mark.parametrize("mutation", ["initial", "final", "drop", "nested", "top", "entry_global"])
def test_summary_binds_gate_decision_to_the_actual_fixed18_global_losses(tmp_path, mutation):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    path = paths["full4"] / "gate.json"
    gate = json.loads(path.read_text())
    if mutation == "initial": gate["initial_global"] = 100.
    elif mutation == "final": gate["final_global"] = .01
    elif mutation == "drop": gate["relative_global_drop"] = .99
    elif mutation == "nested": gate["overfit_metric_gate"]["passed"] = True
    elif mutation == "top": gate["passed"] = True
    else:
        file = paths["full4"] / "fixed_eval_000200.json"
        value = json.loads(file.read_text())
        value["entries"][0]["losses"]["global"] = .01
        write(file, value)
    write(path, gate)
    with pytest.raises(ValueError, match="gate|global|metric|loss|门禁"):
        summary.build_summary(paths, replay_noise=replay_fixture)


@pytest.mark.parametrize("mutation", ["draw", "initialization", "code", "memory", "current_native", "coverage", "active_count"])
def test_summary_rejects_protocol_or_stream_or_coverage_drift(tmp_path, mutation):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    run = paths["no_dense"]
    if mutation == "draw":
        lines = (run / "microsteps.jsonl").read_text().splitlines()
        row = json.loads(lines[0]); row["noise_seed"] += 1
        lines[0] = json.dumps(row)
        (run / "microsteps.jsonl").write_text("\n".join(lines) + "\n")
    elif mutation == "initialization":
        value = json.loads((run / "initialization.json").read_text())
        value["tensor_inventory"]["query_projection.weight"]["sha256"] = "b" * 64
        write(run / "initialization.json", value)
    elif mutation in ("code", "memory", "current_native"):
        value = json.loads((run / "training_protocol.json").read_text())
        if mutation == "code": value["metadata"]["code_signature"] = "different"
        if mutation == "memory": value["memory_budget"]["budget_bytes"] += 1024
        if mutation == "current_native": value["native_inputs"] = "current_geometry_encoded"
        write(run / "training_protocol.json", value)
    elif mutation == "coverage":
        value = json.loads((run / "fixed_eval_000025.json").read_text())
        value["entries"][0]["coverage"]["face"] = .9
        write(run / "fixed_eval_000025.json", value)
    else:
        value = json.loads((run / "gate.json").read_text()); value["observed_active_parameters"] = 999
        write(run / "gate.json", value)
    with pytest.raises(ValueError):
        summary.build_summary(paths, replay_noise=replay_fixture)


def test_summary_never_completes_incomplete_runs_or_missing_noise_proof(tmp_path):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    assert summary.build_summary(paths)["status"] == "pending_noise_proof"
    gate = json.loads((paths["full8"] / "gate.json").read_text())
    gate["status"] = "incomplete"; gate["cursor"]["optimizer_step"] = 24
    write(paths["full8"] / "gate.json", gate)
    result = summary.build_summary(paths, replay_noise=replay_fixture)
    assert result["status"] == "pending" and not result["protocol_matched"]


@pytest.mark.parametrize("mutation", ["cursor", "final_checkpoint", "fixed_ids", "optimizer", "micro_configured"])
def test_summary_rejects_forged_complete_or_protocol_evidence(tmp_path, mutation):
    summary = tool("summarize_pilot")
    paths = pilot_runs(tmp_path)
    if mutation == "cursor":
        value = json.loads((paths["no_contact"] / "gate.json").read_text())
        value["cursor"]["noise_draw_cursor"] -= 1
        write(paths["no_contact"] / "gate.json", value)
    elif mutation == "final_checkpoint":
        with (paths["no_contact"] / "checkpoint_000025/state.pt").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "fixed_ids":
        for arm in ARMS:
            for file in paths[arm].glob("fixed_eval_*.json"):
                value = json.loads(file.read_text())
                for entry in value["entries"][:3]: entry["sample_id"] = "other_sample"
                write(file, value)
    elif mutation == "optimizer":
        for arm in ARMS:
            value = json.loads((paths[arm] / "training_protocol.json").read_text())
            value["optimizer"]["lr"] = .0002
            write(paths[arm] / "training_protocol.json", value)
    else:
        path = paths["no_contact"] / "microsteps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["configured_parameters"] += 1
        path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError):
        summary.build_summary(paths, replay_noise=replay_fixture)


def test_replay_hashes_real_cpu_draws_and_refuses_backend_substitution():
    summary = tool("summarize_pilot")
    import torch
    from src.pose_control.qwen21_pro import training, cache, dataflow
    rows = []
    for i in range(3):
        draw = training.training_draw(train_seed=3407, update=i, microstep=0, sample_id="one",
                                     target_shape=(1, 4, 64), device="cpu", dtype=torch.bfloat16)
        rows.append({k: v for k, v in draw.items() if k != "noise"})
    metadata = {**dataflow.training_execution_protocol("cpu"), "train_seed": 3407}
    frozen = SimpleNamespace(training=training, cache=cache, dataflow=dataflow, torch=torch)
    result = summary.replay_draws(frozen, rows, metadata, {"one": [1, 4, 64]}, device="cpu")
    actual = training.training_draw(train_seed=3407, update=0, microstep=0, sample_id="one",
                                   target_shape=(1, 4, 64), device="cpu", dtype=torch.bfloat16)
    assert result["draws"][0]["noise_sha256"] == cache.tensor_sha256(actual["noise"])
    assert result["hash_source"] == "replayed_not_captured"
    with pytest.raises(ValueError, match="backend|device"):
        summary.replay_draws(frozen, rows, {**metadata, "training_device": "cuda"}, {"one": [1, 4, 64]}, device="cpu")


def test_render_routes_all_six_bound_legacy_inputs_to_actual_api(tmp_path, monkeypatch):
    render = tool("render_pilot")
    from src.pose_control.qwen21_pro import runner, dataflow, inference, refinement, cache, contracts, gates
    import torch
    from contextlib import nullcontext
    rows = [{"dataset": d, "sample_id": f"{d}{i}", "split": "train"} for i in range(2) for d in ("iper", "tiktok", "hi4d")]
    cfg = contracts.ProAdapterConfig(source_dim=1600)
    checkpoint = tmp_path / "checkpoint_000200"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"unit fixture")
    metadata = {"config": cfg.to_dict(), "arm": "full4", "profile": "overfit", "code_signature": V4}
    write(checkpoint / "checkpoint.json", {"metadata": metadata, "cursor": {"optimizer_step": 200},
                                          "state_sha256": cache.file_sha256(checkpoint / "state.pt")})
    samples = {r["sample_id"]: (f"bound_{r['sample_id']}", None,
        SimpleNamespace(source_fingerprint=r["sample_id"], provenance={"geometry": {"legacy": "verified"}})) for r in rows}
    seen = []
    monkeypatch.setattr(dataflow, "records", lambda _: rows)
    def loader(args, records, cfg, **kwargs):
        assert kwargs["with_supervision"] is False
        assert args.kind == "legacy" and args.conditions.endswith("overfit6_conditions.jsonl")
        return samples, [], "source"
    monkeypatch.setattr(dataflow, "_legacy_samples", loader)
    monkeypatch.setattr(dataflow, "_inference_pipeline", lambda args, run, **kwargs: (SimpleNamespace(), cfg, metadata))
    def sample(pipe, condition, **kwargs):
        assert condition.startswith("bound_")
        seen.append((kwargs["seed"], kwargs["num_inference_steps"], kwargs["true_cfg_scale"], kwargs["use_kv_cache"]))
        return SimpleNamespace(audit={"fixture": "API route only"})
    monkeypatch.setattr(inference, "sample_cached", sample)
    monkeypatch.setattr(refinement, "write_coarse_artifact", lambda result, **kwargs: SimpleNamespace(manifest_path=kwargs["output_dir"] / "coarse.json"))
    verified = []
    verification = SimpleNamespace(_verify_sample=lambda condition, pack, sid, required: verified.append((sid, required)))
    frozen = SimpleNamespace(runner=runner, dataflow=dataflow, inference=inference, refinement=refinement, training=verification,
                             cache=cache, gates=gates, code_root=ROOT, signature=V4, torch=torch)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(render, "load_frozen_code", lambda _: frozen)
    args = render.build_parser().parse_args(["--code-root", str(ROOT), "--name", "legacy12_stage200",
        "--output-root", str(tmp_path / "pro"), "--checkpoint", str(checkpoint), "--feature-run", str(tmp_path / "features"),
        "--source-geometry-index", str(tmp_path / "source.json"), "--contact-vertex-parts", str(tmp_path / "parts.npz"),
        "--legacy-workspace", str(tmp_path / "legacy")])
    result = render.render_checkpoint(args)
    assert len(seen) == 6 and set(seen) == {(43, 40, 1., False)}
    assert result["source_inputs"] == "locked_legacy_B" and result["coarse_images"] == 6
    assert verified == [(r["sample_id"], True) for r in rows]
    with pytest.raises(FileExistsError):
        render.render_checkpoint(args)


def test_default_none_manifest_paths_match_the_training_fallback_for_noise_proof(tmp_path):
    summary = tool("summarize_pilot")
    import torch
    from src.pose_control.qwen21_pro import cache, dataflow, training
    legacy, features = tmp_path / "legacy", tmp_path / "features"
    write(legacy / "data/overfit6_conditions.jsonl", {"sample_id": "one"})
    write(legacy / "data/overfit6_supervision.jsonl", {"sample_id": "one"})
    write(features / "report.json", {"source": "bound contract fixture"})
    config = {"legacy_workspace": str(legacy), "feature_run": str(features), "cache_run": None,
              "conditions": None, "supervision": None, "kind": "legacy"}
    cfg = {"source_dim": 1600}
    samples = {"one": (SimpleNamespace(attestation={"fixture": "source"}), None, SimpleNamespace(target_hw=(2, 2)))}
    df = SimpleNamespace(records=lambda _: [{"sample_id": "one"}], configuration=lambda _: SimpleNamespace(to_dict=lambda: cfg),
        _legacy_samples=lambda *a, **k: (samples, [], "source"), training_execution_protocol=dataflow.training_execution_protocol)
    frozen = SimpleNamespace(dataflow=df, training=training, cache=cache, torch=torch,
                              union_runner=SimpleNamespace(model_signature=lambda: {"runtime": {"contract_fixture": True}}))
    draw = training.training_draw(train_seed=3407, update=0, microstep=0, sample_id="one", target_shape=(1, 4, 64),
                                   device="cpu", dtype=torch.bfloat16)
    metadata = {**dataflow.training_execution_protocol("cpu"), "config": cfg, "train_seed": 3407,
        "source_signature": "source", "cache_signature": cache.file_sha256(features / "report.json"),
        "data_signature": cache.stable_hash(cache.asset_hashes({"conditions": str(legacy / "data/overfit6_conditions.jsonl"),
             "supervision": str(legacy / "data/overfit6_supervision.jsonl")}))}
    result = summary._bound_replay(frozen, [{k: v for k, v in draw.items() if k != "noise"}], metadata, config, device="cpu")
    assert result["target_binding_verified"]
    assert result["draws"][0]["noise_sha256"] == cache.tensor_sha256(draw["noise"])


@pytest.mark.parametrize("step", [0, 200])
def test_render_rejects_wrong_same_grid_bound_pack_before_probe_or_sampling(tmp_path, monkeypatch, step):
    render = tool("render_pilot")
    import torch
    from dataclasses import replace
    from src.pose_control.qwen21_pro import runner, dataflow, inference, refinement, cache, contracts, gates, training
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from test_qwen21_pro_preparation import fixture_assets
    record, bound, assets = fixture_assets(tmp_path)
    pack = prepare_task_condition(record, bound, project_root=tmp_path, config=contracts.ProAdapterConfig(), assets=assets)
    wrong = replace(pack, provenance={**pack.provenance, "cache_binding": {**pack.provenance["cache_binding"],
        "condition_fingerprint": "wrong_same_grid"}})
    rows = [{"dataset": d, "sample_id": "fixture" if i == 0 and d == "iper" else f"{d}{i}", "split": "train"}
            for i in range(2) for d in ("iper", "tiktok", "hi4d")]
    cfg = contracts.ProAdapterConfig(source_dim=1600)
    checkpoint = tmp_path / f"checkpoint_{step:06d}"; checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"fixture")
    metadata = {"config": cfg.to_dict(), "arm": "full4", "profile": "overfit", "code_signature": V4}
    write(checkpoint / "checkpoint.json", {"metadata": metadata, "cursor": {"optimizer_step": step},
                                          "state_sha256": cache.file_sha256(checkpoint / "state.pt")})
    monkeypatch.setattr(dataflow, "records", lambda _: rows)
    monkeypatch.setattr(dataflow, "_legacy_samples", lambda *a, **k: ({r["sample_id"]: (bound, None, wrong) for r in rows}, [], "source"))
    monkeypatch.setattr(dataflow, "_inference_pipeline", lambda *a, **k: (SimpleNamespace(transformer=None), cfg, metadata))
    monkeypatch.setattr(gates, "probe_zero_equivalence", lambda *a, **k: pytest.fail("wrong bound pack reached zero probe"))
    monkeypatch.setattr(inference, "sample_cached", lambda *a, **k: pytest.fail("wrong bound pack reached sampler"))
    frozen = SimpleNamespace(runner=runner, dataflow=dataflow, inference=inference, refinement=refinement, cache=cache,
                             gates=gates, training=training, torch=torch, code_root=ROOT, signature=V4)
    monkeypatch.setattr(render, "load_frozen_code", lambda _: frozen)
    args = render.build_parser().parse_args(["--code-root", str(ROOT), "--name", "wrong_bound", "--output-root", str(tmp_path / "pro"),
        "--checkpoint", str(checkpoint), "--feature-run", str(tmp_path / "features"), "--source-geometry-index", str(tmp_path / "source.json"),
        "--contact-vertex-parts", str(tmp_path / "parts.npz"), "--legacy-workspace", str(tmp_path / "legacy")])
    with pytest.raises(ValueError, match="binding"):
        render.render_checkpoint(args)
