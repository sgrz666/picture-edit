"""完整发布权重的只增训练门禁：真实 six、zero/prefix、两次更新与冻结 SHA。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=json_value)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--feature-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--exact-resume", action="store_true")
    args = parser.parse_args()
    if args.updates != (4 if args.exact_resume else 2):
        parser.error("普通门禁固定两次更新；精确续跑门禁固定四次更新")
    legacy, project, feature = (Path(v).resolve() for v in (args.legacy_workspace, args.project_root, args.feature_run))
    run = Path(args.run_dir)
    if not run.is_absolute() or run.resolve().is_relative_to(legacy) or any(p.lower() in {"models", "datasets"} for p in run.parts):
        parser.error("run-dir 必须是旧工作区、datasets/models 之外的新绝对路径")
    run.mkdir(parents=True, exist_ok=False)
    started, phase = time.monotonic(), "imports"
    import torch
    from src.pose_control.qwen21_pro import ProAdapterConfig
    from src.pose_control.qwen21_pro import checkpoint, training
    from src.pose_control.qwen21_pro.cache import ProCache, file_sha256, load_bound_condition, ordinary_cpu, stable_hash
    from src.pose_control.qwen21_pro.features import RawSourceFeatures
    from src.pose_control.qwen21_pro.losses import RegionalLossConfig
    from src.pose_control.qwen21_pro.offload import place_training_model
    from src.pose_control.qwen21_pro.pipeline import build_pro_transformer
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition, resolve_condition_assets
    from src.pose_control.qwen21_union.cache import expected_supervision_fingerprint, load_supervision
    from src.pose_control.qwen21_union.integrity import verify_workspace_inputs
    from src.pose_control.qwen21_union.runner import model_signature
    from diffusers import FlowMatchEulerDiscreteScheduler
    policy = training.configure_determinism()
    torch.set_num_threads(4)
    torch.manual_seed(3407)

    def event(kind, **fields):
        value = {"kind": kind, "phase": phase, "seconds": time.monotonic() - started, **fields}
        with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_value) + "\n")
        print(json.dumps(value, ensure_ascii=False, default=json_value), flush=True)

    try:
        write_json(run / "run.json", {"argv": vars(args), "code_root": str(ROOT), "determinism": policy,
            "script_sha256": file_sha256(__file__), "engineering_only": True, "quality_claim": False})
        phase = "verify_inputs"
        verify_workspace_inputs(legacy)
        producer_path = legacy / "cache/completed_20261016T184014.json"
        if file_sha256(producer_path) != "ffb3077938662e16277d5e4862af8c3466e316de4f0c3b64ec107981d23ad0ae":
            raise ValueError("legacy encoder producer event changed")
        producer = json.loads(producer_path.read_text())["model_signature"]
        current = model_signature()
        for key in ("assets", "configs", "preprocessing_code", "runtime", "videox_commit", "union_revision"):
            if producer[key] != current[key]:
                raise ValueError(f"frozen encoder producer binding differs: {key}")
        vae_config = json.loads((Path(args.base_model) / "vae/config.json").read_text())
        conditions = [json.loads(line) for line in (legacy / "data/overfit6_conditions.jsonl").read_text().splitlines() if line]
        truths = {r["sample_id"]: r for r in map(json.loads, (legacy / "data/overfit6_supervision.jsonl").read_text().splitlines())}
        report = json.loads((feature / "report.json").read_text())
        if not report["passed"] or report["profile"] != "effect1600" or len(conditions) != 6:
            raise ValueError("six verified effect1600 records required")
        cfg = ProAdapterConfig.from_dict(report["config"])
        raw_cache = ProCache(feature / "cache", namespace="raw_source")
        samples, source_models = {}, []
        for row in conditions:
            sid = row["sample_id"]
            condition = load_bound_condition(row, legacy / "cache", model_signature=producer, vae_config=vae_config)
            meta = raw_cache.metadata(sid)
            raw = RawSourceFeatures.from_dict(raw_cache.get(sid, expected_hash=meta["hash"], signature=meta["signature"]))
            assets = resolve_condition_assets(row, project)
            if row["dataset"] == "hi4d":
                assets["contact_vertex_parts"] = feature.parents[1] / "data/smpl_male_v6_parts_20261018.npz"
            pack = prepare_task_condition(row, condition, project_root=project, config=cfg, raw_source_features=raw, assets=assets)
            expected = expected_supervision_fingerprint({**truths[sid], "split": "train"}, producer, vae_config)
            truth = load_supervision(legacy / "cache", sid, expected_hash=expected)
            samples[sid] = condition, truth, pack
            source_models.append(raw.provenance["model_signature"]["extractors"])
        if any(model != source_models[0] for model in source_models):
            raise ValueError("source extractor signatures differ across samples")
        order = tuple((dataset, row["sample_id"]) for i in range(2) for dataset in training.DATASET_ORDER
                      for row in [r for r in conditions if r["dataset"] == dataset][i:i + 1])
        free, total = torch.cuda.mem_get_info()
        budget = min(int(total * .9), free - 1024**3)
        if budget <= 1024**3:
            raise RuntimeError("insufficient currently free CUDA memory")
        torch.cuda.set_per_process_memory_fraction(budget / total)
        event("verified", available_memory_bytes=free, total_memory_bytes=total, process_budget_bytes=budget,
              sample_order=order, cache_report_sha256=file_sha256(feature / "report.json"))
        phase = "load_published_model_cpu"
        event("loading")
        model, audit = build_pro_transformer("joint", "cpu", torch.bfloat16, config=cfg, base_model=args.base_model,
            union_checkpoint=project / "union-adapter/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
        write_json(run / "backend.json", audit)
        phase = "selective_cuda_placement"
        memory = place_training_model(model, "cuda", memory_mode="frozen_linear_cpu")
        model.gradient_checkpointing = True
        parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
        write_json(run / "memory_policy.json", memory)
        event("placed", **memory)
        frozen = training.frozen_parameter_digest(model)
        optimizer = training.make_optimizer(model)
        lr = training.make_lr_scheduler(optimizer, warmup_updates=2)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        package_code = {p.relative_to(ROOT).as_posix(): file_sha256(p)
                        for name in ("qwen21_pro", "qwen21_union") for p in (ROOT / "src/pose_control" / name).glob("*.py")}
        metadata = {"config": cfg.to_dict(), "scope": "joint", "task_signature": stable_hash(RegionalLossConfig().to_dict()),
            "source_signature": stable_hash(source_models[0]), "cache_signature": file_sha256(feature / "report.json"),
            "code_signature": stable_hash(package_code), "data_signature": file_sha256(legacy / "data/lock.json"),
            "trajectory_signature": training.trajectory_signature(order, memory_mode="frozen_linear_cpu"),
            "memory_mode": "frozen_linear_cpu", "control_lora_targets": [], "selection_role": "engineering_probe",
            "engineering_updates": args.updates, "repeat_samples": args.exact_resume}
        checkpoint.save_checkpoint(run / "checkpoint_000000", model, optimizer, lr, metadata=metadata,
                                   cursor=checkpoint.TrainingCursor())
        phase = "zero_equivalence"
        cond, truth, pack = samples[order[0][1]]
        noise = training.training_draw(train_seed=3407, update=0, microstep=0, sample_id=order[0][1],
            target_shape=truth["target_latent"].shape, device="cuda", dtype=torch.bfloat16)["noise"]
        inputs = training.transformer_inputs(cond, noise, torch.tensor([.5]), device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base = model(**inputs, task_condition=pack, task_enabled=False, return_dict=False)[0].cpu()
            initial = model(**inputs, task_condition=pack, return_dict=False)[0].cpu()
        if not torch.equal(base, initial):
            raise ValueError("full published model zero Pro differs from frozen Union")
        event("zero_equivalent", shape=list(base.shape))
        phase = "continuous_updates"
        branch_maxima = {"heads": 0., "geometry": 0., "contacts": 0., "appearance": 0., "identity": 0.}
        prefixes = {"heads": "output_heads.", "geometry": "geometry_encoder.projections.",
                    "contacts": "geometry_encoder.contacts.", "appearance": "banks.source_", "identity": "banks.identity_projection."}
        def micro(value):
            norms = {}
            for group, prefix in prefixes.items():
                squared = sum(float(p.grad.detach().double().square().sum()) for n, p in model.task_adapter.named_parameters()
                              if n.startswith(prefix) and p.grad is not None)
                norms[group] = squared ** .5
                branch_maxima[group] = max(branch_maxima[group], norms[group])
            event("microstep", **value, branch_gradient_norms=norms)
        def update(value):
            event("optimizer_update", **value)
            if args.exact_resume and value["optimizer_step"] == 2 and not (run / "checkpoint_000002").exists():
                checkpoint.save_checkpoint(run / "checkpoint_000002", model, optimizer, lr,
                                           metadata=metadata, cursor=value["cursor"])
        result = training.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
            sample_order=order, load_sample=samples.__getitem__, updates=args.updates, device="cuda", max_memory_bytes=budget,
            on_microstep=micro, on_update=update, repeat_samples=args.exact_resume)
        resume_equivalence = None
        if args.exact_resume:
            import numpy as np
            phase = "exact_resume"
            reference = ordinary_cpu({"task": model.task_adapter.state_dict(), "optimizer": optimizer.state_dict(),
                                      "lr": lr.state_dict()})
            reference_rng = checkpoint.capture_rng_state()
            cursor = checkpoint.load_checkpoint(run / "checkpoint_000002", model, optimizer, lr, expected_metadata=metadata)
            if {name: id(parameter) for name, parameter in model.named_parameters()} != parameter_ids:
                raise ValueError("checkpoint restore replaced Parameter objects")
            tail = training.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
                sample_order=order, load_sample=samples.__getitem__, updates=4, cursor=cursor, device="cuda",
                max_memory_bytes=budget, on_microstep=micro, on_update=update, repeat_samples=True)
            def equal(left, right):
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
                elif isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                elif isinstance(left, dict):
                    if left.keys() != right.keys():
                        raise ValueError("resume tree keys differ")
                    for key in left:
                        equal(left[key], right[key])
                elif isinstance(left, (tuple, list)):
                    if len(left) != len(right):
                        raise ValueError("resume tree length differs")
                    for a, b in zip(left, right):
                        equal(a, b)
                elif left != right:
                    raise ValueError("resume scalar differs")
            equal(reference, {"task": model.task_adapter.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr.state_dict()})
            equal(reference_rng, checkpoint.capture_rng_state())
            if result["trajectory"][6:] != tail["trajectory"] or result["cursor"] != tail["cursor"]:
                raise ValueError("resume sample/noise/timestep/loss trajectory differs")
            for key in ("max_memory_allocated", "max_memory_reserved"):
                result[key] = max(result[key], tail[key])
            resume_equivalence = {"passed": True, "continuous_updates": 4, "resumed_updates": [2, 2],
                                  "tensor_tolerance": 0, "same_parameter_objects": True}
            event("exact_resume_passed", **resume_equivalence)
        if any(value <= 0 for value in branch_maxima.values()):
            raise ValueError(f"missing real branch gradients: {branch_maxima}")
        phase = "frozen_and_prefix_verification"
        if training.frozen_parameter_digest(model) != frozen:
            raise ValueError("base or official Union frozen bytes changed")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            trained = model(**inputs, task_condition=pack, return_dict=False)[0].cpu()
        target_n = pack.target_hw[0] * pack.target_hw[1]
        if not torch.equal(trained[:, :-target_n], base[:, :-target_n]):
            raise ValueError("trained task increment modified the causal reference prefix")
        checkpoint.save_checkpoint(run / f"checkpoint_{args.updates:06d}", model, optimizer, lr, metadata=metadata, cursor=result["cursor"])
        write_json(run / "gate.json", {"passed": True, "engineering_only": True, "updates": args.updates,
            "zero_equivalence": True, "prefix_unchanged": True, "frozen_unchanged": True,
            "resume_equivalence": resume_equivalence,
            "branch_gradient_maxima": branch_maxima, "metadata": metadata,
            "max_memory_allocated": result["max_memory_allocated"], "max_memory_reserved": result["max_memory_reserved"],
            "seconds": time.monotonic() - started})
        event("passed", updates=args.updates, branch_gradient_maxima=branch_maxima)
    except Exception as error:
        write_json(run / "failure.json", {"phase": phase, "type": type(error).__name__, "error": str(error),
            "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
