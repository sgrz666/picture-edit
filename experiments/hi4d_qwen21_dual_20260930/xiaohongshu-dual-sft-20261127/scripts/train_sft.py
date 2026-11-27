"""Launch the pinned V7 stage sequence after matched data and GPU gradient gates."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from qwen_cache import EVAL, EVAL_SHA, INIT, INIT_SHA, MODEL, MODEL_MANIFEST, MODEL_SHA, pinned
from select_pairs import read, ref, write

AUTHORIZATION = {"source": "direct_user_message", "text": "进行我们双人的最新架构的大规模训练吧,先训着吧,可以跟我对齐一下需求,我们显存算力十分充足,可以考虑最好的训练效果的🤔,先训sft吧/data/shangguanrz/datasets/xiaohongshu(two)"}
STAGES = (("S0", 300), ("S1", 400), ("S2", 800), ("S3", 1500))


def smoke_bindings(root):
    rows = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2:
        raise ValueError("The smoke run must have two independently selected pairs")
    groups = read(root / "train-groups.json")
    if len(set(groups.values())) != 2:
        raise ValueError("Smoke train and validation need disjoint notes")
    val_id = rows[1]["pair_id"]
    old = "train\0" + val_id
    new = "validation\0" + val_id
    for split, row in zip(("train", "validation"), rows):
        path = root / f"smoke-{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        write(root / f"smoke-{split}-groups.json", {row["pair_id"]: groups[row["pair_id"]]})
    index = read(root / "cache/index.json")
    for key in ("entries", "condition_bindings"):
        index[key][new] = index[key].pop(old)
    write(root / "cache/smoke-index.json", index)
    preservation = read(root / "preservation.json")
    preservation["entries"][new] = preservation["entries"].pop(old)
    write(root / "smoke-preservation.json", preservation)


def worker(remainder):
    import torch
    from src.pose_control.qwen21_full.native_v7.training import main

    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    if total * .90 > free - 2 * 1024**3:
        raise RuntimeError("Training GPU does not have the required 90% allocator budget and 2 GiB margin")
    torch.cuda.set_per_process_memory_fraction(.90)
    main(remainder)


def train(root, smoke):
    import fcntl
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, stage_preset_v7
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec

    lock = (root / "sft.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    native = read(root / "native-ready.json")
    if smoke:
        smoke_bindings(root)
    prefix = "smoke-" if smoke else ""
    reports = root / "sft"
    reports.mkdir()
    splits = ("train", "validation") if smoke else ("train", "validation", "test")
    if not smoke and any(native["splits"].get(role, 0) == 0 for role in splits):
        raise ValueError("Large-scale training requires nonempty train/validation/test")
    sources = tuple(SourceSpec(role, role=role, manifest=ref(root / f"{prefix}{role}.jsonl"),
                    groups=ref(root / f"{prefix}{role}-groups.json"), data_profile="full_smplx",
                    scene_kind="dual", weight=1. if role == "train" else None) for role in splits)
    cfg = TrainingConfigV7(sources=sources, initialization=pinned(INIT, INIT_SHA),
        model_path=str(MODEL), model_manifest=pinned(MODEL_MANIFEST, MODEL_SHA),
        cache_root=str(root / "cache"), cache_identity=ref(root / "cache" / f"{prefix}index.json"),
        evaluators={"identity": pinned(EVAL, EVAL_SHA)}, output_root=str(root / "output"),
        preservation={"source_identity": .2, "background": 0. if smoke else .1},
        preservation_index=ref(root / f"{prefix}preservation.json"),
        paper_loss={"person_fm": .5, "contact_fm": 0., "identity_contrastive": .1, "temperature": .07},
        save_every_updates=1 if smoke else 100, probe_every_updates=0 if smoke else 100,
        probe_samples=12, generation_every_updates=0, diagnostics=False,
        continuous_stage_authorization=AUTHORIZATION)
    stage_budgets = (("S0", 1),) if smoke else STAGES
    write(reports / "schedule.json", {"stages": stage_budgets, "accumulation": 1 if smoke else 4,
          "initialization": cfg.initialization, "source_manifest": list(native["splits"]), "smoke": smoke,
          "model_manifest": cfg.model_manifest, "evaluators": cfg.evaluators})
    completed = 0
    parent = None
    body_gate = None

    def run(name, action):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--", *map(str, action)]
        print(json.dumps({"event": name, "time": datetime.now(timezone.utc).isoformat(), "command": cmd}), flush=True)
        with (reports / (name + ".log")).open("x", encoding="utf-8") as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)

    for phase, budget in stage_budgets:
        stage = stage_preset_v7(phase, budget=budget)
        stage = replace(stage, accumulation=1 if smoke else 4, optimizer=replace(stage.optimizer,
                        scheduler="constant" if smoke else "warmup_constant",
                        warmup_updates=0 if smoke else min(40, budget // 10)))
        current = replace(cfg, stage=stage, initialization=parent or cfg.initialization,
                          body_baseline_report=body_gate, gradient_report=None)
        config_path = reports / f"{phase}-config.json"
        data_path = reports / f"{phase}-data.json"
        gradient_path = reports / f"{phase}-gradient.json"
        write(config_path, current.record())
        run(f"{phase}-data", ["preflight", "--level", "data", "--config", config_path, "--report", data_path])
        if read(data_path).get("passed") is not True:
            raise RuntimeError(f"{phase} data preflight rejected")
        run(f"{phase}-grad", ["preflight", "--level", "grad", "--config", config_path, "--report", gradient_path])
        gradient = read(gradient_path)
        if not (gradient.get("production_passed") and gradient.get("config_sha256") == current.signature()
                and gradient.get("initialization_sha256") == current.initialization["sha256"]):
            raise RuntimeError(f"{phase} production gradient gate rejected")
        current = replace(current, gradient_report=ref(gradient_path))
        ready = reports / f"{phase}-ready.json"
        write(ready, current.record())
        segments = root / "output/segments"
        before = set(segments.glob("*"))
        action = ["train"] if parent is None else ["advance-stage", "--checkpoint", parent["path"]]
        run(f"{phase}-train", [*action, "--config", ready])
        [summary_path] = [path / "summary.json" for path in set(segments.glob("*")) - before]
        summary = read(summary_path)
        checkpoint = summary["checkpoint"]["checkpoint"]
        state, checkpoint_ref, _ = load_committed(checkpoint, schema=current.schema)
        completed += budget
        training = state["training_state"]
        if not (summary.get("budget_completed") and training["parent"] == parent
                and training["global_update"] == completed and training["stage_update"] == budget
                and training["config_sha256"] == current.signature()):
            raise RuntimeError(f"Committed stage {phase} does not match the approved run")
        del state, training
        parent = checkpoint_ref
        write(reports / f"{phase}-committed.json", {"updates": completed, "checkpoint": parent,
                                                   "summary": ref(summary_path), "gradient": ref(gradient_path)})
        print(json.dumps({"event": "stage_committed", "phase": phase, "updates": completed,
                          "checkpoint": parent}, ensure_ascii=False), flush=True)
        if phase == "S0" and not smoke:
            path = reports / "S0-body-technical-gate.json"
            write(path, {"body_accepted": True, "acceptance_kind": "technical_pilot_gate",
                         "continuous_stage_authorization": AUTHORIZATION, "s0_gradient_report": ref(gradient_path),
                         "checkpoint_sha256": parent["sha256"], "completed_updates": completed})
            body_gate = ref(path)
    write(reports / "complete.json", {"updates": completed, "checkpoint": parent, "smoke": smoke})


if __name__ == "__main__":
    if sys.argv[1:3] == ["worker", "--"]:
        worker(sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--smoke", action="store_true")
        args = parser.parse_args()
        train(args.root.resolve(), args.smoke)
