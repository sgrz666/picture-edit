"""本次已批准pilot的串行运行清单；独占新目录，任一工程失败即停。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

P = Path("/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1")
B = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1")
C = P / "code_task7b_20261018T161055"
PY = B / "env/bin/python"
MAIN = P / "runs/pilot_main200_v4_20261018T161055"
STAMP = "20261021T110000"
QUEUE = P / "runs" / ("pilot_sequence_" + STAMP)
SIGNATURE = "1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e"
COMMON = ["--output-root", str(P), "--feature-run", str(P / "runs/conditions_pilot1600_20261018T142046"),
          "--source-geometry-index", str(P / "runs/train_geometry_20261018T142046/source_geometry_index.json"),
          "--contact-vertex-parts", str(P / "data/smpl_male_v6_parts_20261018.npz"),
          "--memory-mode", "resident", "--memory-budget-gib", "32"]
ARMS = ("no_dense", "no_appearance", "no_contact", "full8")

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_training(run, step):
    gate, info = read(run / "gate.json"), read(run / "run.json")
    if not (gate["engineering_passed"] is True and gate["status"] == "complete"
            and gate["cursor"]["optimizer_step"] == step
            and gate["cursor"]["accumulation_steps"] == 1
            and info["code_signature"] == SIGNATURE):
        raise RuntimeError(f"Training engineering evidence rejected: {run}")
    # A failed quality threshold remains a failed quality threshold.
    return {"engineering_passed": True, "quality_passed": gate["passed"],
            "relative_global_drop": gate["relative_global_drop"],
            "gate_sha256": sha(run / "gate.json")}

def stages():
    rows = []
    for arm in ARMS:
        name = f"pilot_{arm}_25_{STAMP}"
        rows.append({"name": name, "kind": "ablation25", "arm": arm,
                     "argv": [str(PY), "-u", str(C / "scripts/train/qwen21_pro.py"), "ablate",
                              "--name", name, *COMMON, "--arm", arm]})
    for step in (0, 200):
        name = f"pilot_train6_step{step}_{STAMP}"
        rows.append({"name": name, "kind": "legacy_render6", "step": step,
                     "argv": [str(PY), "-u", str(P / "scripts/render_pilot.py"),
                              "--code-root", str(C), "--name", name,
                              "--checkpoint", str(MAIN / f"checkpoint_{step:06d}"),
                              "--feature-run", str(P / "runs/conditions_pilot1600_20261018T142046"),
                              "--source-geometry-index", str(P / "runs/train_geometry_20261018T142046/source_geometry_index.json"),
                              "--contact-vertex-parts", str(P / "data/smpl_male_v6_parts_20261018.npz")]})
    for seed in (42, 43):
        name = f"pilot_valpair_seed{seed}_{STAMP}"
        rows.append({"name": name, "kind": "val_render2", "seed": seed,
                     "argv": [str(PY), "-u", str(C / "scripts/run/qwen21_pro.py"), "infer",
                              "--name", name, "--output-root", str(P), "--kind", "current",
                              "--conditions", str(C / "experiments/qwen21_pro_adapter_v1/data/val12_conditions_20261018.jsonl"),
                              "--cache-run", str(P / "runs/val_conditions1600_sync_20261018T143039"),
                              "--checkpoint", str(MAIN / "checkpoint_000025"), "--arm", "full4",
                              "--sample-id", "iper_val_e36352d6b4b72ae362cb1ce3",
                              "--sample-id", "hi4d_val_1cdfa27cfed29647a42be809",
                              "--seed", str(seed), "--steps", "40", "--true-cfg-scale", "1",
                              "--memory-budget-gib", "32", "--offload", "sync_group_offload", "--engineering-probe"]})
    name = "pilot_summary_" + STAMP
    cmd = [str(PY), "-u", str(P / "scripts/summarize_pilot.py"), "--code-root", str(C),
           "--name", name, "--main-run", str(MAIN), "--noise-replay-device", "cuda"]
    for arm in ARMS:
        cmd.extend(["--" + arm.replace("_", "-") + "-run", str(P / "runs" / f"pilot_{arm}_25_{STAMP}")])
    rows.append({"name": name, "kind": "summary", "argv": cmd})
    return rows

def main():
    plan = stages()
    if sys.argv[1:] == ["--plan"]:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if sys.argv[1:]:
        raise ValueError("Only --plan or no arguments accepted")
    QUEUE.mkdir(exist_ok=False)
    event_file = QUEUE / "events.jsonl"
    def event(event_type, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event_type, **fields}
        with event_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (QUEUE / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    event("main_complete", **verify_training(MAIN, 200), script_sha256=sha(__file__), source_signature=SIGNATURE)
    try:
        for row in plan:
            log = P / "logs" / (row["name"] + ".log")
            if (P / "runs" / row["name"]).exists() or log.exists():
                raise FileExistsError(f"Stage already exists: {row['name']}")
            event("stage_started", **row, log=str(log))
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run(row["argv"], cwd=C, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, check=False)
            if result.returncode:
                raise RuntimeError(f"Stage {row['name']} exited {result.returncode}; inspect {log}")
            run = P / "runs" / row["name"]
            proof = verify_training(run, 25) if row["kind"] == "ablation25" else {}
            report = read(run / "report.json")
            if row["kind"] in ("legacy_render6", "val_render2"):
                expected = 6 if row["kind"] == "legacy_render6" else 2
                if report.get("coarse_images") != expected:
                    raise RuntimeError(f"Unexpected generated image count: {row['name']}")
            if row["kind"] == "summary" and report.get("status") != "complete":
                raise RuntimeError("Summary remains incomplete")
            event("stage_completed", name=row["name"], report_sha256=sha(run / "report.json"), **proof)
        event("complete", stages=len(plan))
    except Exception as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        raise

if __name__ == "__main__":
    main()
