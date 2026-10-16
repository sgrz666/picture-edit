"""按固定预算串行执行方案 B；任一门禁失败即停止，不修改实验变量。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_gate(workspace, name, kind):
    path = Path(workspace) / "runs" / name / "gate.json"
    if not path.is_file():
        raise RuntimeError(f"{kind} gate has no completion evidence: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("kind") != kind or gate.get("gate_pass") is not True:
        raise RuntimeError(f"{kind} gate failed: {path}")
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reuse-smoke")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.name) or args.name in (".", ".."):
        raise ValueError("queue name must be a safe path component")
    work = args.workspace.resolve()
    queue = work / "queues" / args.name
    queue.mkdir(parents=True, exist_ok=False)
    cli = Path(__file__).with_name("run.py")
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    events = queue / "events.jsonl"
    def event(**fields):
        fields = dict(time=datetime.now().isoformat(), **fields)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(fields, ensure_ascii=False) + "\n")
        print(json.dumps(fields, ensure_ascii=False), flush=True)
    def phase(name, *arguments):
        if (queue / "STOP").exists():
            raise RuntimeError("queue STOP requested; no next phase started")
        command = [sys.executable, str(cli), arguments[0], "--workspace", str(work), *arguments[1:]]
        event(stage=name, status="running", command=command)
        started = time.monotonic()
        with (queue / f"{name}.log").open("x", encoding="utf-8") as output:
            result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        event(stage=name, status="passed" if result.returncode == 0 else "failed", returncode=result.returncode, seconds=time.monotonic() - started)
        if result.returncode:
            raise RuntimeError(f"phase {name} failed; inspect {queue / (name + '.log')}")
    names = {key: key + "_" + args.name for key in ("smoke", "overfit", "joint", "target", "diagnostics", "test")}
    def require_training(name):
        path = work / "runs" / name / "summary.json"
        summary = json.loads(path.read_text()) if path.is_file() else {}
        if summary.get("complete") is not True or summary.get("optimizer_step") != 512:
            raise RuntimeError(f"training stopped before the required 512 updates: {name}")
    def require_gate(name, kind):
        gate = read_gate(work, name, kind)
        import run as _entry  # initializes this snapshot's package/search paths
        from src.pose_control.qwen21_union.runner import training_binding, validated_gate_kinds
        if kind not in validated_gate_kinds([gate], training_binding(work)):
            raise RuntimeError(f"{kind} gate signature or required evidence is stale: {name}")
    try:
        phase("audit", "audit")
        phase("cache_overfit6", "cache", "--conditions", "overfit6", "--supervision", "train")
        if args.reuse_smoke:
            require_gate(args.reuse_smoke, "smoke")
            names["smoke"] = args.reuse_smoke
            event(stage="smoke", status="reused", run=args.reuse_smoke)
        else:
            phase("smoke", "smoke", "--name", names["smoke"])
            require_gate(names["smoke"], "smoke")
        phase("overfit", "overfit", "--name", names["overfit"], "--updates", "200")
        require_gate(names["overfit"], "overfit")
        phase("cache_full", "cache", "--conditions", "train,val,test", "--supervision", "train")
        phase("diagnostics", "infer", "--split", "val", "--name", names["diagnostics"], "--seeds", "42",
              "--arms", "native,union_frozen,union_zero,union_target_frozen,wrong_pose")
        phase("evaluate_diagnostics", "evaluate", "--run", names["diagnostics"])
        phase("report_diagnostics", "report", "--run", names["diagnostics"])
        phase("train_joint", "train", "--name", names["joint"], "--arm", "union_lora_joint", "--updates", "512")
        require_training(names["joint"])
        phase("train_target", "train", "--name", names["target"], "--arm", "union_lora_target", "--updates", "512")
        require_training(names["target"])
        phase("test", "infer", "--split", "test", "--name", names["test"], "--seeds", "42,43",
              "--arms", "native,union_frozen,union_lora_joint,union_lora_target",
              "--joint-checkpoint", str(work / "runs" / names["joint"] / "checkpoint_000512"),
              "--target-checkpoint", str(work / "runs" / names["target"] / "checkpoint_000512"))
        phase("evaluate_test", "evaluate", "--run", names["test"])
        phase("report_test", "report", "--run", names["test"])
        phase("final_status", "status")
        final_status = json.loads((queue / "final_status.log").read_text())
        if final_status.get("complete") is not True:
            raise RuntimeError("formal training/result matrix completeness audit failed")
        event(stage="queue", status="complete", runs=names)
    except Exception as error:
        event(stage="queue", status="stopped", error=f"{type(error).__name__}: {error}", runs=names)
        raise


if __name__ == "__main__":
    main()
