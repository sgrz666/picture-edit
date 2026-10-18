#!/usr/bin/env python3
"""Finish the long Qwen21 dual run, then evaluate the held-out Hi4D test split."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/qwen21_hi4d_dual/action_prompt_v2"
PYTHON = sys.executable
TEST_PAIRS = ROOT / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_test.jsonl"
EXPECTED_TEST_ROWS = 606


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_status(state: str, **extra):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "completion_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"state": state, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                     **extra}, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(state, json.dumps(extra), flush=True)


def train_alive(pid: int) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes().replace(b"\x00", b" ")
    except FileNotFoundError:
        return False
    return b"train_hi4d_qwen21_dual.py" in command


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("ab", buffering=0) as stream:
        stream.write(("\nSUPERVISOR COMMAND: " + " ".join(command) + "\n").encode())
        return subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT).returncode


def train_command(resume: Path | None):
    command = [PYTHON, str(ROOT / "train_hi4d_qwen21_dual.py"),
               "--output", str(RUN), "--max-steps", "10000", "--stage1-steps", "2000",
               "--micro-batch-size", "1", "--accumulation", "4", "--iper-replay-rate", "0.1",
               "--save-every", "500", "--val-count", "64", "--gradient-checkpointing"]
    return command + (["--resume", str(resume)] if resume else [])


def test_command(steps: int, seed: int):
    return [PYTHON, str(ROOT / "eval_hi4d_qwen21_dual.py"),
            "--checkpoint", str(RUN / "checkpoint_best.pt"),
            "--pairs", str(TEST_PAIRS), "--output", str(RUN / "test_eval"),
            "--steps", str(steps), "--seed", str(seed), "--resume-generation"]


def metric(summary, mode, name):
    entry = summary["mean"][mode]
    value, count = entry[name], entry[name + "_n"]
    return "N/A" if value is None else f"{value:.4f} (n={count})"


def write_report(summary):
    modes = ("off_detailed", "geometry_short", "full_short")
    fields = (("A 姿态 PCK", "pose_pck_a"), ("B 姿态 PCK", "pose_pck_b"),
              ("接触距离误差", "contact_distance_error"),
              ("A 外观相似度", "body_identity_a"), ("B 外观相似度", "body_identity_b"),
              ("身份交换率", "identity_swap"), ("背景 MAE", "background_mae"),
              ("第三人检出率", "third_person_detected"),
              ("两人姿态可评估率", "two_person_pose_evaluable"))
    lines = ["# Qwen-Image-2.1 Hi4D 双人正式测试结果", "",
             f"检查点：`{summary['checkpoint']}`（第 {summary['checkpoint_step']} 步）", "",
             f"测试配对：{summary['count']}；采样：{summary['steps']} 步、种子 {summary['seed']}。",
             "训练：1312 配对；验证：822 配对，仅验证集选择检查点；测试集在选点后运行。", "",
             "原生模型使用详细动作提示词，不接收目标几何。仅几何组使用简短提示词和目标姿态、部位、法线、深度、相对几何、渲染人物区域；完整双人组再加入目标接触和 A/B 源外观绑定。后两组是目标几何已知控制任务。", "",
             "| 指标 | 原生＋详细 | 仅几何＋简短 | 完整双人＋简短 |", "|---|---:|---:|---:|"]
    for label, field in fields:
        lines.append("| " + label + " | " + " | ".join(metric(summary, mode, field) for mode in modes) + " |")
    native, full = (summary["mean"][mode] for mode in ("off_detailed", "full_short"))
    checks = {
        "A_pose_improved": full["pose_pck_a"] is not None and native["pose_pck_a"] is not None
                           and full["pose_pck_a"] > native["pose_pck_a"],
        "B_pose_improved": full["pose_pck_b"] is not None and native["pose_pck_b"] is not None
                           and full["pose_pck_b"] > native["pose_pck_b"],
        "contact_improved": full["contact_distance_error"] is not None
                            and native["contact_distance_error"] is not None
                            and full["contact_distance_error"] < native["contact_distance_error"],
        "identity_swap_not_increased": full["identity_swap"] is not None
                                       and native["identity_swap"] is not None
                                       and full["identity_swap"] <= native["identity_swap"],
        "metric_coverage_not_reduced": all(
            full[field + "_n"] >= native[field + "_n"]
            for field in ("pose_pck_a", "pose_pck_b", "contact_distance_error", "identity_swap")
        ),
    }
    lines += ["", "## 验收检查", "",
              "逐项结果：" + ", ".join(f"{name}={value}" for name, value in checks.items()) + "。",
              "缺失指标只计入可评估样本，表中的 n 必须同时检查；覆盖率不足不能当作效果达标。",
              "若姿态或接触未优于原生，或身份交换增加，应继续定位和修正。", "",
              "逐例图板与指标见 `test_eval/images/` 和 `test_eval/per_sample.json`；完整配置、数据哈希及代码哈希见 `run_manifest.json`。", ""]
    (RUN / "test_report.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN / "acceptance_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def main():
    args = arguments()
    write_status("waiting_for_training", train_pid=args.train_pid)
    while train_alive(args.train_pid):
        time.sleep(args.poll_seconds)
    for restart in range(args.max_restarts + 1):
        if (RUN / "validation_full.json").exists() and (RUN / "checkpoint_best.pt").exists():
            break
        if restart == args.max_restarts:
            write_status("training_failed", restarts=restart)
            return 1
        resume = RUN / "checkpoint_last.pt"
        write_status("resuming_training", attempt=restart + 1, checkpoint=str(resume) if resume.exists() else None)
        code = run_logged(train_command(resume if resume.exists() else None), RUN / "train.log")
        if code != 0:
            write_status("training_retry_needed", attempt=restart + 1, returncode=code)
            time.sleep(180)
    write_status("training_complete", best_checkpoint=str(RUN / "checkpoint_best.pt"))
    output = RUN / "test_eval"
    output.mkdir(exist_ok=True)
    for attempt in range(args.max_restarts + 1):
        write_status("evaluating_test", attempt=attempt + 1)
        code = run_logged(test_command(args.steps, args.seed), RUN / "test_eval.log")
        if code == 0:
            break
        if attempt == args.max_restarts:
            write_status("test_evaluation_failed", attempts=attempt + 1, returncode=code)
            return 1
        time.sleep(180)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["count"] != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"expected {EXPECTED_TEST_ROWS} test rows, got {summary['count']}")
    checks = write_report(summary)
    write_status("complete" if all(checks.values()) else "test_finished_needs_diagnosis",
                 test_rows=summary["count"], checkpoint_step=summary["checkpoint_step"],
                 acceptance_checks=checks)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status("failed", error=repr(error))
        raise
