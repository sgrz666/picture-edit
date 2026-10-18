#!/usr/bin/env python3
"""Export source-backed training figures for the matched prompt pilot (CPU only).

Reads the real short/structured train_metrics.jsonl files. Partial logs are
supported; no missing steps, validation points, or completion are invented.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


VARIANTS = ("short", "structured")
COLORS = {"short": "#2266A8", "structured": "#B36A16"}
STYLES = {"short": "-", "structured": "--"}
MARKERS = {"short": "o", "structured": "s"}
DATA_SCHEMA = "hi4d_dual_prompt_pilot_v1"
TRAINING_SCHEMA = "qwen21_dual_prompt_training_complete_v1"


@dataclass
class LogSnapshot:
    path: Path
    sha256: str | None
    rows: list[dict[str, Any]]
    warnings: list[str]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def finite_number(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric in {path}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is nonfinite in {path}")
    return result


def read_log(path: Path) -> LogSnapshot:
    if not path.exists():
        return LogSnapshot(path, None, [], ["metrics file has not been created"])
    content = path.read_bytes()
    lines = content.decode("utf-8-sig").splitlines()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    previous = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # A writer can be sampled between writes; only its unfinished tail
            # can be omitted. Corrupt complete records are never hidden.
            if index == len(lines) - 1 and not content.endswith((b"\n", b"\r")):
                warnings.append(f"omitted incomplete final JSONL record at line {index + 1}")
                break
            raise ValueError(f"invalid JSONL record in {path}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected an object in {path}:{index + 1}")
        step = row.get("step")
        if type(step) is not int or not previous < step <= 300:
            raise ValueError(f"steps must increase within 1..300 in {path}:{index + 1}")
        finite_number(row.get("loss"), "loss", path)
        microbatches = row.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != 4:
            raise ValueError(f"expected four real microbatch flow terms in {path}:{index + 1}")
        row["training_flow_loss"] = sum(finite_number(micro.get("flow"), "microbatch.flow", path)
            for micro in microbatches) / 4
        if "validation_loss" in row:
            finite_number(row["validation_loss"], "validation_loss", path)
            if row.get("validation", {}).get("selection") != "true_val_split_only":
                raise ValueError(f"validation point lacks true-val provenance in {path}:{index + 1}")
        previous = step
        rows.append(row)
    return LogSnapshot(path, hashlib.sha256(content).hexdigest(), rows, warnings)


def completion_summary(run_root: Path, logs: dict[str, LogSnapshot]) -> dict[str, Any]:
    """Confirm recorded completion against logs; this does not load checkpoints."""
    root_path = run_root / "training_complete.json"
    if not root_path.exists():
        return {"confirmed": False, "reason": "root training_complete.json absent", "best": {}}
    root = read_json(root_path)
    required = {"status": "success", "schema_version": TRAINING_SCHEMA,
        "production_training": True, "completed_steps": 300, "requested_steps": 300,
        "validation_selection": "true_val_split_only", "backbone_unchanged": True,
        "initialization_sha256_equal": True}
    if any(root.get(key) != value for key, value in required.items()):
        raise ValueError("root completion does not describe the approved production pilot")
    if set(root.get("variants", {})) != set(VARIANTS):
        raise ValueError("root completion requires both short and structured variants")
    best: dict[str, Any] = {}
    hashes = {str(root_path): hashlib.sha256(root_path.read_bytes()).hexdigest()}
    for variant in VARIANTS:
        branch_path = run_root / variant / "training_complete.json"
        if not branch_path.exists():
            raise ValueError(f"root completion exists without {variant} completion")
        branch = read_json(branch_path)
        branch_required = {**required, "variant": variant}
        if any(branch.get(key) != value for key, value in branch_required.items()):
            raise ValueError(f"invalid production completion for {variant}")
        rows = logs[variant].rows
        if [row["step"] for row in rows] != list(range(1, 301)) or logs[variant].warnings:
            raise ValueError(f"completed {variant} run requires all 300 intact log records")
        validation = [(row["step"], float(row["validation_loss"]))
            for row in rows if "validation_loss" in row]
        if [step for step, _ in validation] != list(range(50, 301, 50)):
            raise ValueError(f"completed {variant} validation steps do not match 50..300")
        # min preserves the earliest step on ties, as the trainer's strict < does.
        best_step, best_loss = min(validation, key=lambda point: point[1])
        root_branch = root["variants"][variant]
        for record in (branch, root_branch):
            if (record.get("best_step") != best_step or
                not math.isclose(finite_number(record.get("best_validation"), "best_validation", branch_path),
                    best_loss, rel_tol=1e-12, abs_tol=1e-12) or
                record.get("best_checkpoint", {}).get("step") != best_step):
                raise ValueError(f"{variant} completion best does not match true validation log")
        best[variant] = {"step": best_step, "validation_loss": best_loss}
        hashes[str(branch_path)] = hashlib.sha256(branch_path.read_bytes()).hexdigest()
    return {"confirmed": True, "reason": "both production completion records agree with all logged steps",
        "best": best, "completion_sha256": hashes,
        "scope": "completion/log consistency only; checkpoint tensor/hash validation belongs to evaluator preflight"}


def prompt_table(dataset_path: Path) -> tuple[str, dict[str, Any]]:
    content = dataset_path.read_bytes()
    dataset = json.loads(content.decode("utf-8-sig"))
    if dataset.get("schema_version") != DATA_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {dataset_path}")
    samples = dataset.get("selected_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("dataset has no selected_samples")

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")

    lines = ["# Exact selected prompts", "",
        f"Dataset SHA256: `{hashlib.sha256(content).hexdigest()}`", "",
        "Text is copied from selected_samples[].prompts; it is not regenerated from RGB.", "",
        "| Split | Sample | Template | Short prompt | Structured prompt |",
        "|---|---|---|---|---|"]
    identities = []
    for row in samples:
        prompts = row["prompts"]
        values = (row["split"], row["sample_id"], prompts["provenance"]["template_version"],
            prompts["short"], prompts["structured"])
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("selected sample has incomplete prompt fields")
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        identities.append({"split": row["split"], "sample_id": row["sample_id"],
            "template_version": prompts["provenance"]["template_version"]})
    return "\n".join(lines) + "\n", {"path": str(dataset_path),
        "sha256": hashlib.sha256(content).hexdigest(), "samples": identities}


def plot_training(logs: dict[str, LogSnapshot], completion: dict[str, Any], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 13, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharex=True)
    for variant in VARIANTS:
        rows = logs[variant].rows
        if not rows:
            continue
        x: list[float] = []
        y: list[float] = []
        previous = 0
        for row in rows:
            if previous and row["step"] != previous + 1:
                x.append(float("nan")); y.append(float("nan"))
            x.append(row["step"]); y.append(float(row["training_flow_loss"]))
            previous = row["step"]
        axes[0].plot(x, y, color=COLORS[variant], linestyle=STYLES[variant], linewidth=1.3,
            label=f"{variant} ({rows[-1]['step']} logged steps)")
        validation = [row for row in rows if "validation_loss" in row]
        if validation:
            axes[1].scatter([row["step"] for row in validation],
                [row["validation_loss"] for row in validation], color=COLORS[variant],
                marker=MARKERS[variant], s=45, linewidths=1.1,
                facecolors="none" if variant == "structured" else COLORS[variant], label=variant)
        if completion["confirmed"]:
            best = completion["best"][variant]
            axes[1].scatter([best["step"]], [best["validation_loss"]], marker="*", s=170,
                facecolors=COLORS[variant], edgecolors="#222222", linewidths=.7, zorder=4,
                label=f"{variant} best: step {best['step']}")
    titles = ("Training flow loss", "True validation flow loss")
    for axis, title in zip(axes, titles):
        axis.set_title(title, loc="left", pad=12)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Loss")
        axis.set_xlim(0, 305)
        axis.set_xticks([0, 50, 100, 150, 200, 250, 300])
        axis.grid(axis="y", color="#DDE1E5", linewidth=.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#77808A")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, frameon=False, fontsize=9, loc="best")
        else:
            axis.text(.5, .5, "No logged observations", ha="center", va="center", transform=axis.transAxes,
                color="#68717B")
    status = "Complete: both 300-step production runs confirmed" if completion["confirmed"] else "Partial snapshot: training completion is unconfirmed"
    fig.suptitle("Qwen 2.1 dual-person prompt pilot", x=.075, ha="left", fontsize=16, fontweight="bold")
    fig.text(.075, .865, status, fontsize=10, color="#48525E")
    fig.text(.075, .075, "Training flow: mean of four logged microbatch flow terms; validation disables teacher/budget terms.", fontsize=9, color="#48525E")
    fig.text(.075, .04, "Four training pairs; one validation pair; fixed validation noise seeds. Validation points are not interpolated.", fontsize=9, color="#48525E")
    fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.21, wspace=.26)
    paths = [output / "training_plot.png", output / "training_plot.svg"]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return [str(path) for path in paths]


def main(argv=None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="directory containing short/ and structured/")
    parser.add_argument("--output", type=Path, required=True, help="new figure directory; existing directories are rejected")
    parser.add_argument("--dataset", type=Path, help="optional real dataset.json for exact prompt table")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root does not exist: {run_root}")
    if args.output.exists():
        raise ValueError("choose a new output directory; existing figures are preserved")
    logs = {variant: read_log(run_root / variant / "train_metrics.jsonl") for variant in VARIANTS}
    completion = completion_summary(run_root, logs)
    table, dataset_metadata = prompt_table(args.dataset.resolve()) if args.dataset else (None, None)
    summary = {"schema_version": "qwen21_dual_prompt_figures_v1", "run_root": str(run_root),
        "completion": completion, "variants": {variant: {"path": str(log.path), "sha256": log.sha256,
            "logged_steps": [row["step"] for row in log.rows],
            "training_loss": [{"step": row["step"], "loss": row["loss"]} for row in log.rows],
            "training_flow_loss": [{"step": row["step"], "loss": row["training_flow_loss"]} for row in log.rows],
            "validation_loss": [{"step": row["step"], "loss": row["validation_loss"]}
                for row in log.rows if "validation_loss" in row], "warnings": log.warnings}
            for variant, log in logs.items()}, "dataset": dataset_metadata,
        "plot_rules": {"training_smoothing": False, "validation_interpolation": False,
            "missing_steps_filled": False, "best_markers_require_confirmed_completion": True}}
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figure_paths"] = plot_training(logs, completion, args.output)
    if table is not None:
        table_path = args.output / "selected_prompts.md"
        table_path.write_text(table, encoding="utf-8")
        summary["prompt_table"] = str(table_path)
    summary_path = args.output / "figure_sources.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figure_sources": str(summary_path), "figures": summary["figure_paths"],
        "completion_confirmed": completion["confirmed"],
        "logged_steps": {variant: len(log.rows) for variant, log in logs.items()}}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
