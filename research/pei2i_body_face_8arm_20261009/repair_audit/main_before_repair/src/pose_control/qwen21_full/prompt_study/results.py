from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .protocol import (
    ARMS,
    EVAL_STEPS,
    FULL_VAL_STEPS,
    TRAIN_MONITOR,
    VAL_MONITOR,
)


def eval_jobs(*, include_final_test: bool = False) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def append(arm: str, step: int, split: str, indices: Any, seeds: tuple[int, ...]) -> None:
        for seed in seeds:
            for index in indices:
                jobs.append(dict(arm=arm, step=step, split=split, pair_index=index, seed=seed))

    for arm in ARMS:
        for step in EVAL_STEPS:
            append(arm, step, "train", TRAIN_MONITOR, (42,))
            if step in FULL_VAL_STEPS:
                append(arm, step, "val", range(24), (42, 43))
            else:
                append(arm, step, "val", VAL_MONITOR, (42,))
            if step in (0, 512, 1024):
                append(arm, step, "diagnostic", range(6), (42,))
        if include_final_test:
            for step in (0, 1024):
                append(arm, step, "test", range(24), (42, 43))
    return jobs


def paired_metric(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    right_map = {(r["split"], r["pair_id"]): r for r in right}
    values: list[float] = []
    group_values: dict[str, list[float]] = defaultdict(list)

    def finite(value: Any) -> bool:
        return (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    for row in left:
        other = right_map.get((row["split"], row["pair_id"]))
        a = row["metrics"].get(metric)
        b = None if other is None else other["metrics"].get(metric)
        if finite(a) and finite(b):
            diff = float(b - a)
            values.append(diff)
            grp = str(row.get("group", "default"))
            group_values[grp].append(diff)

    means = {g: sum(v) / len(v) for g, v in group_values.items()} if group_values else {}
    return {
        "expected_n": len(left),
        "valid_pair_n": len(values),
        "missing_or_invalid_n": len(left) - len(values),
        "mean_difference": sum(values) / len(values) if values else None,
        "group_macro_mean_difference": sum(means.values()) / len(means) if means else None,
        "group_means": means,
    }


def compare_private_continuations(
    left_state: dict[str, Any],
    right_state: dict[str, Any],
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.run.run_qwen21_joint_pilot import _state_drift, _loss_drift
    from src.pose_control.qwen21_full.joint_training import assert_exact_state

    errors: list[str] = []
    if (
        left_state["step"] != 4
        or right_state["step"] != 4
        or len(left_rows) != 4
        or len(right_rows) != 4
    ):
        raise ValueError("Private continuation must end at four updates")
    for key in ("step", "sample_order", "joint_signature", "joint_initialization_sha256"):
        assert_exact_state(left_state[key], right_state[key], key)
    assert_exact_state(left_rows[:2], right_rows[:2], "forked_metrics_prefix")
    for a, b in zip(left_rows, right_rows):
        for key in (
            "step",
            "pair_index",
            "pair_id",
            "noise_seed",
            "prompt_sha256",
            "sigma",
            "face_active",
        ):
            assert_exact_state(a[key], b[key], f"metrics.{key}")
    if len(left_rows) != len(right_rows):
        raise ValueError("Continuation metric lengths differ")
    return {
        "adapter": _state_drift(
            left_state["adapter_state_dict"],
            right_state["adapter_state_dict"],
            "adapter",
            errors,
        ),
        "optimizer": _state_drift(
            left_state["optimizer_state_dict"],
            right_state["optimizer_state_dict"],
            "resume.optimizer",
            errors,
        ),
        "losses": _loss_drift(left_rows, right_rows, "losses", errors),
        "errors": errors,
    }


PROHIBITED_EXPORT_EXTENSIONS = {".pt", ".safetensors", ".bin", ".onnx"}
ALLOWED_EXPORT_PREFIXES = ("data", "prompts", "training", "evaluation", "report", "preflight")


def aggregate_pair_metrics(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("identity_A", "pose_pck_002", "bg_l1_A", "pixel_l1_GT"),
) -> dict[str, Any]:
    def finite(val: Any) -> bool:
        return isinstance(val, (float, int)) and not isinstance(val, bool) and math.isfinite(val)

    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pairs[(str(r.get("split", "val")), str(r.get("pair_id", "")))].append(r)

    aggregated_rows: list[dict[str, Any]] = []
    group_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_values: dict[str, list[float]] = defaultdict(list)

    for (split, pair_id), items in sorted(pairs.items()):
        grp = str(items[0].get("group", "default"))
        pair_metrics: dict[str, float | None] = {}
        for m in metrics:
            vals = [
                it["metrics"][m]
                for it in items
                if "metrics" in it and m in it["metrics"] and finite(it["metrics"][m])
            ]
            if vals:
                mean_v = sum(vals) / len(vals)
                pair_metrics[m] = mean_v
                group_values[grp][m].append(mean_v)
                metric_values[m].append(mean_v)
            else:
                pair_metrics[m] = None
        aggregated_rows.append({
            "split": split,
            "pair_id": pair_id,
            "group": grp,
            "metrics": pair_metrics,
            "raw_seeds": [it.get("seed") for it in items],
        })

    group_means: dict[str, dict[str, float]] = defaultdict(dict)
    for g, m_dict in group_values.items():
        for m, vals in m_dict.items():
            if vals:
                group_means[g][m] = sum(vals) / len(vals)

    group_macro_means: dict[str, float | None] = {}
    overall_means: dict[str, float | None] = {}
    valid_counts: dict[str, int] = {}
    for m in metrics:
        g_vals = [group_means[g][m] for g in group_means if m in group_means[g]]
        group_macro_means[m] = sum(g_vals) / len(g_vals) if g_vals else None
        overall_means[m] = sum(metric_values[m]) / len(metric_values[m]) if metric_values[m] else None
        valid_counts[m] = len(metric_values[m])

    return {
        "pairs": aggregated_rows,
        "group_means": dict(group_means),
        "group_macro_means": group_macro_means,
        "overall_means": overall_means,
        "valid_counts": valid_counts,
        "expected_n": len(pairs),
    }


def verify_report_relative_links(report_html_path: Path, run_dir: Path) -> list[str]:
    """Parse report/index.html and verify every href/src relative link points to an existing file."""
    import re
    from urllib.parse import unquote

    if not report_html_path.is_file():
        raise FileNotFoundError(f"Report file not found: {report_html_path}")

    content = report_html_path.read_text(encoding="utf-8")
    report_dir = report_html_path.parent
    run_dir = run_dir.resolve()

    missing_links: list[str] = []
    pattern = re.compile(r'(?:href|src)=["\']([^"\']+)["\']', re.IGNORECASE)
    for match in pattern.finditer(content):
        raw_url = match.group(1).strip()
        if not raw_url or raw_url.startswith(("#", "http://", "https://", "data:", "javascript:")):
            continue
        cleaned = unquote(raw_url.split("?")[0].split("#")[0])
        target = (report_dir / cleaned).resolve()
        if not target.is_relative_to(run_dir):
            missing_links.append(f"Escapes run directory: {raw_url}")
            continue
        if not target.exists():
            missing_links.append(f"Missing referenced file: {raw_url} (resolved: {target})")

    return missing_links


def build_export_manifest(run_dir: Path) -> list[dict[str, Any]]:
    """Collect exportable files excluding .pt, .safetensors, .bin, .onnx, and cache/."""
    from .protocol import sha256_file

    run_dir = run_dir.resolve()
    manifest: list[dict[str, Any]] = []

    for prefix in ALLOWED_EXPORT_PREFIXES:
        pdir = run_dir / prefix
        if not pdir.is_dir():
            continue
        for path in sorted(pdir.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(run_dir):
                raise ValueError(f"File {path} escapes run directory {run_dir}")
            rel = path.relative_to(run_dir)
            rel_parts = [p.lower() for p in rel.parts]
            if any(p in ("cache", ".cache", "__pycache__") for p in rel_parts):
                continue
            if path.suffix.lower() in PROHIBITED_EXPORT_EXTENSIONS:
                continue

            manifest.append({
                "path": rel.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return manifest


def package_export(run_dir: Path, output_tar: Path | None = None) -> dict[str, Any]:
    """Package verified exportable files into export/results.tar.gz and write export/files.json."""
    import shutil
    import tarfile

    run_dir = run_dir.resolve()
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = export_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify report links if index.html exists
    report_html = run_dir / "report" / "index.html"
    if report_html.is_file():
        missing = verify_report_relative_links(report_html, run_dir)
        if missing:
            raise RuntimeError(f"Report links verification failed: {missing[:5]}")

    # 2. Build manifest
    files = build_export_manifest(run_dir)
    files_json = export_dir / "files.json"
    files_json.write_text(json.dumps(files, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # 3. Stage files
    for entry in files:
        rel = Path(entry["path"])
        src = run_dir / rel
        dst = staging_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Copy files.json into staging as well
    shutil.copy2(files_json, staging_dir / "files.json")

    # 4. Create tar.gz
    tar_path = output_tar or (export_dir / "results.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for entry in files:
            rel = entry["path"]
            tar.add(staging_dir / rel, arcname=rel)
        tar.add(staging_dir / "files.json", arcname="files.json")

    return {
        "tar_path": str(tar_path),
        "files_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files_json": str(files_json),
    }
