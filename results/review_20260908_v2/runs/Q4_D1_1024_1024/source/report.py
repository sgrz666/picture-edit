"""Build a dependency-free, evidence-backed CPU-only experiment report."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

SCORE_FIELDS = ("action", "identity", "clothing", "background", "anatomy")
FLAG_FIELDS = ("added_person", "missing_person", "wrong_person")
PROVENANCE_FIELDS = ("reviewer", "note")
STATUSES = {"PASS", "OOM", "INVALID", "TIMEOUT", "CANCELLED", "NOT_RUN"}
METRICS = ("wall_seconds", "peak_allocated_gib", "peak_reserved_gib", "peak_nvml_gib", "step_p50_seconds", "step_p95_seconds", "adapter_parameters")
CSV_FIELDS = ("id", "kind", "group", "task", "seed", "cfg", "steps", "resolution", "preprocess", "status", "validation_passed", *METRICS, *SCORE_FIELDS, *FLAG_FIELDS, *PROVENANCE_FIELDS)
IMAGE_NAMES = ("input_original", "input_model", "output", "output_content", "normal", "depth", "skeleton", "contact")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda _: None)
    except (OSError, ValueError):
        return default


def number(value):
    """Missing/non-finite values remain missing, never become zero."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def png_dimensions(path: Path):
    try:
        with path.open("rb") as stream:
            head = stream.read(24)
        if len(head) == 24 and head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            width, height = struct.unpack(">II", head[16:24])
            return {"width": width, "height": height}
    except OSError:
        pass
    return None


def normalize_scores(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {
        **{key: raw.get(key) if type(raw.get(key)) in (int, float) and raw.get(key) in (0, 1, 2) else None for key in SCORE_FIELDS},
        **{key: raw.get(key) if isinstance(raw.get(key), bool) else None for key in FLAG_FIELDS},
        **{key: raw.get(key) if isinstance(raw.get(key), str) else None for key in PROVENANCE_FIELDS},
    }


def load_task_evidence(directory: Path, config: dict, prefix: str):
    """Keep alternating task inputs and controls together without changing run config."""
    entries = config.get("task_evidence", {})
    if not isinstance(entries, dict):
        return {}
    result = {}
    for task, relative in entries.items():
        if not isinstance(relative, str):
            continue
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_dir():
            continue
        rel = path.relative_to(directory.resolve()).as_posix()
        task_prefix = prefix + quote(rel, safe="/") + "/"
        evidence = {p.name: task_prefix + quote(p.name, safe="") for p in path.iterdir() if p.is_file()}
        images = {name: {"url": evidence[name + ".png"], "dimensions": png_dimensions(path / (name + ".png"))}
                  for name in IMAGE_NAMES if name + ".png" in evidence}
        result[str(task)] = {"relative_path": rel, "images": images, "evidence": evidence,
                             "control_metadata": read_json(path / "control_metadata.json", {}),
                             "config": read_json(path / "config.json", {})}
    return result


def timeline_phases(events):
    """Group consecutive stage starts for display; leave source events untouched."""
    ordered = sorted((event for event in events if isinstance(event, dict)
                      and number(event.get("elapsed_seconds")) is not None),
                     key=lambda event: number(event["elapsed_seconds"]))
    phases = []
    for event in ordered:
        stage = event.get("stage")
        if not isinstance(stage, str) or not stage:
            continue
        elapsed = number(event["elapsed_seconds"])
        if event.get("event") == "start":
            if not phases or phases[-1]["stage"] != stage:
                phases.append({"stage": stage, "start_seconds": elapsed, "last_start_seconds": elapsed,
                               "end_seconds": None, "count": 1, "closed": False})
            else:
                phases[-1]["count"] += 1
                phases[-1]["last_start_seconds"] = elapsed
                phases[-1]["closed"] = False
        elif event.get("event") == "end":
            for phase in reversed(phases):
                if phase["stage"] == stage and not phase["closed"]:
                    phase["end_seconds"] = elapsed
                    phase["closed"] = True
                    break
    return phases


def load_run(directory: Path, root: Path, overrides: dict):
    config = read_json(directory / "config.json", {})
    config = config if isinstance(config, dict) else {}
    summary = read_json(directory / "summary.json")
    present = isinstance(summary, dict)
    summary = summary if present else {}
    run_id = str(config.get("id") or directory.name)
    status = str(summary.get("status", "NOT_RUN")).upper()
    warning = None
    if not present:
        warning = "summary.json 缺失或无法解析；未确认运行结果"
    elif status not in STATUSES:
        warning = "summary.json 中状态无法识别"
        status = "INVALID"
    prefix = "runs/" + quote(directory.name, safe="") + "/"
    evidence = {p.name: prefix + quote(p.name, safe="") for p in directory.iterdir() if p.is_file()}
    images = {}
    for name in IMAGE_NAMES:
        if name + ".png" in evidence:
            images[name] = {"url": evidence[name + ".png"], "dimensions": png_dimensions(directory / (name + ".png"))}
    events = []
    try:
        for line in (directory / "events.jsonl").read_text(encoding="utf-8-sig").splitlines():
            try:
                event = json.loads(line, parse_constant=lambda _: None)
                if isinstance(event, dict):
                    events.append(event)
            except ValueError:
                continue
    except OSError:
        pass
    samples = []
    try:
        with (directory / "gpu_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
            for sample in csv.DictReader(stream):
                samples.append({"elapsed_seconds": number(sample.get("elapsed_seconds")), "used_gib": number(sample.get("used_gib")), "stage": sample.get("stage")})
    except (OSError, csv.Error):
        pass
    original_scores = summary.get("scores", {})
    original_scores = original_scores if isinstance(original_scores, dict) else {}
    override = overrides.get(run_id, {})
    if isinstance(override, dict) and isinstance(override.get("scores"), dict):
        override = {**override, **override["scores"]}
    scores = normalize_scores({**original_scores, **(override if isinstance(override, dict) else {})})
    return {
        "id": run_id, "directory": directory.name, "config": config, "summary": summary,
        "status": status, "summary_present": present, "warning": warning,
        **{field: number(summary.get(field)) for field in METRICS},
        "stages": summary.get("stages") if isinstance(summary.get("stages"), list) else [],
        "validation_passed": summary.get("validation_passed") if isinstance(summary.get("validation_passed"), bool) else None,
        "control_metadata": read_json(directory / "control_metadata.json", {}),
        "task_evidence": load_task_evidence(directory, config, prefix),
        "source_version": read_json(directory / "source_version.json", summary.get("source_version", config.get("source_version"))),
        "scores": scores, "images": images, "evidence": evidence, "events": events, "gpu_samples": samples,
        "timeline_phases": timeline_phases(events),
    }


def aggregate(runs):
    """Aggregate only real observations; failure rows are included in counts."""
    counts = Counter(run["status"] for run in runs)
    completed = [run for run in runs if run["status"] == "PASS"]
    means = {}
    for field in METRICS:
        values = [run[field] for run in completed if run[field] is not None]
        means[field] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    score_means = {}
    for field in SCORE_FIELDS:
        values = [run["scores"][field] for run in completed if run["scores"][field] is not None]
        score_means[field] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    return {"total": len(runs), "counts": dict(counts), "pass_metrics": means, "pass_scores": score_means}


def csv_text(runs):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for run in runs:
        row = {key: run["config"].get(key) for key in CSV_FIELDS}
        row.update({key: run.get(key) for key in ("id", "status", "validation_passed", *METRICS)})
        row.update(run["scores"])
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                row[key] = "'" + value
        writer.writerow(row)
    return output.getvalue()


def build_report(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    overrides = read_json(root / "scores.json", {})
    overrides = overrides if isinstance(overrides, dict) else {}
    run_root = root / "runs"
    runs = [load_run(path, root, overrides) for path in sorted(run_root.iterdir()) if path.is_dir()] if run_root.is_dir() else []
    data = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "runs": runs, "aggregate": aggregate(runs),
            "capacity": read_json(root / "capacity.json"),
            "root_evidence": {name: name for name in ("capacity.json", "MEMORY_REPORT.md") if (root / name).is_file()}}
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, default=str)
    # The payload is a non-executable JSON script; escaping '<' prevents closing it.
    inline = serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
    (root / "index.html").write_text(template.replace("__REPORT_DATA__", inline), encoding="utf-8")
    (root / "results.json").write_text(serialized + "\n", encoding="utf-8")
    (root / "summary.csv").write_text(csv_text(runs), encoding="utf-8-sig")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="从已有实验记录生成中文 CPU-only HTML 报告")
    parser.add_argument("--root", required=True, type=Path, help="实验根目录，内含 runs/<id>/")
    args = parser.parse_args(argv)
    report = build_report(args.root)
    print(f"Report: {args.root.resolve() / 'index.html'} | {len(report['runs'])} runs")


if __name__ == "__main__":
    main()
