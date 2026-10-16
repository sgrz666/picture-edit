"""续训时继承 checkpoint 对应的完整日志前缀，不改写父 run。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .checkpoint import FORMAT_VERSION

PROVENANCE_NAME = "resume_log_provenance.json"


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value in training log: {value}")


def _prefix(path, count):
    if not path.is_file():
        if count:
            raise FileNotFoundError(f"Checkpoint requires {count} log rows: {path}")
        data, parent_sha = b"", None
    else:
        # Take one immutable byte snapshot. Parent may contain later updates or
        # an unfinished later row; only the selected prefix needs to parse.
        data = path.read_bytes()
        parent_sha = hashlib.sha256(data).hexdigest()
    lines = data.splitlines(keepends=True)
    if len(lines) < count:
        raise ValueError(f"Truncated parent log {path}: {len(lines)} rows < {count}")
    selected, records = lines[:count], []
    for index, line in enumerate(selected):
        if not line.endswith(b"\n"):
            raise ValueError(f"Truncated log line {index + 1}: {path}")
        try:
            record = json.loads(line, parse_constant=_invalid_constant)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"Invalid log JSON at line {index + 1}: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Log row must be an object at line {index + 1}: {path}")
        records.append(record)
    prefix = b"".join(selected)
    return prefix, records, {"parent_log": str(path), "parent_log_sha256": parent_sha,
                             "parent_bytes": len(data), "selected_lines": count,
                             "selected_prefix_bytes": len(prefix),
                             "selected_prefix_sha256": hashlib.sha256(prefix).hexdigest()}


def _expect_integer(record, name, expected, label):
    if type(record.get(name)) is not int or record[name] != expected:
        raise ValueError(f"Log continuity differs at {label}: {name} must be {expected}")


def _validate_continuity(microsteps, updates, accumulation):
    for position, record in enumerate(microsteps):
        _expect_integer(record, "position", position, f"microstep row {position + 1}")
        _expect_integer(record, "update", position // accumulation, f"microstep row {position + 1}")
        _expect_integer(record, "microstep", position % accumulation, f"microstep row {position + 1}")
    for index, record in enumerate(updates):
        _expect_integer(record, "optimizer_step", index + 1, f"update row {index + 1}")
        if "trajectory" not in record:
            continue
        trajectory = record["trajectory"]
        if not isinstance(trajectory, list) or len(trajectory) != accumulation:
            raise ValueError(f"Update trajectory length differs at step {index + 1}")
        for microstep, entry in enumerate(trajectory):
            if not isinstance(entry, dict):
                raise ValueError(f"Update trajectory must contain objects at step {index + 1}")
            position = index * accumulation + microstep
            actual = microsteps[position]
            for key in ("position", "update", "microstep", "dataset", "sample_id", "index",
                        "noise_seed", "timestep_seed", "sigma", "timestep", "loss"):
                if key in entry and (key not in actual or entry[key] != actual[key]):
                    raise ValueError(f"Update/microstep trajectory differs at position {position}: {key}")


def prepare_resume_logs(new_run, resume_checkpoint, cursor):
    """校验来源后以 x 模式创建两份前缀和 provenance，返回 provenance。

    调用者应先 load_checkpoint 恢复并验证完整训练状态，再调用本函数；
    本函数重新校验 checkpoint 文件 SHA256、manifest 与传入 cursor。
    所有连续性/源数据检查和已有目标检查都在首次写入前完成。
    """
    cursor.validate()
    if cursor.loss_log_cursor != cursor.microstep or cursor.gradient_log_cursor != cursor.optimizer_step:
        raise ValueError("Full-history log cursors must match completed microsteps and optimizer updates")
    new_run = Path(new_run).resolve(strict=True)
    checkpoint = Path(resume_checkpoint).resolve(strict=True)
    if not new_run.is_dir() or not checkpoint.is_dir():
        raise ValueError("new_run and resume_checkpoint must be existing directories")
    parent_run = checkpoint.parent
    if new_run == parent_run or parent_run in new_run.parents or new_run in parent_run.parents:
        raise ValueError("Resume log destination must be a separate run directory")
    outputs = {name: new_run / name for name in ("microsteps.jsonl", "updates.jsonl", PROVENANCE_NAME)}
    for path in outputs.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Resume logs refuse to overwrite: {path}")
    manifest_path = checkpoint / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint format version differs")
    if manifest.get("cursor") != asdict(cursor):
        raise ValueError("Checkpoint manifest cursor differs from restored cursor")
    state_sha = _sha_file(checkpoint / "state.pt")
    if state_sha != manifest.get("state_sha256"):
        raise ValueError("Parent checkpoint state checksum differs")
    micro_prefix, microsteps, micro_meta = _prefix(parent_run / "microsteps.jsonl", cursor.loss_log_cursor)
    update_prefix, updates, update_meta = _prefix(parent_run / "updates.jsonl", cursor.gradient_log_cursor)
    _validate_continuity(microsteps, updates, cursor.accumulation_steps)
    provenance = {"format_version": "qwen21_union_resume_logs_v1", "new_run": str(new_run),
                  "parent_run": str(parent_run), "parent_checkpoint": str(checkpoint),
                  "checkpoint_manifest_sha256": _sha_file(manifest_path), "state_sha256": state_sha,
                  "cursor": asdict(cursor),
                  "logs": {"microsteps.jsonl": micro_meta, "updates.jsonl": update_meta}}
    for name, data in (("microsteps.jsonl", micro_prefix), ("updates.jsonl", update_prefix)):
        with outputs[name].open("xb") as stream:
            stream.write(data)
    with outputs[PROVENANCE_NAME].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return provenance
