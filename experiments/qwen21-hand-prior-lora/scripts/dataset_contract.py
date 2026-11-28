"""Read immutable Parquet exports in manifest order, including the legacy single shard."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export_fingerprint(shards):
    return shards[0]["sha256"] if len(shards) == 1 else fingerprint(shards)


def read_export(root, split, *, verify=True):
    root = Path(root).resolve()
    record = json.loads(root.with_name(root.name + ".json").read_text(encoding="utf-8"))
    if record["split"] != split or record["pairs"] < 1:
        raise ValueError(f"Expected a completed {split} Parquet export")
    shards = record.get("shards")
    if shards is None:
        shards = [{"file": record["parquet"], "sha256": record["parquet_sha256"], "rows": record["pairs"]}]
    names = [shard["file"] for shard in shards]
    if (not names or len(names) != len(set(names)) or set(names) != {p.name for p in root.iterdir()}
            or any(Path(name).name != name or not name.endswith(".parquet") for name in names)
            or any(shard["rows"] < 1 for shard in shards) or sum(x["rows"] for x in shards) != record["pairs"]):
        raise ValueError("Parquet shard manifest disagrees with the dataset directory")
    paths = [root / name for name in names]
    if verify:
        for path, shard in zip(paths, shards):
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Parquet SHA256 mismatch: {path}")
    record["dataset_sha256"] = export_fingerprint(shards)
    record["condition_columns"] = record.get("condition_columns", ["source"])
    return record, paths


def load_export(paths, split, cache_dir):
    from datasets import load_dataset

    return load_dataset("parquet", data_files={split: list(map(str, paths))}, split=split, cache_dir=cache_dir)
