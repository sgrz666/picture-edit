"""Read-only probe: does the new Hi4D canvas guard open the 1389 caches already on disk?

The guard has two halves. ``canvas_hw`` disagrees with ``SIZE`` -> refuse. Pre-contract files have
no record at all, so every square >16 tensor is checked instead. That second half can only be
trusted against the real key/shape vocabulary of the existing files, which is what this measures.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import hi4d_qwen21_dataset as hi4d


def would_flag(key: str, value) -> bool:
    return (torch.is_tensor(value) and value.ndim >= 2 and value.shape[-1] == value.shape[-2] > 16
            and tuple(value.shape[-2:]) != (hi4d.SIZE, hi4d.SIZE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.cache_dir.glob("control_*.pt"))
    shapes = collections.Counter()
    with_record = 0
    flagged = []
    keys = collections.Counter()
    for path in files[: args.limit]:
        control = torch.load(path, map_location="cpu", weights_only=True)
        if "canvas_hw" in control:
            with_record += 1
        for key, value in control.items():
            keys[key] += 1
            if torch.is_tensor(value):
                shapes[(key, tuple(value.shape))] += 1
                if would_flag(key, value):
                    flagged.append({"file": path.name, "key": key, "shape": list(value.shape)})
    report = {
        "files_on_disk": len(files),
        "files_probed": min(len(files), args.limit),
        "size": hi4d.SIZE,
        "version": hi4d.VERSION,
        "files_with_canvas_record": with_record,
        "distinct_key_shape_pairs": len(shapes),
        "square_non_canvas_keys": sorted(
            [{"key": key, "shape": list(shape), "count": count}
             for (key, shape), count in shapes.items()
             if len(shape) >= 2 and shape[-1] == shape[-2] > 16 and shape[-2:] != (hi4d.SIZE, hi4d.SIZE)],
            key=lambda item: item["key"]),
        "guard_would_reject": flagged,
        "canvas_sized_squares_ok": sum(
            count for (key, shape), count in shapes.items()
            if len(shape) >= 2 and shape[-2:] == (hi4d.SIZE, hi4d.SIZE)),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
