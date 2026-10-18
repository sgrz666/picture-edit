"""Qwen21 Pro 共享入口；status仅标准库，生产运行按新name独占输出。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[3]
P_ROOT = "/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1"
B_ROOT = "/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def allocate_run(output_root, name, *, legacy_workspace):
    root, legacy = Path(output_root), Path(legacy_workspace).resolve()
    if not root.is_absolute() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name or ""):
        raise ValueError("output-root must be absolute; name must be a single new descriptive directory name")
    root = root.resolve()
    if root == legacy or root.is_relative_to(legacy) or any(p.lower() in {"models", "datasets"} for p in root.parts):
        raise ValueError("Cannot write into legacy workspace, models or datasets")
    run = root / "runs" / name
    if not run.resolve().is_relative_to(root):
        raise ValueError("Run escapes output root")
    run.mkdir(parents=True, exist_ok=False)
    return run


def source_snapshot(root=CODE_ROOT, destination=None):
    """实际import源码的字节清单；从来不以manifest自报SHA替代当前文件。"""
    root = Path(root).resolve()
    files = [p for package in ("qwen21_pro", "qwen21_union")
             for p in (root / "src/pose_control" / package).glob("*.py")]
    dependencies = ("src/__init__.py", "src/pose_control/__init__.py", "src/data/__init__.py",
        "src/data/qwen21_union_dataset.py", "src/pose_edit/__init__.py", "src/pose_edit/geometry.py",
        "scripts/train/qwen21_pro.py", "scripts/run/qwen21_pro.py",
        "experiments/qwen21_pro_adapter_v1/scripts/run.py")
    files += [root / p for p in dependencies if (root / p).is_file()]
    inventory = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(files)}
    if not inventory:
        raise ValueError("Actual source snapshot is empty")
    if destination is not None:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=False)
        for name in inventory:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    return {"signature": _json_sha(inventory), "files": inventory, "code_root": str(root)}


def status(output_root, *, name=None):
    root = Path(output_root).resolve() / "runs"
    runs = []
    paths = [root / name] if name else (sorted(root.iterdir()) if root.is_dir() else [])
    for path in paths:
        if not path.is_dir():
            continue
        row = {"name": path.name, "path": str(path)}
        for kind in ("run", "gate", "report", "failure"):
            file = path / f"{kind}.json"
            if file.is_file():
                try:
                    value = json.loads(file.read_text(encoding="utf-8"))
                    row[kind] = ({k: v for k, v in value.items() if k in ("command", "profile", "status", "passed",
                        "engineering_passed", "cursor", "completed", "type", "error", "samples", "code_signature")}
                        if kind != "failure" else value)
                except (ValueError, OSError) as error:
                    row[kind] = {"unreadable": str(error)}
        runs.append(row)
    return {"output_root": str(Path(output_root).resolve()), "runs": runs}


def _dispatch(name):
    def invoke(args, run):
        from . import dataflow
        return getattr(dataflow, name)(args, run)
    return invoke


COMMANDS = {"prepare": _dispatch("prepare"), "cache": _dispatch("cache"), "audit": _dispatch("audit"),
            "smoke": _dispatch("train"), "train": _dispatch("train"), "overfit": _dispatch("train"),
            "ablate": _dispatch("train"), "infer": _dispatch("infer"), "refine": _dispatch("refine")}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (*COMMANDS, "status"):
        p = commands.add_parser(name)
        p.add_argument("--output-root", default=P_ROOT)
        p.add_argument("--name", required=name != "status")
        if name == "status":
            continue
        p.add_argument("--legacy-workspace", default=B_ROOT)
        p.add_argument("--project-root", default="/home/shangguanrz/project/pic-edit")
        p.add_argument("--conditions", help="explicit condition JSONL; default locked overfit6")
        p.add_argument("--supervision", help="independent supervision JSONL; default locked overfit6")
        p.add_argument("--sample-id", action="append", default=[])
        p.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
        p.add_argument("--union-checkpoint")
        p.add_argument("--source-root", help="pinned VideoX source root (optional upstream default)")
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        p.add_argument("--memory-mode", choices=("resident", "frozen_linear_cpu"), default="resident")
        p.add_argument("--memory-budget-gib", type=float,
                       help="fixed explicit GiB; default fixed once at startup to min(.9 total, free-1GiB)")
        p.add_argument("--arm", choices=("full4", "no_dense", "no_appearance", "no_contact", "full8"), default="full4")
        p.add_argument("--feature-run", help="verified effect1600 raw source cache run")
        p.add_argument("--cache-run", help="shared CLI cache run containing encoded/bound conditions")
        p.add_argument("--kind", choices=("legacy", "current"), default="legacy")
        p.add_argument("--source-geometry-index")
        p.add_argument("--current-geometry-index")
        p.add_argument("--contact-vertex-parts")
        p.add_argument("--geometry-python", help="existing body model Python; Qwen process never imports smplx")
        p.add_argument("--geometry-mode", choices=("source", "current", "retarget", "all"), default="all")
        p.add_argument("--canvas-hw", nargs=2, type=int, default=[512, 512])
        p.add_argument("--encode-supervision", action="store_true")
        p.add_argument("--resume", help="strict complete Pro checkpoint, always into this new name")
        p.add_argument("--checkpoint")
        p.add_argument("--exact-resume", action="store_true", help="smoke4 continuous=2+2; default smoke2")
        p.add_argument("--steps", type=int, default=40)
        p.add_argument("--seed", type=int, default=43)
        p.add_argument("--true-cfg-scale", type=float, default=1.)
        p.add_argument("--offload", choices=("none", "sync_group_offload"), default="none")
        p.add_argument("--coarse", help="generated coarse manifest for refine")
        p.add_argument("--roi-kind", choices=("face", "paired_hands"), default="face")
        p.add_argument("--roi-person", type=int)
        p.add_argument("--hand-endpoint", action="append", default=[], help="stable person_id:part_id (4/7), twice")
        p.add_argument("--engineering-probe", action="store_true", help="explicit low-step sampling role")
    return parser


def main(argv=None):
    parser = build_parser()
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(actual_argv)
    if args.command == "status":
        if args.name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.name):
            parser.error("status name must be a single run name")
        print(json.dumps(status(args.output_root, name=args.name), ensure_ascii=False))
        return 0
    # Normalize caller paths before any run manifest is produced.
    for key in ("output_root", "legacy_workspace", "project_root", "conditions", "supervision", "base_model",
                "union_checkpoint", "source_root", "feature_run", "cache_run", "source_geometry_index",
                "current_geometry_index", "contact_vertex_parts", "geometry_python", "resume", "checkpoint", "coarse"):
        value = getattr(args, key, None)
        if value:
            setattr(args, key, str(Path(value).resolve()))
    if args.conditions is None:
        args.conditions = str(Path(args.legacy_workspace) / "data/overfit6_conditions.jsonl")
    if args.supervision is None:
        args.supervision = str(Path(args.legacy_workspace) / "data/overfit6_supervision.jsonl")
    run = allocate_run(args.output_root, args.name, legacy_workspace=args.legacy_workspace)
    source = source_snapshot(destination=run / "code_snapshot")
    inputs = {key: {"path": value, "sha256": _sha(value)} for key, value in vars(args).items()
              if isinstance(value, str) and key in {"conditions", "supervision", "source_geometry_index",
                  "current_geometry_index", "contact_vertex_parts"} and Path(value).is_file()}
    _write(run / "run.json", {"schema": "qwen21_pro_run_v1", "command": args.command,
        "created_utc": datetime.now(timezone.utc).isoformat(), "argv": actual_argv,
        "config": vars(args), "architecture": {"hidden_dim": 4096, "width": 512, "heads": 8,
            "source_dim": 1600, "stage_indices": list(range(0, 16, 2)) if args.arm == "full8" else [0, 4, 8, 12],
            "control_lora": False, "official_control_blocks": 16},
        "seeds": {"initialization": 3407, "train": 3407, "sampler": 20261015, "fixed18": 43, "sampling": args.seed},
        "run_dir": str(run), "code_root": str(CODE_ROOT), "import_module": str(Path(__file__).resolve()),
        "code_signature": source["signature"], "source_files": source["files"], "inputs": inputs})
    started = time.monotonic()
    try:
        result = COMMANDS[args.command](args, run)
        _write(run / "report.json", {"command": args.command, "seconds": time.monotonic() - started, **result})
        print(json.dumps({"run_dir": str(run), **result}, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        _write(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                      "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
