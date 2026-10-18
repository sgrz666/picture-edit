#!/usr/bin/env python3
"""CPU acceptance for the unchanged V6 copy and its legacy architecture tests.

Writes generated test copies only below the experiment acceptance directory.
The only source transform is the control namespace (plus config file lookup).
No original tests or control modules are edited.
"""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    "test_condition_injector_v1.py",
    "test_adapter_reasoning_v1.py",
    "test_condition_bridge_contract.py",
    "test_adapter_v6_architecture.py",
    "test_deepgen_control_interface_v1.py",
    "test_v6_detail_contracts.py",
    "test_v6_detail_preparer.py",
    "test_v6_detail_branch.py",
    "test_v65_face_foundation.py",
    "test_v65_face_integration.py",
    "test_v66_hand_core.py",
)


class Results:
    def __init__(self):
        self.tests = []
        self.collection_errors = []

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed:
            self.tests.append({
                "nodeid": report.nodeid, "phase": report.when,
                "outcome": report.outcome, "seconds": report.duration,
                "failure": str(report.longrepr) if report.failed else None,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/home/shangguanrz/nihaochen/qwen21_full/acceptance"))
    args = parser.parse_args()
    # Set before importing torch/diffusers. This program never uses a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    import torch
    import pytest
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mirror = args.output_dir / "legacy_tests_new_namespace"
    mirror.mkdir(exist_ok=True)
    sources = []
    # Mirror fixture dependencies too; collect only the explicitly selected suites.
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name.startswith("test_qwen21_full"):
            continue
        original = path.read_text()
        converted = original.replace("src.pose_control.v6", "src.pose_control.qwen21_full")
        converted = converted.replace("Path(__file__).parents[1]", f"Path({str(ROOT)!r})")
        # All fixture imports must target the mirrored test modules; an installed
        # third-party ``tests`` package otherwise shadows this repo's namespace.
        converted = converted.replace("from tests.test_", "from test_")
        (mirror / path.name).write_text(converted)
        if path.name in SUITES:
            sources.append({"original": str(path), "executed_copy": str(mirror / path.name),
                            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                            "executed_sha256": hashlib.sha256(converted.encode()).hexdigest()})
    sys.path[:0] = [str(mirror), str(ROOT)]
    os.chdir(ROOT)
    results = Results()
    started = time.monotonic()
    exit_code = pytest.main([
        str(ROOT / "tests/test_qwen21_full_preservation.py"),
        *(str(mirror / name) for name in SUITES),
        "-q", "--disable-warnings", "--tb=short",
        f"--junitxml={args.output_dir / 'preservation_junit.xml'}",
    ], plugins=[results])
    counts = dict(Counter(t["outcome"] for t in results.tests if t["phase"] == "call"))
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU architecture preservation; not Qwen GPU bridge or training acceptance",
        "device": "cpu", "threads": 4, "torch_version": torch.__version__,
        "source_namespace": "src.pose_control.v6",
        "tested_namespace": "src.pose_control.qwen21_full",
        "file_manifest": "/home/shangguanrz/nihaochen/qwen21_full/migration_manifest.json",
        "copied_source_files": 47,
        "parity": "Identical inputs and state_dict; exact FP32 forward and parameter-gradient parity with activated residual heads",
        "coverage": ["geometry", "single/mixed/dual reasoner", "appearance and identity binding",
                     "recurrent control attention", "detail", "Face preprocessing and Adapter",
                     "Hand preprocessing and Adapter", "strength schedules", "checkpoint migration"],
        "not_claimed": ["Real-data Hand training", "Real-data dual-person training",
                        "Qwen interface acceptance", "Image quality or generalization"],
        "test_source_transforms": ["src.pose_control.v6 -> src.pose_control.qwen21_full",
                                   "Keep original repo path for config assertions",
                                   "Route tests.test_* fixture imports to mirrored test_* modules"],
        "reused_suites": sources, "counts": counts, "exit_code": int(exit_code),
        "seconds": time.monotonic() - started,
        "collection_errors": results.collection_errors, "tests": results.tests,
    }
    (args.output_dir / "preservation_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"report": str(args.output_dir / "preservation_results.json"),
                      "counts": counts, "exit_code": int(exit_code)}))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
