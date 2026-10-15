#!/usr/bin/env python3
"""Prepare and lock dataset splits, geometry assets, and metadata for the PE-I2I study.

Performs deterministic test24 selection, video group disjointness audit,
DWPose confidence-filtering rendering, and lock.json generation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    INIT_SHA256,
    TRAIN_SHA256,
    VAL_SHA256,
    atomic_json,
    sha256_file,
    text_sha256,
)
from src.pose_control.qwen21_full.prompt_study.sampling import make_order
from src.pose_control.qwen21_full.prompt_study.pairs import (
    select_test24,
    audit_video_groups,
    normalize_diagnostic_row,
    TEST_GROUPS,
)
from src.pose_control.qwen21_full.prompt_study.geometry import render_dwpose


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def prepare_data_lock(
    source_pilot: Path,
    test_candidates_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    face_source_run: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data_dir = output_dir if output_dir.name == "data" else output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy train512 and val24 from source pilot
    src_train = source_pilot / "train512_pairs.jsonl"
    src_val = source_pilot / "val24_pairs.jsonl"
    if not src_train.is_file() or not src_val.is_file():
        # Maybe source_pilot is experiment root, check data subdir
        if (source_pilot / "data/train512_pairs.jsonl").is_file():
            src_train = source_pilot / "data/train512_pairs.jsonl"
            src_val = source_pilot / "data/val24_pairs.jsonl"
        else:
            raise FileNotFoundError(f"Missing train512/val24 under {source_pilot}")

    dst_train = data_dir / "train512_pairs.jsonl"
    dst_val = data_dir / "val24_pairs.jsonl"
    shutil.copyfile(src_train, dst_train)
    shutil.copyfile(src_val, dst_val)

    train_sha = sha256_file(dst_train)
    val_sha = sha256_file(dst_val)
    if train_sha != TRAIN_SHA256:
        raise ValueError(f"train512 hash mismatch: expected {TRAIN_SHA256}, got {train_sha}")
    if val_sha != VAL_SHA256:
        raise ValueError(f"val24 hash mismatch: expected {VAL_SHA256}, got {val_sha}")

    train_rows = read_jsonl(dst_train)
    val_rows = read_jsonl(dst_val)

    # 2. Select test24
    test_candidates = read_jsonl(test_candidates_path)
    rejected_candidates = []
    eligible_candidates = []

    for row in test_candidates:
        # Check basic asset presence if accessible
        group = row.get("source_video_group")
        if group not in TEST_GROUPS:
            rejected_candidates.append({"pair_id": row.get("pair_id"), "reason": f"group {group} not in test groups"})
            continue
        eligible_candidates.append(row)

    test_rows = select_test24(eligible_candidates)
    dst_test = data_dir / "test24_pairs.jsonl"
    write_jsonl(dst_test, test_rows)
    test_sha = sha256_file(dst_test)

    dst_rejected = data_dir / "rejected_test_candidates.jsonl"
    write_jsonl(dst_rejected, rejected_candidates)

    # 3. Normalize diagnostic rows
    raw_diag = read_jsonl(diagnostics_path)
    diag_rows = [normalize_diagnostic_row(r) for r in raw_diag]
    dst_diag = data_dir / "diagnostic6_pairs.jsonl"
    write_jsonl(dst_diag, diag_rows)
    diag_sha = sha256_file(dst_diag)

    # 4. Audit video groups
    group_audit = audit_video_groups(train_rows, val_rows, test_rows)

    # 5. Build geometry previews for rows with available dwpose_json
    geom_dir = data_dir / "geometry"
    assets_records = []
    all_rows = [("train", r) for r in train_rows] + [("val", r) for r in val_rows] + [("test", r) for r in test_rows] + [("diag", r) for r in diag_rows]
    for split_name, row in all_rows:
        pid = row["pair_id"]
        dwpose_json = row.get("target_dwpose_json")
        if dwpose_json and Path(dwpose_json).is_file():
            pair_geom = geom_dir / pid
            pair_geom.mkdir(parents=True, exist_ok=True)
            rec = json.loads(Path(dwpose_json).read_text(encoding="utf-8"))
            filt_img, diag = render_dwpose(rec)
            filt_path = pair_geom / "dwpose_filtered.png"
            filt_img.save(filt_path)
            orig_png = row.get("target_dwpose_png") or row.get("target_dwpose")
            if orig_png and Path(orig_png).is_file():
                shutil.copyfile(orig_png, pair_geom / "dwpose_original.png")
            assets_records.append({
                "pair_id": pid,
                "split": split_name,
                "filtered_dwpose": str(filt_path),
                "diagnostics": diag,
            })

    write_jsonl(data_dir / "assets.jsonl", assets_records)

    # 6. Build and write lock.json
    lock_payload = {
        "schema": "qwen21_pei2i_data_lock_v1",
        "epoch_order": make_order(),
        "splits": {
            "train": {"path": "train512_pairs.jsonl", "count": len(train_rows), "sha256": train_sha},
            "val": {"path": "val24_pairs.jsonl", "count": len(val_rows), "sha256": val_sha},
            "test": {"path": "test24_pairs.jsonl", "count": len(test_rows), "sha256": test_sha},
            "diagnostic": {"path": "diagnostic6_pairs.jsonl", "count": len(diag_rows), "sha256": diag_sha},
        },
        "video_group_audit": group_audit,
        "initial_checkpoint_sha256": INIT_SHA256,
    }
    lock_path = data_dir / "lock.json"
    atomic_json(lock_path, lock_payload)
    return lock_payload


def verify_data_lock(data_dir: Path) -> bool:
    lock_path = data_dir / "lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing lock.json in {data_dir}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "qwen21_pei2i_data_lock_v1":
        raise ValueError("Invalid lock schema")

    splits = lock["splits"]
    for split_key, expected in splits.items():
        split_file = data_dir / expected["path"]
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        actual_sha = sha256_file(split_file)
        if actual_sha != expected["sha256"]:
            raise ValueError(f"Hash mismatch on {split_key}: expected {expected['sha256']}, got {actual_sha}")
        rows = read_jsonl(split_file)
        if len(rows) != expected["count"]:
            raise ValueError(f"Count mismatch on {split_key}: expected {expected['count']}, got {len(rows)}")

    if splits["train"]["sha256"] != TRAIN_SHA256:
        raise ValueError("train512 does not match expected TRAIN_SHA256")
    if splits["val"]["sha256"] != VAL_SHA256:
        raise ValueError("val24 does not match expected VAL_SHA256")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or verify PE-I2I prompt study data lock")
    parser.add_argument("--source-pilot", type=Path, help="Directory containing original pilot train512 and val24")
    parser.add_argument("--face-source-run", type=Path, help="Face run source directory")
    parser.add_argument("--test-candidates", type=Path, help="Manifest for test candidates")
    parser.add_argument("--diagnostics", type=Path, help="Manifest for diagnostic pairs")
    parser.add_argument("--output", type=Path, help="Output data directory")
    parser.add_argument("--verify", action="store_true", help="Verify existing data lock")
    args = parser.parse_args()

    if args.verify:
        target_dir = args.output if args.output else Path("data")
        verify_data_lock(target_dir)
        print("Data lock verification succeeded.")
    else:
        if not all([args.source_pilot, args.test_candidates, args.diagnostics, args.output]):
            parser.error("Preparing lock requires --source-pilot, --test-candidates, --diagnostics, --output")
        prepare_data_lock(
            args.source_pilot,
            args.test_candidates,
            args.diagnostics,
            args.output,
            args.face_source_run,
        )
        print("Data lock prepared successfully.")


if __name__ == "__main__":
    main()
