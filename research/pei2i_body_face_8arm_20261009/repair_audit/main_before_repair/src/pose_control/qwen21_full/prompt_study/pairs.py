from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    INIT_SHA256,
    TRAIN_SHA256,
    VAL_SHA256,
    sha256_file,
    text_sha256,
)

TEST_GROUPS = ("seq_00010_00028", "seq_00241_00244", "seq_00289_00307")

DIAGNOSTIC_WHITELIST_FIELDS = (
    "pair_id",
    "source_appearance",
    "target_appearance",
    "sequence_id",
    "source_stem",
    "target_stem",
    "source_rgb",
    "target_rgb",
    "cross_person",
    "dataset_type",
    "target_depth_vis",
    "target_normal",
    "target_dwpose_json",
    "target_dwpose_png",
    "filtered_dwpose",
    "part_512",
    "v6_conditions",
    "smplestx_path",
)


def select_test24(eligible_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in TEST_GROUPS:
        candidates = sorted(
            (r for r in eligible_rows if r.get("source_video_group") == group),
            key=lambda r: (text_sha256("pei2i_20261009:" + str(r["pair_id"])), str(r["pair_id"])),
        )
        chosen: list[dict[str, Any]] = []
        frames: dict[str, list[int]] = defaultdict(list)
        targets: set[tuple[str, str]] = set()
        for row in candidates:
            seq = str(row["sequence_id"])
            target_stem_str = str(row["target_stem"] if "target_stem" in row else row["target"]["stem"])
            frame = int(target_stem_str)
            key = (seq, target_stem_str)
            if key in targets or any(abs(frame - old) < 8 for old in frames[seq]):
                continue
            chosen.append(dict(row))
            frames[seq].append(frame)
            targets.add(key)
            if len(chosen) == 8:
                break
        if len(chosen) != 8:
            raise ValueError(f"Cannot lock eight spaced targets for {group}: got {len(chosen)}")
        selected.extend(chosen)
    return selected


def audit_video_groups(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    face_train_groups: set[str] | None = None,
) -> dict[str, Any]:
    def extract_groups(rows: list[dict[str, Any]]) -> set[str]:
        groups = set()
        for r in rows:
            g = r.get("source_video_group")
            if g:
                groups.add(str(g))
        return groups

    train_groups = extract_groups(train_rows)
    val_groups = extract_groups(val_rows)
    test_groups = extract_groups(test_rows)
    face_groups = set(face_train_groups or ())

    overlap_train = test_groups & train_groups
    overlap_val = test_groups & val_groups
    overlap_face = test_groups & face_groups

    if overlap_train:
        raise ValueError(f"Test video groups overlap with train: {overlap_train}")
    if overlap_val:
        raise ValueError(f"Test video groups overlap with val: {overlap_val}")
    if overlap_face:
        raise ValueError(f"Test video groups overlap with face training groups: {overlap_face}")

    return {
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "face_groups": sorted(face_groups),
        "disjoint": True,
    }


def normalize_diagnostic_row(
    row: Mapping[str, Any],
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a diagnostic row (e.g. tiktok_00..05), stripping non-whitelisted fields."""
    target_app = str(row.get("target_appearance", row.get("sequence_id", "")))
    source_app = str(row.get("source_appearance", row.get("appearance", target_app)))
    source_stem = str(row.get("source_stem", row["source"]["stem"] if "source" in row else ""))
    target_stem = str(row.get("target_stem", row["target"]["stem"] if "target" in row else ""))
    pair_id = str(row.get("pair_id", f"{source_app}_{source_stem}__{target_app}_{target_stem}"))

    normalized: dict[str, Any] = {
        "pair_id": pair_id,
        "source_appearance": source_app,
        "target_appearance": target_app,
        "sequence_id": target_app,  # sequence_id binds to target geometry
        "source_stem": source_stem,
        "target_stem": target_stem,
        "source_rgb": str(row["source_rgb"]),
        "target_rgb": str(row["target_rgb"]),
        "cross_person": source_app != target_app,
        "dataset_type": "tiktok_v3",
    }

    # Pass-through whitelisted condition paths if present
    for field in (
        "target_depth_vis",
        "target_normal",
        "target_dwpose_json",
        "target_dwpose_png",
        "filtered_dwpose",
        "part_512",
        "v6_conditions",
        "smplestx_path",
    ):
        if field in row:
            normalized[field] = str(row[field])
        elif assets_root is not None:
            # Check standard path construction
            root = Path(assets_root)
            if field == "target_dwpose_json":
                p = root / "dwpose" / target_app / f"{target_stem}.json"
                if p.is_file():
                    normalized[field] = str(p)
            elif field == "part_512":
                p = root / "part_512" / target_app / f"{target_stem}.png"
                if p.is_file():
                    normalized[field] = str(p)
            elif field == "v6_conditions":
                p = root / "v6_conditions" / f"{target_app}.pt"
                if p.is_file():
                    normalized[field] = str(p)

    return normalized
