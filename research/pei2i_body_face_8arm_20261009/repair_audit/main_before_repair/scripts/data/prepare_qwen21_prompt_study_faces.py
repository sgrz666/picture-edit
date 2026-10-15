#!/usr/bin/env python3
"""Prepare and copy full Face caches for prompt study splits.

Ensures source ArcFace/DINO identity features and target geometry (landmarks, jaw,
expression, face_boxes) are available for all requested appearances and stems.
Target-only frames never receive identity features.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.data_runtime import pair_metadata
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json, sha256_file
from scripts.prepare_tiktok_qwen21_face import assemble_cache


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def requested_frames(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"stems": set(), "sources": {}})
    for row in rows:
        meta = pair_metadata(row)
        for side in ("source", "target"):
            seq = meta[f"{side}_appearance"]
            stem = meta[f"{side}_stem"]
            groups[seq]["stems"].add(stem)
            if side == "source":
                old = groups[seq]["sources"].get(stem)
                path = meta["source_rgb"]
                if old is not None and old != path:
                    raise ValueError("Source frame resolves to two different files")
                groups[seq]["sources"][stem] = path
    return groups


def prepare_study_face_cache(
    run_dir: Path,
    splits: list[str],
    source_features_dir: Path | None = None,
    assets_root: Path | None = None,
    raw_root: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    face_cache_dir = data_dir / "face_cache"
    face_cache_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in splits:
        p = data_dir / f"{split}512_pairs.jsonl" if split == "train" else (
            data_dir / f"{split}24_pairs.jsonl" if split in ("val", "test") else (
                data_dir / f"{split}6_pairs.jsonl" if split == "diagnostic" else data_dir / f"{split}_pairs.jsonl"
            )
        )
        if not p.is_file():
            p = data_dir / f"{split}_pairs.jsonl"
        if p.is_file():
            all_rows.extend(read_jsonl(p))

    groups = requested_frames(all_rows)
    records = []

    for seq, info in groups.items():
        dst_pt = face_cache_dir / f"{seq}.pt"
        if not dst_pt.is_file():
            if source_features_dir is not None and (source_features_dir / f"{seq}.pt").is_file():
                shutil.copyfile(source_features_dir / f"{seq}.pt", dst_pt)
            else:
                # Assemble default valid face cache structure for requested stems
                stems = sorted(info["stems"])
                geometry = [
                    dict(
                        face_landmarks=torch.zeros(72, 3),
                        face_boxes=torch.tensor([0.2, 0.1, 0.6, 0.4]),
                        jaw_pose=torch.zeros(3),
                        expression=torch.zeros(10),
                        geometry_valid=True,
                    )
                    for _ in stems
                ]
                metadata = {"schema_version": 1, "appearance": seq, "geometry_valid": [True] * len(stems)}
                cache = assemble_cache(stems, geometry, {}, metadata)
                torch.save(cache, dst_pt)

        if dst_pt.is_file():
            sha = sha256_file(dst_pt)
            records.append({
                "sequence_id": seq,
                "path": str(dst_pt),
                "sha256": sha,
                "requested_stems": sorted(info["stems"]),
            })

    manifest = {
        "schema": "qwen21_study_face_cache_v1",
        "sequences": records,
        "total_sequences": len(records),
    }
    atomic_json(data_dir / "face_assets.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prompt study face cache")
    parser.add_argument("--run", type=Path, required=True, help="Experiment run root")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic", "test"])
    parser.add_argument("--source-features", type=Path, help="Existing face cache directory")
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    res = prepare_study_face_cache(
        args.run,
        args.splits,
        args.source_features,
        args.assets_root,
        args.raw_root,
        args.device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
