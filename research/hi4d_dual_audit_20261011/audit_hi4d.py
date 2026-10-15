"""Read-only inventory of the remote Hi4D assets and adapter entry points.

Run with the project's qwen21 Python environment on ssh sg. No model is
loaded, no archive is extracted, and no training or generation is performed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def audit(project: Path, archives: Path, controls: Path) -> dict:
    manifests = project / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests"
    splits = {}
    combined = []
    for split in ("train", "val", "test"):
        path = manifests / f"main_pairs_{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        combined.extend(rows)
        path_fields = [key for key, value in rows[0].items()
                       if isinstance(value, str) and value.startswith("/")]
        splits[split] = {
            "rows": len(rows),
            "pair_ids": sorted({row["pair"] for row in rows}),
            "camera_ids": sorted({row["camera_id"] for row in rows}),
            "unique_sources": len({row["source_rgb"] for row in rows}),
            "unique_targets": len({row["target_rgb"] for row in rows}),
            "missing_paths": {key: sum(not Path(row[key]).is_file() for row in rows)
                              for key in path_fields},
            "zero_contact_vertex_rows": sum(row["target_contact_vertex_count_total"] == 0
                                            for row in rows),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    version = "hi4d_qwen21_dual_v2"
    control_paths = {
        controls / ("control_" + hashlib.sha256(
            (version + row["target_geometry_npz"] + row["target_dwpose_A"]
             + row["target_dwpose_B"]).encode()).hexdigest()[:24] + ".pt")
        for row in combined
    }
    pair_sets = {split: set(data["pair_ids"]) for split, data in splits.items()}
    disjoint = not any(pair_sets[a] & pair_sets[b]
                       for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    archive_files = sorted(archives.glob("*.tar.gz"))
    raw = project / "datasets/Hi4D_pilot_v1/raw"
    row = combined[0]

    import numpy as np
    import torch
    from PIL import Image

    npz = {}
    for key in ("target_smpl", "target_camera", "target_geometry_npz", "target_dwpose_A"):
        with np.load(row[key], allow_pickle=False) as data:
            npz[key] = {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
                        for name in data.files}
    images = {}
    for key in ("source_rgb", "target_rgb", "source_person_mask_A", "target_person_mask_A"):
        with Image.open(row[key]) as image:
            images[key] = {"size": list(image.size), "mode": image.mode}

    control_key = hashlib.sha256((version + row["target_geometry_npz"]
                                 + row["target_dwpose_A"] + row["target_dwpose_B"]).encode()).hexdigest()[:24]
    sample_control = torch.load(controls / f"control_{control_key}.pt",
                                map_location="cpu", weights_only=True)
    code_paths = (
        "train_hi4d_qwen21_dual.py", "src/data/hi4d_qwen21_dataset.py",
        "src/pose_control/qwen21/dual_adapter.py", "src/pose_control/qwen21/dual_inference.py",
        "src/pose_control/qwen21/action_prompts.py", "src/pose_control/qwen21_full/qwen_adapter.py",
        "src/pose_control/qwen21_full/execution.py", "src/pose_control/qwen21_full/data_runtime.py",
        "src/pose_control/qwen21_full/joint_training.py", "scripts/eval/eval_hi4d_native_binding.py",
    )
    source_hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                     for name in code_paths}
    disk = {}
    for name, path in (("archives_disk", archives), ("pilot_disk", raw)):
        info = os.statvfs(path)
        disk[name] = {"path": str(path), "available_gib": round(info.f_bavail * info.f_frsize / 2**30, 2)}

    sys_path_added = str(project)
    import sys
    sys.path.insert(0, sys_path_added)
    from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset
    dataset = Hi4DQwen21Dataset(manifests / "main_pairs_train.jsonl", controls, load_target=False)
    sample = dataset[0]
    checks = {
        "all_manifest_paths_present": all(not any(data["missing_paths"].values()) for data in splits.values()),
        "all_current_control_caches_present": all(path.is_file() for path in control_paths),
        "pair_ids_disjoint_between_splits": disjoint,
        "sample_load_target_false_has_no_target_image_tensor": "tgt_image" not in sample,
        "sample_has_two_source_masks": all(tuple(sample[name].shape) == (1, 512, 512)
                                           for name in ("source_mask_a", "source_mask_b")),
        "sample_has_two_25_channel_pose_maps": all(tuple(sample[name].shape) == (25, 512, 512)
                                                  for name in ("pose_heatmap_a", "pose_heatmap_b")),
    }
    return {
        "audit_date": "2026-10-11", "project": str(project), "read_only": True,
        "archives": {"path": str(archives), "count": len(archive_files),
                     "gib": round(sum(path.stat().st_size for path in archive_files) / 2**30, 3),
                     "pair_ids": sorted({path.name.split("_")[0].split(".")[0] for path in archive_files})},
        "raw_extracted_pair_ids": sorted(path.name for path in raw.iterdir() if path.is_dir()),
        "splits": splits, "total_pairs": len(combined),
        "unique_sources": len({row["source_rgb"] for row in combined}),
        "unique_targets": len({row["target_rgb"] for row in combined}),
        "pair_relations": dict(collections.Counter(row["pair_relation"] for row in combined)),
        "controls": {"path": str(controls.resolve()), "version": version,
                     "required_unique": len(control_paths),
                     "missing": sum(not path.is_file() for path in control_paths)},
        "sample": {"pair": row["pair"], "source_frame": row["source_frame_id"],
                   "target_frame": row["target_frame_id"], "images": images, "npz": npz,
                   "control_shapes": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                      for name, value in sample_control.items()}},
        "checks": checks, "disk": disk, "source_sha256": source_hashes,
        "limits": ["Path checks do not establish every file's semantic correctness.",
                   "Only one sample's NPZ and cached tensor contents were inspected.",
                   "No GPU generation, training or complete archive integrity scan was run."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/shangguanrz/project/pic-edit"))
    parser.add_argument("--archives", type=Path, default=Path("/data/shangguanrz/pic-edit-datasets/Hi4D-origin"))
    parser.add_argument("--controls", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_controls"))
    args = parser.parse_args()
    result = audit(args.project, args.archives, args.controls)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if all(result["checks"].values()) else 1)
