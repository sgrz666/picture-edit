"""Read-only CPU audit of the existing TikTok Full Body condition path.

Run on sg with its qwen21 Python. No model weights or CUDA are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/home/shangguanrz/project/pic-edit")
    parser.add_argument("--baseline", default="/home/shangguanrz/nihaochen/qwen21_full/tiktok_body_fullpass_20261004")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import torch
    from src.pose_control.qwen21_full.data_runtime import QwenFullDataset, qwen_full_collate
    from src.pose_control.qwen21_full.execution import prepare

    # Capture the real preparation function's arguments without building a model.
    class CaptureAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.legacy = SimpleNamespace(condition_injector=self.capture_bundle)

        @staticmethod
        def capture_bundle(**kwargs):
            return kwargs

        @staticmethod
        def prepare_conditioning(**kwargs):
            return kwargs

    tracked = (
        "scripts/build_tiktok_v6_conditions.py",
        "src/data/tiktok_dataset.py",
        "src/pose_control/qwen21_full/data_runtime.py",
        "src/pose_control/qwen21_full/execution.py",
        "src/pose_control/qwen21_full/token_encoders.py",
        "src/pose_control/qwen21_full/detail/hand.py",
    )
    report = {
        "audit_date": "2026-10-08",
        "repo": str(repo),
        "baseline": args.baseline,
        "method": "Two actual dataset items and real Body prepare() with a CPU capture adapter; no DiT forward/backward.",
        "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in tracked},
        "global_schema": {"betas": [0, 10], "root_rotation_6d": [10, 16], "translation": [16, 19], "camera": [19, 26]},
        "samples": [],
    }
    assets = repo / "datasets/TikTokDataset/TikTok_3d_assets_native"
    explicit_fields = (
        "body_pose", "jaw_pose", "expression", "left_hand_pose", "right_hand_pose",
        "face_condition", "hand_condition", "detail_condition", "smplx_detail",
    )
    for manifest_name, index in (("train_pairs.jsonl", 96), ("val24_pairs.jsonl", 0)):
        manifest = Path(args.baseline) / "data" / manifest_name
        dataset = QwenFullDataset(manifest, resolution=512, dataset_type="tiktok_v3", align_mode="letterbox")
        row = dataset.rows[index]
        sample = dataset[index]
        seq_dir = assets / row["sequence_id"]
        bundle = torch.load(seq_dir / "v6_conditions.pt", map_location="cpu", weights_only=False, mmap=True)
        raw = torch.load(seq_dir / "smplestx/params.pt", map_location="cpu", weights_only=False, mmap=True)
        frame_index = int(bundle["stem_to_idx"][row["target_stem"]])
        assert raw["betas"].shape[0] == bundle["smplx_global"].shape[0]
        raw_parameters = {}
        for field in ("body_pose", "jaw_pose", "expression", "left_hand_pose", "right_hand_pose"):
            values = torch.as_tensor(raw[field])[frame_index].float()
            raw_parameters[field] = {
                "all_frames_shape": list(raw[field].shape),
                "target_shape": list(values.shape),
                "finite": bool(torch.isfinite(values).all()),
                "target_abs_sum": float(values.abs().sum()),
            }
            assert raw_parameters[field]["finite"] and raw_parameters[field]["target_abs_sum"] > 0
        batch = qwen_full_collate([sample])
        captured = prepare(
            CaptureAdapter(), batch,
            {"resolution": 512, "source_latent": torch.zeros(1, 1, 1)},
            batch["pair_metadata"], None, with_face=False,
        )
        assert sample["smplx_global"].shape == (26,)
        assert not any(field in sample for field in explicit_fields)
        assert not any(field in captured for field in ("face_condition", "hand_condition", "detail_condition", "hand_features"))
        translation_duplicated = bool(torch.equal(sample["smplx_global"][16:19], sample["smplx_global"][23:26]))
        assert translation_duplicated
        report["samples"].append({
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "dataset_index": index, "pair_id": row["pair_id"], "sequence_id": row["sequence_id"],
            "target_stem": row["target_stem"], "target_frame_index": frame_index,
            "raw_parameters": raw_parameters,
            "dataset_keys": sorted(sample),
            "tensor_shapes": {key: list(value.shape) for key, value in sample.items() if torch.is_tensor(value)},
            "explicit_parameter_presence": {field: field in sample for field in explicit_fields},
            "prepare_argument_keys": sorted(captured),
            "geometry_argument_keys": sorted(captured["condition_bundle"]),
            "translation_repeated_in_camera": translation_duplicated,
        })
    report["passed"] = True
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "samples": len(report["samples"]), "output": str(output)}))


if __name__ == "__main__":
    main()
