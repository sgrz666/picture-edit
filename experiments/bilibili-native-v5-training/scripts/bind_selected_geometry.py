"""Lock all split rows to the user-selected v1 files; stdlib-only, no GPU."""
import argparse
import json
from pathlib import Path
from split_dataset import rows, sha, write, SPLITS


def reference(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    outputs = [root / s / name for s in SPLITS for name in ("frames_sapiens_v1.jsonl", "pairs_sapiens_v1.jsonl")]
    if args.report.exists() or any(p.exists() for p in outputs):
        raise FileExistsError("Selected geometry snapshot already exists")
    frames, pairs = {}, {}
    selected = {}
    for split in SPLITS:
        frames[split], pairs[split] = rows(root / split / "frames.jsonl"), rows(root / split / "pairs.jsonl")
        for frame in frames[split]:
            clip, fid = frame["clip_id"], frame["frame_id"]
            rgb = Path(frame["rgb_path"])
            if sha(rgb) != frame["rgb_sha256"]:
                raise ValueError("RGB changed: " + fid)
            paths = {
                "smplx": root / "geometry/smplx/sapiens_v1" / clip / (fid + ".pt"),
                "depth": root / "geometry/depth/sapiens_v1" / clip / (fid + "_depth.png"),
                "normal": root / "geometry/normal/sapiens_v1" / clip / (fid + "_normal.png")}
            selected[fid] = {"method": "sapiens_v1", **{k: reference(v) for k, v in paths.items()},
                "depth_encoding": "uint16; z_model_unit_times_1000; zero_background",
                "normal_encoding": "RGB_uint8_view_space_xyz; Native_camera_cv_signs_1_minus1_minus1",
                "smplx_mesh_policy": "reconstruct_FK_from_selected_parameters; original_mesh_cam_can_be_stale"}
            if len(selected) % 100 == 0:
                print(json.dumps({"verified_frames": len(selected), "total_frames": 1960}), flush=True)
    artifacts = {}
    for split in SPLITS:
        frame_index = {f["frame_id"]: f for f in frames[split]}
        selected_frames = [{**frame, "training_geometry": selected[frame["frame_id"]]} for frame in frames[split]]
        selected_pairs = []
        for pair in pairs[split]:
            source, target = (frame_index[pair[k]] for k in ("source_frame_id", "target_frame_id"))
            selected_pairs.append({**pair, "geometry_method": "sapiens_v1",
                "source_training_geometry": selected[source["frame_id"]],
                "target_training_geometry": selected[target["frame_id"]],
                "source_rgb_path": source["rgb_path"], "source_pose_path": source["pose_path"],
                "target_rgb_path": target["rgb_path"], "target_pose_path": target["pose_path"],
                "source_rgb_sha256": source["rgb_sha256"], "target_rgb_sha256": target["rgb_sha256"],
                "input_contract": "bilibili_sapiens_v1_inputs_v1", "native_v5_sidecars_required": True})
        artifacts[split] = {"frames": write(root / split / "frames_sapiens_v1.jsonl", selected_frames, True),
            "pairs": write(root / split / "pairs_sapiens_v1.jsonl", selected_pairs, True),
            "frame_count": len(selected_frames), "pair_count": len(selected_pairs)}
    result = {"schema": "bilibili_sapiens_v1_asset_binding_v1", "passed": True,
        "geometry_method": "sapiens_v1", "frames_verified": len(selected), "geometry_files_verified": len(selected) * 3,
        "rgb_files_sha256_verified": len(selected), "pairs_preserved": sum(len(v) for v in pairs.values()),
        "quality_filter_applied": False, "strict_subset_created": False, "artifacts": artifacts,
        "native_v5_manifest": False, "native_v5_sidecars_required": True, "training_started": False}
    write(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    main()
