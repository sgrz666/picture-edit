"""Run the frozen batch producer with one-entry same-target physical reuse.

Physical frame producers remain byte-identical; the additional pair producer is
separately sealed into every pair identity and the final preparation report.
"""
import importlib.util
import json
from pathlib import Path
import sys

from split_dataset import sha


def argument(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def seal(path, output):
    digest = sha(path)
    destination = output / "source" / (digest + "_" + path.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
    if sha(destination) != digest:
        raise ValueError("Cached producer snapshot differs")
    return {"path": str(destination.resolve()), "sha256": digest}


def main():
    scripts = Path(__file__).resolve().parent
    output = Path(argument("--output-root")).resolve()
    repository = Path(argument("--repo-root", "/home/shangguanrz/project/pic-edit"))
    sys.path.insert(0, str(repository))
    from pair_target_cache import SingleTargetCache
    import pair_target_cache
    from local_scene_depth import LocalSceneDepth
    import local_scene_depth
    from src.pose_control.qwen21_full import native_full_data
    spec = importlib.util.spec_from_file_location("bilibili_frozen_batch", scripts / "prepare_native_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    producer = {"pair_target_cache": seal(Path(pair_target_cache.__file__), output),
                "local_scene_depth": seal(Path(local_scene_depth.__file__), output),
                "pair_cache_launcher": seal(Path(__file__), output),
                "native_full_data_original": seal(Path(native_full_data.__file__), output)}
    original_assemble = native_full_data.assemble_full_cache
    aligned = LocalSceneDepth(original_assemble)
    cache = SingleTargetCache(aligned.assemble)
    def assemble(job, *, renderer):
        sample, local = cache.assemble(job, renderer=renderer)
        for metadata in local.metadata:
            bound = metadata.get("local_scene_depth", {})
            if (bound.get("policy") != local_scene_depth.POLICY
                    or bound.get("source_asset_sha256") != job["driver_asset"]["sha256"]):
                raise ValueError("Selected v1 local occlusion must use the exact driver's float scene")
        return sample, local
    native_full_data.assemble_full_cache = assemble
    original_pair, original_write, original_frame = batch.prepare_pair, batch.write, batch.prepare_frame
    physical_frames, boundary_pixels = set(), 0

    def frame(*args, **kwargs):
        nonlocal boundary_pixels
        frame_row, options, helper = args[:3]
        if tuple(frame_row["original_hw"]) == helper.canvas_for(frame_row["original_hw"], options.canvas_long_side).canvas_hw:
            raise ValueError("This private preparation requires a distinct downscaled training canvas for Source float visibility")
        result = original_frame(*args, **kwargs)
        if result["frame_id"] not in physical_frames:
            physical_frames.add(result["frame_id"])
            boundary_pixels += result.get("geometry_boundary_pixels_excluded", 0)
        return result

    def pair(pair_row, source, target, frame_row, args, helper, renderer, physical_producer):
        return original_pair(pair_row, source, target, frame_row, args, helper, renderer,
                             {**physical_producer, **producer})

    def write(path, value, jsonl=False):
        if isinstance(value, dict) and value.get("schema") == "bilibili_native_v5_preparation_v1":
            value["single_target_cache"] = {"hits": cache.hits, "misses": cache.misses, "fallbacks": cache.fallbacks,
                "max_targets": 1, "producers": producer, "math_and_precision": "cache_equal_to_aligned_assembler_FP32"}
            value["local_scene_depth_policy"] = "same_sapiens_v1_float_mesh_on_target_canvas; body_PNG_depth_normal_unchanged"
            value["local_scene_depth_targets_aligned_this_attempt"] = aligned.aligned
            value["selected_geometry_images_consumed_current_attempt"] = value["selected_geometry_images_consumed"]
            value["selected_geometry_frames_bound"] = len(physical_frames)
            value["geometry_boundary_pixels_excluded"] = boundary_pixels
            value["resumed_physical_frames_reverified"] = True
        return original_write(path, value, jsonl)

    batch.prepare_frame, batch.prepare_pair, batch.write = frame, pair, write
    print(json.dumps({"event": "single_target_reuse_enabled", "max_targets": 1, "producer": producer}), flush=True)
    try:
        batch.main()
    finally:
        native_full_data.assemble_full_cache = original_assemble


if __name__ == "__main__":
    main()
