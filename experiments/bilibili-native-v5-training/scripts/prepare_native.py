"""Full Bilibili -> Native V5 preparation using the selected Sapiens2 v1 assets.

No quality selection, optimizer construction, gradient test or training is performed.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import uuid

from split_dataset import rows, sha, write


def ref(path, schema=None):
    result = {"path": str(Path(path).resolve()), "sha256": sha(path)}
    if schema:
        result["schema"] = schema
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value, jsonl=False):
    """Resume immutable publishing only when every previously written row matches."""
    if Path(path).exists():
        previous = rows(path) if jsonl else load(path)
        if previous != json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)):
            raise ValueError("Existing published content differs: " + str(path))
        return ref(path)
    return write(path, value, jsonl)


def check_ready(marker, identity):
    if not marker.exists():
        return None
    result = load(marker)
    if result["identity"] != identity:
        raise ValueError("Completed record identity differs: " + str(marker))
    for item in result["artifacts"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Completed artifact changed: " + item["path"])
    return result


def producer_snapshot(args):
    files = [Path(__file__), Path(__file__).with_name("appearance.py"), args.hia_tool]
    records = {}
    for path in files:
        identity = sha(path)
        target = args.output_root / "source" / (identity + "_" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
        records[path.name] = ref(target)
    return records


class SelectedGeometryRenderer:
    """Use published v1 PNGs for global Normal/Depth; same-fit FK supplies Part14.

    Local face/hand submeshes are rasterized from the SAME v1 fit. The published
    global normals are RGB view-space; the typed renderer contract is camera CV.
    """
    def __init__(self, delegate):
        self.delegate = delegate
        self.selected = {}
        self.used = set()
        self.boundary_pixels = {}

    def register(self, person, camera, hw, depth, normal):
        self.selected[sha_array(person["mesh_cam"])] = (camera.copy(), tuple(hw), depth, normal, len(person["faces"]))

    def render(self, person, camera, hw):
        import numpy as np
        import cv2
        result = self.delegate.render(person, camera, hw)
        item = self.selected.get(sha_array(person["mesh_cam"]))
        if item is None or tuple(hw) != item[1] or len(person["faces"]) != item[4] or not np.allclose(camera, item[0]):
            return result
        depth = cv2.imread(str(item[2]), cv2.IMREAD_UNCHANGED)
        normal = cv2.imread(str(item[3]), cv2.IMREAD_COLOR)
        if depth is None or depth.dtype != np.uint16 or depth.shape != tuple(hw) or normal is None or normal.shape[:2] != tuple(hw):
            raise ValueError("Published v1 geometry image shape/dtype differs")
        published_valid = depth > 0
        # Original GPU and Native FK rasterization differ by edge rounding; the
        # real v1 audit measured 2 pixels / 609152, one pixel from the boundary.
        # Keep the PNG values unchanged and use their shared physical support.
        valid = published_valid & normal.any(-1) & result["valid"]
        self.boundary_pixels[str(item[2])] = int((published_valid & ~valid).sum())
        rgb = normal[..., ::-1].astype(np.float32) / 127.5 - 1.
        camera_normal = rgb * np.array([1., -1., -1.], np.float32)
        camera_normal /= np.maximum(np.linalg.norm(camera_normal, axis=-1, keepdims=True), 1e-8)
        self.used.add(str(item[2]))
        return {"z": depth.astype(np.float32) / 1000., "valid": valid,
                "normal": np.where(valid[..., None], camera_normal, 0.),
                "part": np.where(valid, result["part"], 0).astype(np.uint8)}


def sha_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


class GeometryReferences:
    """Explicit first pass; the second pass must populate every Source reference."""
    def __init__(self, models):
        self.models = {**models, "phase": "physical_assets_before_batched_source_features"}

    def extract(self, frame, rgb, canvas, asset, empty):
        return empty


def populate_references(frames, prepared, args, features, producer):
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_full_assets import read_record, write_record
    keys = list(frames)
    completed_count = 0
    for start in range(0, len(keys), args.feature_batch_size):
        pending = []
        for key in keys[start:start + args.feature_batch_size]:
            base = prepared[key]
            identity = fingerprint({"physical_frame": base["identity"], "models": features.models,
                "feature_producer": producer["appearance.py"], "batch_limit": args.feature_batch_size})
            directory = args.output_root / "appearance" / key / identity[:16]
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "ready.json"
            previous = check_ready(marker, identity)
            if previous:
                if sha(previous["rgb"]["path"]) != previous["rgb"]["sha256"]:
                    raise ValueError("Source RGB changed during reference resume")
                prepared[key] = previous
                completed_count += 1
                continue
            pending.append({"key": key, "frame": frames[key], "rgb": base["rgb"],
                "canvas": CanvasSpec.from_record(base["canvas"]), "asset": base["artifacts"]["asset"],
                "empty_references": read_record(base["artifacts"]["references"], schema="native_local_reference_v4"),
                "base": base, "identity": identity, "directory": directory, "marker": marker})
        if pending:
            features.prime_dino(pending, max_batch=args.feature_batch_size)
        for item in pending:
            result = features.extract(item["frame"], item["rgb"], item["canvas"], item["asset"], item["empty_references"])
            refs = write_record(item["directory"] / ("references_" + uuid.uuid4().hex[:12] + ".npz"), result,
                                schema="native_local_reference_v4")
            built = {**item["base"], "identity": item["identity"], "identity_sha256": item["identity"],
                "physical_identity": item["base"]["identity"], "source_features_complete": True,
                "feature_protocol": "FP32_same_models_and_preprocessing_batched_" + str(args.feature_batch_size),
                "features_valid": {"arcface": result.arcface_valid.tolist(), "face_dino": result.face_dino_valid.tolist(),
                                   "hand_dino": result.hand_dino_valid.tolist()},
                "artifacts": {**item["base"]["artifacts"], "references": refs}}
            write(item["marker"], built)
            prepared[item["key"]] = built
            completed_count += 1
        print(json.dumps({"event": "source_features", "completed": completed_count, "total": len(keys)}), flush=True)
    return prepared


def prepare_frame(frame, args, helper, importer, renderer, selected_renderer, features, producer):
    import numpy as np
    import torch
    from PIL import Image
    from src.pose_control.qwen21_full.native_full_assets import write_record
    from src.pose_control.qwen21_full.native_v5.data import write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import DETAIL_SCHEMA
    from src.pose_edit.geometry import smplx137_to_dwpose134, render_poseguide

    clip, fid = frame["clip_id"], frame["frame_id"]
    paths = {
        "parameters": args.root / "geometry/smplx/sapiens_v1" / clip / (fid + ".pt"),
        "depth": args.root / "geometry/depth/sapiens_v1" / clip / (fid + "_depth.png"),
        "normal": args.root / "geometry/normal/sapiens_v1" / clip / (fid + "_normal.png")}
    original = {k: ref(p) for k, p in paths.items()}
    identity = fingerprint({"frame": frame, "selected": original, "producer": producer,
                            "smplx_model_sha256": importer.model_sha, "model_options": importer.options,
                            "expression_basis_sha256": importer.basis_sha,
                            "canvas_long_side": args.canvas_long_side, "appearance": features.models})
    directory = args.output_root / "frames" / fid / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ready.json"
    completed = check_ready(marker, identity)
    if completed:
        if sha(completed["rgb"]["path"]) != completed["rgb"]["sha256"]:
            raise ValueError("Original RGB changed since preparation: " + fid)
        from src.pose_control.qwen21_full.native_full_assets import read_record
        asset = read_record(completed["artifacts"]["asset"], schema="native_smplx_asset_v4")
        selected_renderer.register(asset["people"][0], asset["K"], frame["original_hw"], paths["depth"], paths["normal"])
        return completed
    attempt = directory / ("build_" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    rgb = ref(frame["rgb_path"])
    if rgb["sha256"] != frame["rgb_sha256"]:
        raise ValueError("Original RGB fingerprint differs: " + fid)
    raw = torch.load(paths["parameters"], map_location="cpu", weights_only=False)
    if (int(raw["height"].item()), int(raw["width"].item())) != tuple(frame["original_hw"]):
        raise ValueError("Selected parameters and RGB original canvas differ")
    names = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "betas", "expression", "transl")
    with torch.no_grad():
        effective_mesh = importer.model(**{k: raw[k].float() for k in names}).vertices.detach()
    # The original producer changes pose parameters but retains the old mesh_cam
    # auxiliary field. Its own geometry renderer uses FK from the new parameters.
    # Preserve all original files and explicitly reconstruct that identical FK.
    stale_error = float((effective_mesh - raw["mesh_cam"]).abs().max())
    effective = {**raw, "mesh_cam": effective_mesh}
    canonical = attempt / "selected_parameters_with_fk_mesh.pt"
    with canonical.open("xb") as stream:
        torch.save(effective, stream)
    asset, joints137 = importer.convert(ref(canonical), {"selected_geometry": original}, frame, attempt)
    person_id = "bilibili:" + clip + ":0"
    asset["people"][0]["metadata"]["person_id"] = person_id
    asset["metadata"].update(camera_id="bilibili:" + clip + ":camera", scale_group_id="bilibili:" + clip + ":model",
        dataset="bilibili", geometry_method="sapiens_v1", original_geometry=original,
        original_mesh_cam_max_difference=stale_error, mesh_policy="FK_from_selected_original_parameters_no_refit")
    asset_ref = write_record(attempt / "asset.npz", asset, schema="native_smplx_asset_v4")
    selected_renderer.register(asset["people"][0], asset["K"], frame["original_hw"], paths["depth"], paths["normal"])
    spec = helper.canvas_for(frame["original_hw"], args.canvas_long_side)
    empty_refs, detail = helper.source_references(asset, rgb, spec, asset_ref, frame, renderer)
    refs = features.extract(frame, rgb, spec, asset_ref, empty_refs)
    refs_ref = write_record(attempt / "source_references.npz", refs, schema="native_local_reference_v4")
    detail_ref = write_sidecar(attempt / "source_detail.npz", detail, DETAIL_SCHEMA)
    uv = joints137 @ asset["K"].T
    uv = uv[:, :2] / uv[:, 2:]
    points = spec.pixel_points(helper.mask_guide_points(smplx137_to_dwpose134(uv), frame))
    guide = attempt / "pose_guide.png"
    render_poseguide(points, (spec.canvas_hw[1], spec.canvas_hw[0])).save(guide)
    import cv2
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError("Selected published depth must be uint16")
    mask = attempt / "source_mask.png"
    global_maps = renderer.render(asset["people"][0], asset["K"], frame["original_hw"])
    Image.fromarray(global_maps["valid"].astype(np.uint8) * 255).save(mask)
    result = {"identity": identity, "identity_sha256": identity, "frame_id": fid, "clip_id": clip,
        "split": frame["split"], "group_id": clip, "rgb": rgb, "canvas": spec.metadata(),
        "geometry_method": "sapiens_v1", "original_geometry": original, "loss_mask": frame["loss_mask"],
        "original_mesh_cam_max_difference": stale_error,
        "geometry_support_policy": "intersection_of_selected_depth_normal_and_same_fit_raster; RGB_coverage_unchanged",
        "geometry_boundary_pixels_excluded": selected_renderer.boundary_pixels.get(str(paths["depth"]), 0),
        "features_valid": {"arcface": refs.arcface_valid.tolist(), "face_dino": refs.face_dino_valid.tolist(), "hand_dino": refs.hand_dino_valid.tolist()},
        "artifacts": {"asset": asset_ref, "references": refs_ref, "detail": detail_ref,
                      "guide": ref(guide), "source_mask": ref(mask)}}
    write(marker, result)
    return result


def prepare_pair(pair, source, target, frame, args, helper, renderer, producer):
    import numpy as np
    import torch
    from src.pose_control.qwen21_full.native_full_assets import read_record, write_record
    from src.pose_control.qwen21_full.native_full_data import assemble_full_cache, BODY_SCHEMA, PAIR_SCHEMA
    from src.pose_control.qwen21_full.native_v5.data import training_sidecar, write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import SUPERVISION_SCHEMA
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA
    from src.data.canvas_contract import CanvasSpec
    identity = fingerprint({"pair": pair, "source": source["identity"], "target": target["identity"], "producer": producer})
    directory = args.output_root / "pairs" / pair["pair_id"] / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ready.json"
    completed = check_ready(marker, identity)
    if completed:
        return completed
    if pair["split"] != source["split"] or pair["split"] != target["split"] or source["clip_id"] != target["clip_id"]:
        raise ValueError("Original video/split mismatch")
    if pair["source_loss_mask"] != source["loss_mask"] or pair["target_loss_mask"] != target["loss_mask"]:
        raise ValueError("Original loss masks changed")
    attempt = directory / ("build_" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    sample, local = assemble_full_cache({"pair_id": pair["pair_id"],
        "source_asset": source["artifacts"]["asset"], "driver_asset": target["artifacts"]["asset"],
        "source_canvas": source["canvas"], "target_canvas": target["canvas"],
        "source_masks": [source["artifacts"]["source_mask"]], "source_rgb": source["rgb"], "target_rgb": target["rgb"],
        "source_indices": [0, -1], "source_slot_ids": [0, 1], "units": "model_unit", "scale_status": "model_scale",
        "references": source["artifacts"]["references"], "expression_mode": "driver", "distances": "absent_single_person"}, renderer=renderer)
    local = helper.apply_regional_validity(local, frame)
    sample["pair_metadata"].update(dataset_type="bilibili_native_full_smplx_v4", geometry_method="sapiens_v1",
        original_split=pair["split"], original_clip_id=pair["clip_id"], original_group_id=frame["group_id"],
        loss_mask_scope=helper.MASK_SCOPE, target_loss_mask=pair["target_loss_mask"], source_loss_mask=pair["source_loss_mask"],
        selected_source_geometry=source["original_geometry"], selected_target_geometry=target["original_geometry"])
    body_ref = write_record(attempt / "body.npz", sample, schema=BODY_SCHEMA)
    local_ref = write_record(attempt / "local.npz", local, schema="native_local_condition_v4")
    v4 = {"schema": PAIR_SCHEMA, "pair_id": pair["pair_id"], "scene_kind": "single", "data_profile": "full_smplx",
          "body": body_ref, "local": local_ref, "references": source["artifacts"]["references"], "source_rgb": source["rgb"], "target_rgb": target["rgb"]}
    v4_ref = write(attempt / "v4.jsonl", [v4], True)
    references = read_record(source["artifacts"]["references"], schema="native_local_reference_v4")
    alignment = frame.get("detection", {}).get("face") or {}
    landmarks = np.asarray(alignment.get("landmarks5", []), dtype=np.float32)
    auxiliary = landmarks.shape == (5, 2) and np.isfinite(landmarks).all() and frame["loss_mask"]["face"]
    declaration = {"availability": "unavailable", "reason": "Existing target five-point alignment unavailable"}
    if auxiliary:
        target_record = write(attempt / "target_alignment.json", {"role": "training_only", "coordinate_space": "original_pixels",
            "driver_rgb_sha256": target["rgb"]["sha256"], "target_person_ids": local.metadata[0]["target_person_ids"],
            "landmarks5": [landmarks.tolist(), np.zeros((5, 2)).tolist()], "valid": [True, False]})
        declaration = {"availability": "available", "record": target_record}
    supervision = training_sidecar({**sample, "local_condition_v4": local, "local_references_v4": references}, declaration)
    supervision = replace(supervision, weights=helper.mask_regional_weights(supervision.weights, frame)).validate()
    supervision_ref = write_sidecar(attempt / "supervision.npz", supervision, SUPERVISION_SCHEMA)
    source_spec, target_spec = CanvasSpec.from_record(source["canvas"]), CanvasSpec.from_record(target["canvas"])
    prompt = "Change the person's pose to match the pose in <image2>. Preserve the identity, face, clothing and background of the person in <image1>."
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    prompt_bank = write(attempt / "prompt_bank.json", {"schema": "bilibili_native_prompt_v1", "pair_id": pair["pair_id"],
        "compiled_prompt_inference": prompt, "compiled_prompt_inference_sha256": prompt_sha,
        "producer": "deterministic_source_and_selected_sapiens_v1_pose", "original_pair": pair})
    condition_ref = write(attempt / "condition.json", {"schema": SCHEMA, "profile": "single_2ref", "pair_id": pair["pair_id"],
        "primary_source_index": 0, "target_canvas_id": target_spec.fingerprint(), "images": [
            {"role": "source", "file": source["rgb"], "canvas": source["canvas"], "canvas_id": source_spec.fingerprint(), "space": "original"},
            {"role": "target_pose", "file": target["artifacts"]["guide"], "canvas": target["canvas"], "canvas_id": target_spec.fingerprint(), "space": "canvas"}],
        "prompt": {"text": prompt, "sha256": prompt_sha, "field": "compiled_prompt_inference", "bank": prompt_bank,
                   "schema": "bilibili_native_prompt_v1", "pair_id": pair["pair_id"]}})
    row = {"schema": "native_full_pair_v5", "pair_id": pair["pair_id"], "v4_manifest": v4_ref,
           "references": source["artifacts"]["references"], "detail": source["artifacts"]["detail"],
           "supervision": supervision_ref, "qwen_condition": condition_ref}
    result = {"identity": identity, "row": row, "group_id": "bilibili:" + frame["group_id"], "split": pair["split"],
        "artifacts": {"body": body_ref, "local": local_ref, "v4_manifest": v4_ref, "supervision": supervision_ref,
                      "condition": condition_ref, "prompt_bank": prompt_bank}}
    write(marker, result)
    return result


def run(args):
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.hia_tool.parent))
    spec = importlib.util.spec_from_file_location("hia_training_reuse", args.hia_tool)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from appearance import SourceFeatureExtractor
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, SourceSpec, stage_preset
    args.output_root.mkdir(parents=True, exist_ok=True)
    producer = producer_snapshot(args)
    print(json.dumps({"event": "producer_pinned", "geometry_method": "sapiens_v1"}), flush=True)
    models_dir = args.repo_root / "models/antelopev2"
    dino_dir = models_dir / "dinov2-giant"
    models = {"arcface_model": str(models_dir / "glintr100.onnx"), "arcface_sha256": sha(models_dir / "glintr100.onnx"),
        "dino_model": str(dino_dir), "dino_files": {str(p.relative_to(dino_dir)): sha(p) for p in sorted(dino_dir.rglob("*"))
        if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")}, "device": "cuda"}
    features = SourceFeatureExtractor(models)
    geometry_references = GeometryReferences(models)
    print(json.dumps({"event": "appearance_models_pinned"}), flush=True)
    importer = helper.Importer(args.smplx_model, args.output_root / "model_metadata", producer[Path(__file__).name])
    print(json.dumps({"event": "smplx_model_loaded", "sha256": importer.model_sha}), flush=True)
    selected_renderer = SelectedGeometryRenderer(IndependentMeshRasterizer("cuda"))
    renderer = helper.BoundedRenderer(selected_renderer, max_bytes=512 * 1024 * 1024)
    print(json.dumps({"event": "renderer_ready"}), flush=True)
    all_frames = {s: rows(args.root / s / "frames.jsonl") for s in ("train", "val", "test")}
    helper.validate_frame_groups(f for values in all_frames.values() for f in values)
    frames = {f["frame_id"]: f for values in all_frames.values() for f in values}
    prepared = {}
    for i, frame in enumerate(frames.values()):
        prepared[frame["frame_id"]] = prepare_frame(frame, args, helper, importer, renderer, selected_renderer, geometry_references, producer)
        if (i + 1) % 20 == 0:
            print(json.dumps({"event": "geometry_frames", "completed": i + 1, "total": len(frames)}), flush=True)
    prepared = populate_references(frames, prepared, args, features, producer)
    if not all(item.get("source_features_complete") is True for item in prepared.values()):
        raise RuntimeError("Full Source feature phase is incomplete; manifests must not be published")
    del features
    torch.cuda.empty_cache()
    manifests, sources = {}, []
    for split in ("train", "val", "test"):
        built = []
        pairs = rows(args.root / split / "pairs.jsonl")
        for i, pair in enumerate(pairs):
            source, target = (prepared[pair[k]] for k in ("source_frame_id", "target_frame_id"))
            built.append(prepare_pair(pair, source, target, frames[pair["target_frame_id"]], args, helper, renderer, producer))
            if (i + 1) % 20 == 0:
                print(json.dumps({"event": "pairs", "split": split, "completed": i + 1, "total": len(pairs)}), flush=True)
        destination = args.output_root / "manifests" / split
        destination.mkdir(parents=True, exist_ok=True)
        manifest = publish(destination / "pairs.jsonl", [b["row"] for b in built], True)
        groups = publish(destination / "groups.json", {b["row"]["pair_id"]: b["group_id"] for b in built})
        manifests[split] = {"manifest": manifest, "groups": groups, "pairs": len(built)}
        if split != "test":
            sources.append(SourceSpec("bilibili_" + split, role="train" if split == "train" else "validation", manifest=manifest,
                groups=groups, data_profile="full_smplx", scene_kind="single", weight=1.))
    baseline = Path("/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1")
    config = TrainingConfig(stage=stage_preset("S0", budget=args.steps), sources=tuple(sources),
        initialization=ref(baseline / "initialization/native_v5_init_v2.pt"),
        model_path="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1", model_manifest=ref(baseline / "source/qwen_model_files.json"),
        cache_root=str(args.output_root / "qwen_frozen"), output_root=str(args.output_root / "training"),
        qwen_condition_profile="single_2ref", save_every_updates=500, probe_every_updates=500)
    config_ref = publish(args.output_root / "S0_prepare.json", config.record())
    result = {"schema": "bilibili_native_v5_preparation_v1", "geometry_method": "sapiens_v1", "manifests": manifests,
        "frames_prepared": len(prepared), "pairs_prepared": sum(m["pairs"] for m in manifests.values()), "config": config_ref,
        "all_original_rows_preserved": True, "strict_subset_created": False, "producer": producer,
        "published_normal_depth_used_for_global_conditions": True, "selected_geometry_images_consumed": len(selected_renderer.used),
        "geometry_boundary_pixels_excluded": sum(selected_renderer.boundary_pixels.values()),
        "geometry_local_maps": "same_sapiens_v1_fit_submesh_render", "training_ready": False,
        "missing_training_bindings": config.missing(), "training_started": False, "optimizer_created": False,
        "next": "formal frozen Qwen preprocessing and explicit training budget; no smoke training performed"}
    write(args.report, result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/shangguanrz/datasets/bilibili"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path("/home/shangguanrz/project/pic-edit"))
    p.add_argument("--method", choices=("sapiens_v1",), required=True)
    p.add_argument("--canvas-long-side", type=int, default=512)
    p.add_argument("--steps", type=int)
    p.add_argument("--feature-batch-size", type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.feature_batch_size <= 16:
        p.error("Source feature batch size must be in [1,16]")
    args.hia_tool = args.repo_root / "experiments/6.qwen-controlnet union/hia_geometry_training_20261029/scripts/hia_training.py"
    args.smplx_model = args.repo_root / "add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
    if not str(args.output_root.resolve()).startswith("/data/shangguanrz/"):
        p.error("New data must be on the persistent data volume")
    try:
        run(args)
    except Exception:
        failure = args.output_root / ("failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        write(failure, {"error": traceback.format_exc(), "training_started": False})
        raise


if __name__ == "__main__":
    main()
