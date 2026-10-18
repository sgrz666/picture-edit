"""真实 current-geometry、单/双人 coarse 与768 face/paired-hand 工程门禁。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PAIRED_HAND_SAMPLE = "hi4d_train_aeb388ac9f51195bf39345f3"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def records_for_audit(legacy):
    from src.pose_control.qwen21_pro.cache import file_sha256
    lock = json.loads((legacy / "data/lock.json").read_text())
    rows = {}
    for name in ("overfit6_conditions.jsonl", "train_conditions.jsonl"):
        path = legacy / "data" / name
        if lock["selected_manifest_sha256"][name] != file_sha256(path):
            raise ValueError(f"Locked manifest changed: {name}")
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
        rows[name] = values
    six = rows["overfit6_conditions.jsonl"]
    if len(six) != 6:
        raise ValueError("Expected locked overfit6")
    hand = next(r for r in rows["train_conditions.jsonl"] if r["sample_id"] == PAIRED_HAND_SAMPLE)
    single = next(r for r in six if r["dataset"] == "tiktok")
    return six, (single, hand)


def prepare_geometry(args, run):
    """在已安装 SMPL 的独立环境运行；source worker 不读 driver。"""
    import torch
    from PIL import Image
    from src.pose_control.qwen21_pro.geometry_preparation import _job_main, source_geometry_paths
    from src.pose_control.qwen21_pro.preparation import resolve_condition_assets
    torch.set_num_threads(4)
    project, legacy = Path(args.project_root), Path(args.legacy_workspace)
    six, selected = records_for_audit(legacy)
    model = project / "add/tools/SMPLest-X/human_models/human_model_files/smpl/SMPL_MALE.pkl"
    jobs, source_index, current_index = [], {}, {}
    sources = [r for r in six if r["dataset"] == "hi4d"] + [selected[1]]
    for record in sources:
        sid = record["sample_id"]
        assets = resolve_condition_assets(record, project)
        paths = source_geometry_paths(record, project)
        with Image.open(assets["source_rgb"]) as image:
            source_hw = [image.height, image.width]
        out = run / sid / "source"
        job = {"source_geometry_only": True, "dataset": "hi4d", "body_model_path": str(model),
               "source_params": str(paths["source_raw_parameters"]),
               "source_camera_path": str(paths["source_camera"]),
               "source_camera_id": paths["source_camera_id"], "source_image_hw": source_hw,
               "source_rgb": str(assets["source_rgb"]), "source_masks": list(map(str, assets["source_masks"])),
               "person_order": record["person_order"], "output_dir": str(out)}
        jobs.append(job)
        _job_main(job)
        source_index[sid] = str(out / "source_geometry.json")
    for record in selected:
        sid, dataset = record["sample_id"], record["dataset"]
        assets = resolve_condition_assets(record, project)
        job = {"dataset": dataset, "mode": "driver_exact", "for_training": True,
               "output_dir": str(run / sid / "driver_exact"), "canvas_hw": [512, 512]}
        if dataset == "hi4d":
            paths = source_geometry_paths(record, project)
            with Image.open(assets["source_rgb"]) as image:
                image_hw = [image.height, image.width]
            job.update(body_model_path=str(model), source_params=str(paths["source_raw_parameters"]),
                       driver_params=str(assets["contact_mesh"]), person_order=record["person_order"],
                       camera_path=str(assets["camera"]), camera_id=assets["camera_id"], image_hw=image_hw)
        else:
            base = project / "datasets/TikTokDataset/TikTok_3d_assets_native" / record["metadata"]["sequence_id"] / "smplestx"
            job.update(smplest_root=str(project / "add/tools/SMPLest-X"), params=str(base / "params.pt"),
                       frame_map=str(base / "frame_map.json"), source_frame=Path(assets["source_rgb"]).stem,
                       driver_frame=Path(assets["target_keypoints"][0]).stem, person_id=record["person_order"][0])
        jobs.append(job)
        _job_main(job)
        current_index[sid] = str(Path(job["output_dir"]) / "manifest.json")
        if dataset != "hi4d":
            retarget = {**job, "mode": "source_shape", "for_training": False,
                        "output_dir": str(run / sid / "source_shape")}
            jobs.append(retarget)
            _job_main(retarget)
    write_json(run / "jobs.json", jobs)
    write_json(run / "source_geometry_index.json", source_index)
    write_json(run / "current_geometry_index.json", current_index)
    write_json(run / "selected_conditions.json", selected)
    return {"source_geometry_samples": len(source_index), "driver_exact_samples": len(current_index),
            "source_shape_unpaired_samples": 1, "geometry_worker_jobs": len(jobs)}


def run_inference(args, run):
    import copy
    import torch
    from src.pose_control.qwen21_pro import ProAdapterConfig
    from src.pose_control.qwen21_pro.cache import ProCache, file_sha256, stable_hash
    from src.pose_control.qwen21_pro.encoded import EncodedConditionCache, encode_condition, encode_prepared_condition
    from src.pose_control.qwen21_pro.encoded import sample_validator
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor, RawSourceFeatures
    from src.pose_control.qwen21_pro.geometry_preparation import load_geometry_task_condition, load_source_geometry
    from src.pose_control.qwen21_pro.geometry import PixelCanvasTransform
    from src.pose_control.qwen21_pro.inference import sample_cached
    from src.pose_control.qwen21_pro.pipeline import build_pro_pipeline
    from src.pose_control.qwen21_pro.preparation import prepare_source_task_condition, resolve_condition_assets
    from src.pose_control.qwen21_pro.refinement import (
        derive_refinement_rois, encode_roi_source_bank, localize_task_condition, make_refiner_training_sample,
        prepare_refinement_request, run_refinement, write_coarse_artifact, write_source_crop,
    )
    from src.pose_control.qwen21_union.training import configure_determinism
    configure_determinism()
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    project, geometry, feature, checkpoint = map(Path, (args.project_root, args.geometry_run, args.feature_run, args.checkpoint))
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())["metadata"]
    cfg = ProAdapterConfig.from_dict(metadata["config"])
    code = {p.relative_to(ROOT).as_posix(): file_sha256(p)
            for package in ("qwen21_pro", "qwen21_union")
            for p in (ROOT / "src/pose_control" / package).glob("*.py")}
    if stable_hash(code) != metadata["code_signature"]:
        raise ValueError("Checkpoint code differs from this actual inference source snapshot")
    models = project / "models/antelopev2"
    extractor = FrozenSourceFeatureExtractor.from_local(models / "dinov2-giant", vae_root=Path(args.base_model) / "vae",
                                                       device="cuda", face_root=models)
    if stable_hash(extractor.model_signature) != metadata["source_signature"]:
        raise ValueError("Active source extractor differs from checkpoint signature")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(total * .9), free - 1024**3)
    if budget < 8 * 1024**3:
        raise RuntimeError("Insufficient currently free GPU memory for inference smoke")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    expected = {k: metadata[k] for k in ("config", "scope", "task_signature", "source_signature", "code_signature")}
    pipe, backend = build_pro_pipeline("joint", "cuda", "sync_group_offload", torch.bfloat16, config=cfg,
            cached=False, base_model=args.base_model,
            union_checkpoint=project / "union-adapter/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors",
            checkpoint_path=checkpoint, expected_metadata=expected)
    write_json(run / "backend.json", backend)
    current_index = json.loads((geometry / "current_geometry_index.json").read_text())
    source_index = json.loads((geometry / "source_geometry_index.json").read_text())
    selected = json.loads((geometry / "selected_conditions.json").read_text())
    raw_cache = ProCache(feature / "cache", namespace="raw_source")
    encoded_cache = EncodedConditionCache(run / "encoded")
    refiner_cache = EncodedConditionCache(run / "refiner_encoded")
    legacy = Path(args.legacy_workspace)
    lock = json.loads((legacy / "data/lock.json").read_text())
    truth_path = legacy / "data/train_supervision.jsonl"
    if file_sha256(truth_path) != lock["selected_manifest_sha256"][truth_path.name]:
        raise ValueError("Locked training supervision manifest changed")
    truths = {row["sample_id"]: row for row in map(json.loads, truth_path.read_text().splitlines())}
    producer = {"source": "verified_published_vae_vlm", "code_root": str(ROOT),
                "checkpoint_code": metadata["code_signature"], "base_model": args.base_model}
    results = []
    negative_prompt = "" if args.true_cfg_scale > 1 else None
    for record in selected:
        sid = record["sample_id"]
        manifest_path = Path(current_index[sid])
        manifest = json.loads(manifest_path.read_text())
        assets = resolve_condition_assets(record, project)
        bundle = None
        if record["dataset"] == "hi4d":
            bundle = load_source_geometry(source_index[sid], record, assets)
            assets["source_geometry"] = bundle
            bootstrap = encode_condition(pipe, sample_id=sid, source_images=[record["source_rgb"]],
                    pose_images=manifest["pose_images"], control_image=manifest["control_image"],
                    canvas_hw=(512, 512), person_ids=tuple(record["person_order"]), prompt=record["prompt"],
                    producer_signature=producer, geometry_signature=manifest["geometry_signature"],
                    source_fingerprint="unsealed_source_extraction_only", negative_prompt=negative_prompt)
            raw = extractor.extract(record, bootstrap, project_root=project, assets=assets)
            source_pack = prepare_source_task_condition(raw, config=cfg)
            pack = load_geometry_task_condition(source_pack, manifest_path, config=cfg)
            condition = bootstrap.with_source_binding(pack, config=cfg)
        else:
            meta = raw_cache.metadata(sid)
            raw = RawSourceFeatures.from_dict(raw_cache.get(sid, expected_hash=meta["hash"], signature=meta["signature"]))
            source_pack = prepare_source_task_condition(raw, config=cfg)
            condition, pack = encode_prepared_condition(pipe, source_pack, manifest_path=manifest_path,
                    config=cfg, sample_id=sid, source_images=[record["source_rgb"]], prompt=record["prompt"],
                    producer_signature=producer, negative_prompt=negative_prompt)
        encoded_cache.put(condition, pack, config=cfg)
        condition, pack = encoded_cache.load(sid, expected_binding=condition.binding, config=cfg)
        print(json.dumps({"kind": "encoded", "sample_id": sid, "contact_edges": int(pack.relations.valid.sum()),
                          "source_tokens": list(pack.source_tokens.shape)}), flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = sample_cached(pipe, condition, task_condition=pack, num_inference_steps=2, seed=43,
                                    output_type="pil", true_cfg_scale=args.true_cfg_scale)
        coarse = write_coarse_artifact(sampled, output_dir=run / sid / "coarse", stage1_checkpoint=checkpoint,
                stage1_signatures=metadata, source_fingerprint=pack.source_fingerprint,
                geometry_signature=pack.provenance["geometry"], selection_role="engineering_probe")
        rois = derive_refinement_rois(pack, manifest["rois"], canvas_hw=(512, 512))
        kind = "paired_hands" if record["dataset"] == "hi4d" else "face"
        roi = next((roi for roi in rois if roi.kind == kind and roi.invalid_reason is None), None)
        if roi is None:
            raise ValueError(f"Real {kind} ROI unavailable: {sid}; no substitute ROI allowed")
        crops = []
        for endpoint in roi.endpoints:
            slot = pack.person_ids.index(endpoint.person_id)
            part_name = {0: "face", 4: "left_hand", 7: "right_hand"}[endpoint.part_id]
            if bundle is not None:
                box = bundle["rois"][slot][part_name]
            else:
                coverage = raw.provenance["coverage"][slot]["crops"]
                box = next(item["box"] for item in coverage if item["kind"] == part_name)
            crops.append(write_source_crop(person_id=endpoint.person_id, part_id=endpoint.part_id,
                    source_image=assets["source_rgb"], source_mask=assets["source_masks"][slot], box=box,
                    source_geometry_provenance=bundle,
                    output_dir=run / sid / f"source_pid{endpoint.person_id}_part{endpoint.part_id}"))
        roi_bank = encode_roi_source_bank(pipe, extractor, crops, person_ids=roi.person_ids, endpoints=roi.endpoints)
        local = localize_task_condition(pack, roi, source_features=roi_bank, config=cfg)
        request = prepare_refinement_request(coarse, roi=roi, source_crops=crops,
                geometry_image=manifest["control_image"], output_dir=run / sid / "request", seed=43,
                model_signature=copy.deepcopy(pipe.trained_model_signature),
                source_bank_fingerprint=roi_bank.fingerprint, task_condition=local, config=cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            refined = run_refinement(pipe, request, task_condition=local, output_dir=run / sid / "refined",
                                     num_inference_steps=2, true_cfg_scale=args.true_cfg_scale,
                                     negative_prompt=negative_prompt)
        if refined.audit["fallback_reason"] is not None or not refined.audit["outside_edit_mask_bitwise_equal"]:
            raise ValueError("Real refiner did not execute or changed pixels outside edit mask")
        truth = truths[sid]
        if file_sha256(truth["target_rgb"]) != truth["metadata"]["target_sha256"]:
            raise ValueError("Locked GT bytes changed before independent supervision encoding")
        canvas = manifest["geometry_signature"]["canvas"]
        transform = PixelCanvasTransform(torch.tensor(canvas["matrix"]), tuple(canvas["input_hw"]), tuple(canvas["output_hw"]))
        training_sid = f"{sid}_roi_{kind}"
        sample, supervision, training_pack = make_refiner_training_sample(pipe, request, task_condition=local,
                supervision_image=truth["target_rgb"], config=cfg, cache=refiner_cache, sample_id=training_sid,
                producer_signature=producer, output_dir=run / sid / "refiner_training_sample",
                prompt="Refine the person while preserving identity and the requested pose.",
                negative_prompt=negative_prompt, engineering_probe=True, supervision_transform=transform)
        sample_validator(sample, training_pack, training_sid).validate(training_pack, training_sid)
        if (tuple(supervision["target_latent"].shape) != (1, 2304, 64)
                or tuple(sample["keep_mask"].shape) != (1, 2304, 1)
                or tuple(sample["inpaint_latents"].shape) != (1, 2304, 64)):
            raise ValueError("Real refiner training cache shape differs from768 unpatched64")
        results.append({"sample_id": sid, "dataset": record["dataset"], "coarse": str(coarse.manifest_path),
                        "refined": str(refined.image_path), "kind": kind, "source_crops": len(crops),
                        "local_grid": list(local.target_hw), "person_ids": list(local.person_ids),
                        "edited_pixels": refined.audit["edited_pixels"], "outside_edit_mask_bitwise_equal": True,
                        "refiner_training_cache_verified": True,
                        "supervision_valid_area": float(supervision["valid_target_tokens"].sum())})
        print(json.dumps({"kind": "sample_passed", **results[-1]}), flush=True)
    return {"samples": results, "coarse_images": len(results), "refined_images": len(results),
            "steps": 2, "true_cfg_scale": args.true_cfg_scale,
            "max_memory_allocated": torch.cuda.max_memory_allocated(),
            "max_memory_reserved": torch.cuda.max_memory_reserved(), "initial_free_memory_bytes": free}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("geometry", "inference"))
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    parser.add_argument("--geometry-run")
    parser.add_argument("--feature-run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--true-cfg-scale", type=float, choices=(1., 4.), default=1.)
    args = parser.parse_args()
    from src.pose_control.qwen21_pro.geometry_preparation import P_ROOT
    from src.pose_control.qwen21_pro.cache import file_sha256
    run = Path(args.run_dir).resolve()
    if not Path(args.run_dir).is_absolute() or not run.is_relative_to(P_ROOT.resolve()):
        parser.error("run-dir 必须是 Pro P root 下的新绝对目录")
    if args.phase == "inference" and not all((args.geometry_run, args.feature_run, args.checkpoint)):
        parser.error("inference 需要 geometry-run、feature-run 与 checkpoint")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(run / "run.json", {"argv": vars(args), "code_root": str(ROOT),
               "script_sha256": file_sha256(__file__), "engineering_only": True, "quality_claim": False})
    try:
        result = prepare_geometry(args, run) if args.phase == "geometry" else run_inference(args, run)
        write_json(run / "report.json", {"passed": True, "phase": args.phase, **result,
                                          "seconds": time.monotonic() - started})
        print(json.dumps({"passed": True, "phase": args.phase, "seconds": time.monotonic() - started}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
