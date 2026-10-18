"""锁定val的4条mixed Hi4D：真实重建/渲染及无smplx的Qwen几何交接。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MALE_PARTS = "f6ac0bb73e15ae1b567bc0b2070f966de52fdba55023c9fa88f1f10ee66dc7d5"
FEMALE_PARTS = "5d486c82a83e3bca518ade5da9df9fc4771dab6e1f102d80e06157bfc5ea46df"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cpu", "geometry", "handoff"))
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--geometry-run")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from src.pose_control.qwen21_pro.cache import file_sha256, tensor_sha256
    from src.pose_control.qwen21_pro.geometry_preparation import P_ROOT, _job_main, source_geometry_paths
    from src.pose_control.qwen21_pro.preparation import resolve_condition_assets
    run = Path(args.run_dir).resolve()
    if not Path(args.run_dir).is_absolute() or not run.is_relative_to(P_ROOT.resolve()):
        parser.error("run-dir必须是新建P目录")
    if args.phase == "handoff" and not args.geometry_run:
        parser.error("handoff需要geometry-run")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    project, legacy = Path(args.project_root), Path(args.legacy_workspace)
    write_json(run / "run.json", {"argv": vars(args), "script_sha256": file_sha256(__file__),
                                 "code_root": str(ROOT), "geometry_only": True, "quality_claim": False})
    try:
        path = legacy / "data/val_conditions.jsonl"
        lock = json.loads((legacy / "data/lock.json").read_text())
        if file_sha256(path) != lock["selected_manifest_sha256"][path.name]:
            raise ValueError("Locked val manifest changed")
        original = {r["sample_id"]: r for r in map(json.loads, path.read_text().splitlines())}
        records = [r for r in map(json.loads, Path(args.conditions).read_text().splitlines()) if r["dataset"] == "hi4d"]
        if len(records) != 4 or any(original.get(r["sample_id"]) != r for r in records):
            raise ValueError("Expected four unchanged preregistered val Hi4D records")
        if args.phase == "cpu":
            from dataclasses import replace
            from src.pose_control.qwen21_pro.geometry import (
                GeometryState, load_hi4d_gendered_backend, load_hi4d_parameters, vertex_parts_for,
            )
            from src.pose_control.qwen21_pro.geometry_preparation import verify_hi4d_forward
            rows = []
            for record in records:
                assets = resolve_condition_assets(record, project)
                source = source_geometry_paths(record, project)
                source_params, driver_params = Path(source["source_raw_parameters"]), Path(assets["contact_mesh"])
                order = tuple(record["person_order"])
                backend = load_hi4d_gendered_backend(
                    project / "add/tools/SMPLest-X/human_models/human_model_files", source_params.parent.parent / "meta.npz",
                    parameters_path=source_params, person_order=order, driver_gender_metadata=driver_params.parent.parent / "meta.npz",
                    driver_parameters_path=driver_params, expected_pair=record["group_id"])
                _, driver, _, _ = load_hi4d_parameters(source_params, driver_params, person_order=order)
                _, original_source, _, _ = load_hi4d_parameters(source_params, source_params, person_order=order)
                source_audit = verify_hi4d_forward(source_params, GeometryState(original_source, backend.forward(original_source)),
                                                 person_order=order, body_model=backend)
                body = backend.forward(driver)
                driver_audit = verify_hi4d_forward(driver_params, GeometryState(driver, body), person_order=order, body_model=backend)
                reversed_driver = replace(driver, person_ids=driver.person_ids[::-1], betas=driver.betas.flip(0),
                    global_orient=driver.global_orient.flip(0), body_pose=driver.body_pose.flip(0), transl=driver.transl.flip(0))
                reversed_body = backend.forward(reversed_driver)
                torch.testing.assert_close(reversed_body.vertices, body.vertices.flip(0), rtol=0, atol=0)
                torch.testing.assert_close(reversed_body.joints, body.joints.flip(0), rtol=0, atol=0)
                labels = vertex_parts_for(backend, driver.person_ids)
                if not torch.equal(vertex_parts_for(backend, reversed_driver.person_ids), labels.flip(0)):
                    raise ValueError("Real mixed body permutation changed anatomical parts")
                rows.append({"sample_id": record["sample_id"], "person_order": list(order), "source": source_audit,
                             "driver": driver_audit, "body_and_parts_permutation_exact": True,
                             "labels_sha256": [tensor_sha256(row) for row in labels]})
                print(json.dumps(rows[-1]), flush=True)
            results = {"samples": rows, "cuda_initialized": torch.cuda.is_initialized()}
        elif args.phase == "geometry":
            index, jobs = {}, []
            for record in records:
                sid = record["sample_id"]
                assets = resolve_condition_assets(record, project)
                source = source_geometry_paths(record, project)
                source_params = Path(source["source_raw_parameters"])
                driver_params = Path(assets["contact_mesh"])
                with Image.open(assets["source_rgb"]) as image:
                    source_hw = [image.height, image.width]
                source_job = {
                    "source_geometry_only": True, "dataset": "hi4d",
                    "body_model_root": str(project / "add/tools/SMPLest-X/human_models/human_model_files"),
                    "gender_metadata": str(source_params.parent.parent / "meta.npz"),
                    "source_params": str(source_params), "source_camera_path": str(source["source_camera"]),
                    "source_camera_id": source["source_camera_id"], "source_image_hw": source_hw,
                    "source_rgb": str(assets["source_rgb"]), "source_masks": list(map(str, assets["source_masks"])),
                    "group_id": record["group_id"], "person_order": record["person_order"], "output_dir": str(run / sid / "source"),
                }
                jobs.append(source_job)
                _job_main(source_job)
                target_hw = list(reversed(record["target_transform"]["original_wh"]))
                current_job = {
                    "dataset": "hi4d", "mode": "driver_exact", "for_training": True,
                    "body_model_root": source_job["body_model_root"],
                    "gender_metadata": source_job["gender_metadata"],
                    "driver_gender_metadata": str(driver_params.parent.parent / "meta.npz"),
                    "source_params": str(source_params), "driver_params": str(driver_params),
                    "group_id": record["group_id"], "person_order": record["person_order"], "camera_path": str(assets["camera"]),
                    "camera_id": assets["camera_id"], "image_hw": target_hw, "canvas_hw": [512, 512],
                    "output_dir": str(run / sid / "driver_exact"),
                }
                jobs.append(current_job)
                _job_main(current_job)
                data = torch.load(Path(current_job["output_dir"]) / "geometry.pt", weights_only=True, map_location="cpu")
                labels = data["vertex_parts"]
                if tuple(labels.shape) != (2, 6890):
                    raise ValueError("Mixed labels must be saved in current stable-person order")
                for slot, raw_id in enumerate(record["person_order"]):
                    expected = (MALE_PARTS, FEMALE_PARTS)[raw_id]
                    if tensor_sha256(labels[slot]) != expected:
                        raise ValueError("Real matched female/male part labels follow wrong stable person")
                index[sid] = {"source": str(Path(source_job["output_dir"]) / "source_geometry.json"),
                              "current": str(Path(current_job["output_dir"]) / "manifest.json")}
                print(json.dumps({"sample_id": sid, "person_order": record["person_order"],
                                  "labels_sha256": [tensor_sha256(row) for row in labels]}), flush=True)
            write_json(run / "jobs.json", jobs)
            write_json(run / "geometry_index.json", index)
            results = {"samples": len(records), "worker_jobs": len(jobs), "state_order_labels_verified": True}
        else:
            from src.pose_control.qwen21_pro import ProAdapterConfig
            from src.pose_control.qwen21_pro.features import RawSourceFeatures
            from src.pose_control.qwen21_pro.geometry_preparation import load_geometry_task_condition, load_source_geometry
            from src.pose_control.qwen21_pro.preparation import prepare_source_task_condition
            if any(name == "smplx" or name.startswith("smplx.") for name in sys.modules):
                raise ValueError("Qwen geometry handoff must run in a fresh environment without smplx")
            index = json.loads((Path(args.geometry_run) / "geometry_index.json").read_text())
            cfg, rows = ProAdapterConfig(source_dim=1600), []
            for record in records:
                sid, pids = record["sample_id"], tuple(record["person_order"])
                assets = resolve_condition_assets(record, project)
                bundle = load_source_geometry(index[sid]["source"], record, assets)
                # Only exercises real geometry deserialization/routing. These dummy source
                # tensors are never cached, sampled, trained or claimed as extracted features.
                dummy = RawSourceFeatures(pids, torch.zeros(1, 2, 1, 1600), torch.ones(1, 2, 1, dtype=torch.bool),
                    torch.zeros(1, 2, 512), torch.zeros(1, 2, dtype=torch.bool), "geometry_probe_only",
                    {"audit_only": True, "appearance_not_extracted": True})
                source_pack = prepare_source_task_condition(dummy, config=cfg)
                pack = load_geometry_task_condition(source_pack, index[sid]["current"], config=cfg)
                pack.validate(cfg)
                rows.append({"sample_id": sid, "person_ids": list(pack.person_ids), "target_hw": list(pack.target_hw),
                             "contact_edges": int(pack.relations.valid.sum()), "source_bundle_verified": True,
                             "source_support_pixels": bundle["support_pixels"],
                             "spatial_valid": {k: v.tolist() for k, v in pack.spatial_valid.items()}})
            if any(name == "smplx" or name.startswith("smplx.") for name in sys.modules):
                raise ValueError("Geometry handoff unexpectedly imported smplx")
            results = {"samples": rows, "without_smplx": True, "appearance_not_extracted": True,
                       "cached_sampled_or_trained": False}
        write_json(run / "report.json", {"passed": True, "phase": args.phase, **results,
                                          "seconds": time.monotonic() - started})
        print(json.dumps({"passed": True, "phase": args.phase, "seconds": time.monotonic() - started}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
