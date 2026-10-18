"""实读 overfit6 的身体参数，验证重定向、相机、GT守卫及可选三角面渲染。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def audit_known_triangles(api, renderer, torch):
    """独立于数据集的 GPU 行序、metric Z 与跨人遮挡断言。"""
    class PlaneBody:
        family = "smpl"
        vertex_count = 3
        faces = torch.tensor([[0, 1, 2]])
        vertex_parts = torch.tensor([1, 1, 1])
        provenance = {"backend": "known_CV_triangles"}
    params = api.BodyParameters("smpl", (41, 7), torch.zeros(2, 10), torch.zeros(2, 3),
                                torch.zeros(2, 69), torch.zeros(2, 3), "camera_cv")
    vertices = torch.tensor([[[-.5, -.5, 3.], [.5, -.5, 3.], [0., 0., 3.]],
                             [[-1/3, -1/3, 2.], [1/3, -1/3, 2.], [0., 0., 2.]]])
    state = api.GeometryState(params, api.BodyOutput(vertices, vertices))
    camera = api.Camera(torch.tensor([[100., 0, 40], [0, 100, 30], [0, 0, 1.]]), (60, 80))
    result = renderer.render(state, camera, PlaneBody())
    mask = result.mask[1]
    checks = {"filled_triangle": int(mask.sum()) > 100,
              "metric_z": torch.allclose(result.depth_m[1][mask], torch.full_like(result.depth_m[1][mask], 2), atol=1e-5),
              "nearest_owner": bool((result.visible_owner[mask] == 2).all()),
              "cv_normal": bool((result.normal_cv[1][mask][:, 2] > .999).all()),
              "top_down_rows": float(mask.nonzero()[:, 0].float().mean()) < 30}
    narrow = api.Camera(torch.tensor([[100., 0, 10], [0, 100, 30], [0, 0, 1.]]), (60, 20))
    canvas = api.PixelCanvasTransform.letterbox((60, 20), (60, 80))
    clipped = renderer.render(state, narrow, PlaneBody(), canvas=canvas)
    outside = torch.ones((60, 80), dtype=torch.bool, device=clipped.mask.device)
    outside[:, 30:50] = False
    checks["physical_fov_no_padding"] = bool(clipped.mask.any()) and not bool(clipped.mask[:, outside].any())
    checks["zero_modalities_in_padding"] = not any(bool(value[:, outside].count_nonzero()) for value in
        (clipped.normal_cv, clipped.depth_m, clipped.part, clipped.contact))
    if not all(checks.values()):
        raise ValueError(f"known triangle renderer failed: {checks}")
    return {"passed": True, "checks": checks, "foreground_pixels": int(mask.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--solver-probe", action="store_true", help="另做已登记的 2cm root 扰动求解审计")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    legacy, project = Path(args.legacy_workspace).resolve(), Path(args.project_root).resolve()
    run = Path(args.run_dir)
    if not run.is_absolute():
        parser.error("run-dir 必须是绝对路径")
    run = run.resolve()
    if run.is_relative_to(legacy) or any(p.lower() in {"models", "datasets"} for p in run.parts):
        parser.error("产物不能写入旧工作区、models 或 datasets")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import numpy as np
    import torch
    from src.pose_control.qwen21_pro import geometry as api
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.preparation import resolve_condition_assets
    torch.set_num_threads(4)
    write_json(run / "run.json", {"stage": "geometry_audit", "argv": vars(args),
                                 "code_root": str(ROOT), "script_sha256": file_sha256(__file__)})
    try:
        manifest = legacy / "data/overfit6_conditions.jsonl"
        manifest_sha256 = file_sha256(manifest)
        lock = json.loads((legacy / "data/lock.json").read_text(encoding="utf-8"))
        if lock["selected_manifest_sha256"][manifest.name] != manifest_sha256:
            raise ValueError("overfit6 manifest differs from its locked SHA")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 6:
            raise ValueError("expected exactly six locked records")
        smplest_root = project / "add/tools/SMPLest-X"
        smplx_backend = api.load_smplestx_backend(smplest_root)
        smpl_backend = api.load_hi4d_smpl_backend(smplest_root / "human_models/human_model_files/smpl/SMPL_MALE.pkl")
        renderer = api.NVDiffDenseRenderer(device=args.device) if args.render else None
        renderer_probe = audit_known_triangles(api, renderer, torch) if renderer is not None else None
        results = []
        for record in records:
            sample_id, dataset = record["sample_id"], record["dataset"]
            assets = resolve_condition_assets(record, project)
            if dataset in {"iper", "tiktok"}:
                sequence = record["metadata"]["sequence_id"]
                if dataset == "iper":
                    base = project / "datasets/iPER/iper_sampled_6src64tgt" / sequence / "smplestx"
                else:
                    base = project / "datasets/TikTokDataset/TikTok_3d_assets_native" / sequence / "smplestx"
                params_path, frame_map = base / "params.pt", base / "frame_map.json"
                shape, driver, camera, provenance = api.load_smplestx_parameters(params_path, frame_map,
                    source_frame=Path(assets["source_rgb"]).stem, driver_frame=Path(assets["target_keypoints"][0]).stem,
                    dataset=dataset, person_id=record["person_order"][0])
                parameters = torch.load(params_path, map_location="cpu", weights_only=True)
                index = provenance["driver_tensor_index"]
                expected_mesh = parameters["mesh_cam"][index:index + 1]
                # Estimator output is 12x16 body-crop heatmap coordinates, not native pixels.
                crop = parameters["joint_proj_model"][index:index + 1]
                fx, fy = camera.K[0, 0], camera.K[1, 1]
                cx, cy = camera.K[0, 2], camera.K[1, 2]
                crop_w, crop_h = fx * 192 / 5000, fy * 256 / 5000
                expected_joints = torch.stack([cx - crop_w / 2 + crop[..., 0] / 12 * crop_w,
                                               cy - crop_h / 2 + crop[..., 1] / 16 * crop_h], -1)
                backend, contacts = smplx_backend, ()
            else:
                source_rgb = Path(assets["source_rgb"])
                pair, camera_name, action, source_frame = source_rgb.parts[-5:-1]
                source_raw = project / "datasets/Hi4D_pilot_v1/raw" / pair / action / "smpl" / f"{source_frame}.npz"
                shape, driver, raw_contact, provenance = api.load_hi4d_parameters(source_raw, assets["contact_mesh"],
                    person_order=tuple(record["person_order"]))
                if driver.person_ids != tuple(record["person_order"]):
                    raise ValueError("Hi4D stable IDs differ from locked source order")
                with np.load(assets["contact_mesh"], allow_pickle=False) as raw:
                    expected_mesh = torch.from_numpy(raw["verts"][list(record["person_order"])].copy()).float()
                with np.load(assets["geometry"], allow_pickle=False) as geo, np.load(assets["camera"], allow_pickle=False) as cameras:
                    camera_index = list(cameras["ids"]).index(assets["camera_id"])
                    camera = api.Camera(torch.from_numpy(geo["K_runtime"].copy()).float(),
                                        (int(geo["height"]), int(geo["width"])),
                                        torch.from_numpy(cameras["extrinsics"][camera_index].copy()).float())
                backend, expected_joints = smpl_backend, None
                contacts = api.contacts_from_raw_flags(expected_mesh, raw_contact, person_ids=driver.person_ids, max_pairs=8)
            request = api.retarget_request(shape, driver, camera, backend)
            mesh_error = float((request.driver.body.vertices - expected_mesh).abs().max())
            if mesh_error > 1e-4:
                raise ValueError(f"original model reconstruction differs from stored mesh: {sample_id}: {mesh_error}")
            projection_error = None
            if expected_joints is not None:
                projected = camera.project(request.driver.body.joints, frame=driver.frame, z_epsilon=1e-4)
                projection_error = float((projected - expected_joints).abs().max())
                if projection_error > .25:
                    raise ValueError(f"raw camera projection mismatch: {sample_id}: {projection_error}")
            unchanged = api.solve_contacts(request, (), backend)
            if not torch.equal(unchanged.solved.body.vertices, request.requested.body.vertices):
                raise ValueError("empty contact solver changed requested geometry")
            solved = api.solve_contacts(request, contacts, backend)
            if not torch.equal(solved.camera.K, request.camera.K):
                raise ValueError("contact solver changed camera intrinsics")
            if camera.world_to_camera is not None and not torch.equal(solved.camera.world_to_camera, camera.world_to_camera):
                raise ValueError("contact solver changed camera extrinsics")
            root_delta = (solved.solved.parameters.transl - request.requested.parameters.transl).norm(dim=-1)
            if (root_delta > request.source_rest_heights_m * .05 + 1e-7).any():
                raise ValueError("contact solver exceeded source-height root bounds")
            if not solved.solver_log["success"] and not torch.equal(solved.solved.body.vertices, request.requested.body.vertices):
                raise ValueError("failed contact solver did not return requested geometry")
            solver_probe = None
            if args.solver_probe and contacts:
                from dataclasses import replace
                translation = driver.transl.clone()
                translation[0, 0] += .02
                diagnostic = api.retarget_request(shape, replace(driver, transl=translation), camera, backend)
                repaired = api.solve_contacts(diagnostic, contacts, backend)
                solver_probe = {"kind": "synthetic_2cm_root_perturbation_on_real_body",
                                "delta_m": [.02, 0, 0], "slot": 0, "log": repaired.solver_log}
                if not repaired.solver_log["success"] or repaired.solver_log["iterations"] == 0:
                    raise ValueError(f"real body gradient solver probe did not repair perturbation: {solver_probe}")
            exact_pair = True
            try:
                request.require_exact_gt_pairing()
            except ValueError:
                exact_pair = False
            beta_changed = not torch.equal(driver.betas, shape.betas)
            if beta_changed and exact_pair:
                raise ValueError("source beta edit incorrectly allowed old RGB GT pairing")
            canvas = api.PixelCanvasTransform.letterbox(camera.image_hw, (512, 512))
            result = {"sample_id": sample_id, "dataset": dataset, "person_ids": request.person_ids,
                      "parameter_provenance": provenance, "body_provenance": request.provenance,
                      "driver_mesh_max_error_m": mesh_error, "driver_projection_max_error_px": projection_error,
                      "source_rest_heights_m": request.source_rest_heights_m.tolist(),
                      "source_beta_changed": beta_changed, "original_rgb_exact_pair_allowed": exact_pair,
                      "requested_vs_driver_mesh_max_m": float((request.requested.body.vertices - request.driver.body.vertices).abs().max()),
                      "no_contact_identity": True, "annotated_contact_pairs": len(contacts),
                      "contact_solver": solved.solver_log,
                      "synthetic_solver_probe": solver_probe,
                      "requested_part_rois": api.projected_part_rois(request.requested, camera, backend, canvas=canvas)}
            arrays = {"driver_vertices": request.driver.body.vertices.numpy(),
                      "requested_vertices": request.requested.body.vertices.numpy(),
                      "requested_joints": request.requested.body.joints.numpy(), "camera_K": camera.K.numpy()}
            arrays["solved_vertices"] = solved.solved.body.vertices.numpy()
            if renderer is not None:
                dense = renderer.render(request.requested, camera, backend, canvas=canvas, contacts=contacts)
                spatial = dense.typed_spatial(target_hw=(32, 32))
                for key, value in spatial.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"nonfinite typed render: {key}")
                    arrays[f"typed_{key}"] = value.detach().cpu().numpy()
                result["renderer"] = dense.metadata
                result["rendered_person_pixels"] = dense.mask.flatten(1).sum(1).cpu().tolist()
                result["rendered_owner_labels"] = dense.visible_owner.unique().cpu().tolist()
                result["typed_shapes"] = {key: list(value.shape) for key, value in spatial.items()}
            output = run / f"{sample_id}.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            result["output"] = str(output)
            result["output_sha256"] = file_sha256(output)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("sample_id", "driver_mesh_max_error_m", "source_beta_changed",
                                                    "original_rgb_exact_pair_allowed", "annotated_contact_pairs")}), flush=True)
        if file_sha256(manifest) != manifest_sha256:
            raise ValueError("locked manifest changed during geometry audit")
        write_json(run / "report.json", {"schema": "qwen21_pro_geometry_audit_v1", "passed": True,
            "manifest_sha256": manifest_sha256, "renderer_probe": renderer_probe,
            "samples": results, "seconds": time.monotonic() - started})
        print(json.dumps({"passed": True, "samples": len(results), "run_dir": str(run)}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
