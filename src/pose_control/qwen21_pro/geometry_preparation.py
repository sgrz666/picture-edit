"""把当前 driver/requested/solved mesh 真正接入原生控制与 typed 条件。

可在既有 SMPL 环境先运行 ``python -m ...geometry_preparation job.json``，
再在 Qwen 环境编码生成的控制图；不要求 Qwen 环境安装 smplx/nvdiffrast。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from .cache import (
    _payload_digest,
    asset_hashes,
    file_sha256,
    ordinary_cpu,
    tensor_sha256,
)
from .contracts import ContactRelations
from .geometry import (
    Camera,
    GeometryState,
    PixelCanvasTransform,
    SourceShape,
    dwpose134,
    projected_part_rois,
    retarget_request,
)

P_ROOT = Path("/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1")
ROI_POLICY = {
    "min_support_pixels": 8,
    "bounds": "source_mesh_parts_native_xyxy",
    "landmarks": "head_bounds_not_face_landmarks;SMPL_hands_not_21_finger_joints",
    "visibility": "source_visible_mask_only;occlusion_unknown",
    "margin_px": 0.0,
}


def _exclusive_directory(path, output_root):
    path, root = Path(path).resolve(), Path(output_root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(
            "Output must be a new child directory of the explicit Pro output root"
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _save_image(path, image):
    with Path(path).open("xb") as stream:
        image.save(stream, format="PNG")


def _geometry_code():
    return {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("geometry.py", "geometry_preparation.py")
    }


def load_hi4d_camera(path, camera_id, image_hw):
    with np.load(path, allow_pickle=False) as data:
        indices = np.nonzero(data["ids"] == camera_id)[0]
        if len(indices) != 1:
            raise ValueError(
                "Hi4D source camera ID must occur exactly once in its own action"
            )
        index = int(indices[0])
        if "dist_coeffs" in data and np.any(np.abs(data["dist_coeffs"][index]) > 1e-10):
            raise ValueError(
                "Nonzero camera distortion requires explicit undistortion before projection"
            )
        k = torch.from_numpy(data["intrinsics"][index].copy()).float()
        extrinsic = torch.from_numpy(data["extrinsics"][index].copy()).float()
    return Camera(k, tuple(image_hw), extrinsic)


def verify_hi4d_forward(path, state, *, person_order):
    """Matched male backend 只覆盖原始 male mesh；拒绝性别/网格不一致的 GT。"""
    with np.load(path, allow_pickle=False) as raw:
        expected = torch.from_numpy(raw["verts"][list(person_order)].copy()).to(
            state.body.vertices
        )
        genders = None
        for key in ("genders", "gender"):
            if key in raw:
                declared = np.asarray(raw[key]).reshape(-1)
                if declared.dtype.kind in "US":
                    genders = [
                        (
                            value.decode() if isinstance(value, bytes) else str(value)
                        ).lower()
                        for value in declared
                    ]
                    if any(value not in ("male", "m") for value in genders):
                        raise ValueError(
                            "Matched male Hi4D backend cannot reconstruct female/mixed gender records"
                        )
                break
    if (
        expected.shape != state.body.vertices.shape
        or not torch.isfinite(expected).all()
    ):
        raise ValueError("Hi4D raw mesh shape/values differ from matched body forward")
    error = float((expected - state.body.vertices).abs().max())
    if error > 1e-4:
        raise ValueError(f"Hi4D matched male forward mismatch: {error}m")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "max_abs_error_m": error,
        "genders": genders,
        "coverage": "matched_male_only;no_full_mixed_gender_coverage_claim",
    }


def source_geometry_paths(record, project_root):
    """由 original SOURCE 路径解析 own source action，不读取 target action/camera。"""
    original = Path(
        record.get("metadata", {}).get("original_source_rgb", record["source_rgb"])
    )
    pair, camera, action, frame = original.parts[-5:-1]
    if not pair.startswith("pair") or not camera.startswith("cam"):
        raise ValueError(
            "Hi4D original source path must identify pair/camera/source_action/frame"
        )
    root = Path(project_root) / "datasets/Hi4D_pilot_v1/raw" / pair / action
    return {
        "source_raw_parameters": root / "smpl" / f"{frame}.npz",
        "source_camera": root / "cameras/rgb_cameras.npz",
        "source_camera_id": int(camera[3:]),
    }


def validate_source_geometry(bundle, record, assets):
    if tuple(bundle["person_ids"]) != tuple(record["person_order"]):
        raise ValueError("source geometry stable person order differs")
    if file_sha256(assets["source_rgb"]) != bundle["source_rgb_sha256"]:
        raise ValueError("source geometry RGB SHA256 differs")
    if asset_hashes(bundle["asset_files"]) != bundle["asset_hashes"]:
        raise ValueError("source geometry asset SHA256 differs")
    if (
        bundle.get("code_signature", {}).get("geometry.py") is not None
        and bundle["code_signature"] != _geometry_code()
    ):
        raise ValueError("source geometry helper code changed")
    with Image.open(assets["source_rgb"]) as image:
        if list(image.size) != list(bundle["source_image_wh"]):
            raise ValueError("source geometry image dimensions differ")
    if len(bundle["rois"]) != len(record["person_order"]):
        raise ValueError("source geometry ROI person count differs")
    return bundle


def build_source_geometry(
    record, assets, *, body_model, state, camera, asset_files, source_camera_id=None
):
    """原始源 mesh + 自己的 camera + 可见源 mask；不猜隐藏面部与手指。"""
    if state.parameters.person_ids != tuple(record["person_order"]):
        raise ValueError("source geometry state/person binding differs")
    original = Path(assets["source_rgb"])
    if (
        len(original.parts) >= 5
        and original.parts[-5].startswith("pair")
        and original.parts[-4].startswith("cam")
    ):
        pair, camera_token, action, frame = original.parts[-5:-1]
        parameters = Path(asset_files["source_raw_parameters"])
        calibration = Path(asset_files["source_camera"])
        if tuple(parameters.parts[-4:]) != (
            pair,
            action,
            "smpl",
            f"{frame}.npz",
        ) or tuple(calibration.parts[-4:]) != (
            pair,
            action,
            "cameras",
            "rgb_cameras.npz",
        ):
            raise ValueError(
                "Source mesh and camera must come from their own source action/frame"
            )
        if source_camera_id is None or source_camera_id != int(
            camera_token.removeprefix("cam")
        ):
            raise ValueError(
                "Source camera ID must explicitly match the original source RGB camera"
            )
    with Image.open(assets["source_rgb"]) as opened:
        wh = opened.size
    if camera.image_hw != wh[::-1]:
        raise ValueError("source geometry camera and original RGB dimensions differ")
    raw_rois = projected_part_rois(state, camera, body_model)
    rois, coverage = [], []
    for slot, path in enumerate(assets["source_masks"]):
        with Image.open(path) as image:
            if image.size != wh:
                raise ValueError(
                    "source person mask dimensions must match original source"
                )
            mask = np.asarray(image.convert("L")) > 0
        valid, counts = {}, {}
        for name, box in raw_rois[slot]["mesh_part_bounds_xyxy"].items():
            if box is None or not np.isfinite(box).all():
                valid[name], counts[name] = None, 0
                continue
            x0, y0 = np.floor(box[:2]).astype(int).clip(0, wh)
            x1, y1 = np.ceil(box[2:]).astype(int).clip(0, wh)
            count = int(mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
            valid[name] = [int(x0), int(y0), int(x1), int(y1)] if count >= 8 else None
            counts[name] = count
        rois.append(valid)
        coverage.append(counts)
    files = {key: str(Path(value).resolve()) for key, value in asset_files.items()}
    for key in ("model_path", "loader_path", "smplx_loader_path"):
        if body_model.provenance.get(key):
            files[f"source_{key}"] = str(body_model.provenance[key])
    files.update(
        {f"source_mask_{i}": str(p) for i, p in enumerate(assets["source_masks"])}
    )
    bundle = {
        "schema": "source_mesh_rois_v1",
        "person_ids": list(state.parameters.person_ids),
        "rois": rois,
        "support_pixels": coverage,
        "source_rgb_sha256": file_sha256(assets["source_rgb"]),
        "source_image_wh": list(wh),
        "asset_files": files,
        "asset_hashes": asset_hashes(files),
        "source_camera_id": source_camera_id,
        "body_model": body_model.provenance,
        "source_parameters_digest": _payload_digest(asdict(state.parameters)),
        "source_camera": {
            "K": camera.K.tolist(),
            "world_to_camera": (
                None
                if camera.world_to_camera is None
                else camera.world_to_camera.tolist()
            ),
        },
        "vertex_parts_sha256": tensor_sha256(body_model.vertex_parts),
        "code_signature": _geometry_code(),
        "policy": ROI_POLICY,
        "landmark_availability": raw_rois[0]["landmark_availability"],
    }
    return validate_source_geometry(bundle, record, assets)


def load_source_geometry(path, record, assets):
    return validate_source_geometry(
        json.loads(Path(path).read_text(encoding="utf-8")), record, assets
    )


@dataclass(frozen=True)
class PreparedGeometry:
    request: object
    dense: object
    control_image: Path
    pose_images: tuple[Path, ...]
    rois: tuple[dict, ...]
    canvas: PixelCanvasTransform
    contacts: tuple
    vertex_parts: torch.Tensor
    manifest: dict


def _pose_guide(state, camera, canvas, slot):
    smplx = state.parameters.family == "smplx"
    joints = dwpose134(state.body, family="smplx") if smplx else state.body.joints
    xy = (
        camera.project(joints, frame=state.parameters.frame, canvas=canvas)[slot]
        .detach()
        .cpu()
        .numpy()
    )
    image = Image.new("RGB", canvas.output_hw[::-1])
    draw = ImageDraw.Draw(image)
    # SMPL24 body parents. SMPL-X134 retains a separate actual point layout;
    # drawing its actual points is truthful without inventing unavailable joints.
    parents = (
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    )
    if not smplx:
        for child, parent in enumerate(parents[: len(xy)]):
            if parent >= 0 and np.isfinite(xy[[child, parent]]).all():
                draw.line(
                    [tuple(xy[parent]), tuple(xy[child])], fill=(80, 220, 160), width=3
                )
    for point in xy:
        if np.isfinite(point).all():
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 180, 80))
    pixels = np.array(image)
    pixels[~canvas.valid_fov_mask().cpu().numpy()] = 0
    return Image.fromarray(pixels)


def prepare_geometry(
    request,
    *,
    body_model,
    renderer,
    output_dir,
    output_root=P_ROOT,
    canvas_hw=(512, 512),
    mode="source_shape",
    for_training=False,
    contacts=(),
    input_assets=None,
):
    """每次都渲染当前 native controls；绝不复用旧 driver control_latents/pose refs。

    driver_exact 是与原 GT 精确配对的模式；任何 beta/solve 变化都被 guard 拒绝。
    source_shape 选择 solved state 并明确标记 unpaired，不能给旧 GT 训练。
    """
    if mode not in {"driver_exact", "source_shape"}:
        raise ValueError("mode must be driver_exact or source_shape")
    if mode == "driver_exact":
        request.require_exact_gt_pairing()
    elif for_training:
        raise ValueError("source_shape request is unpaired with original GT")
    if any(v % 32 for v in canvas_hw):
        raise ValueError("native canvas must be divisible by32")
    state = request.driver if mode == "driver_exact" else request.solved
    canvas = PixelCanvasTransform.letterbox(request.camera.image_hw, tuple(canvas_hw))
    dense = renderer.render(
        state, request.camera, body_model, canvas=canvas, contacts=contacts
    )
    if (
        dense.person_ids != request.person_ids
        or (dense.mask & ~canvas.valid_fov_mask(device=dense.mask.device)[None]).any()
    ):
        raise ValueError("current rendered geometry person/FoV binding differs")
    path = _exclusive_directory(output_dir, output_root)
    spatial = dense.typed_spatial()
    # Current metric depth is a truthful native Union control for SMPL24. No
    # synthetic face/finger keypoints are inferred from a body-only mesh.
    control = (
        (spatial["depth"][0, :, 0] * dense.visibility)
        .sum(0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )
    # Constant-depth foreground still has visible support in its native guide.
    control = np.where(
        dense.mask.any(0).cpu().numpy(), np.maximum(control, 32), 0
    ).astype(np.uint8)
    control_path = path / "control_current_depth.png"
    _save_image(control_path, Image.fromarray(control).convert("RGB"))
    pose_paths = []
    for slot, person_id in enumerate(request.person_ids):
        pose_path = path / f"pose_person_{person_id}.png"
        _save_image(pose_path, _pose_guide(state, request.camera, canvas, slot))
        pose_paths.append(pose_path)
    rois = tuple(projected_part_rois(state, request.camera, body_model, canvas=canvas))
    data = {
        "spatial": ordinary_cpu(
            dense.typed_spatial(target_hw=tuple(v // 16 for v in canvas_hw))
        ),
        "person_ids": request.person_ids,
        "rois": rois,
        "canvas": canvas.to_dict(),
        "state": ordinary_cpu(asdict(state)),
        "camera": ordinary_cpu(asdict(request.camera)),
        "source_rest_heights_m": ordinary_cpu(request.source_rest_heights_m),
        "contacts": [asdict(c) for c in contacts],
        "vertex_parts": ordinary_cpu(body_model.vertex_parts),
        "visible_owner": ordinary_cpu(dense.visible_owner),
    }
    with (path / "geometry.pt").open("xb") as stream:
        torch.save(data, stream)
    signature = {
        "mode": mode,
        "states": {
            name: _payload_digest(asdict(getattr(request, name)))
            for name in ("driver", "requested", "solved")
        },
        "camera": _payload_digest(asdict(request.camera)),
        "body_model": body_model.provenance,
        "canvas": canvas.to_dict(),
        "code": _geometry_code(),
        "dense_sha256": file_sha256(path / "geometry.pt"),
        "input_asset_files": {
            k: str(Path(v).resolve()) for k, v in (input_assets or {}).items()
        },
        "input_asset_hashes": asset_hashes(input_assets or {}),
        "request_provenance": request.provenance,
    }
    manifest = {
        "schema": "current_geometry_preparation_v1",
        "geometry_signature": signature,
        "paired_supervision": mode == "driver_exact",
        "person_ids": list(request.person_ids),
        "native_control_style": (
            "current_depth_smplx134"
            if state.parameters.family == "smplx"
            else "current_depth_body24"
        ),
        "control_image": str(control_path),
        "pose_images": list(map(str, pose_paths)),
        "rois": rois,
        "solver_log": request.solver_log,
        "asset_hashes": asset_hashes(
            {
                "control": control_path,
                **{f"pose_{i}": p for i, p in enumerate(pose_paths)},
            }
        ),
    }
    _save_json(path / "manifest.json", manifest)
    return PreparedGeometry(
        request,
        dense,
        control_path,
        tuple(pose_paths),
        rois,
        canvas,
        tuple(contacts),
        body_model.vertex_parts.detach().cpu().clone(),
        manifest,
    )


def geometry_task_condition(source_pack, prepared, *, config):
    """在同一新 canvas 上构建 spatial/owner/ROI/显式 contact；源身份 bank 保持绑定。"""
    if source_pack.person_ids != prepared.request.person_ids:
        raise ValueError("source bank and current geometry stable person order differ")
    hw = tuple(v // 16 for v in prepared.canvas.output_hw)
    spatial = ordinary_cpu(prepared.dense.typed_spatial(target_hw=hw))
    valid = {name: source_pack.person_valid.clone() for name in spatial}
    masks = spatial["mask"].flatten(2)
    owner = torch.minimum(masks, spatial["visibility"].flatten(2))
    owner *= source_pack.person_valid[..., None]
    roi_masks = torch.zeros(1, len(source_pack.person_ids), 3, *hw)
    for slot, record in enumerate(prepared.rois):
        for part, name in enumerate(("face", "left_hand", "right_hand")):
            box = record["mesh_part_bounds_xyxy"].get(name)
            if box is not None:
                x0, y0 = np.floor(np.array(box[:2]) / 16).astype(int).clip(0, hw[::-1])
                x1, y1 = np.ceil(np.array(box[2:]) / 16).astype(int).clip(0, hw[::-1])
                roi_masks[0, slot, part, y0:y1, x0:x1] = 1
    roi_masks *= spatial["visibility"]
    contacts = prepared.contacts
    pids = source_pack.person_ids
    relation = source_pack.relations
    relation_metadata = {
        "normalization": "mean_current_parameter_frame_mesh_max_axis_extent_m",
        "relative_frame": "camera_cv_xyz",
        "endpoint_visibility_available": True,
        "endpoint_lookup": "round_projected_canvas_xy;0=background;slot+1=visible_person",
        "visibility_resolution_hw": list(prepared.canvas.output_hw),
    }
    if contacts:
        state, camera = prepared.request.solved, prepared.request.camera
        vertices = camera.to_camera(
            state.body.vertices, frame=state.parameters.frame
        ).cpu()
        # Same ABI as preparation.build_contact_relations: max-axis extent is
        # measured in the declared parameter frame before camera conversion.
        extent = state.body.vertices.amax(1) - state.body.vertices.amin(1)
        scale = float(extent.amax(1).mean())
        if not math.isfinite(scale) or scale <= 1e-8:
            raise ValueError("Current geometry contact normalization is degenerate")
        relation_metadata.update(parameter_frame=state.parameters.frame, extent_m=scale)
        src, dst, src_parts, dst_parts, relative = [], [], [], [], []
        labels = prepared.vertex_parts
        projected = (
            camera.project(
                state.body.vertices,
                frame=state.parameters.frame,
                canvas=prepared.canvas,
            )
            .detach()
            .cpu()
        )
        visible_owner = prepared.dense.visible_owner.detach().cpu()

        def visible(slot, vertex):
            point = projected[slot, vertex]
            if not torch.isfinite(point).all():
                return 0.0
            x, y = point.round().long().tolist()
            h, w = visible_owner.shape
            return float(
                0 <= x < w and 0 <= y < h and int(visible_owner[y, x]) == slot + 1
            )

        for contact in contacts:
            a, b = contact.a, contact.b
            sa, sb = pids.index(a.person_id), pids.index(b.person_id)
            delta = (
                vertices[sb, b.vertex_index] - vertices[sa, a.vertex_index]
            ) / scale
            src.append(sa)
            dst.append(sb)
            src_parts.append(int(labels[a.vertex_index]))
            dst_parts.append(int(labels[b.vertex_index]))
            relative.append(
                [
                    *delta.tolist(),
                    float(delta.norm()),
                    visible(sa, a.vertex_index),
                    visible(sb, b.vertex_index),
                ]
            )
        enabled = source_pack.person_valid[:, src] & source_pack.person_valid[:, dst]
        relation = ContactRelations(
            torch.tensor([src]),
            torch.tensor([src_parts]),
            torch.tensor([dst]),
            torch.tensor([dst_parts]),
            torch.tensor([relative]),
            torch.ones(1, len(src)),
            enabled,
        )
    else:
        # Old driver relations may describe different positions after retargeting.
        empty = torch.zeros(1, 0, dtype=torch.long)
        relation = ContactRelations(
            empty,
            empty.clone(),
            empty.clone(),
            empty.clone(),
            torch.zeros(1, 0, 6),
            torch.zeros(1, 0),
            torch.zeros(1, 0, dtype=torch.bool),
        )
    provenance = {
        "source": source_pack.provenance.get("source", {}),
        "geometry": prepared.manifest["geometry_signature"],
        "geometry_rois": prepared.rois,
        "geometry_relations": relation_metadata,
        "availability": {name: bool(value.all()) for name, value in valid.items()},
    }
    return replace(
        source_pack,
        target_hw=hw,
        spatial=spatial,
        spatial_valid=valid,
        person_mask=masks,
        owner=owner,
        face_mask=roi_masks[:, :, 0].flatten(2),
        hand_mask=roi_masks[:, :, 1:].flatten(3),
        relations=relation,
        provenance=provenance,
    ).validate(config)


def load_geometry_task_condition(source_pack, manifest_path, *, config):
    """Qwen 环境消费 SMPL worker 的文件交接；无需 smplx/nvdiffrast。

    验证当前 native pose/control 字节、typed payload、helper code，再组装新 pack。
    """
    from .geometry import BodyOutput, BodyParameters, VertexContact, VertexEndpoint

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    signature = manifest["geometry_signature"]
    if asset_hashes(signature.get("input_asset_files", {})) != signature.get(
        "input_asset_hashes", {}
    ):
        raise ValueError("Current geometry raw input asset SHA256 differs")
    assets = {
        "control": manifest["control_image"],
        **{f"pose_{i}": p for i, p in enumerate(manifest["pose_images"])},
    }
    if (
        asset_hashes(assets) != manifest["asset_hashes"]
        or file_sha256(path.parent / "geometry.pt") != signature["dense_sha256"]
    ):
        raise ValueError("Current geometry handoff asset SHA256 differs")
    if signature["code"] != _geometry_code():
        raise ValueError("Current geometry helper code differs from handoff")
    data = torch.load(
        path.parent / "geometry.pt", weights_only=True, map_location="cpu"
    )
    if tuple(data["person_ids"]) != source_pack.person_ids:
        raise ValueError("Current geometry handoff person binding differs")
    params = BodyParameters(**data["state"]["parameters"])
    state = GeometryState(params, BodyOutput(**data["state"]["body"]))
    contacts = tuple(
        VertexContact(
            VertexEndpoint(**c["a"]),
            VertexEndpoint(**c["b"]),
            **{k: v for k, v in c.items() if k not in ("a", "b")},
        )
        for c in data["contacts"]
    )
    canvas = signature["canvas"]
    prepared = SimpleNamespace(
        request=SimpleNamespace(
            person_ids=params.person_ids,
            solved=state,
            camera=Camera(**data["camera"]),
            source_rest_heights_m=data["source_rest_heights_m"],
        ),
        dense=SimpleNamespace(
            typed_spatial=lambda **kwargs: data["spatial"],
            visible_owner=data["visible_owner"],
        ),
        rois=data["rois"],
        canvas=PixelCanvasTransform(
            torch.tensor(canvas["matrix"]),
            tuple(canvas["input_hw"]),
            tuple(canvas["output_hw"]),
        ),
        contacts=contacts,
        vertex_parts=data["vertex_parts"],
        manifest=manifest,
    )
    return geometry_task_condition(source_pack, prepared, config=config)


def _job_main(job):
    """JSON worker：只读参数/相机，全部新产物写入显式 P root 子目录。"""
    from .geometry import (
        load_hi4d_parameters,
        load_hi4d_smpl_backend,
        load_smplestx_backend,
        load_smplestx_parameters,
        NVDiffDenseRenderer,
        contacts_from_raw_flags,
        solve_contacts,
    )

    if Path(job.get("output_root", P_ROOT)).resolve() != P_ROOT.resolve():
        raise ValueError("Production geometry worker outputs must stay in Pro P root")
    mode = job.get("mode", "driver_exact")
    dataset = job["dataset"]
    if job.get("source_geometry_only"):
        if dataset != "hi4d":
            raise ValueError(
                "source mesh ROI worker currently needs Hi4D raw source/camera inputs"
            )
        backend = load_hi4d_smpl_backend(job["body_model_path"])
        _, source, _, _ = load_hi4d_parameters(
            job["source_params"],
            job["source_params"],
            person_order=tuple(job["person_order"]),
        )
        source_camera = load_hi4d_camera(
            job["source_camera_path"],
            job["source_camera_id"],
            tuple(job["source_image_hw"]),
        )
        state = GeometryState(source, backend.forward(source))
        forward_audit = verify_hi4d_forward(
            job["source_params"], state, person_order=tuple(job["person_order"])
        )
        record = {"person_order": job["person_order"]}
        assets = {"source_rgb": job["source_rgb"], "source_masks": job["source_masks"]}
        bundle = build_source_geometry(
            record,
            assets,
            body_model=backend,
            state=state,
            camera=source_camera,
            asset_files={
                "source_raw_parameters": job["source_params"],
                "source_camera": job["source_camera_path"],
            },
            source_camera_id=job["source_camera_id"],
        )
        path = _exclusive_directory(job["output_dir"], P_ROOT)
        bundle["forward_audit"] = forward_audit
        _save_json(path / "source_geometry.json", bundle)
        return
    if dataset == "hi4d":
        backend = load_hi4d_smpl_backend(job["body_model_path"])
        shape, driver, flags, provenance = load_hi4d_parameters(
            job["source_params"],
            job["driver_params"],
            person_order=tuple(job["person_order"]),
        )
        camera = load_hi4d_camera(
            job["camera_path"], job["camera_id"], tuple(job["image_hw"])
        )
        _, source, _, _ = load_hi4d_parameters(
            job["source_params"],
            job["source_params"],
            person_order=tuple(job["person_order"]),
        )
        forward_audit = {
            "source": verify_hi4d_forward(
                job["source_params"],
                GeometryState(source, backend.forward(source)),
                person_order=tuple(job["person_order"]),
            ),
            "driver": verify_hi4d_forward(
                job["driver_params"],
                GeometryState(driver, backend.forward(driver)),
                person_order=tuple(job["person_order"]),
            ),
        }
        input_assets = {
            "source_params": job["source_params"],
            "driver_params": job["driver_params"],
            "camera": job["camera_path"],
        }
    else:
        backend = load_smplestx_backend(job["smplest_root"])
        shape, driver, camera, provenance = load_smplestx_parameters(
            job["params"],
            job["frame_map"],
            source_frame=job["source_frame"],
            driver_frame=job["driver_frame"],
            dataset=dataset,
            person_id=job.get("person_id", 0),
        )
        flags = None
        forward_audit = {}
        input_assets = {"params": job["params"], "frame_map": job["frame_map"]}
    for key in ("model_path", "loader_path", "smplx_loader_path"):
        if backend.provenance.get(key):
            input_assets[key] = backend.provenance[key]
    if mode == "driver_exact":
        shape = SourceShape(driver.family, driver.person_ids, driver.betas.clone())
    request = retarget_request(shape, driver, camera, backend)
    request = replace(
        request, provenance={**request.provenance, "raw_forward_audit": forward_audit}
    )
    contacts = (
        ()
        if flags is None
        else contacts_from_raw_flags(
            request.driver.body.vertices, flags, person_ids=driver.person_ids
        )
    )
    if job.get("solve_contacts", False):
        request = solve_contacts(
            request, contacts, backend, **job.get("solver_options", {})
        )
    prepare_geometry(
        request,
        body_model=backend,
        renderer=NVDiffDenseRenderer(),
        output_dir=job["output_dir"],
        output_root=P_ROOT,
        canvas_hw=tuple(job.get("canvas_hw", [512, 512])),
        mode=mode,
        for_training=job.get("for_training", False),
        contacts=contacts,
        input_assets=input_assets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="JSON job，见 _job_main 支持的显式参数")
    arguments = parser.parse_args()
    _job_main(json.loads(arguments.job.read_text(encoding="utf-8")))
