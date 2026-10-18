"""只读锁定 B 条件，生成带有效性门控的 Pro 条件，不读取目标 RGB。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .cache import BoundUnionCondition, asset_hashes, file_sha256, ordinary_cpu, source_feature_fingerprint
from .contracts import ContactRelations, ProAdapterConfig, SPATIAL_CHANNELS, TaskCondition
from ..qwen21_union.contracts import CanvasTransform

GEOMETRY_CODEC = {
    "version": "qwen21_pro_geometry_v1",
    "normal": "PIL_RGB/127.5-1;cameraCV=renderer*[1,-1,-1];unit;masked_background0",
    "normal_hi4d": "unverified_basis:invalid",
    "depth": "near0_far1;metric=shared_scene_minmax;visual=(255-gray)/254_frameP1P99;masked_bg0",
    "part": "v6_14_zero_based;invalid=-1;render_plus1_bg0_decoded_only_when_declared",
    "contact": "per_person_soft_contact_[0,1];zero_is_valid_when_asset_exists",
    "raw_contact": "finite_nonnegative_annotation_membership_gt0;annotated_pair_confidence1",
    "relative": "camera_dx_dy_dz,distance / scene_mean_max_axis_mesh_extent,src_visible,dst_visible",
    "roi": "DWPose134:body24,face68,left21,right21;confidence>=.25;soft_ellipse",
}


def _record(record):
    return record.to_dict() if hasattr(record, "to_dict") else dict(record)


def _transform(value):
    return CanvasTransform.from_dict(value)


def map_points(points, transform, *, inverse=False):
    """按实际 W/H resize 比例映射，避免 round 后用同一个 scale。"""
    transform = _transform(transform)
    ratio = np.asarray(transform.resized_wh) / transform.original_wh
    points = np.asarray(points, dtype=np.float64)
    return (points - transform.offset_xy) / ratio if inverse else points * ratio + transform.offset_xy


def transform_map(value, transform, *, nearest=False, fill=0):
    """资产是完整 FoV；存成 512 方形的 part 也直接拉回 resized_wh。"""
    transform = _transform(transform)
    value = np.asarray(value)
    if value.ndim not in (2, 3):
        raise ValueError("spatial asset must be HxW or HxWxC")
    source = torch.from_numpy(np.ascontiguousarray(value)).float()
    source = source[None, None] if value.ndim == 2 else source.permute(2, 0, 1)[None]
    resized = F.interpolate(source, size=transform.resized_wh[::-1], mode="nearest" if nearest else "bilinear",
                            **({} if nearest else {"align_corners": False}))[0]
    channels = () if value.ndim == 2 else (value.shape[2],)
    result = np.full((*transform.canvas_wh[::-1], *channels), fill, dtype=value.dtype if nearest else np.float32)
    x, y = transform.offset_xy
    w, h = transform.resized_wh
    data = resized[0].numpy() if value.ndim == 2 else resized.permute(1, 2, 0).numpy()
    result[y:y + h, x:x + w] = data
    return result


def decode_renderer_normal(rgb, mask):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("renderer normal must be PIL RGB")
    result = (rgb / 127.5 - 1.) * np.array([1., -1., -1.], dtype=np.float32)
    length = np.linalg.norm(result, axis=-1, keepdims=True)
    return result / np.maximum(length, 1e-8) * np.asarray(mask, bool)[..., None]


def _path(root, value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _exists(value):
    return value is not None and Path(value).is_file()


def resolve_condition_assets(condition, project_root):
    """由已锁定 original_condition_assets 找几何，不访问 supervision/target_rgb。

    所有 people/source_masks/source_keypoints/target_keypoints 列表均已按 slot 排列。
    显式 assets 可增加真正 part maps、contact_mesh、contact_vertex_parts、camera。
    """
    record = _record(condition)
    metadata = record.get("metadata", {})
    order = list(record["person_order"])
    root = Path(project_root)
    source = _path(root, metadata.get("original_source_rgb", record["source_rgb"]))
    paths = [_path(root, value) for value in metadata.get("original_condition_assets", {})]
    dataset = record["dataset"]
    masks = [_path(root, p) for p in metadata.get("original_source_masks", [])]
    result = {"source_rgb": source, "source_masks": masks,
              "source_keypoints": [None] * len(order), "target_keypoints": [None] * len(order)}
    if dataset == "hi4d":
        by_raw = {raw: next((p for p in paths if p.name == f"dwpose_{'AB'[raw]}.npz" and "targets" in p.parts), None)
                  for raw in order}
        result["target_keypoints"] = [by_raw[raw] for raw in order]
        first = next((p for p in by_raw.values() if p is not None), None)
        if first:
            result["geometry"] = first.parent / "geometry.npz"
            # final_assets.../targets/pair/action/cam/frame -> raw/pair/action/smpl/frame
            native_root = root / "datasets/Hi4D_pilot_v1/raw"
            pair, action, camera, frame = first.parts[-5:-1]
            result["contact_mesh"] = native_root / pair / action / "smpl" / f"{frame}.npz"
            result["camera"] = native_root / pair / action / "cameras/rgb_cameras.npz"
            result["camera_id"] = int(camera.removeprefix("cam"))
        result["source_keypoints"] = [source.parent / f"dwpose_{'AB'[raw]}.npz" for raw in order]
        # Own source action/camera; target raw paths above are different roles.
        from .geometry_preparation import source_geometry_paths
        try:
            result.update(source_geometry_paths(record, root))
        except ValueError:
            pass  # Small explicit CPU fixtures need not use dataset path layout.
    elif dataset == "iper":
        source_kp = next((p for p in paths if p.suffix == ".npz" and p.stem == source.stem), None)
        target_kp = next((p for p in paths if p.suffix == ".npz" and p != source_kp), None)
        result["source_keypoints"], result["target_keypoints"] = [source_kp], [target_kp]
        appearance = metadata.get("sequence_id", source.parts[-4])
        base = root / "datasets/iPER/iper_assets_512_nvdiffrast" / appearance
        def assets(role, stem):
            return {"normal": base / "normal_512" / role / f"{stem}.png",
                    "depth": base / "depth_npy_512" / role / f"{stem}.npy",
                    "mask": base / "mask_512" / role / f"{stem}.png",
                    "part": base / "part_512" / role / f"{stem}.png", "depth_kind": "metric_meters",
                    "part_codec": "v6_14_plus1_bg0"}
        source_assets = assets(source.parent.name, source.stem)
        if not masks:
            result["source_masks"] = [source_assets["mask"]]
        result["people"] = [assets(target_kp.parent.name, target_kp.stem) if target_kp else {}]
    elif dataset == "tiktok":
        keypoints = [p for p in paths if p.suffix == ".json" and "dwpose_json" in p.parts]
        source_kp = next((p for p in keypoints if p.stem == source.stem), None)
        target_kp = next((p for p in keypoints if p != source_kp), None)
        result["source_keypoints"], result["target_keypoints"] = [source_kp], [target_kp]
        seq = metadata.get("sequence_id", source.parent.parent.name)
        base = root / "datasets/TikTokDataset/TikTok_3d_assets_native" / seq
        if not masks:
            result["source_masks"] = [base / "smplx_mask" / f"{source.stem}.png"]
        result["people"] = [{"normal": base / "normal" / f"{target_kp.stem}.png",
                             "depth": base / "depth_vis" / f"{target_kp.stem}.png",
                             "mask": base / "smplx_mask" / f"{target_kp.stem}.png",
                             "part": base / "part_512" / f"{target_kp.stem}.png",
                             "depth_kind": "visual_normalized", "part_codec": "v6_14_plus1_bg0"} if target_kp else {}]
    else:
        result["people"] = [{} for _ in order]
    if len(result["source_masks"]) != len(order):
        result["source_masks"] = [None] * len(order)
    return result


def source_asset_roles(assets, condition=None):
    result = {"rgb": assets["source_rgb"]}
    if condition is not None:
        result["prepared_rgb"] = _record(condition)["source_rgb"]
    for key in ("source_masks", "source_keypoints"):
        result.update({f"{key}_{i}": value for i, value in enumerate(assets.get(key, []))})
    if assets.get("source_geometry") is not None:
        bundle = assets["source_geometry"]
        if condition is not None:
            from .geometry_preparation import validate_source_geometry
            validate_source_geometry(bundle, _record(condition), assets)
        result.update({f"source_geometry_{k}": v for k, v in bundle["asset_files"].items()})
        result["source_geometry_binding"] = {key: value for key, value in bundle.items() if key != "asset_files"}
    return result


def validate_source_assets(condition, assets, project_root):
    """拒绝锁定原源图漂移，防止旧 source VAE 与新 DINO 外观静默混合。"""
    record = _record(condition)
    metadata = record.get("metadata", {})
    declared = {str(_path(project_root, path).resolve()): sha
                for path, sha in metadata.get("original_condition_assets", {}).items()}
    expected_rgb = _path(project_root, metadata.get("original_source_rgb", record["source_rgb"]))
    expected_masks = [_path(project_root, path) for path in metadata.get("original_source_masks", [])]
    roles = [("rgb", expected_rgb, assets["source_rgb"])]
    roles += [(f"mask_{i}", expected, assets.get("source_masks", [])[i])
              for i, expected in enumerate(expected_masks) if i < len(assets.get("source_masks", []))]
    roles += [(f"keypoints_{i}", path, path) for i, path in enumerate(assets.get("source_keypoints", [])) if path]
    for role, expected, actual in roles:
        key = str(Path(expected).resolve())
        if key in declared:
            if not _exists(expected) or file_sha256(expected) != declared[key]:
                raise ValueError(f"locked source asset SHA256 mismatch: {role}")
            if not _exists(actual) or file_sha256(actual) != declared[key]:
                raise ValueError(f"locked source slot asset mismatch: {role}")
        elif role == "rgb" and Path(actual).resolve() != Path(expected).resolve():
            raise ValueError("locked source RGB role cannot be replaced without declared hash")
    for path in assets.get("source_keypoints", []):
        if _exists(path):
            path = Path(path)
            if record["dataset"] == "hi4d":
                valid = path.parent.resolve() == expected_rgb.parent.resolve()
            else:
                valid = path.stem == expected_rgb.stem
            if not valid:
                raise ValueError("source keypoints must belong to the source frame")


def geometry_asset_roles(assets):
    result = {key: assets.get(key) for key in ("geometry", "contact_mesh", "contact_vertex_parts", "camera")}
    result["semantic_options"] = {key: assets.get(key) for key in ("camera_id", "part_codec", "normal_basis")}
    result["semantic_options"]["people"] = [{key: person.get(key) for key in ("part_codec", "depth_kind", "normal_basis")}
                                              for person in assets.get("people", [])]
    vertex_parts = assets.get("contact_vertex_parts")
    if vertex_parts is not None and isinstance(vertex_parts, (str, Path)):
        result["vertex_parts_codec"] = Path(vertex_parts).with_suffix(".json")
    for key in ("part_A", "part_B"):
        result[key] = assets.get(key)
    for i, person in enumerate(assets.get("people", [])):
        result.update({f"person_{i}_{key}": person[key] for key in SPATIAL_CHANNELS if key in person})
    result.update({f"keypoints_{i}": value for i, value in enumerate(assets.get("target_keypoints", []))})
    return result


def validate_cached_binding(condition, cached):
    if not isinstance(cached, BoundUnionCondition):
        raise ValueError("cached_condition must be a verified BoundUnionCondition from load_bound_condition")
    cached.validate_binding(condition)
    record = _record(condition)
    order = tuple(record["person_order"])
    if not 1 <= len(order) <= 2 or len(set(order)) != len(order):
        raise ValueError("person_order must bind one or two unique people")
    if tuple(cached.get("person_order", ())) != order:
        raise ValueError("cached person_order differs from locked condition")
    if len(record.get("reference_pose_images", [])) != len(order):
        raise ValueError("one pose reference per bound person is required")
    hw = tuple(cached["target_hw"])
    if len(hw) != 2 or any(type(v) is not int or v <= 0 for v in hw):
        raise ValueError("invalid cached target_hw")
    shapes = cached["img_shapes"]
    if len(shapes) != 1 or len(shapes[0]) != len(order) + 2:
        raise ValueError("cached img_shapes must be [source,*pose,target]")
    if any(tuple(value) != (1, *hw) for value in shapes[0]):
        raise ValueError("cached reference/target grid mismatch")
    transforms = cached.get("transforms", {})
    for key in ("source_transform", "target_transform"):
        expected = _transform(record[key])
        if key not in transforms or _transform(transforms[key]) != expected:
            raise ValueError("cached transform differs from locked condition")
        if tuple(cached["canvas_hw"]) != expected.canvas_wh[::-1]:
            raise ValueError("cached canvas grid mismatch")
    latent = cached["reference_latents"]
    n = hw[0] * hw[1]
    if tuple(latent.shape) != (1, n * (len(order) + 1), 64) or not torch.isfinite(latent).all():
        raise ValueError("ordered references must be [1,(source+poses)*H*W,64]")
    return record, order, hw, ordinary_cpu(latent[:, :n])


def _read_array(path):
    if not _exists(path):
        return None
    if Path(path).suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image).copy()


def load_source_masks(assets, transform):
    result = []
    for path in assets.get("source_masks", []):
        array = _read_array(path)
        if array is None:
            result.append(np.zeros(_transform(transform).canvas_wh[::-1], dtype=np.float32))
        else:
            if array.ndim == 3:
                array = array.max(-1)
            result.append(transform_map((array > 0).astype(np.float32), transform, nearest=True))
    return result


def _reduce(value, hw, *, labels=False):
    array = np.asarray(value)
    tensor = torch.as_tensor(array.copy()).float()
    tensor = tensor[None, None] if tensor.ndim == 2 else tensor.permute(2, 0, 1)[None]
    return F.interpolate(tensor, size=hw, mode="nearest" if labels else "area")[0]


def _roi_from_keypoints(path, dataset, transform, hw, support):
    result = torch.zeros(3, *hw)
    if not _exists(path):
        return result
    from src.data.qwen21_union_dataset import _keypoints
    payload = _keypoints(path, dataset, _transform(transform))
    points = np.asarray(payload["xy_canvas"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    yy, xx = np.meshgrid(np.arange(hw[0]) + .5, np.arange(hw[1]) + .5, indexing="ij")
    canvas_w, canvas_h = _transform(transform).canvas_wh
    xy = points * [hw[1] / canvas_w, hw[0] / canvas_h]
    for index, block in enumerate((slice(24, 92), slice(92, 113), slice(113, 134))):
        usable = xy[block][valid[block]]
        if len(usable) < 3:
            continue
        lo, hi = usable.min(0), usable.max(0)
        if np.any(hi - lo < .05):
            continue
        center, radius = (lo + hi) / 2, np.maximum((hi - lo) * .75, .6)
        ellipse = np.exp(-.5 * (((xx - center[0]) / radius[0]) ** 2 + ((yy - center[1]) / radius[1]) ** 2))
        result[index] = torch.from_numpy(ellipse).float() * support
    return result


def _empty_relations():
    integers = {key: torch.empty(1, 0, dtype=torch.long) for key in ("src_person", "src_part", "dst_person", "dst_part")}
    return ContactRelations(**integers, relative=torch.empty(1, 0, 6), confidence=torch.empty(1, 0),
                            valid=torch.empty(1, 0, dtype=torch.bool))


def _load_spatial(record, order, assets, hw):
    p = len(order)
    transform = record["target_transform"]
    spatial = {name: torch.full((1, p, channels, *hw), -1 if name == "part" else 0,
                               dtype=torch.long if name == "part" else torch.float32)
               for name, channels in SPATIAL_CHANNELS.items()}
    valid = {name: torch.zeros(1, p, dtype=torch.bool) for name in spatial}
    provenance = {"depth_units": "unavailable", "depth_metric": False,
                  "depth_convention": "near0_far1;foreground_mask_separate;background0",
                  "normal_basis": "unavailable", "codec": GEOMETRY_CODEC}
    entries = []
    scene_depth = None
    if record["dataset"] == "hi4d" and _exists(assets.get("geometry")):
        with np.load(assets["geometry"], allow_pickle=False) as geometry:
            scene_depth = np.asarray(geometry["depth_scene"], dtype=np.float32).copy()
            for raw in order:
                side = "AB"[raw]
                mask = geometry[f"semantic_{side}"] > 0
                entry = {"mask": mask, "visibility": (geometry["front_owner"] == raw + 1).astype(np.float32),
                         "depth": geometry[f"depth_{side}"].copy(), "depth_kind": "metric_meters"}
                if f"contact_{side}" in geometry:
                    entry["contact"] = geometry[f"contact_{side}"].copy()
                # Binary semantic is amodal support, never a body-part label map.
                if f"part_{side}" in assets:
                    entry["part"] = _read_array(assets[f"part_{side}"])
                    entry["part_codec"] = assets.get("part_codec")
                entries.append(entry)
    else:
        for person in assets.get("people", [{} for _ in order]):
            entry = {name: _read_array(person.get(name)) for name in SPATIAL_CHANNELS}
            entry.update(depth_kind=person.get("depth_kind"), part_codec=person.get("part_codec"))
            entries.append(entry)
    if len(entries) != p:
        raise ValueError("spatial people list must already follow person_order")
    depth_values = []
    for entry in entries:
        mask = entry.get("mask")
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.max(-1)
            entry["mask"] = mask > 0
        depth = entry.get("depth")
        if depth is not None and depth.ndim == 3:
            if np.array_equal(depth[..., 0], depth[..., 1]) and np.array_equal(depth[..., 0], depth[..., 2]):
                depth = depth[..., 0]
                entry["depth"] = depth
            else:
                entry["depth"] = None  # Colorized depth has no invertible scalar codec.
                depth = None
        if mask is not None and depth is not None and entry.get("depth_kind") in {"metric_meters", "visual_normalized"}:
            mask_for_depth = np.asarray(Image.fromarray(entry["mask"]).resize(depth.shape[::-1], Image.Resampling.NEAREST))
            values = depth[mask_for_depth & np.isfinite(depth) & (depth > 0)]
            if len(values):
                depth_values.append(values)
    if scene_depth is not None:
        finite = scene_depth[np.isfinite(scene_depth) & (scene_depth > 0)]
        depth_values.append(finite)
    nonempty = [values for values in depth_values if len(values)]
    bounds = (float(min(v.min() for v in nonempty)), float(max(v.max() for v in nonempty))) if nonempty else None
    provenance["depth_range"] = list(bounds) if bounds else None
    for slot, entry in enumerate(entries):
        mask = entry.get("mask")
        if mask is None or not mask.any():
            continue
        canvas_mask = transform_map(mask.astype(np.float32), transform, nearest=True)
        support = _reduce(canvas_mask, hw).clamp(0, 1)
        spatial["mask"][0, slot] = support
        valid["mask"][0, slot] = True
        visibility = entry.get("visibility")
        if visibility is None and p == 1:
            visibility = mask.astype(np.float32)
        if visibility is not None:
            value = _reduce(transform_map(visibility.astype(np.float32), transform, nearest=True), hw).clamp(0, 1)
            spatial["visibility"][0, slot] = torch.minimum(value, support)
            valid["visibility"][0, slot] = True
        normal = entry.get("normal")
        if normal is not None and record["dataset"] in {"iper", "tiktok"}:
            normal_mask = np.asarray(Image.fromarray(mask).resize(normal.shape[1::-1], Image.Resampling.NEAREST))
            normal = decode_renderer_normal(normal, normal_mask)
            mapped = _reduce(transform_map(normal, transform), hw)
            mapped = F.normalize(mapped, dim=0, eps=1e-8) * support
            spatial["normal"][0, slot] = mapped
            valid["normal"][0, slot] = True
            provenance["normal_basis"] = "cameraCV:xright,ydown,zforward"
        depth = entry.get("depth")
        kind = entry.get("depth_kind")
        if depth is not None and bounds and kind in {"metric_meters", "visual_normalized"}:
            lo, hi = bounds
            if kind == "visual_normalized":
                # Producer depth_vis: bg0; fg 1+clip((P99-Z)/(P99-P1))*254.
                # Only its within-frame order is recoverable, never metric Z.
                if depth.dtype != np.uint8:
                    raise ValueError("visual depth codec requires original uint8 grayscale")
                depth = np.where(depth > 0, (255 - depth.astype(np.float32)) / 254, 0)
                provenance["depth_range"] = [1, 255]
                provenance["visual_depth_codec"] = "per_frame_P1_P99_inverse_linear_Z;original_metric_range_unavailable"
            else:
                depth = np.where(np.isfinite(depth) & (depth > 0), (depth - lo) / max(hi - lo, 1e-6), 0)
            mapped = _reduce(transform_map(depth.astype(np.float32), transform), hw) * support
            spatial["depth"][0, slot] = mapped
            valid["depth"][0, slot] = True
            provenance["depth_units"], provenance["depth_metric"] = kind, kind == "metric_meters"
        contact = entry.get("contact")
        if contact is not None:
            if not np.isfinite(contact).all() or np.any(contact < 0) or np.any(contact > 1):
                raise ValueError("contact must preserve finite soft values in [0,1]")
            spatial["contact"][0, slot] = _reduce(transform_map(contact.astype(np.float32), transform), hw) * support
            valid["contact"][0, slot] = True
        part = entry.get("part")
        codec = entry.get("part_codec")
        if part is not None and codec in {"v6_14_plus1_bg0", "v6_14_zero_based_invalid_minus1"}:
            if part.ndim != 2:
                raise ValueError("part requires integer scalar label PNG")
            labels = part.astype(np.int64) - (1 if codec == "v6_14_plus1_bg0" else 0)
            if np.any(labels < -1) or np.any(labels >= 14):
                raise ValueError("part codec labels outside v6 14 classes")
            labels = _reduce(transform_map(labels, transform, nearest=True, fill=-1), hw, labels=True).long()
            labels[support <= 0] = -1
            spatial["part"][0, slot] = labels
            valid["part"][0, slot] = True
    return spatial, valid, provenance


def prepare_task_condition(condition, cached_condition, *, project_root, config: ProAdapterConfig,
                           raw_source_features=None, assets=None):
    """单条锁定记录 -> B=1 TaskCondition。效果 profile 必须传 raw1600。

    assets 字典路径均只读；任何 target_rgb 字段都忽略。cached_condition 必须是
    Union conditions namespace payload，且包含原 reference 顺序、grid、transform。
    """
    record, order, hw, source_latent = validate_cached_binding(condition, cached_condition)
    assets = resolve_condition_assets(record, project_root) if assets is None else assets
    validate_source_assets(record, assets, project_root)
    masks = load_source_masks(assets, record["source_transform"])
    if len(masks) != len(order):
        raise ValueError("source masks must be in locked slot order")
    person_valid = torch.tensor([[bool(mask.any()) for mask in masks]])
    spatial, spatial_valid, geometry_provenance = _load_spatial(record, order, assets, hw)
    person_mask = spatial["mask"].flatten(2)
    owner = torch.minimum(person_mask, spatial["visibility"].flatten(2))
    owner /= owner.sum(1, keepdim=True).clamp(min=1)
    owner *= person_valid[..., None]
    roi = torch.stack([_roi_from_keypoints(path, record["dataset"], record["target_transform"], hw,
                                          person_mask[0, i].reshape(hw))
                       for i, path in enumerate(assets.get("target_keypoints", [None] * len(order)))])[None]
    source_assets = source_asset_roles(assets, record)
    if raw_source_features is None:
        if config.source_dim != 64:
            raise ValueError("source_dim1600 requires frozen raw source features; no VAE-only fallback")
        tokens = source_latent[:, None].repeat(1, len(order), 1, 1)
        token_valid = torch.stack([_reduce(mask, hw)[0].flatten() > 0 for mask in masks])[None]
        identity = torch.zeros(1, len(order), config.identity_dim)
        identity_valid = torch.zeros(1, len(order), dtype=torch.bool)
        fingerprint = source_feature_fingerprint(source_assets=source_assets,
                    model_signature={"profile": "locked_B_source_vae64", "source_vae_producer": cached_condition.producer_signature}, person_order=order,
                    source_transform=record["source_transform"], source_latent=source_latent)
        source_provenance = {"profile": "vae64_baseline", "asset_hashes": asset_hashes(source_assets),
                             "source_vae_producer": cached_condition.producer_signature,
                             "prepared_source_sha256": cached_condition.attestation["prepared_source_sha256"],
                             "source_latent_sha256": cached_condition.attestation["source_latent_sha256"]}
    else:
        raw_source_features.validate()
        if raw_source_features.person_ids != order or raw_source_features.tokens.shape[-1] != config.source_dim:
            raise ValueError("raw source feature person order/config mismatch")
        # A pack from another source/crop/model must never be accepted by shape alone.
        raw_source_features.validate_binding(record, source_latent, assets, cached_condition.producer_signature)
        tokens, token_valid = raw_source_features.tokens, raw_source_features.valid
        identity, identity_valid = raw_source_features.face_identity, raw_source_features.face_identity_valid
        fingerprint, source_provenance = raw_source_features.fingerprint, raw_source_features.provenance
    token_valid = token_valid & person_valid[..., None]
    identity_valid = identity_valid & person_valid
    for name in spatial_valid:
        spatial_valid[name] &= person_valid
    relations, relation_metadata = build_contact_relations(assets, order, person_valid)
    availability = {name: {"valid": bool(value.all()), "reason": "available" if value.all() else "missing_or_invalid_person"}
                    for name, value in spatial_valid.items()}
    availability["faceid"] = {"valid": bool(identity_valid.all()), "reason": "source_face_quality_gate"}
    binding_keys = ("sample_id", "condition_fingerprint", "payload_sha256", "prepared_source_sha256", "source_latent_sha256")
    provenance = {"source": source_provenance,
                  "cache_binding": {key: cached_condition.attestation[key] for key in binding_keys},
                  "geometry": {**geometry_provenance, "relations": relation_metadata,
                   "asset_hashes": asset_hashes(geometry_asset_roles(assets)), "target_transform": record["target_transform"]},
                  "availability": availability}
    result = TaskCondition(hw, order, spatial, spatial_valid, person_mask, owner, roi[:, :, 0].flatten(2),
                           roi[:, :, 1:].flatten(3), person_valid, tokens, token_valid, identity, identity_valid,
                           relations, fingerprint, provenance)
    return TaskCondition.from_dict(ordinary_cpu(result.to_dict())).validate(config)


def build_contact_relations(assets, person_order, person_valid):
    """已标记 raw mesh 顶点对按距离去重 parts；缺任一 codec 输入返回 E0+invalid。"""
    metadata = {"valid": False, "reason": "missing_mesh_or_vertex_parts", "codec": GEOMETRY_CODEC["relative"]}
    labels = assets.get("contact_vertex_parts")
    if not _exists(assets.get("contact_mesh")) or labels is None or len(person_order) != 2:
        return _empty_relations(), metadata
    if not isinstance(labels, np.ndarray):
        if not _exists(labels):
            return _empty_relations(), metadata
        with np.load(labels, allow_pickle=False) as data:
            labels = data["parts"].copy()
    with np.load(assets["contact_mesh"], allow_pickle=False) as data:
        vertices, flags = data["verts"].copy(), data["contact"].copy()
    if vertices.ndim != 3 or vertices.shape[0] != 2 or vertices.shape[-1] != 3:
        raise ValueError("Hi4D verts must be [2,V,3]")
    if flags.shape != vertices.shape[:2] or labels.shape != (vertices.shape[1],):
        raise ValueError("contact flags / vertex part topology mismatch")
    if (not np.isfinite(vertices).all() or not np.isfinite(flags).all() or np.any(flags < 0)
            or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= 14)):
        raise ValueError("invalid raw mesh contact or v6 vertex part codec")
    if not _exists(assets.get("camera")):
        metadata["reason"] = "camera_extrinsics_unavailable"
        return _empty_relations(), metadata
    with np.load(assets["camera"], allow_pickle=False) as cameras:
        indices = np.nonzero(cameras["ids"] == assets.get("camera_id"))[0]
        if len(indices) != 1:
            raise ValueError("camera_id not uniquely present in Hi4D cameras")
        index = int(indices[0])
        extrinsic = cameras["extrinsics"][index]
        metadata["dist_coeffs"] = cameras["dist_coeffs"][index].tolist() if "dist_coeffs" in cameras else None
    if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("Hi4D extrinsics must be finite3x4")
    cam_vertices = vertices @ extrinsic[:, :3].T + extrinsic[:, 3]
    extent = float(np.ptp(vertices, axis=1).max(axis=1).mean())
    if extent <= 1e-8:
        raise ValueError("degenerate mesh has no normalization extent")
    metadata.update(valid=True, reason="mesh_contact_annotations", flag_threshold=0.,
                    raw_contact_codec="finite_nonnegative_annotation_membership_gt0",
                    confidence_codec="annotated_pair=1;numeric_annotation_is_not_probability",
                    normalization="mean_of_two_world_mesh_max_axis_extents", extent_m=extent,
                    endpoints_visibility_available=False)
    if not bool(person_valid.all()):
        metadata.update(valid=False, reason="invalid_source_person")
        return _empty_relations(), metadata
    # The official raw contact array contains positive annotation values (e.g.
    # talk09:50..6335), not probabilities. Do not infer an unverified index codec.
    ia, ib = np.nonzero(flags[0] > 0)[0], np.nonzero(flags[1] > 0)[0]
    metadata["flagged_vertices"] = [len(ia), len(ib)]
    if not len(ia) or not len(ib):
        metadata["reason"] = "valid_zero_contact"
        return _empty_relations(), metadata
    from scipy.spatial import cKDTree
    distances, nearest = cKDTree(cam_vertices[1, ib]).query(cam_vertices[0, ia], k=1)
    candidates = sorted(zip(distances.tolist(), ia.tolist(), ib[nearest].tolist()))
    chosen, seen = [], set()
    for distance, a, b in candidates:
        pair = (int(labels[a]), int(labels[b]))
        if pair not in seen:
            chosen.append((distance, a, b, *pair))
            seen.add(pair)
        if len(chosen) == 8:
            break
    owner = intrinsics = None
    if _exists(assets.get("geometry")):
        with np.load(assets["geometry"], allow_pickle=False) as geometry:
            if "front_owner" in geometry and "K_runtime" in geometry:
                owner, intrinsics = geometry["front_owner"].copy(), geometry["K_runtime"].copy()
                metadata["endpoints_visibility_available"] = True
    def visible(raw, vertex):
        if owner is None:
            return 0.
        point = cam_vertices[raw, vertex]
        if point[2] <= 0:
            return 0.
        projected = intrinsics @ point
        x, y = np.rint(projected[:2] / projected[2]).astype(int)
        return float(0 <= x < owner.shape[1] and 0 <= y < owner.shape[0] and owner[y, x] == raw + 1)
    src_slot, dst_slot = person_order.index(0), person_order.index(1)
    n = len(chosen)
    relative, confidence = [], []
    for distance, a, b, _, _ in chosen:
        delta = (cam_vertices[1, b] - cam_vertices[0, a]) / extent
        relative.append([*delta, distance / extent, visible(0, a), visible(1, b)])
        confidence.append(1.)  # Both endpoints have annotations; numeric values carry no calibrated confidence.
    relations = ContactRelations(torch.full((1, n), src_slot, dtype=torch.long),
             torch.tensor([[row[3] for row in chosen]], dtype=torch.long),
             torch.full((1, n), dst_slot, dtype=torch.long),
             torch.tensor([[row[4] for row in chosen]], dtype=torch.long),
             torch.tensor([relative], dtype=torch.float32), torch.tensor([confidence], dtype=torch.float32),
             torch.ones(1, n, dtype=torch.bool))
    return relations, metadata


def prepare_source_task_condition(raw_source_features, *, config, target_hw=(32,32)):
    """新记录没有旧 B cache 时的 source-only typed 起点；几何保持显式 unavailable。

    后续必须经 geometry_task_condition/load_geometry_task_condition 装配当前几何，
    再 EncodedCondition.with_source_binding；不能从 GT/driver 外观构造身份 bank。
    """
    raw_source_features.validate()
    order=raw_source_features.person_ids; p=len(order); h,w=target_hw; n=h*w
    if raw_source_features.tokens.shape[-1]!=config.source_dim:
        raise ValueError("Raw source feature width differs from task config")
    person_valid=raw_source_features.valid.any(-1)
    spatial={name:torch.full((1,p,channels,h,w),-1 if name=="part" else 0,
        dtype=torch.long if name=="part" else torch.float32) for name,channels in
        (("normal",3),("depth",1),("mask",1),("visibility",1),("contact",1),("part",1))}
    zero=torch.zeros(1,p,n)
    return TaskCondition(tuple(target_hw),order,spatial,{name:torch.zeros(1,p,dtype=torch.bool) for name in spatial},
        zero,zero.clone(),zero.clone(),torch.zeros(1,p,2,n),person_valid,raw_source_features.tokens,
        raw_source_features.valid&person_valid[...,None],raw_source_features.face_identity,
        raw_source_features.face_identity_valid&person_valid,_empty_relations(),raw_source_features.fingerprint,
        {"source":raw_source_features.provenance,"availability":{name:False for name in spatial}}).validate(config)
