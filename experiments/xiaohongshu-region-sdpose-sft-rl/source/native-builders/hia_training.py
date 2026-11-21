#!/usr/bin/env python3
"""HIA 原发布 -> Native V5 离线数据接口；本工具从不创建 optimizer 或启动训练。"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

def repository_root(start):
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise ValueError("Cannot locate repository root containing src/ and scripts/")


ROOT = repository_root(__file__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "hia_native_v5_preparation_v1"
RELEASE = "/data/shangguanrz/pic-edit-datasets/HIA_processed_v2_full_20261027/release"
TIKTOK = "/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1"
SPLITS = ("train", "val", "test")
MASK_SCOPE = "regional_auxiliary_supervision"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


LOADED_SOURCE_SHA = sha(__file__)


def ref(path, schema=None):
    result = {"path": str(Path(path).resolve()), "sha256": sha(path)}
    if schema:
        result["schema"] = schema
    return result


def verified(value):
    path = Path(value["path"])
    if not path.is_absolute() or sha(path) != value["sha256"]:
        raise ValueError("Asset path/SHA256 differs: " + str(path))
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def jsonl(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


def write_jsonl(path, values):
    with Path(path).open("x", encoding="utf8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    return ref(path)


def regional_policy(frame):
    """HIA a/b 是原 DWPose 身体右/左腕；Native 手轴是解剖学 left/right。"""
    blocks = frame.get("hand_block_slices", {})
    if (set(blocks) != {"hand_a", "hand_b"} or sorted(blocks.values()) != [[92, 113], [113, 134]] or
            frame.get("hand_body_wrist_indices") != {"hand_a": 4, "hand_b": 7}):
        raise ValueError("Explicit HIA hand anatomical order must match wrists4/right and7/left")
    mask = frame.get("loss_mask", {})
    if set(mask) != {"body", "face", "hand_a", "hand_b"} or any(type(value) is not bool for value in mask.values()):
        raise ValueError("Explicit HIA regional boolean loss_mask required")
    if frame.get("loss_mask_scope", MASK_SCOPE) != MASK_SCOPE:
        raise ValueError("HIA loss_mask scope must remain regional_auxiliary_supervision")
    return {"face": mask["face"], "left": mask["hand_b"], "right": mask["hand_a"]}


def control_policy(frame):
    """可见清晰不等于原手部 Pose 可信；沿用原发布的独立原始点门禁。"""
    policy = regional_policy(frame)
    raw = frame.get("hand_raw_pose_usable")
    if raw is not None:
        if set(raw) != {"hand_a", "hand_b"} or any(type(v) is not bool for v in raw.values()):
            raise ValueError("Explicit raw hand pose validity required")
        policy["left"] &= raw["hand_b"]
        policy["right"] &= raw["hand_a"]
    return policy


def validate_frame_groups(frames):
    clips, groups, ids = {}, {}, set()
    for frame in frames:
        split = frame["split"]
        if split not in SPLITS or frame["frame_id"] in ids:
            raise ValueError("Unique frame IDs and original train/val/test labels required")
        ids.add(frame["frame_id"])
        regional_policy(frame)
        for table, key in ((clips, frame["clip_id"]), (groups, frame["group_id"])):
            if key in table and table[key] != split:
                raise ValueError("Original-video/duplicate-group split leakage: " + key)
            table[key] = split


def mask_regional_weights(weights, frame):
    policy = control_policy(frame)
    result = weights.clone()
    if not policy["face"]:
        result[:, 0] = 0
    for side, name in enumerate(("left", "right")):
        if not policy[name]:
            result[:, 2 + side] = 0
    return result


def apply_regional_validity(local, frame):
    """在序列化前关闭真实 Native gate，不能只把 HIA flags 附在 JSON 后。"""
    policy = control_policy(frame)
    names = ("face_roi_valid", "face_point_valid", "jaw_valid", "expression_valid",
             "hand_pose_valid", "hand_point_valid", "mesh_valid", "hand_roi_valid",
             "region_masks", "visibility", "local_z", "local_z_valid", "part_visible")
    changed = {name: getattr(local, name).clone() for name in names}
    if not policy["face"]:
        for name in names[:4]:
            changed[name][:, 0] = False
        for name in names[8:]:
            changed[name][:, 0] = 0
    for side, label in enumerate(("left", "right")):
        if policy[label]:
            continue
        for name in names[4:8]:
            changed[name][:, 0, side] = False
        for name in names[8:]:
            changed[name][:, 2 + side] = 0
    changed["metadata"] = tuple({**m, "hia_regional_policy": policy, "loss_mask": frame["loss_mask"],
                                  "hand_raw_pose_usable": frame.get("hand_raw_pose_usable"),
                                  "disabled_hand_reason": "loss_mask_false_or_hand_raw_pose_unreliable",
                                  "loss_mask_scope": MASK_SCOPE} for m in local.metadata)
    return replace(local, **changed).validate()


def mask_guide_points(points, frame):
    import numpy as np
    result = np.asarray(points, dtype=np.float32).copy()
    if result.shape != (134, 2):
        raise ValueError("Original camera DWPose134 xy required")
    policy = control_policy(frame)
    valid = frame.get("valid_keypoint_mask")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (134,):
            raise ValueError("HIA original valid_keypoint_mask requires134 points")
        # Raw detector blocks can flip (209/22276 release frames). Geometry has
        # fixed anatomical left/right order; move validity through the wrist map.
        canonical = valid.copy()
        for start, raw_name in ((92, "hand_b"), (113, "hand_a")):
            low, high = frame["hand_block_slices"][raw_name]
            canonical[start:start + 21] = valid[low:high]
        result[~canonical] = np.nan
    if not policy["face"]:
        result[24:92] = np.nan
    for side, label in enumerate(("left", "right")):
        if not policy[label]:
            result[92 + side * 21:113 + side * 21] = np.nan
            result[7 if side == 0 else 4] = np.nan
    return result


def canvas_for(hw, long_side):
    from src.data.canvas_contract import CanvasSpec, native_pad
    if long_side == 0:
        return native_pad(hw)
    if long_side < 32 or long_side % 32:
        raise ValueError("canvas-long-side must be0 (native) or a positive multiple of32")
    scale = min(1., long_side / max(hw))
    resized = tuple(max(1, round(value * scale)) for value in hw)
    canvas = tuple(((value + 31) // 32) * 32 for value in resized)
    offset = ((canvas[1] - resized[1]) // 2, (canvas[0] - resized[0]) // 2)
    return CanvasSpec(tuple(hw), resized, canvas, offset, policy="aspect_preserving")


class BoundedRenderer:
    """只缓存最近的渲染结果，硬限制CPU数组总字节数；不累计全量原生图。"""
    def __init__(self, delegate, max_bytes=64 * 1024 * 1024):
        self.delegate, self.max_bytes = delegate, max_bytes
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def render(self, person, camera, hw):
        import numpy as np
        digest = hashlib.sha256()
        for value in (person["mesh_cam"], person["faces"], person["face_part"], camera):
            value = np.ascontiguousarray(value)
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        digest.update(str(tuple(hw)).encode())
        key = digest.hexdigest()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        result = self.delegate.render(person, camera, hw)
        size = sum(value.nbytes for value in result.values())
        if size <= self.max_bytes:
            while self.cache and self.bytes + size > self.max_bytes:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.bytes -= old_size
            self.cache[key] = (result, size)
            self.bytes += size
        return result


class Importer:
    """复用一个原模型，仍由现有 import_smplx 执行全 mesh FK 校验。"""
    def __init__(self, model_file, directory, producer):
        import numpy as np
        import smplx
        from src.pose_control.qwen21_full.dual_hi4d_geometry import SMPL_TO_PART
        from src.pose_control.qwen21_full.conditions import PART_NAMES
        from smplx.joint_names import JOINT_NAMES
        self.factory = smplx.SMPLX
        self.producer = producer
        self.options = dict(use_pca=False, num_pca_comps=6, flat_hand_mean=False,
                            num_expression_coeffs=10, gender="neutral", use_face_contour=True)
        self.model = self.factory(str(model_file), batch_size=1, **self.options).float().eval().requires_grad_(False)
        self.model_file, self.model_sha = str(model_file), sha(model_file)
        self.basis_sha = hashlib.sha256(self.model.expr_dirs.numpy().astype("<f4").tobytes()).hexdigest()
        mapping = np.asarray([*SMPL_TO_PART[:22], 0, 0, 0, *([4] * 15), *([7] * 15)])
        lbs = self.model.lbs_weights.numpy()
        part_weights = np.stack([lbs[:, mapping == part].sum(1) for part in range(14)], 1)
        vertex_parts = part_weights.argmax(1)
        self.segmentation = Path(directory) / self.model_sha / "smplx_part14.npz"
        self.segmentation.parent.mkdir(parents=True, exist_ok=True)
        if not self.segmentation.exists():
            with self.segmentation.open("xb") as stream:
                np.savez_compressed(stream, face_part14=part_weights[self.model.faces].sum(1).argmax(1),
                                    left_hand_vertices=np.flatnonzero(vertex_parts == 4),
                                    right_hand_vertices=np.flatnonzero(vertex_parts == 7))
        self.parts = list(PART_NAMES)
        self.hand_ids = []
        for side in ("left", "right"):
            names = [side + "_wrist"]
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                names += [side + "_" + finger + str(i) for i in (1, 2, 3)] + [side + "_" + finger]
            self.hand_ids.append([JOINT_NAMES.index(name) for name in names])
        tree = ast.parse((ROOT / "add/tools/SMPLest-X/human_models/human_models.py").read_text())
        maps = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "joint_idx" for t in node.targets)
                and isinstance(node.value, (ast.Tuple, ast.List))]
        maps = [m for m in maps if len(m) == 137]
        if len(maps) != 1:
            raise ValueError("One verified original SMPLest-X137 model joint mapping required")
        self.joint_map = np.asarray(maps[0])

    def convert(self, raw_ref, record, frame, directory):
        import numpy as np
        import smplx
        import torch
        from src.pose_control.qwen21_full.native_full_assets import import_smplx
        raw = torch.load(verified(raw_ref), map_location="cpu", weights_only=False)
        keys = ("betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "expression", "transl")
        params = {key: np.asarray(raw[key], dtype=np.float32) for key in keys}
        for key in ("leye_pose", "reye_pose"):
            params[key] = np.zeros((1, 3), np.float32)
        k = np.eye(3, dtype=np.float32)
        k[0, 0], k[1, 1] = np.asarray(raw["focal_length_xy"]).reshape(-1).tolist()
        k[:2, 2] = np.asarray(raw["princpt"]).reshape(2)
        params.update(mesh_cam=np.asarray(raw["mesh_cam"], dtype=np.float32), K=k,
                      camera_R=np.eye(3, dtype=np.float32), camera_t=np.zeros(3, np.float32))
        path = directory / "original_parameters.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **params)
        options = dict(self.options)
        job = {"parameters": str(path), "people": [{"person_id": "hia:" + frame["clip_id"] + ":0",
               "model_file": self.model_file, "model_sha256": self.model_sha, "model_options": options,
               "segmentation": str(self.segmentation), "expression_basis_sha256": self.basis_sha,
               "hand_joint_indices": self.hand_ids, "part_names": self.parts}],
               "original_hw": frame["original_hw"], "camera_id": "hia:" + frame["clip_id"] + ":camera",
               "scale_group_id": "hia:" + frame["clip_id"] + ":model",
               "provenance": {"rgb_sha256": frame["rgb_sha256"], "original_parameters": raw_ref,
                              "geometry_record_sha256": signature(record), "producer": self.producer,
                              "eye_pose_source": "SMPLest-X model default zero; whole-mesh FK verified"}}
        # import_smplx validates complete model conventions and every reconstructed vertex.
        original = smplx.SMPLX
        def same_model(path, batch_size=1, **kwargs):
            if str(path) != self.model_file or batch_size != 1 or kwargs != options:
                raise ValueError("Unexpected SMPL-X model factory request")
            return self.model
        smplx.SMPLX = same_model
        try:
            asset = import_smplx(job)
        finally:
            smplx.SMPLX = original
        with torch.no_grad():
            joints = self.model(**{key: torch.from_numpy(params[key]) for key in (*keys, "leye_pose", "reye_pose")}).joints[0].numpy()
        selected = joints[self.joint_map]
        face_order = [65, 66, 67, 68, *range(120, 137), *range(69, 120)]
        asset["people"][0]["face72_cam"] = selected[face_order].copy()
        asset["people"][0]["face72_valid"][:] = True
        asset["people"][0]["metadata"].update(face72_layout="smplx137_65_136_first4_extras_then_dw68",
                                                   face72_joint_indices_137=face_order)
        return asset, selected


def source_references(asset, rgb, spec, geometry_ref, frame, renderer):
    import numpy as np
    import torch
    from src.pose_control.qwen21_full.native_full_assets import local_geometry
    from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA
    from src.pose_control.qwen21_full.native_v5.contracts import NativeDetailReferencesV5, DETAIL_SCHEMA
    local = local_geometry(asset, spec, renderer=renderer)
    face_boxes = torch.zeros(1, 2, 4)
    hand_boxes = torch.zeros(1, 2, 2, 4)
    face_boxes[0, 0] = torch.from_numpy(local["boxes"][0, 0])
    hand_boxes[0, 0] = torch.from_numpy(local["boxes"][0, 1:])
    person_ids = [asset["people"][0]["metadata"]["person_id"], "__inactive__"]
    source = {"rgb": rgb, "canvas": spec.metadata(), "canvas_id": spec.fingerprint(),
              "geometry_sha256": geometry_ref["sha256"], "person_ids": person_ids}
    metadata = ({"schema": REFERENCE_SCHEMA, "role": "source_only", "source_rgb_sha256": rgb["sha256"],
                 "source_canvas_id": spec.fingerprint(), "source_person_ids": person_ids,
                 "reference_count": 1, "reference_sources": [source],
                 "availability": {"arcface": "unavailable", "face_dino": "unavailable", "hand_dino": "unavailable"},
                 "reason": "Geometry preparation does not extract learned appearance; all feature validity is false"},)
    references = NativeLocalReferencesV4(torch.zeros(1, 2, 1, 512), torch.zeros(1, 2, 1, dtype=torch.bool),
        torch.zeros(1, 2, 1, 256, 1536), torch.zeros(1, 2, 1, dtype=torch.bool),
        torch.zeros(1, 2, 2, 1, 256, 1536), torch.zeros(1, 2, 2, 1, dtype=torch.bool), face_boxes, hand_boxes, metadata).validate()
    boxes = torch.cat((face_boxes, hand_boxes.flatten(1, 2)), 1)
    support = torch.zeros(1, 6, *spec.canvas_hw, dtype=torch.bool)
    policy = control_policy(frame)
    surface = local["surfaces"][0]
    tolerance = 1e-4 * asset["people"][0]["body_height"]
    for index, label in enumerate(("face", "left", "right")):
        region = local["regions"][0][index]
        visible = region["valid"] & surface["valid"] & (np.abs(region["z"] - surface["z"]) <= tolerance)
        if policy[label]:
            support[0, (0, 2, 3)[index]] = torch.from_numpy(visible)
    contested = support.sum(1, keepdim=True) > 1
    support &= ~contested
    valid = support.flatten(2).any(-1) & (boxes[..., 2:] > boxes[..., :2]).all(-1)
    detail = NativeDetailReferencesV5(boxes, support, valid, torch.tensor([[0, -1]]),
        ({"schema": DETAIL_SCHEMA, "role": "primary_source_only", "source_canvas_id": spec.fingerprint(),
          "source_rgb_sha256": rgb["sha256"], "source_person_ids": person_ids,
          "loss_mask": frame["loss_mask"], "loss_mask_scope": MASK_SCOPE, "source_geometry_sha256": geometry_ref["sha256"]},)).validate()
    return references, detail


def prepare_frame(frame, record_path, args, importer, renderer):
    import torch
    from src.pose_control.qwen21_full.native_full_assets import write_record, read_record
    from src.pose_control.qwen21_full.native_local_condition import REFERENCE_SCHEMA
    from src.pose_control.qwen21_full.native_v5.data import write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import DETAIL_SCHEMA
    from src.pose_edit.geometry import smplx137_to_dwpose134, render_poseguide
    record = load_json(record_path)
    if record.get("schema") != "hia_geometry_frame_v1" or record.get("status") != "complete":
        raise ValueError("Complete hia_geometry_frame_v1 required")
    identity = {"schema": VERSION, "geometry_record": ref(record_path), "frame": frame,
                "canvas_long_side": args.canvas_long_side, "model_sha256": importer.model_sha, "producer_sha256": args.producer["sha256"]}
    key = signature(identity)
    directory = Path(args.output_root) / "frames" / frame["frame_id"] / key[:16]
    marker = directory / "ready.json"
    if marker.exists():
        existing = load_json(marker)
        if existing["identity_sha256"] != key:
            raise ValueError("Frame cache identity differs")
        verified(existing["rgb"])
        for item in existing["artifacts"].values():
            verified(item)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ("build_" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    rgb = {"path": frame["rgb_path"], "sha256": frame["rgb_sha256"]}
    verified(rgb)
    if record["input"]["image_sha256"] != rgb["sha256"]:
        raise ValueError("Geometry and release RGB fingerprint differ")
    if record["camera"]["model_sha256"] != importer.model_sha or record["camera"]["model_options"] != importer.options:
        raise ValueError("Geometry model identity/options differ from the native importer")
    if record["supervision"]["loss_mask"] != frame["loss_mask"]:
        raise ValueError("Geometry and immutable release regional supervision differ")
    asset, joints137 = importer.convert(record["artifacts"]["smplx_params"], record, frame, attempt)
    asset_ref = write_record(attempt / "asset.npz", asset, schema="native_smplx_asset_v4")
    spec = canvas_for(frame["original_hw"], args.canvas_long_side)
    references, detail = source_references(asset, rgb, spec, asset_ref, frame, renderer)
    refs = write_record(attempt / "source_references.npz", references, schema=REFERENCE_SCHEMA)
    detail_ref = write_sidecar(attempt / "source_detail.npz", detail, DETAIL_SCHEMA)
    uv = joints137 @ asset["K"].T
    uv = uv[:, :2] / uv[:, 2:]
    points = spec.pixel_points(mask_guide_points(smplx137_to_dwpose134(uv), frame))
    guide = attempt / "pose_guide.png"
    render_poseguide(points, (spec.canvas_hw[1], spec.canvas_hw[0])).save(guide)
    mask = record["artifacts"]["smplx_mask"]
    verified(mask)
    result = {"schema": "hia_native_frame_v1", "identity_sha256": key, "frame_id": frame["frame_id"],
              "clip_id": frame["clip_id"], "split": frame["split"], "group_id": frame["group_id"],
              "rgb": rgb, "canvas": spec.metadata(), "canvas_long_side": args.canvas_long_side,
              "loss_mask": frame["loss_mask"], "loss_mask_scope": MASK_SCOPE,
              "hand_control_policy": control_policy(frame), "hand_raw_pose_usable": frame.get("hand_raw_pose_usable"),
              "artifacts": {"asset": asset_ref, "references": refs, "detail": detail_ref,
                            "guide": ref(guide), "source_mask": mask}}
    write_json(marker, result)
    return result


def prepare_pair(pair, source, target, source_frame, target_frame, args, renderer):
    import torch
    from src.pose_control.qwen21_full.native_full_assets import read_record, write_record
    from src.pose_control.qwen21_full.native_full_data import assemble_full_cache, BODY_SCHEMA, PAIR_SCHEMA
    from src.pose_control.qwen21_full.native_v5.data import training_sidecar, write_sidecar
    from src.pose_control.qwen21_full.native_v5.contracts import SUPERVISION_SCHEMA
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA
    if pair["split"] != source["split"] or pair["split"] != target["split"] or source["clip_id"] != target["clip_id"]:
        raise ValueError("Pair must preserve the original video/split")
    if pair["source_loss_mask"] != source["loss_mask"] or pair["target_loss_mask"] != target["loss_mask"]:
        raise ValueError("Pair loss_mask differs from immutable frame release")
    identity = signature({"pair": pair, "source": source["identity_sha256"], "target": target["identity_sha256"], "producer_sha256": args.producer["sha256"]})
    directory = Path(args.output_root) / "pairs" / pair["pair_id"] / identity[:16]
    marker = directory / "ready.json"
    if marker.exists():
        existing = load_json(marker)
        if existing["identity_sha256"] != identity:
            raise ValueError("Pair preparation identity differs")
        for value in existing["artifacts"].values():
            verified(value)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ("build_" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    job = {"pair_id": pair["pair_id"], "source_asset": source["artifacts"]["asset"],
           "driver_asset": target["artifacts"]["asset"], "source_canvas": source["canvas"], "target_canvas": target["canvas"],
           "source_masks": [source["artifacts"]["source_mask"]], "source_rgb": source["rgb"], "target_rgb": target["rgb"],
           "source_indices": [0, -1], "source_slot_ids": [0, 1], "units": "model_unit", "scale_status": "model_scale",
           "references": source["artifacts"]["references"], "expression_mode": "driver", "distances": "absent_single_person"}
    sample, local = assemble_full_cache(job, renderer=renderer)
    local = apply_regional_validity(local, target_frame)
    sample["pair_metadata"].update(dataset_type="hia_native_full_smplx_v4", original_split=pair["split"],
        original_clip_id=pair["clip_id"], original_group_id=target_frame["group_id"], loss_mask_scope=MASK_SCOPE,
        target_loss_mask=pair["target_loss_mask"], source_loss_mask=pair["source_loss_mask"], canvas_long_side=args.canvas_long_side)
    body = write_record(attempt / "body.npz", sample, schema=BODY_SCHEMA)
    local_ref = write_record(attempt / "local.npz", local, schema="native_local_condition_v4")
    v4 = {"schema": PAIR_SCHEMA, "pair_id": pair["pair_id"], "scene_kind": "single", "data_profile": "full_smplx",
          "body": body, "local": local_ref, "references": source["artifacts"]["references"], "source_rgb": source["rgb"], "target_rgb": target["rgb"]}
    v4_ref = write_jsonl(attempt / "v4.jsonl", [v4])
    references = read_record(source["artifacts"]["references"], schema="native_local_reference_v4")
    supervised = {**sample, "local_condition_v4": local, "local_references_v4": references}
    supervision = training_sidecar(supervised, {"availability": "unavailable", "reason": "No target identity/perceptual evaluator qualification requested"})
    supervision = replace(supervision, weights=mask_regional_weights(supervision.weights, target_frame),
        metadata=tuple({**m, "loss_mask": pair["target_loss_mask"], "loss_mask_scope": MASK_SCOPE,
                        "global_velocity_scope": "all_valid_image_pixels"} for m in supervision.metadata)).validate()
    supervision_ref = write_sidecar(attempt / "supervision.npz", supervision, SUPERVISION_SCHEMA)
    from src.data.canvas_contract import CanvasSpec
    source_spec, target_spec = CanvasSpec.from_record(source["canvas"]), CanvasSpec.from_record(target["canvas"])
    prompt = "Change the person's pose to match the pose in <image2>. Preserve the identity, face, clothing and background of the person in <image1>."
    prompt_bank = write_json(attempt / "prompt_bank.json", {"schema": "hia_native_prompt_v1", "pair_id": pair["pair_id"],
        "compiled_prompt_inference": prompt, "compiled_prompt_inference_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "producer": "deterministic_source_and_pose_instruction", "source_release_pair": pair})
    condition = {"schema": SCHEMA, "profile": "single_2ref", "pair_id": pair["pair_id"], "primary_source_index": 0,
        "target_canvas_id": target_spec.fingerprint(), "images": [
            {"role": "source", "file": source["rgb"], "canvas": source["canvas"], "canvas_id": source_spec.fingerprint(), "space": "original"},
            {"role": "target_pose", "file": target["artifacts"]["guide"], "canvas": target["canvas"], "canvas_id": target_spec.fingerprint(), "space": "canvas"}],
        "prompt": {"text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest(), "field": "compiled_prompt_inference",
                   "bank": prompt_bank, "schema": "hia_native_prompt_v1", "pair_id": pair["pair_id"]}}
    condition_ref = write_json(attempt / "condition.json", condition)
    v5 = {"schema": "native_full_pair_v5", "pair_id": pair["pair_id"], "v4_manifest": v4_ref,
          "references": source["artifacts"]["references"], "detail": source["artifacts"]["detail"],
          "supervision": supervision_ref, "qwen_condition": condition_ref}
    result = {"identity_sha256": identity, "row": v5, "group_id": "hia:" + target_frame["group_id"],
              "split": pair["split"], "loss_mask_scope": MASK_SCOPE, "canvas_long_side": args.canvas_long_side,
              "artifacts": {"body": body, "local": local_ref, "v4_manifest": v4_ref, "supervision": supervision_ref,
                            "condition": condition_ref, "prompt_bank": prompt_bank}}
    write_json(marker, result)
    return result


def geometry_index(root):
    root = Path(root)
    base = root / "output/frames" if (root / "output/frames").is_dir() else root / "frames"
    current = load_json(root / "pipeline_signature.json")["sha256"]
    result = {}
    for path in base.glob("*/*/record.json"):
        record = load_json(path)
        if record.get("status") == "complete" and record.get("pipeline_sha256") == current:
            key = record.get("frame_id", record.get("input", {}).get("frame_id"))
            if key in result:
                raise ValueError("Multiple geometry records for frame " + str(key))
            result[key] = path
    return result


def prepare(args):
    import torch
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, SourceSpec, stage_preset
    torch.set_num_threads(4)
    output, release = Path(args.output_root).resolve(), Path(args.release).resolve()
    if not str(output).startswith("/data/shangguanrz/"):
        raise ValueError("New cache artifacts must be on /data/shangguanrz/")
    if sha(__file__) != LOADED_SOURCE_SHA:
        raise RuntimeError("Preparation source changed after import; restart with frozen code")
    producer = output / "source/asset_producers" / (LOADED_SOURCE_SHA + ".py")
    producer.parent.mkdir(parents=True, exist_ok=True)
    if not producer.exists():
        with producer.open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
    args.producer = ref(producer)
    if args.producer["sha256"] != LOADED_SOURCE_SHA:
        raise ValueError("Immutable preparation producer snapshot SHA differs")
    from hia_status import inspect_geometry
    coverage = inspect_geometry(release, args.geometry_root, verify_hashes=True)
    coverage_path = output / "reports" / ("geometry_verification_" + uuid.uuid4().hex + ".json")
    coverage_ref = write_json(coverage_path, coverage)
    if coverage["errors"]:
        raise ValueError("Geometry provenance/hash verification failed: " + str(coverage_path))
    frames = {}
    for split in SPLITS:
        for item in jsonl(release / "manifests" / (split + ".jsonl")):
            if item["split"] != split:
                raise ValueError("Release filename/split assertion differs")
            if item["frame_id"] in frames:
                raise ValueError("Duplicate frame ID across original release manifests")
            frames[item["frame_id"]] = item
    validate_frame_groups(frames.values())
    index = geometry_index(args.geometry_root)
    importer = Importer(Path(args.smplx_model).resolve(), output / "model_metadata", args.producer)
    renderer = BoundedRenderer(IndependentMeshRasterizer(args.device))
    built = {split: [] for split in SPLITS}
    missing = {split: 0 for split in SPLITS}
    frame_cache = {}
    for split in args.splits:
        for pair in jsonl(release / ("pairs_" + split + ".jsonl")):
            if args.limit_per_split and len(built[split]) >= args.limit_per_split:
                break
            ids = (pair["source_frame_id"], pair["target_frame_id"])
            if any(key not in index for key in ids):
                missing[split] += 1
                continue
            for key in ids:
                if key not in frames:
                    raise ValueError("Pair references a frame outside release")
                if key not in frame_cache:
                    frame_cache[key] = prepare_frame(frames[key], index[key], args, importer, renderer)
            built[split].append(prepare_pair(pair, frame_cache[ids[0]], frame_cache[ids[1]], frames[ids[0]], frames[ids[1]], args, renderer))
            if sum(map(len, built.values())) % 25 == 0:
                print(json.dumps({"event": "pair_progress", "counts": {key: len(value) for key, value in built.items()}, "missing_geometry": missing}), flush=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / "manifests" / tag
    destination.mkdir(parents=True)
    sources, manifests = [], {}
    for split, rows in built.items():
        if not rows:
            continue
        manifest = write_jsonl(destination / (split + ".jsonl"), [r["row"] for r in rows])
        groups = write_json(destination / (split + "_groups.json"), {r["row"]["pair_id"]: r["group_id"] for r in rows})
        manifests[split] = {"manifest": manifest, "groups": groups, "pairs": len(rows)}
        if split != "test":
            sources.append(SourceSpec("hia_" + split, role="train" if split == "train" else "validation",
                manifest=manifest, groups=groups, scene_kind="single", data_profile="full_smplx", weight=1.))
    cfg = TrainingConfig(stage=stage_preset("S0", budget=args.steps), sources=tuple(sources),
        initialization=ref(args.initialization), model_path=args.base_model, model_manifest=ref(args.model_manifest),
        cache_root=str(output / "frozen" / tag), output_root=str(output / "training" / tag),
        qwen_condition_profile="single_2ref", save_every_updates=1)
    config = write_json(destination / "S0_prepare.json", cfg.record())
    selected_complete = not any(missing.values()) and all(built[split] for split in args.splits)
    full_ready = selected_complete and coverage["geometry_complete"] and not args.limit_per_split and set(args.splits) == set(SPLITS)
    disabled_raw = sum(int(frame["loss_mask"][region] and not frame.get("hand_raw_pose_usable", {}).get(region, True))
                       for key, frame in frames.items() if key in frame_cache for region in ("hand_a", "hand_b"))
    result = {"schema": VERSION, "manifests": manifests, "config": config, "geometry_frames_available": len(index),
              "prepared_frames": len(frame_cache), "missing_geometry_pairs": missing, "limit_per_split": args.limit_per_split,
              "release_frames": len(frames), "geometry_complete": coverage["geometry_complete"],
              "geometry_verification": coverage_ref, "current_geometry_pipeline": load_json(Path(args.geometry_root) / "pipeline_signature.json")["sha256"],
              "producer": args.producer,
              "canvas_long_side": args.canvas_long_side, "loss_mask_scope": MASK_SCOPE,
              "full_release_ready": full_ready, "partial": not full_ready,
              "training_ready": False,
              "subset_only": bool(args.limit_per_split), "extra_disabled_raw_unreliable_hands": disabled_raw,
              "global_velocity_scope": "all_valid_image_pixels", "appearance_features": "explicit_unavailable_no_extraction",
              "optimizer_created": False, "training_started": False, "test_in_training_sources": False,
              "renderer_memo": {"hits": renderer.hits, "misses": renderer.misses, "max_bytes": renderer.max_bytes,
                                "resident_bytes": renderer.bytes, "entries": len(renderer.cache)},
              "next": {"data_preflight": "preflight --config " + config["path"] + " --report <new_report.json>",
                       "frozen_cache": "scripts/train/train_qwen21_native_full_v5.py prepare-cache --config " + config["path"] + " --cache-output " + cfg.cache_root}}
    report_path = Path(args.report) if args.report else output / "reports" / ("prepare_" + tag + ".json")
    result["report_path"] = str(report_path)
    write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def preflight(args):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    config = TrainingConfig.from_dict(load_json(args.config))
    result = NativeV5Trainer(config).preflight_data(args.report)
    print(json.dumps({"passed": result["passed"], "level": "data", "model_constructed": False,
                      "optimizer_created": False, "training_started": False, "report": str(args.report)}, ensure_ascii=False))
    if result.get("passed") is not True:
        raise RuntimeError("Native V5 data preflight did not pass; inspect " + str(args.report))
    return result


def audit_consumers(args):
    """读回真实 Native 数据与原 loss 函数，验证局部手门禁及全图 RGB 监督。"""
    import torch
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from src.pose_control.qwen21_full.native_v5.losses import LocalLossConfig, local_velocity_loss
    from src.pose_control.qwen21_full.native_v5.contracts import NativeTrainingSupervisionV5, SUPERVISION_SCHEMA
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, RouteSpec
    from src.pose_control.qwen21_full.native_v5.training.data import sample_status
    records = []
    for manifest in args.manifest:
        dataset = NativeFullDatasetV5(manifest, load_target=True)
        for index in range(len(dataset)):
            sample = dataset[index]
            local = sample["local_condition_v4"]
            supervision = sample["training_supervision_v5"]
            policy = local.metadata[0]["hia_regional_policy"]
            if not torch.equal(sample["valid_image_mask"], CanvasSpec.from_record(sample["target_canvas"]).valid_mask()):
                raise ValueError("Regional hand masks must never remove full-image RGB coverage")
            status = sample_status(sample, RouteSpec("audit_hand", "hand", 1.), TrainingConfig())
            disabled = []
            for side, name in enumerate(("left", "right")):
                if policy[name]:
                    continue
                region = 2 + side
                disabled.append(name)
                for field in ("hand_pose_valid", "hand_point_valid", "mesh_valid", "hand_roi_valid"):
                    if getattr(local, field)[0, 0, side].any():
                        raise ValueError("Invalid hand gate restarted: " + field)
                for field in ("region_masks", "visibility", "local_z", "local_z_valid", "part_visible"):
                    if getattr(local, field)[0, region].any():
                        raise ValueError("Invalid hand has live local geometry: " + field)
                if supervision.weights[0, region].any() or status["present"]["hand_geometry"][0][0][side]:
                    raise ValueError("Native regional supervision/hand route ignores disabled hand")
            records.append({"pair_id": sample["pair_metadata"]["pair_id"], "disabled_hands": disabled,
                            "global_coverage_preserved": True, "native_hand_geometry": status["present"]["hand_geometry"]})
    weights = torch.zeros(1, 6, 32, 32)
    weights[:, 2, :16, :16] = 1
    weights[:, 3, 16:, 16:] = 1
    frame = {"hand_body_wrist_indices": {"hand_a": 4, "hand_b": 7},
             "hand_block_slices": {"hand_a": [113, 134], "hand_b": [92, 113]},
             "loss_mask": {"body": True, "face": True, "hand_a": False, "hand_b": True}}
    weights = mask_regional_weights(weights, frame)
    metadata = ({"schema": SUPERVISION_SCHEMA, "role": "training_only", "driver_record_sha256": "d" * 64,
                 "driver_rgb_sha256": "r" * 64, "target_canvas_id": "synthetic_32px_consumer_test",
                 "target_person_ids": ["audit_person", "__inactive__"]},)
    truth = NativeTrainingSupervisionV5(weights, torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, 5, 2),
                                       torch.ones(1, dtype=torch.bool), metadata).validate()
    prediction = torch.zeros(1, 4, 64)
    prediction[:, 3] = 1
    total, loss = local_velocity_loss(prediction, torch.zeros_like(prediction), torch.ones(1, 4), truth,
                                    LocalLossConfig(profile="local_velocity"))
    if float(loss["velocity"]) != .25 or float(loss["hand_velocity"]) != 0. or float(total) != .25:
        raise ValueError("Actual Native velocity loss did not preserve the declared regional-mask semantics")
    result = {"schema": "hia_native_consumer_audit_v1", "passed": True, "records": records,
              "samples": len(records), "native_loss_probe": {"full_image_velocity": float(loss["velocity"]),
              "masked_hand_velocity": float(loss["hand_velocity"]), "total": float(total)},
              "loss_mask_scope": MASK_SCOPE, "model_constructed": False, "optimizer_created": False}
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare", help="Resume frame/pair caches and export a new immutable manifest/config snapshot")
    p.add_argument("--release", default=RELEASE)
    p.add_argument("--geometry-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--limit-per-split", type=int, default=0, help="Explicit smoke subset;0 exports every available original pair")
    p.add_argument("--canvas-long-side", type=int, default=512, help="Uniform aspect-preserving global canvas;0 is native_pad")
    p.add_argument("--smplx-model", default=str(ROOT / "add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"))
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--initialization", default=TIKTOK + "/initialization/native_v5_init_v2.pt")
    p.add_argument("--model-manifest", default=TIKTOK + "/source/qwen_model_files.json")
    p.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    p.add_argument("--steps", type=int, help="Record an explicit future update budget; absent remains unbound")
    p.add_argument("--report")
    q = commands.add_parser("preflight", help="Use actual Native V5 DataPlan; no model, cache encoding or optimizer")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--report", type=Path, required=True)
    a = commands.add_parser("audit-consumers", help="Read actual Native V5 samples and prove mask behavior through native loss")
    a.add_argument("--manifest", type=Path, action="append", required=True)
    a.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare" and (args.limit_per_split < 0 or (args.steps is not None and args.steps < 1)):
        parser.error("limit-per-split>=0 and optional steps>0 required")
    return {"prepare": prepare, "preflight": preflight, "audit-consumers": audit_consumers}[args.action](args)


if __name__ == "__main__":
    main()
