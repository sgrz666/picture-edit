"""单帧 Source-only 原始外观缓存；冻结的 ArcFace/DINO 按需加载并跨帧常驻。

这里只生成 NativeLocalReferencesV4，不缓存 Adapter 的可训练特征。
导入及构造无需 numpy、torch 或模型运行库，便于单独检查发布标记。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import struct


HAND_RELEASE_KEYS = ("hand_b", "hand_a")  # Native 的顺序：anatomical left, right
ARCFACE_PREPROCESSING = "source_5point_similarity; RGB112; (x-127.5)/127.5"
DINO_PREPROCESSING = "local_processor;224;no_center_crop"


def _coordinates(values, count):
    try:
        if len(values) != count or any(isinstance(v, (bool, str)) for v in values):
            return None
        result = tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def valid_five_points(points):
    """与 Native V5 相同的五点距离/二维秩门禁，不导入 torch。"""
    try:
        if len(points) != 5:
            return False
        values = [_coordinates(point, 2) for point in points]
    except TypeError:
        return False
    if any(point is None for point in values):
        return False
    if any(math.dist(values[i], values[j]) <= 1e-4 for i in range(5) for j in range(i)):
        return False
    mx, my = (sum(p[axis] for p in values) / 5 for axis in (0, 1))
    xx = sum((x - mx) ** 2 for x, y in values)
    yy = sum((y - my) ** 2 for x, y in values)
    xy = sum((x - mx) * (y - my) for x, y in values)
    largest = (xx + yy + math.hypot(xx - yy, 2 * xy)) / 2
    if largest <= 1e-8:
        return False
    smallest = max(0.0, xx * yy - xy * xy) / largest
    return smallest > largest * 1e-10


def source_five_points(frame, *, image_hw=None):
    """仅取原图检测器五点；不从 DWPose/SMPL-X 猜对齐点。"""
    detection = frame.get("detection") or {}
    face = detection.get("face") or {}
    points = face.get("landmarks5")
    if not valid_five_points(points):
        return None
    points = tuple(tuple(float(v) for v in point) for point in points)
    if image_hw is not None:
        height, width = image_hw
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            return None
    return points


def arcface_canvas_points(frame, canvas_spec):
    """原图像素五点通过该帧 CanvasSpec.pixel_points 映射一次。"""
    points = source_five_points(frame, image_hw=canvas_spec.original_hw)
    if points is None:
        return None
    import numpy as np
    mapped = canvas_spec.pixel_points(np.asarray(points, dtype=np.float32))
    if not valid_five_points(mapped):
        return None
    return mapped


def valid_canvas_roi(box):
    """允许跨边界的 unclipped ROI，排除零面积及完全落在 canvas 外的框。"""
    box = _coordinates(box, 4)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return x1 > x0 and y1 > y0 and x1 > 0 and y1 > 0 and x0 < 1 and y0 < 1


def source_appearance_policy(frame, face_box, hand_boxes):
    """只降低特征可用性，保留原 loss_mask/raw 标记且独立处理每只手。"""
    mask = frame.get("loss_mask", {})
    if set(mask) != {"body", "face", "hand_a", "hand_b"} or any(type(v) is not bool for v in mask.values()):
        raise ValueError("Explicit Bilibili regional boolean loss_mask required")
    raw = frame.get("hand_raw_pose_usable")
    if raw is not None and (set(raw) != {"hand_a", "hand_b"} or any(type(v) is not bool for v in raw.values())):
        raise ValueError("Explicit raw hand pose validity required")
    if len(hand_boxes) != 2:
        raise ValueError("Source hand boxes must be anatomical left, right")
    hands = tuple(mask[key] and (raw is None or raw[key]) and valid_canvas_roi(hand_boxes[side])
                  for side, key in enumerate(HAND_RELEASE_KEYS))
    return {"arcface": mask["face"] and source_five_points(frame, image_hw=frame.get("original_hw")) is not None,
            "face_dino": mask["face"] and valid_canvas_roi(face_box), "hand_dino": hands}


def _batch_limit(max_batch):
    if type(max_batch) is not int or not 1 <= max_batch <= 16:
        raise ValueError("DINO batch size must be an integer in [1,16]")


def dino_cache_key(frame_id, box):
    """缓存坐标使用 Native 框的精确 FP32 值，不取整或按像素近似合并。"""
    coordinates = _coordinates(box, 4)
    if not isinstance(frame_id, str) or not frame_id or coordinates is None:
        raise ValueError("DINO cache requires a Source frame ID and finite xyxy")
    return frame_id, struct.unpack("4f", struct.pack("4f", *coordinates))


def plan_dino_batches(rows, max_batch=16):
    """纯 CPU 规划：rows 含 frame、face_box、hand_boxes；批大小按 ROI 数计。"""
    _batch_limit(max_batch)
    rows = list(rows)
    if len(rows) > 16:
        raise ValueError("Prime at most 16 Source frames at once")
    tasks, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or {"frame", "face_box", "hand_boxes"} - row.keys():
            raise ValueError("Missing Source DINO crop planning fields")
        frame = row["frame"]
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen:
            raise ValueError("Missing or duplicate Source frame ID in DINO batch")
        seen.add(frame_id)
        fb = dino_cache_key(frame_id, row["face_box"])[1]
        hb = [dino_cache_key(frame_id, box)[1] for box in row["hand_boxes"]]
        policy = source_appearance_policy(frame, fb, hb)
        for region, box, enabled in (("face", fb, policy["face_dino"]),
                                     ("left", hb[0], policy["hand_dino"][0]),
                                     ("right", hb[1], policy["hand_dino"][1])):
            if enabled:
                tasks.append({"frame_id": frame_id, "region": region, "box": box,
                              "key": dino_cache_key(frame_id, box)})
    return [tasks[start:start + max_batch] for start in range(0, len(tasks), max_batch)]


def _prime_samples(samples, max_batch):
    _batch_limit(max_batch)
    samples = list(samples)
    if len(samples) > 16:
        raise ValueError("Prime at most 16 Source frames at once")
    seen = set()
    for sample in samples:
        required = {"frame", "rgb", "canvas", "asset", "empty_references"}
        if not isinstance(sample, dict) or required - sample.keys() or any(sample[key] is None for key in required):
            raise ValueError("Missing Source prime_dino sample fields")
        frame = sample["frame"]
        frame_id = frame.get("frame_id") if isinstance(frame, dict) else None
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen:
            raise ValueError("Missing or duplicate Source frame ID in prime_dino")
        seen.add(frame_id)
        if not isinstance(sample["rgb"], dict) or {"path", "sha256"} - sample["rgb"].keys():
            raise ValueError("Missing Source RGB path/SHA in prime_dino")
        if not isinstance(sample["asset"], dict) or "sha256" not in sample["asset"]:
            raise ValueError("Missing Source geometry SHA in prime_dino")
    return samples


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_transform(box, canvas_spec):
    import numpy as np
    height, width = canvas_spec.canvas_hw
    xyxy = np.asarray(box, dtype=np.float64)
    pixels = xyxy * [width, height, width, height]
    x0, y0 = np.floor(pixels[:2])
    x1, y1 = np.ceil(pixels[2:])
    affine = None if x1 <= x0 or y1 <= y0 else [
        [224 / (x1 - x0), 0.0, -x0 * 224 / (x1 - x0)],
        [0.0, 224 / (y1 - y0), -y0 * 224 / (y1 - y0)], [0.0, 0.0, 1.0]]
    return {"original_xyxy": canvas_spec.unmap_points(xyxy.reshape(2, 2)).reshape(4).tolist(),
            "canvas_xyxy": xyxy.tolist(), "pixel_crop_unclipped": [float(v) for v in (x0, y0, x1, y1)],
            "canvas_to_dino224": affine}


class SourceFeatureExtractor:
    """每个预处理进程复用一个实例；只将冻结原始特征交给离线 writer。

    models: arcface_model、arcface_sha256、dino_model、dino_files、device。
    首次实际请求某模型时核对文件 SHA 并加载；无可用模态时不加载模型。
    """

    def __init__(self, models):
        required = {"arcface_model", "arcface_sha256", "dino_model", "dino_files"}
        if required - models.keys():
            raise ValueError("Missing appearance model configuration: " + str(sorted(required - models.keys())))
        self.models = deepcopy(models)
        self.models.setdefault("device", "cpu")
        self._session = self._processor = self._dino = None
        self._dino_cache, self._dino_cache_uses, self._primed_images = {}, {}, {}
        self._active_frame_id = None

    def _arcface_session(self):
        if self._session is None:
            if _sha256(self.models["arcface_model"]) != self.models["arcface_sha256"]:
                raise ValueError("ArcFace checkpoint SHA differs")
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.models["arcface_model"]), sess_options=options,
                                                providers=["CPUExecutionProvider"])
        return self._session

    def _dino_models(self):
        if self._dino is None:
            directory = Path(self.models["dino_model"])
            if not self.models["dino_files"]:
                raise ValueError("Declare DINO model/config/processor fingerprints")
            for name, digest in self.models["dino_files"].items():
                if _sha256(directory / name) != digest:
                    raise ValueError("DINO model/processor SHA differs: " + name)
            from transformers import AutoImageProcessor, AutoModel
            processor = AutoImageProcessor.from_pretrained(str(directory), local_files_only=True)
            dino = AutoModel.from_pretrained(str(directory), local_files_only=True).float().eval()
            dino.requires_grad_(False)
            dino = dino.to(self.models["device"])
            self._processor, self._dino = processor, dino
        return self._processor, self._dino

    @staticmethod
    def _crop(image, box):
        if not valid_canvas_roi(box):
            raise ValueError("Declared available Source texture has no valid canvas ROI")
        x0, y0, x1, y1 = box
        # 同原提取器：PIL 在越界处补像素，不把原 ROI clamp 到图像内。
        return image.crop((math.floor(x0 * image.width), math.floor(y0 * image.height),
                           math.ceil(x1 * image.width), math.ceil(y1 * image.height)))

    def _dino_patches(self, crops):
        import torch
        processor, dino = self._dino_models()
        inputs = processor(images=crops, return_tensors="pt", size={"height": 224, "width": 224}, do_center_crop=False)
        with torch.no_grad():
            tokens = dino(**{key: value.to(dino.device) for key, value in inputs.items()}).last_hidden_state
        if tuple(tokens.shape) != (len(crops), 257, 1536) or not torch.isfinite(tokens).all():
            raise ValueError("Require finite ViT-g/14 16x16 patches without register tokens")
        return tokens[:, 1:].detach().to(device="cpu", dtype=torch.float32)

    def _patches(self, image, box):
        key = dino_cache_key(self._active_frame_id, box) if self._active_frame_id is not None else None
        if key in self._dino_cache:
            remaining = self._dino_cache_uses.get(key, 1) - 1
            if remaining:
                self._dino_cache_uses[key] = remaining
                return self._dino_cache[key]
            self._dino_cache_uses.pop(key, None)
            return self._dino_cache.pop(key)
        return self._dino_patches([self._crop(image, box)])[0]

    @staticmethod
    def _source_signature(rgb_ref, canvas_spec, asset_ref):
        return str(rgb_ref["path"]), rgb_ref["sha256"], canvas_spec.fingerprint(), asset_ref["sha256"]

    def prime_dino(self, samples, max_batch=16):
        """为最多 16 个 Source 帧批量预取；随后逐帧调用 extract 消费并清理。

        samples 每项为 {frame,rgb,canvas,asset,empty_references}。max_batch 限制
        单次模型前向的 ROI 数；不改变 FP32、224 processor、无 center crop 的约定。
        """
        samples = _prime_samples(samples, max_batch)
        if self._primed_images or self._dino_cache or self._active_frame_id is not None:
            raise ValueError("Consume the previous primed Source frames before priming another batch")
        from PIL import Image
        images, rows, cache, uses = {}, [], {}, {}
        for sample in samples:
            frame, rgb, spec, asset, empty = (sample[key] for key in ("frame", "rgb", "canvas", "asset", "empty_references"))
            _, face_boxes, hand_boxes = self._source_inputs(frame, rgb, spec, asset, empty)
            fb, hb = face_boxes[0, 0].tolist(), hand_boxes[0, 0].tolist()
            source_appearance_policy(frame, fb, hb)
            with Image.open(rgb["path"]) as im:
                image = spec.image(im.convert("RGB"), fill=(128, 128, 128))
            images[frame["frame_id"]] = (self._source_signature(rgb, spec, asset), image)
            rows.append({"frame": frame, "face_box": fb, "hand_boxes": hb})
        batches = plan_dino_batches(rows, max_batch=max_batch)
        for batch in batches:
            crops = [self._crop(images[task["frame_id"]][1], task["box"]) for task in batch]
            values = self._dino_patches(crops)
            for task, value in zip(batch, values):
                # 完全相同的跨模态 ROI 共享特征，最后一次消费才 pop。
                cache.setdefault(task["key"], value)
                uses[task["key"]] = uses.get(task["key"], 0) + 1
        self._dino_cache, self._dino_cache_uses, self._primed_images = cache, uses, images
        return {"frames": len(samples), "crops": sum(map(len, batches)), "batches": len(batches)}

    def _identity(self, image, points):
        import cv2
        import numpy as np
        import torch
        template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
        affine, _ = cv2.estimateAffinePartial2D(points, template, method=cv2.LMEDS)
        if affine is None or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:, :2])) <= 1e-12:
            raise ValueError("Source face alignment failed")
        crop = cv2.warpAffine(np.asarray(image), affine, (112, 112), borderValue=0)
        value = (crop.astype(np.float32) - 127.5) / 127.5
        session = self._arcface_session()
        output = session.run(None, {session.get_inputs()[0].name: value.transpose(2, 0, 1)[None]})[0]
        if output.shape != (1, 512) or not np.isfinite(output).all():
            raise ValueError("ArcFace must output finite [1,512]")
        return torch.from_numpy(np.asarray(output[0], dtype=np.float32).copy()), affine.tolist()

    @staticmethod
    def _source_inputs(frame, rgb_ref, canvas_spec, asset_ref, empty_references):
        import torch
        empty_references.validate()
        if tuple(empty_references.arcface.shape) != (1, 2, 1, 512):
            raise ValueError("Single-frame Source extraction requires B=1, R=1")
        if "target_rgb" in frame or "driver_rgb" in frame:
            raise ValueError("Appearance extractor accepts Source only")
        if tuple(frame["original_hw"]) != tuple(canvas_spec.original_hw):
            raise ValueError("Source frame original coordinates differ from CanvasSpec")
        for name in ("arcface", "face_dino", "hand_dino", "arcface_valid", "face_dino_valid", "hand_dino_valid"):
            value = getattr(empty_references, name)
            if value.requires_grad or value.any():
                raise ValueError("Only empty raw references may be populated; never cache trainable Adapter features")
        metadata = deepcopy(empty_references.metadata[0])
        if metadata["source_rgb_sha256"] != rgb_ref["sha256"] or metadata["source_canvas_id"] != canvas_spec.fingerprint():
            raise ValueError("Empty Source reference RGB/canvas provenance differs")
        if len(metadata.get("source_person_ids", [])) != 2 or len(metadata.get("reference_sources", [])) != 1:
            raise ValueError("Single Source frame requires stable person IDs and one reference provenance record")
        source = metadata["reference_sources"][0]
        if (source.get("rgb", rgb_ref) != rgb_ref or source.get("geometry_sha256", asset_ref["sha256"]) != asset_ref["sha256"]
                or source.get("person_ids", metadata["source_person_ids"]) != metadata["source_person_ids"]):
            raise ValueError("Source RGB/geometry/person provenance differs")
        if _sha256(rgb_ref["path"]) != rgb_ref["sha256"]:
            raise ValueError("Source RGB SHA differs")
        face_boxes = empty_references.source_face_boxes.detach().to(device="cpu", dtype=torch.float32).clone()
        hand_boxes = empty_references.source_hand_boxes.detach().to(device="cpu", dtype=torch.float32).clone()
        return metadata, face_boxes, hand_boxes

    def extract(self, frame, rgb_ref, canvas_spec, asset_ref, empty_references) -> "NativeLocalReferencesV4":
        """沿用原 API；消费本帧预取结果，未命中仍真实提取，结束即清理本帧。"""
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("Single-frame Source extraction requires a frame ID")
        if self._active_frame_id is not None:
            raise ValueError("SourceFeatureExtractor is single-consumer; concurrent extraction is unsupported")
        self._active_frame_id = frame_id
        try:
            return self._extract_one(frame, rgb_ref, canvas_spec, asset_ref, empty_references)
        finally:
            self._primed_images.pop(frame_id, None)
            for key in [key for key in self._dino_cache if key[0] == frame_id]:
                self._dino_cache.pop(key)
                self._dino_cache_uses.pop(key, None)
            self._active_frame_id = None

    def _extract_one(self, frame, rgb_ref, canvas_spec, asset_ref, empty_references):
        import torch
        from PIL import Image
        from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA

        metadata, face_boxes, hand_boxes = self._source_inputs(frame, rgb_ref, canvas_spec, asset_ref, empty_references)
        source = metadata["reference_sources"][0]
        fb, hb = face_boxes[0, 0].tolist(), hand_boxes[0, 0].tolist()
        policy = source_appearance_policy(frame, fb, hb)
        points = arcface_canvas_points(frame, canvas_spec) if policy["arcface"] else None
        policy["arcface"] = policy["arcface"] and points is not None
        arc = torch.zeros(1, 2, 1, 512, dtype=torch.float32)
        face = torch.zeros(1, 2, 1, 256, 1536, dtype=torch.float32)
        hand = torch.zeros(1, 2, 2, 1, 256, 1536, dtype=torch.float32)
        av = torch.zeros(1, 2, 1, dtype=torch.bool)
        fv = torch.zeros_like(av)
        hv = torch.zeros(1, 2, 2, 1, dtype=torch.bool)
        crop_record = {"face": _crop_transform(fb, canvas_spec),
                       "hands": [_crop_transform(box, canvas_spec) for box in hb], "canvas_to_arcface112": None}
        primed = self._primed_images.get(frame["frame_id"])
        if primed is not None:
            if primed[0] != self._source_signature(rgb_ref, canvas_spec, asset_ref):
                raise ValueError("Primed Source RGB/canvas/geometry provenance differs")
            image = primed[1]
        else:
            with Image.open(rgb_ref["path"]) as im:
                image = canvas_spec.image(im.convert("RGB"), fill=(128, 128, 128))
        if policy["arcface"]:
            arc[0, 0, 0], crop_record["canvas_to_arcface112"] = self._identity(image, points)
            av[0, 0, 0] = True
        if policy["face_dino"]:
            face[0, 0, 0] = self._patches(image, fb)
            fv[0, 0, 0] = True
        for side, enabled in enumerate(policy["hand_dino"]):
            if enabled:
                hand[0, 0, side, 0] = self._patches(image, hb[side])
                hv[0, 0, side, 0] = True
        source.update(rgb=deepcopy(rgb_ref), canvas=canvas_spec.metadata(), canvas_id=canvas_spec.fingerprint(),
                      geometry_sha256=asset_ref["sha256"], original_to_canvas=canvas_spec.transform.tolist(),
                      crop_transforms=[crop_record], roi_and_alignment=[{
                          "face_box": fb, "hand_boxes": hb, "identity_valid": bool(av[0, 0, 0]),
                          "face_texture_valid": bool(fv[0, 0, 0]), "hand_texture_valid": hv[0, 0, :, 0].tolist(),
                          "arcface_five_points_canvas_pixels": None if points is None else points.tolist()}])
        metadata.pop("reason", None)  # 几何准备阶段的全空说明已不再代表本缓存。
        extracted = {"arcface": bool(av.any()), "face_dino": bool(fv.any()), "hand_dino": bool(hv.any())}
        metadata.update(schema=REFERENCE_SCHEMA, role="source_only", reference_count=1,
                        arcface_sha256=self.models["arcface_sha256"], dino_files=deepcopy(self.models["dino_files"]),
                        arcface_preprocessing=ARCFACE_PREPROCESSING, dino_preprocessing=DINO_PREPROCESSING,
                        models_loaded={"arcface": self._session is not None, "dino": self._dino is not None},
                        modalities_requested={"arcface": True, "face_dino": True, "hand_dino": True},
                        modalities_extracted=extracted,
                        availability={key: "available" if valid else "unavailable" for key, valid in extracted.items()},
                        loss_mask=deepcopy(frame["loss_mask"]), hand_raw_pose_usable=deepcopy(frame.get("hand_raw_pose_usable")),
                        hand_release_mapping={"anatomical_left": "hand_b", "anatomical_right": "hand_a"},
                        appearance_policy=policy, source_frame_id=frame["frame_id"],
                        cache_scope="frozen_source_arcface_and_dino_only; no_trainable_adapter_features")
        return NativeLocalReferencesV4(arc, av, face, fv, hand, hv, face_boxes, hand_boxes, (metadata,)).validate()
