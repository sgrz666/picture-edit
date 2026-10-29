import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

SCHEMA = "dual_pose_benchmark_v1"
MODES = ("dwpose_auto", "dwpose_gtbox", "sapiens_yolox", "sapiens_detr070", "sapiens_gtbox")
BODY_JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SAPIENS_DIR = Path("/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose")
META_PATH = SAPIENS_DIR / "configs/_base_/keypoints308.py"
CONFIG_PATH = SAPIENS_DIR / "configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
CHECKPOINT = Path("/data/shangguanrz/models/meta/sapiens2-pose-5b/sapiens2_5b_pose.safetensors")
DETR_PATH = Path("/data/shangguanrz/models/meta/detector/detr-resnet-101-dc5")
GIB = 1024 ** 3


def load_sapiens_metadata(path=META_PATH):
    return runpy.run_path(str(path))["dataset_info"]


def sapiens_body_indices(meta):
    names = meta.get("keypoint_id2name")
    if names is None:
        names = {i: item["name"] for i, item in meta["keypoint_info"].items()}
    names = {int(i): name for i, name in names.items()}
    by_name = {name: i for i, name in names.items()}
    indices = [by_name.get(name, -1) for name in BODY_JOINT_NAMES]
    if set(names) != set(range(308)) or indices != [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]:
        raise ValueError("expected official Sapiens308 names (wrists 62/41), not COCO")
    return indices


def as_boxes(value):
    import numpy as np

    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.shape == (4,):
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("bbox must be finite Nx4 xyxy")
    if not (boxes[:, 2:] > boxes[:, :2]).all():
        raise ValueError("bbox must have positive width and height")
    return boxes


def normalise_instances(boxes, xy, scores, indices, keypoint_count):
    import numpy as np

    boxes = as_boxes(boxes)
    n, k = len(boxes), keypoint_count
    xy, scores = np.asarray(xy, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if n == 1 and xy.shape == (k, 2):
        xy = xy[None]
    if xy.shape == (n, 1, k, 2):
        xy = xy[:, 0]
    if n == 1 and scores.shape == (k,):
        scores = scores[None]
    if scores.shape == (n, 1, k):
        scores = scores[:, 0]
    if scores.shape == (n, k, 1):
        scores = scores[..., 0]
    if xy.shape != (n, k, 2) or scores.shape != (n, k):
        raise ValueError(f"pose shape mismatch: boxes={n}, xy={xy.shape}, scores={scores.shape}, K={k}")
    if not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError("pose contains non-finite values")
    if len(indices) != 12 or len(set(indices)) != 12 or any(i < 0 or i >= k for i in indices):
        raise ValueError("invalid body indices")
    return [
        {"bbox": box.tolist(), "xy": points[indices].tolist(),
         "scores": confidence[indices].tolist(), "joint_names": list(BODY_JOINT_NAMES),
         "extra_keypoints": points.tolist(), "extra_scores": confidence.tolist()}
        for box, points, confidence in zip(boxes, xy, scores)
    ]


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json(raw):
    return json.loads(raw, parse_constant=reject_constant)


def fingerprint(record, mode, request_sha256):
    return {"schema": SCHEMA, "id": record["id"], "mode": mode,
            "image_sha256": record["image_sha256"], "request_sha256": request_sha256}


def should_skip(path, expected):
    if not path.exists():
        return False
    try:
        saved = parse_json(path.read_bytes())
        if saved.get("status") == "ok" and all(saved.get(k) == v for k, v in expected.items()):
            return True
    except (ValueError, AttributeError):
        pass
    raise FileExistsError(f"refusing to overwrite mismatched, failed or incomplete result: {path}")


def write_json_exclusive(path, data):
    raw = (json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def cached_yolox(out_dir, record, request_sha256):
    path = Path(out_dir) / "dwpose_auto" / (record["id"] + ".json")
    raw = path.read_bytes()
    saved = parse_json(raw)
    expected = fingerprint(record, "dwpose_auto", request_sha256)
    if saved.get("status") != "ok" or any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError(f"invalid dwpose_auto fingerprint/status: {path}")
    boxes = as_boxes([instance["bbox"] for instance in saved["instances"]])
    if saved["det_count"] != len(boxes):
        raise ValueError(f"dwpose_auto det_count does not match instances: {path}")
    return boxes, {"source": "dwpose_auto", "source_file": str(path),
                   "source_sha256": hashlib.sha256(raw).hexdigest()}


def postprocess_detr(boxes, scores, nms):
    import numpy as np

    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError("invalid DETR scores")
    keep = nms(np.column_stack((boxes, scores)), 0.7) if len(boxes) else []
    selected = boxes[keep]
    return selected, {"source": "detr", "bbox_thr": 0.3, "nms_thr": 0.7,
                      "raw_boxes": boxes.tolist(), "raw_scores": scores.tolist(),
                      "post_nms_boxes": selected.tolist(), "post_nms_scores": scores[keep].tolist()}


def provenance(project_root, mode):
    root = Path(project_root)
    dw_code = root / "src/evaluation/dwpose_onnx"
    weights = root / "add/tools/dwpose_mimicmotion_official/models"
    info = {"runner": str(Path(__file__).resolve()), "python": sys.executable,
            "project_root": str(root), "body_joint_names": list(BODY_JOINT_NAMES),
            "latency_scope": "detection/box reuse + pose + normalization; excludes image IO, model load and JSON write"}
    if mode.startswith("dwpose"):
        info.update(pose_code=str(dw_code / "onnxpose.py"),
                    pose_checkpoint=str(weights / "dw-ll_ucoco_384.onnx"),
                    providers=["CPUExecutionProvider"], intra_op_threads=4, inter_op_threads=1,
                    timing="CPU wall clock, ORT <=4 threads; not comparable to GPU latency",
                    extra_keypoint_layout="raw COCO133", body_indices=list(range(5, 17)))
        if mode == "dwpose_auto":
            info.update(detector_code=str(dw_code / "onnxdet.py"),
                        detector_checkpoint=str(weights / "yolox_l.onnx"),
                        detector_nms_thr=0.45, detector_score_thr=0.3)
    else:
        info.update(pose_code=str(SAPIENS_DIR), pose_config=str(CONFIG_PATH),
                    pose_checkpoint=str(CHECKPOINT), metainfo=str(META_PATH),
                    timing="CUDA synchronized wall clock; includes CPU preprocessing/decoding",
                    extra_keypoint_layout="official dataset_info Sapiens308",
                    peak_scope="per-frame CUDA allocator peaks, includes resident model(s)")
        if mode == "sapiens_detr070":
            info.update(detector_checkpoint=str(DETR_PATH), bbox_thr=0.3, nms_thr=0.7)
        elif mode == "sapiens_yolox":
            info["detector_source"] = "saved dwpose_auto boxes only; no detector inference"
    if mode.endswith("gtbox"):
        info["detector_source"] = "request.gt_boxes"
    return info


def load_module(path):
    spec = importlib.util.spec_from_file_location("dual_pose_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Models:
    def __init__(self, request, mode, request_sha256):
        self.request, self.mode, self.request_sha256 = request, mode, request_sha256
        self.provenance = provenance(request["project_root"], mode)
        self.torch = None
        self.device = "cpu" if mode.startswith("dwpose") else "cuda:0"
        if self.device == "cpu":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.pose_session = ort.InferenceSession(self.provenance["pose_checkpoint"],
                                                    sess_options=options, providers=["CPUExecutionProvider"])
            self.pose_fn = load_module(Path(self.provenance["pose_code"])).inference_pose
            self.indices, self.keypoint_count = list(range(5, 17)), 133
            if mode == "dwpose_auto":
                self.det_session = ort.InferenceSession(self.provenance["detector_checkpoint"],
                                                       sess_options=options, providers=["CPUExecutionProvider"])
                self.det_fn = load_module(Path(self.provenance["detector_code"])).inference_detector
            return

        self.indices = sapiens_body_indices(load_sapiens_metadata())
        self.keypoint_count = 308
        sys.path.insert(0, str(SAPIENS_DIR.parents[1]))
        import torch
        from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
        from sapiens.pose.models import init_model

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Sapiens requires deepgen CUDA; CPU fallback is disabled")
        free, total = torch.cuda.mem_get_info(self.device)
        self.provenance.update(gpu_free_before_load_gib=free / GIB, gpu_total_gib=total / GIB,
                               body_indices=self.indices)
        if mode == "sapiens_detr070":
            from transformers import DetrForObjectDetection, DetrImageProcessor
            from sapiens.pose.evaluators import nms

            self.det_proc = DetrImageProcessor.from_pretrained(str(DETR_PATH), local_files_only=True)
            self.det_model = DetrForObjectDetection.from_pretrained(str(DETR_PATH), local_files_only=True).eval().to(self.device)
            self.nms = nms
        self.model = init_model(str(CONFIG_PATH), str(CHECKPOINT), device=self.device)
        self.model.pose_metainfo = parse_pose_metainfo(dict(from_file=str(META_PATH)))
        if sapiens_body_indices(self.model.pose_metainfo) != self.indices:
            raise ValueError("parsed pose metadata differs from dataset_info")
        codec = dict(self.model.cfg.codec)
        if codec.pop("type") != "UDPHeatmap":
            raise ValueError("expected UDPHeatmap codec")
        self.model.codec = UDPHeatmap(**codec)

    def detect(self, record, bgr):
        if self.mode.endswith("gtbox"):
            return as_boxes(record["gt_boxes"]), {"source": "request.gt_boxes"}
        if self.mode == "dwpose_auto":
            return as_boxes(self.det_fn(self.det_session, bgr)), {"source": "project YOLOX"}
        if self.mode == "sapiens_yolox":
            return cached_yolox(self.request["out_dir"], record, self.request_sha256)
        from PIL import Image

        h, w = bgr.shape[:2]
        inputs = self.det_proc(images=Image.fromarray(bgr[:, :, ::-1]), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.det_model(**inputs)
        sizes = self.torch.tensor([[h, w]], device=self.device)
        result = self.det_proc.post_process_object_detection(output, target_sizes=sizes, threshold=0.3)[0]
        persons = result["labels"] == 1
        boxes = result["boxes"][persons].cpu().numpy()
        scores = result["scores"][persons].cpu().numpy()
        return postprocess_detr(boxes, scores, self.nms)

    def pose(self, bgr, boxes):
        import numpy as np

        if len(boxes) == 0:
            return np.empty((0, self.keypoint_count, 2)), np.empty((0, self.keypoint_count))
        if self.device == "cpu":
            return self.pose_fn(self.pose_session, boxes, bgr)
        inputs, samples = [], []
        with self.torch.no_grad():
            for box in boxes:
                data = self.model.pipeline(dict(img=bgr, bbox=box[None].astype(np.float32),
                                                bbox_score=np.ones(1, dtype=np.float32)))
                data = self.model.data_preprocessor(data)
                inputs.append(data["inputs"])
                samples.append(data["data_samples"])
            prediction = self.model(self.torch.cat(inputs, dim=0)).cpu().numpy()
        xy, scores = [], []
        for heatmap, sample in zip(prediction, samples):
            points, confidence = self.model.codec.decode(heatmap)
            meta = sample["meta"]
            points = points / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
            xy.append(points[0])
            scores.append(confidence[0])
        return xy, scores


def empty_result(device, info):
    return {"status": "error", "det_count": 0, "instances": [], "latency_ms": 0.0,
            "peak_allocated_gib": None, "peak_reserved_gib": None,
            "device": device, "provenance": dict(info)}


def infer_frame(models, record, bgr, detection=None):
    result = empty_result(models.device, models.provenance)
    start = None
    try:
        if models.torch is not None:
            models.torch.cuda.synchronize(models.device)
            models.torch.cuda.reset_peak_memory_stats(models.device)
        start = time.perf_counter()
        boxes, audit = models.detect(record, bgr) if detection is None else detection
        boxes = as_boxes(boxes)
        result.update(det_count=len(boxes), detection=audit)
        if len(boxes):
            xy, scores = models.pose(bgr, boxes)
            result["instances"] = normalise_instances(boxes, xy, scores, models.indices, models.keypoint_count)
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if models.torch is not None:
                models.torch.cuda.synchronize(models.device)
                result["peak_allocated_gib"] = models.torch.cuda.max_memory_allocated(models.device) / GIB
                result["peak_reserved_gib"] = models.torch.cuda.max_memory_reserved(models.device) / GIB
        except Exception as exc:
            result["status"] = "error"
            result["error"] = result.get("error", "") + f" CUDA metrics: {exc}"
        if start is not None:
            result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


def validate_request(request):
    import numpy as np

    if request.get("schema") != SCHEMA or request.get("body_joint_names") != BODY_JOINT_NAMES:
        raise ValueError("invalid request schema or body_joint_names order")
    for name in ("project_root", "out_dir"):
        if not Path(request[name]).is_absolute():
            raise ValueError(f"{name} must be absolute")
    if not isinstance(request["records"], list):
        raise ValueError("records must be a list")
    seen = set()
    for record in request["records"]:
        digest = record["image_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("image_sha256 must be lowercase SHA256 hex")
        if record["id"] != digest[:16] or record["id"] in seen:
            raise ValueError("record id must be a unique image SHA256 prefix")
        seen.add(record["id"])
        if not Path(record["image"]).is_absolute():
            raise ValueError("image must be absolute")
        if as_boxes(record["gt_boxes"]).shape != (2, 4):
            raise ValueError("gt_boxes must contain exactly two xyxy boxes")
        k = np.asarray(record["K"], dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise ValueError("K must be finite 3x3")
        if type(record["camera_id"]) is not int or any(not isinstance(record[n], str) for n in ("category", "pair", "action")):
            raise ValueError("invalid camera_id/category/pair/action")


def read_verified_image(record):
    raw = Path(record["image"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["image_sha256"]:
        raise ValueError(f"image sha256 mismatch: {record['image']}")
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {record['image']}")
    return bgr


def run(request_path, mode, limit=None):
    if mode not in MODES or (limit is not None and limit < 0):
        raise ValueError("invalid mode or negative limit")
    raw = Path(request_path).read_bytes()
    request, request_sha256 = parse_json(raw), hashlib.sha256(raw).hexdigest()
    validate_request(request)
    records = request["records"] if limit is None else request["records"][:limit]
    out_dir = Path(request["out_dir"]) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    models, load_error, failures = None, None, 0
    info = provenance(request["project_root"], mode)
    device = "cpu" if mode.startswith("dwpose") else "cuda:0"
    for record in records:
        path = out_dir / (record["id"] + ".json")
        expected = fingerprint(record, mode, request_sha256)
        try:
            if should_skip(path, expected):
                print(f"skip {mode}/{record['id']}")
                continue
        except Exception as exc:
            failures += 1
            print(str(exc), file=sys.stderr)
            continue
        result = empty_result(device, models.provenance if models is not None else info)
        try:
            bgr = read_verified_image(record)
            detection = cached_yolox(request["out_dir"], record, request_sha256) if mode == "sapiens_yolox" else None
            if load_error is not None:
                raise RuntimeError(load_error)
            if models is None:
                try:
                    models = Models(request, mode, request_sha256)
                except Exception as exc:
                    load_error = f"model load failed: {type(exc).__name__}: {exc}"
                    raise RuntimeError(load_error) from exc
            result = infer_frame(models, record, bgr, detection)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result.update(expected)
        try:
            write_json_exclusive(path, result)
        except Exception as exc:
            failures += 1
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        print(f"{mode}/{record['id']}: {result['status']} det={result['det_count']}")
        if result["status"] != "ok":
            failures += 1
            print(result["error"], file=sys.stderr)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dual pose comparison; DWPose timing is CPU, Sapiens is CUDA.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        return run(args.request, args.mode, args.limit)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
