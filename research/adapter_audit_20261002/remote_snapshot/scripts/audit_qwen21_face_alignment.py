#!/usr/bin/env python3
"""CPU-only crop/alignment audit on existing single-case generated images.

This does not synthesize images, train, load adapter checkpoints, or change
generation conditions. Target five points are offline supervision metadata.
Rerun after a new repair generation exists to include it in the same audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

# Restrict this standalone audit process; other training processes are unaffected.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.evaluate_qwen21_weight_inheritance import cpu_identity_metric, face_detection
from src.pose_control.qwen21_full.data_runtime import pair_metadata, read_pairs
from src.pose_control.qwen21_full.face_loss_alignment import (
    align_face_rgb, build_arcface_alignment_grid,
)
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore

RUN_ROOT = Path("/home/shangguanrz/nihaochen/qwen21_full")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rgb(path):
    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        raise ValueError(f"Audit requires the same 512 grid without resizing: {path}, {image.size}")
    return np.asarray(image).copy()


def image_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255


def legacy_box_crop(image, box):
    # Exact endpoint/align_corners convention in execution.crop_batch.
    axis = torch.linspace(0., 1., 112)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x0, y0, x1, y1 = box
    grid = torch.stack(((x0 + xx * (x1 - x0)) * 2 - 1,
                        (y0 + yy * (y1 - y0)) * 2 - 1), dim=-1)[None]
    return F.grid_sample(image_tensor(image), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


@torch.inference_mode()
def ts_feature(model, crop, *, normalized=True):
    value = model(crop * 2 - 1 if normalized else crop).flatten(1)
    if not bool(torch.isfinite(value).all()) or float(value.norm()) == 0:
        raise ValueError("Non-finite or zero ArcFace feature")
    return value[0].cpu().numpy().astype(np.float32)


def cosine(left, right):
    if left is None or right is None:
        return None
    value = np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.clip(value, -1., 1.))


def select_stage(path, step):
    rows = read_json(path)
    selected = [row for row in rows if row["added_step"] == step]
    if len(selected) != 1:
        raise ValueError(f"Expected one case at step {step}: {path}")
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, default=RUN_ROOT / "face_single_sample")
    parser.add_argument("--repair-root", type=Path, default=RUN_ROOT / "face_repair")
    parser.add_argument("--face-supervision", type=Path, default=RUN_ROOT / "face_supervision.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.repair_root / "alignment_audit.json"
    torch.set_num_threads(4)

    pairs = read_pairs(args.single_root / "single_pair.jsonl")
    if len(pairs) != 1:
        raise ValueError("This audit is for the already fixed single training pair")
    metadata = pair_metadata(pairs[0])
    supervision = FaceSupervisionStore(args.face_supervision)
    target_record = supervision.record(metadata)
    eligibility = target_record["by_resolution"]["512"]
    if not eligibility["valid"]:
        raise ValueError("Fixed Target lacks reliable five-point supervision")
    detection = target_record["detections"][eligibility["selected_detection"]]
    target_points = torch.tensor(detection["landmarks5_original_px"], dtype=torch.float32)[None]
    target_points *= target_points.new_tensor([512 / target_record["original_width"],
                                              512 / target_record["original_height"]])
    target_box = target_record["geometry_box_normalized"]

    body_path = args.single_root / "body_stage_metrics.json"
    face_path = args.single_root / "face_stage_metrics.json"
    baseline = select_stage(body_path, 500)
    candidates = [
        ("body_500", baseline, body_path),
        ("old_face_300", select_stage(face_path, 300), face_path),
        ("old_face_2000", select_stage(face_path, 2000), face_path),
    ]
    absent = []
    for experiment in ("roi_same_aux", "roi_aligned_aux"):
        path = args.repair_root / "generations" / experiment / "step_000300" / "metrics.json"
        if path.exists():
            candidates.append((experiment + "_300", read_json(path), path))
        else:
            absent.append({"case": experiment + "_300", "metrics_path": str(path),
                           "reason": "generation not yet available; rerun the audit to add it"})
    if not any(name == "roi_same_aux_300" for name, _, _ in candidates):
        raise ValueError("ROI 300 generation is required for this audit")

    source_path, target_path = Path(baseline["source"]), Path(baseline["target"])
    source_image, target_image = rgb(source_path), rgb(target_path)
    metric = cpu_identity_metric()
    for backend in (metric.det, metric.recognition):
        if backend.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Audit ONNX models must use only the CPU provider")
    evaluator_path = ROOT / "models/face_eval/identity_encoder.ts"
    model = torch.jit.load(str(evaluator_path), map_location="cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grid = build_arcface_alignment_grid(target_points, (512, 512))

    def fixed_opencv(image):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        aligned = metric.face_align.norm_crop(bgr, landmark=target_points[0].numpy(), image_size=112)
        return image_tensor(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

    reference_crops = {
        "legacy_box_ts": legacy_box_crop(target_image, target_box),
        "fixed_target5_opencv_ts": fixed_opencv(target_image),
        "fixed_target5_differentiable_ts": align_face_rgb(image_tensor(target_image), grid=grid),
    }
    target_features = {name: ts_feature(model, crop) for name, crop in reference_crops.items()}
    target_detection, target_raw, target_reliable = face_detection(metric, target_image, target_box)
    source_box = baseline["face_boxes_normalized"]["source"]
    source_detection, source_raw, source_reliable = face_detection(metric, source_image, source_box)

    # Test the same aligned source pixels against both actual model wrappers.
    bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    _, source_landmarks = metric.det.detect(bgr)
    index = source_detection["reliable_index"]
    if index is None:
        raise ValueError("Reliable Source detection is required for evaluator parity")
    source_aligned = metric.face_align.norm_crop(bgr, landmark=source_landmarks[index], image_size=112)
    source_crop = image_tensor(cv2.cvtColor(source_aligned, cv2.COLOR_BGR2RGB))
    onnx_feature = metric.recognition.get_feat(source_aligned).reshape(-1).astype(np.float32)
    normalized_feature = ts_feature(model, source_crop)
    legacy_feature = ts_feature(model, source_crop, normalized=False)
    normalized_unit = normalized_feature / np.linalg.norm(normalized_feature)
    onnx_unit = onnx_feature / np.linalg.norm(onnx_feature)
    model_parity = {
        "source_image": str(source_path), "source_reliable_detection_index": index,
        "same_pixels_and_five_point_alignment": True,
        "onnx_input_mean": float(metric.recognition.input_mean),
        "onnx_input_std": float(metric.recognition.input_std),
        "normalized_rgb_ts_vs_onnx_cosine": cosine(normalized_feature, onnx_feature),
        "normalized_feature_max_abs_difference": float(np.abs(normalized_unit - onnx_unit).max()),
        "raw_feature_relative_l2_difference": float(np.linalg.norm(normalized_feature - onnx_feature)
                                                    / np.linalg.norm(onnx_feature)),
        "torchscript_feature_norm": float(np.linalg.norm(normalized_feature)),
        "onnx_raw_feature_norm": float(np.linalg.norm(onnx_feature)),
        "output_scale_note": "TorchScript explicitly L2-normalizes its embedding; ONNX returns the raw vector. "
                             "Raw-vector L2 difference is expected and is not a parity failure. Compare normalized "
                             "embeddings and cosine, as the identity losses and evaluation do.",
        "legacy_zero_one_ts_vs_onnx_cosine": cosine(legacy_feature, onnx_feature),
        "tolerance": {"cosine_at_least": .999, "normalized_max_abs_at_most": .001},
    }
    model_parity["passed"] = (model_parity["normalized_rgb_ts_vs_onnx_cosine"] >= .999
                               and model_parity["normalized_feature_max_abs_difference"] <= .001)

    interpolation_rows = []
    rows = []
    for case, row, metrics_path in candidates:
        if row.get("resolution", 512) != 512 or row["metadata"]["pair_id"] != metadata["pair_id"]:
            raise ValueError(f"Candidate does not use the fixed 512 case: {case}")
        image_path = Path(row["image"])
        image = rgb(image_path)
        crops = {"legacy_box_ts": legacy_box_crop(image, target_box),
                 "fixed_target5_opencv_ts": fixed_opencv(image),
                 "fixed_target5_differentiable_ts": align_face_rgb(image_tensor(image), grid=grid)}
        features = {name: ts_feature(model, crop) for name, crop in crops.items()}
        methods = {name: {"target_cosine": cosine(features[name], target_features[name]), "effective_n": 1,
                           "validity": "fixed reliable Target metadata; not generated-face detection certification"}
                   for name in crops}
        generated_detection, generated_raw, generated_reliable = face_detection(metric, image, target_box)
        methods["official_detection_onnx"] = {
            "target_cosine_raw": cosine(generated_raw, target_raw),
            "target_cosine_reliable": cosine(generated_reliable, target_reliable),
            "source_cosine_raw": cosine(generated_raw, source_raw),
            "source_cosine_reliable": cosine(generated_reliable, source_reliable),
            "raw_effective_n": int(generated_raw is not None and target_raw is not None),
            "reliable_effective_n": int(generated_reliable is not None and target_reliable is not None),
        }
        difference = (crops["fixed_target5_opencv_ts"] - crops["fixed_target5_differentiable_ts"]).abs()
        interpolation_rows.append({"case": case, "rgb_mae_zero_one": float(difference.mean()),
                                   "rgb_max_abs_zero_one": float(difference.max()),
                                   "same_image_alignment_feature_cosine": cosine(
                                       features["fixed_target5_opencv_ts"], features["fixed_target5_differentiable_ts"])})
        rows.append({"case": case, "image": str(image_path), "image_sha256": sha(image_path),
                     "metrics_path": str(metrics_path), "methods": methods,
                     "generated_face_detection": generated_detection,
                     "historical_target_cosine_raw": row["metrics"].get("identity_target_cosine_raw"),
                     "historical_reference_note": "Historical formal evaluation may read 1024 originals then resize; "
                                                  "this audit uses the same saved 512 Source/Target for every method."})

    payload = {
        "schema": "qwen21_full_face_alignment_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True, "no_generation_or_training": True,
        "target_landmarks_enter_generation_conditions": False,
        "scope": "one existing fixed training pair; not a test-set or generalization claim",
        "metadata": metadata, "resolution": 512, "fixed_sample": str(args.single_root / "single_pair.jsonl"),
        "audit_script": str(Path(__file__).resolve()), "audit_script_sha256": sha(__file__),
        "alignment_implementation": str(ROOT / "src/pose_control/qwen21_full/face_loss_alignment.py"),
        "alignment_implementation_sha256": sha(ROOT / "src/pose_control/qwen21_full/face_loss_alignment.py"),
        "source_image": str(source_path), "target_image": str(target_path),
        "source_sha256": sha(source_path), "target_sha256": sha(target_path),
        "face_supervision": str(args.face_supervision), "face_supervision_sha256": sha(args.face_supervision),
        "landmarks_original_px": detection["landmarks5_original_px"],
        "landmarks_at_512_px": target_points[0].tolist(), "target_box_normalized": target_box,
        "target_supervision_eligibility": eligibility,
        "models": {"torchscript": str(evaluator_path), "torchscript_sha256": sha(evaluator_path),
                   "onnx": str(ROOT / "models/antelopev2/glintr100.onnx"),
                   "onnx_sha256": sha(ROOT / "models/antelopev2/glintr100.onnx"),
                   "detector": str(ROOT / "models/antelopev2/scrfd_10g_bnkps.onnx"),
                   "providers": metric.recognition.session.get_providers()},
        "methods": {
            "legacy_box_ts": "SMPL-X normalized square box, 112 endpoint grid, align_corners=True; RGB[-1,1] TorchScript",
            "fixed_target5_opencv_ts": "Both images use Target five points and cv2 norm_crop; RGB[-1,1] TorchScript",
            "fixed_target5_differentiable_ts": "Both images share detached Target5 inverse affine float grid; "
                                               "bilinear grid_sample align_corners=False; RGB[-1,1] TorchScript",
            "official_detection_onnx": "Each image detected independently by existing SCRFD; own five points; "
                                        "existing norm_crop and glintr100 ONNX; raw and reliable counts reported separately",
        },
        "source_face_detection": source_detection, "target_face_detection": target_detection,
        "source_target_identity_official_raw": cosine(source_raw, target_raw),
        "source_target_identity_official_reliable": cosine(source_reliable, target_reliable),
        "evaluator_parity": model_parity,
        "interpolation_note": "OpenCV INTER_LINEAR uses quantized fractions and uint8 rounding; PyTorch samples "
                               "floating-point RGB continuously. Matching geometry does not mean bitwise equal crops.",
        "interpolation_comparison": interpolation_rows,
        "rows": rows, "pending_cases": absent,
        "counts": {"generated_cases": len(rows), "unique_source_target_pairs": 1,
                   "fixed_target_landmark_faces": 1,
                   "reliable_generated_target_comparisons": sum(
                       r["methods"]["official_detection_onnx"]["reliable_effective_n"] for r in rows),
                   "common_reliable_sample_n_across_all_cases": int(all(
                       r["methods"]["official_detection_onnx"]["reliable_effective_n"] for r in rows))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(output)
    print(json.dumps({"output": str(output), "evaluator_parity": model_parity, "scores": [
        {"case": row["case"], **{name: value.get("target_cosine", value.get("target_cosine_raw"))
                                  for name, value in row["methods"].items()}} for row in rows]}, ensure_ascii=False))
    if not model_parity["passed"]:
        raise AssertionError("RGB[-1,1] TorchScript and original ONNX evaluator parity failed; see audit record")


if __name__ == "__main__":
    main()
