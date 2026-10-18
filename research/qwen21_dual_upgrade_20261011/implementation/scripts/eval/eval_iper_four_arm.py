#!/usr/bin/env python3
"""Evaluate saved DeepGen/Qwen21 images using the same CPU metrics and pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.evaluation.dwpose_onnx.onnxpose import inference_pose


ROOT = Path(__file__).resolve().parent
ARMS = ("deepgen_off", "deepgen_on", "qwen21_off", "qwen21_on")
# DWPose's first 17 outputs follow COCO order; these are the corresponding
# BODY_25 indices in the iPER asset bundle.
COCO_TO_BODY25 = (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl")
    parser.add_argument("--train-pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/dev_train_pairs.jsonl")
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "datasets/iPER/iper_assets_512_nvdiffrast")
    parser.add_argument("--deepgen-results", type=Path, default=ROOT / "experiments/qwen21_iper_body/deepgen_eval_50")
    parser.add_argument("--qwen21-results", type=Path, default=ROOT / "experiments/qwen21_iper_body/eval_50")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/qwen21_iper_body/four_arm_eval_50")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-identity", action="store_true", help="For a quick metric smoke run")
    return parser.parse_args()


class PoseMetric:
    def __init__(self) -> None:
        paths = ROOT / "models/DWPose"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.det = ort.InferenceSession(str(paths / "yolox_l.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(str(paths / "dw-ll_ucoco_384.onnx"), sess_options=options, providers=["CPUExecutionProvider"])

    def detect(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray] | None:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        boxes = inference_detector(self.det, bgr)
        if len(boxes) == 0:
            return None
        # The fixed benchmark is single-person; the largest detected body is
        # used if the detector returns several people.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        points, scores = inference_pose(self.pose, boxes[np.argmax(areas):np.argmax(areas) + 1], bgr)
        return points[0, :17], scores[0, :17]

    @staticmethod
    def compare(detected: tuple[np.ndarray, np.ndarray] | None, target: np.ndarray) -> dict[str, float | bool | None]:
        if detected is None:
            return {"pose_detected": False, "pose_nme_diagonal": None, "pose_pck_002": 0.0, "pose_pck_005": 0.0, "pose_joint_recall": 0.0, "pose_matched_joints": 0}
        points, scores = detected
        target_coco = target[np.array(COCO_TO_BODY25)]
        target_visible = target_coco[:, 2] > 0.3
        matched = target_visible & (scores > 0.3)
        count = int(matched.sum())
        denominator = int(target_visible.sum())
        if denominator == 0:
            return {"pose_detected": True, "pose_nme_diagonal": None, "pose_pck_002": None, "pose_pck_005": None, "pose_joint_recall": None, "pose_matched_joints": 0}
        distance = np.linalg.norm(points[matched] - target_coco[matched, :2] * 512, axis=1) / np.hypot(512, 512)
        return {
            "pose_detected": True,
            "pose_nme_diagonal": float(distance.mean()) if count else None,
            "pose_pck_002": float((distance < 0.02).sum() / denominator),
            "pose_pck_005": float((distance < 0.05).sum() / denominator),
            "pose_joint_recall": float(count / denominator),
            "pose_matched_joints": count,
        }


class IdentityMetric:
    def __init__(self) -> None:
        from insightface import model_zoo
        from insightface.utils import face_align

        root = ROOT / "models/antelopev2"
        self.face_align = face_align
        self.det = model_zoo.get_model(str(root / "scrfd_10g_bnkps.onnx"), providers=["CPUExecutionProvider"])
        self.recognition = model_zoo.get_model(str(root / "glintr100.onnx"), providers=["CPUExecutionProvider"])
        self.det.prepare(ctx_id=-1, input_size=(640, 640))
        self.recognition.prepare(ctx_id=-1)

    def embedding(self, image: Image.Image) -> np.ndarray | None:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        boxes, landmarks = self.det.detect(bgr)
        if len(boxes) == 0:
            return None
        index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        crop = self.face_align.norm_crop(bgr, landmark=landmarks[index])
        embedding = self.recognition.get_feat(crop).reshape(-1).astype(np.float32)
        return embedding / max(float(np.linalg.norm(embedding)), 1e-12)

    @staticmethod
    def cosine(source: np.ndarray | None, generated: np.ndarray | None) -> float | None:
        return float(source @ generated) if source is not None and generated is not None else None


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)


def image_metrics(image: Image.Image, target: Image.Image, source: Image.Image, target_mask: np.ndarray, source_mask: np.ndarray) -> dict[str, float]:
    generated = np.asarray(image, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    reference = np.asarray(source, dtype=np.float32)
    mask = target_mask > 0.5
    background = ~(mask | (source_mask > 0.5))
    difference = np.abs(generated - truth)
    mse = float(np.mean((generated - truth) ** 2))
    human_mae = float(difference[mask].mean()) if mask.any() else float("nan")
    background_mae = float(np.abs(generated - reference)[background].mean()) if background.any() else float("nan")
    human_composite = np.where(mask[..., None], generated, truth)
    if mask.any():
        y, x = np.where(mask)
        crop = np.s_[y.min():y.max() + 1, x.min():x.max() + 1]
        human_ssim = float(structural_similarity(human_composite[crop], truth[crop], channel_axis=2, data_range=255))
    else:
        human_ssim = float("nan")
    background_composite = np.where(background[..., None], generated, reference)
    return {
        "mae": float(difference.mean()),
        "psnr": float("inf") if mse == 0 else float(-10 * np.log10(mse / 255**2)),
        "ssim": float(structural_similarity(generated, truth, channel_axis=2, data_range=255)),
        "human_mae": human_mae,
        "human_ssim": human_ssim,
        "background_mae_source": background_mae,
        "background_ssim_source": float(structural_similarity(background_composite, reference, channel_axis=2, data_range=255)),
    }


def strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    side = 384
    canvas = Image.new("RGB", (side * len(images), side + 34), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((side, side), Image.Resampling.LANCZOS), (index * side, 34))
        draw.text((index * side + 5, 8), label, fill="white")
    return canvas


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    strips = args.output / "strips"
    strips.mkdir(exist_ok=True)
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()][:args.max_samples]
    train_appearances = {json.loads(line)["appearance"] for line in args.train_pairs.read_text().splitlines() if line.strip()}
    test_appearances = {item["appearance"] for item in pairs}
    if train_appearances & test_appearances:
        raise ValueError("evaluation contains appearances present in the training split")
    qwen_rows = json.loads((args.qwen21_results / "per_sample.json").read_text())
    qwen_by_stem = {row["image_stem"]: row for row in qwen_rows}
    deepgen_metrics = json.loads((args.deepgen_results / "eval_metrics.json").read_text())
    deepgen_by_index = {
        "deepgen_on": {row["pair_idx"]: row for row in deepgen_metrics["per_sample"]},
        "deepgen_off": {row["pair_idx"]: row for row in deepgen_metrics["per_sample_adapter_off"]},
    }
    if not args.allow_partial and len(qwen_rows) < len(pairs):
        raise ValueError("Qwen21 generation has fewer rows than the requested fixed pairs")
    pose = PoseMetric()
    identity = None if args.skip_identity else IdentityMetric()
    condition_cache: dict[str, dict] = {}
    rows = []
    for index, sample in enumerate(pairs):
        app = sample["appearance"]
        source_meta, target_meta = sample["source"], sample["target"]
        stem = f"{index:03d}_{app}_{source_meta['stem']}_{target_meta['stem']}"
        paths = {
            "deepgen_off": args.deepgen_results / "images" / f"{stem}_deepgen_off.png",
            "deepgen_on": args.deepgen_results / "images" / f"{stem}_deepgen_on.png",
            "qwen21_off": args.qwen21_results / "images" / f"{stem}_off.png",
            "qwen21_on": args.qwen21_results / "images" / f"{stem}_on.png",
        }
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            if args.allow_partial:
                continue
            raise FileNotFoundError(f"{stem}: missing {absent}")
        source = open_rgb(args.sampled_root / app / source_meta["rgb_1024"])
        target = open_rgb(args.sampled_root / app / target_meta["rgb_1024"])
        source_mask_path = args.assets_root / app / "mask_512" / source_meta.get("role", "source") / f"{source_meta['stem']}.png"
        target_mask_path = args.assets_root / app / "mask_512" / target_meta.get("role", "target") / f"{target_meta['stem']}.png"
        source_mask = np.asarray(Image.open(source_mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32) / 255
        target_mask = np.asarray(Image.open(target_mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32) / 255
        if app not in condition_cache:
            condition_cache[app] = torch.load(args.assets_root / app / "v6_conditions.pt", map_location="cpu", weights_only=False)
        conditions = condition_cache[app]
        target_points = conditions["kps_25_coords"][conditions["stem_to_idx"][target_meta["stem"]]].numpy()
        source_embedding = identity.embedding(source) if identity else None
        target_embedding = identity.embedding(target) if identity else None
        panels = [source, target]
        result = {"pair_idx": index, "image_stem": stem, "appearance": app, "arms": {}}
        if identity:
            result["source_target_identity_cosine"] = identity.cosine(source_embedding, target_embedding)
        for arm in ARMS:
            generated = open_rgb(paths[arm])
            panels.append(generated)
            measurements = image_metrics(generated, target, source, target_mask, source_mask)
            measurements.update(pose.compare(pose.detect(generated), target_points))
            if identity:
                generated_embedding = identity.embedding(generated)
                measurements["identity_cosine"] = identity.cosine(source_embedding, generated_embedding)
                measurements["identity_detected"] = measurements["identity_cosine"] is not None
                measurements["identity_target_cosine"] = identity.cosine(target_embedding, generated_embedding)
            result["arms"][arm] = measurements
        matching_qwen = qwen_by_stem[stem]
        if matching_qwen["pair_idx"] != index:
            raise ValueError(f"Qwen21 generation order differs from the fixed pairs at {index}")
        for key, value in matching_qwen["seconds"].items():
            result["arms"][f"qwen21_{key}"]["seconds"] = value
            result["arms"][f"qwen21_{key}"]["peak_cuda_gib"] = matching_qwen["peak_cuda_gib"][key]
        for arm in ("deepgen_off", "deepgen_on"):
            timing = deepgen_by_index[arm][index]
            if (timing["appearance"], timing["source_stem"], timing["target_stem"]) != (
                app, source_meta["stem"], target_meta["stem"]
            ):
                raise ValueError(f"DeepGen generation order differs from the fixed pairs at {index}")
            for key in ("seconds", "peak_cuda_gib"):
                if key in timing:
                    result["arms"][arm][key] = timing[key]
        strip(panels, ["Source", "Target", "DeepGen native", "DeepGen control", "Qwen21 native", "Qwen21 control"]).save(strips / f"{stem}.png")
        rows.append(result)
        (args.output / "per_sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{index + 1}/{len(pairs)} {stem}", flush=True)
    if not rows:
        raise ValueError("no complete four-arm samples were found")
    qwen_summary = json.loads((args.qwen21_results / "summary.json").read_text())
    summary = {
        "num_samples": len(rows), "pairs": str(args.pairs),
        "train_appearance_count": len(train_appearances),
        "test_appearance_count": len(test_appearances),
        "appearance_overlap_count": 0,
        "generation_settings": {
            "deepgen": deepgen_metrics.get("generation_settings", {"checkpoint": deepgen_metrics.get("adapter_checkpoint")}),
            "qwen21": {key: qwen_summary.get(key) for key in ("model", "model_revision", "checkpoint", "steps", "seed", "prompt", "kv_cache")},
        },
        "metric_definitions": {
            "bootstrap_ci95": "Percentile interval from 5000 paired-sample bootstrap resamples with seed 20260927; descriptive uncertainty for the fixed 50 pairs.",
            "pose_nme_diagonal": "Mean Euclidean error across jointly detected COCO17 body keypoints, divided by the 512x512 diagonal; lower is better.",
            "pose_pck_005": "Fraction of target-visible COCO17 joints detected within 0.05 of image diagonal; missing generated joints count as failures.",
            "pose_pck_002": "The same pose accuracy measure with a stricter 0.02 image-diagonal threshold.",
            "identity_cosine": "AntelopeV2 face embedding cosine against source; only pairs with detected source and generated faces are included.",
            "identity_target_cosine": "AntelopeV2 face embedding cosine against target, which shows the same person at the target pose; only jointly detected faces are included.",
            "human_mae": "RGB MAE against target inside target person mask (0-255).",
            "human_ssim": "SSIM in the target person bounding crop after filling outside-mask pixels from target.",
            "background_mae_source": "RGB MAE against source outside the union of source and target person masks (0-255).",
        },
        "arms": {},
    }
    source_target_identity = [row.get("source_target_identity_cosine") for row in rows]
    source_target_identity = [value for value in source_target_identity if value is not None]
    summary["source_target_identity_cosine"] = {
        "mean": float(np.mean(source_target_identity)) if source_target_identity else None,
        "count": len(source_target_identity),
    }
    for arm in ARMS:
        measurements = [row["arms"][arm] for row in rows]
        keys = set().union(*(entry.keys() for entry in measurements))
        summary["arms"][arm] = {}
        for key in sorted(keys):
            valid = [float(entry[key]) for entry in measurements if isinstance(entry.get(key), (bool, int, float)) and np.isfinite(entry[key])]
            summary["arms"][arm][key] = {"mean": float(np.mean(valid)) if valid else None, "count": len(valid)}
    comparisons = {
        "backbone_native_change": ("deepgen_off", "qwen21_off"),
        "deepgen_control_gain": ("deepgen_off", "deepgen_on"),
        "qwen21_control_gain": ("qwen21_off", "qwen21_on"),
        "controlled_model_change": ("deepgen_on", "qwen21_on"),
    }
    summary["paired_deltas_second_minus_first"] = {}
    for label, (first, second) in comparisons.items():
        delta = {}
        for key in ("pose_pck_002", "pose_pck_005", "pose_nme_diagonal", "identity_cosine", "identity_target_cosine", "human_mae", "human_ssim", "background_mae_source", "background_ssim_source", "psnr", "ssim", "seconds", "peak_cuda_gib"):
            values = []
            for row in rows:
                a, b = row["arms"][first].get(key), row["arms"][second].get(key)
                if isinstance(a, (bool, int, float)) and isinstance(b, (bool, int, float)) and np.isfinite(a) and np.isfinite(b):
                    values.append(float(b - a))
            if values:
                observed = np.asarray(values, dtype=np.float64)
                generator = np.random.default_rng(20260927)
                draws = observed[generator.integers(0, len(observed), size=(5000, len(observed)))].mean(axis=1)
                confidence = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
            else:
                observed = np.asarray([], dtype=np.float64)
                confidence = None
            delta[key] = {
                "mean": float(observed.mean()) if values else None,
                "median": float(np.median(observed)) if values else None,
                "bootstrap_ci95": confidence,
                "count": len(values),
            }
        summary["paired_deltas_second_minus_first"][label] = delta
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        keys = sorted(set().union(*(entry.keys() for row in rows for entry in row["arms"].values())))
        writer.writerow(["pair_idx", "image_stem", "source_target_identity_cosine", "arm", *keys])
        for row in rows:
            for arm, entry in row["arms"].items():
                writer.writerow([row["pair_idx"], row["image_stem"], row.get("source_target_identity_cosine"), arm, *(entry.get(key) for key in keys)])


if __name__ == "__main__":
    main()
