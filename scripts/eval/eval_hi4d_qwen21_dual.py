#!/usr/bin/env python3
"""Generate and score native / geometry / full Qwen21 Hi4D controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw
from torchvision.models import vgg16

from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset, _rgb, letterbox_points
from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.evaluation.dwpose_onnx.onnxpose import inference_pose
from src.pose_control.qwen21.action_prompts import PROMPT_VERSION, build_dual_action_prompts
from src.pose_control.qwen21.dual_adapter import Qwen21DualAdapter
from src.pose_control.qwen21.dual_inference import (
    DualGenerationInput, NativeGenerationInput, generation_inputs_from_sample,
    generate_controlled, generate_native,
)
from src.pose_control.qwen21.runtime import MODEL_REVISION, Qwen21IPERCache, dataset_image_to_pil
from train_hi4d_qwen21_dual import DUAL_TARGET_CACHE_NAMESPACE, ROOT


PART_TO_COCO = (0, 5, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "experiments/qwen21_hi4d_dual/action_prompt_v2/checkpoint_best.pt")
    p.add_argument("--pairs", type=Path, default=ROOT / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_test.jsonl")
    p.add_argument("--control-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_controls"))
    p.add_argument("--latent-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_latents"))
    p.add_argument("--output", type=Path, default=ROOT / "experiments/qwen21_hi4d_dual/action_prompt_v2/test_eval")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-metrics", action="store_true", help="Quick generation smoke test")
    p.add_argument("--reuse-images", action="store_true", help="Recompute metrics from existing generated images")
    p.add_argument("--resume-generation", action="store_true", help="Continue a matching interrupted evaluation from completed rows")
    p.add_argument("--prompt-matrix", action="store_true", help="Also run native-short and adapter-detailed prompt cross-arms")
    p.add_argument("--allow-legacy-checkpoint", action="store_true", help="Diagnostic use of a checkpoint trained with the old generic prompt")
    p.add_argument("--allow-val", action="store_true", help="Use only for development, never final test selection")
    return p.parse_args()


class DualPoseMetric:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        paths = ROOT / "models/DWPose"
        self.det = ort.InferenceSession(str(paths / "yolox_l.onnx"), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(str(paths / "dw-ll_ucoco_384.onnx"), sess_options=options,
                                         providers=["CPUExecutionProvider"])

    def detect(self, image: Image.Image, boxes_target: list[np.ndarray]):
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        boxes = inference_detector(self.det, bgr)
        if len(boxes) == 0:
            return [None, None], 0
        points, scores = inference_pose(self.pose, boxes, bgr)
        available = list(range(len(boxes)))
        assigned = []
        for target in boxes_target:
            if not available:
                assigned.append(None)
                continue
            center = (target[:2] + target[2:]) * 256
            best = min(available, key=lambda i: np.linalg.norm((boxes[i, :2] + boxes[i, 2:]) / 2 - center))
            available.remove(best)
            assigned.append((points[best, :17], scores[best, :17]))
        return assigned, len(boxes)


class BodyIdentityMetric:
    """Clothing/body appearance proxy; unavailable faces are not called IDs."""
    def __init__(self):
        model = vgg16(weights=None)
        model.load_state_dict(torch.load(
            Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth",
            map_location="cpu", weights_only=True))
        self.model = model.features.eval().requires_grad_(False)

    @torch.no_grad()
    def feature(self, image: Image.Image, mask: np.ndarray):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            return None
        rgb = np.asarray(image.convert("RGB")).copy()
        rgb[~mask] = 127
        crop = Image.fromarray(rgb[yy.min():yy.max() + 1, xx.min():xx.max() + 1]).resize((224, 224))
        value = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float()[None] / 255
        value = (value - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        feature = self.model(value).mean(dim=(2, 3)).squeeze(0)
        return F.normalize(feature, dim=0).numpy()


def _target_keypoints(path: str, width: int, height: int):
    with np.load(path) as data:
        xy = letterbox_points(data["xy_native"][:17], width, height)
        score = data["scores"][:17]
    return xy, score


def _pose_pck(detected, target):
    if detected is None:
        return None
    points, scores = detected
    xy, target_scores = target
    valid = target_scores > 0.3
    if not valid.any():
        return None
    correct = (np.linalg.norm(points - xy, axis=1) / np.hypot(512, 512) < 0.05) & (scores > 0.3) & valid
    return float(correct.sum() / valid.sum())


def _contact_error(detected, target, relations):
    if any(person is None for person in detected):
        return None
    rows = []
    valid = relations.valid_mask[0].numpy()
    for index in np.flatnonzero(valid):
        person_a, person_b = int(relations.src_person[0, index]), int(relations.dst_person[0, index])
        part_a, part_b = int(relations.src_part[0, index]), int(relations.dst_part[0, index])
        joint_a, joint_b = PART_TO_COCO[part_a], PART_TO_COCO[part_b]
        ga, sa = detected[person_a]
        gb, sb = detected[person_b]
        if sa[joint_a] <= .3 or sb[joint_b] <= .3:
            continue
        generated_distance = np.linalg.norm(ga[joint_a] - gb[joint_b])
        target_distance = np.linalg.norm(target[person_a][0][joint_a] - target[person_b][0][joint_b])
        rows.append(abs(generated_distance - target_distance) / np.hypot(512, 512))
    return float(np.mean(rows)) if rows else None


def _panel(images, labels):
    size = 320
    canvas = Image.new("RGB", (size * len(images), size + 32), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((size, size)), (index * size, 32))
        draw.text((index * size + 6, 8), label, fill="white")
    return canvas


def _metric_summary(rows, modes):
    result = {}
    for mode in modes:
        result[mode] = {}
        for field in ("pose_pck_a", "pose_pck_b", "contact_distance_error", "body_identity_a",
                      "body_identity_b", "background_mae", "identity_swap", "detected_person_count",
                      "assigned_person_count", "third_person_detected", "two_person_pose_evaluable"):
            values = [row["metrics"][mode][field] for row in rows if row["metrics"][mode][field] is not None]
            result[mode][field] = float(np.mean(values)) if values else None
            result[mode][field + "_n"] = len(values)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_manifest(args, modes, checkpoint_step):
    return {"checkpoint_sha256": _file_sha256(args.checkpoint),
            "checkpoint_step": int(checkpoint_step),
            "pairs_sha256": _file_sha256(args.pairs),
            "model_revision": MODEL_REVISION, "prompt_version": PROMPT_VERSION,
            "control_cache": str(args.control_cache.resolve()),
            "modes": modes, "seed": args.seed, "steps": args.steps,
            "max_samples": args.max_samples}


def main():
    args = arguments()
    if args.reuse_images and args.resume_generation:
        raise ValueError("--reuse-images and --resume-generation cannot be combined")
    if not args.allow_val and "test" not in args.pairs.name:
        raise ValueError("final evaluation requires the held-out test manifest; --allow-val is for development")
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)
    # Target RGB is loaded only after all generation calls for this row.
    dataset = Hi4DQwen21Dataset(args.pairs, args.control_cache, load_target=False)
    indices = range(min(args.max_samples, len(dataset)) if args.max_samples else len(dataset))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_prompt_version = saved.get("args", {}).get("prompt_version", "legacy_generic")
    if checkpoint_prompt_version != PROMPT_VERSION and not args.allow_legacy_checkpoint:
        raise ValueError("checkpoint was trained with a different prompt version; use --allow-legacy-checkpoint only for diagnostics")
    modes = ["off_detailed", "geometry_short", "full_short"]
    if args.prompt_matrix:
        modes.extend(("off_short", "full_detailed"))
    manifest_path = args.output / "generation_manifest.json"
    generation_manifest = _generation_manifest(args, modes, saved.get("step", 0))
    previous_rows = []
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != generation_manifest:
            raise ValueError("output directory belongs to a different checkpoint, dataset, prompt or sampling run")
    elif args.reuse_images or (args.resume_generation and (args.output / "per_sample.json").exists()):
        raise ValueError("cannot reuse or resume images without a matching generation manifest")
    else:
        manifest_path.write_text(json.dumps(generation_manifest, indent=2), encoding="utf-8")
    if args.reuse_images:
        previous_rows = json.loads((args.output / "per_sample.json").read_text(encoding="utf-8"))
        if len(previous_rows) != len(indices):
            raise ValueError("--reuse-images requires all samples to be completed")
    elif args.resume_generation and (args.output / "per_sample.json").exists():
        previous_rows = json.loads((args.output / "per_sample.json").read_text(encoding="utf-8"))
        if len(previous_rows) > len(indices):
            raise ValueError("completed rows exceed the requested sample count")
        indices = range(len(previous_rows), len(indices))
    if not args.reuse_images and len(indices) > 0:
        pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer.eval().requires_grad_(False)
        adapter = Qwen21DualAdapter().to(device="cuda", dtype=torch.float32)
        adapter.load_state_dict(saved["adapter_state_dict"])
        adapter.eval()
        cache = Qwen21IPERCache(args.latent_cache, prompt=DUAL_TARGET_CACHE_NAMESPACE)
    pose = None if args.skip_metrics else DualPoseMetric()
    identity = None if args.skip_metrics else BodyIdentityMetric()
    rows = previous_rows if args.resume_generation else []
    for index in indices:
        sample = dataset[index]
        row = sample["row"]
        detailed, short = build_dual_action_prompts(
            row["target_action"], (sample["source_mask_a"], sample["source_mask_b"]))
        if args.reuse_images:
            previous = previous_rows[index]
            if (previous["source"] != row["source_rgb"] or previous["target"] != row["target_rgb"]
                    or previous["prompts"] != {"native_detailed": detailed, "adapter_short": short}):
                raise ValueError("--reuse-images source, target or prompt changed")
        native_input, dual_input = generation_inputs_from_sample(sample, detailed, short)
        source = native_input.source_image
        generated = {}
        if args.reuse_images:
            for mode in modes:
                generated[mode] = Image.open(images_dir / f"{index:04d}_{mode}.png").convert("RGB")
        else:
            for mode in modes:
                if mode == "off_detailed":
                    image = generate_native(pipe, native_input, steps=args.steps, seed=args.seed)
                elif mode == "off_short":
                    image = generate_native(pipe, NativeGenerationInput(source, short),
                                            steps=args.steps, seed=args.seed)
                else:
                    controlled_input = (dual_input if mode != "full_detailed" else
                                        DualGenerationInput(source, detailed, dual_input.controls))
                    image = generate_controlled(pipe, adapter, cache, controlled_input,
                                                mode="geometry" if mode == "geometry_short" else "full",
                                                steps=args.steps, seed=args.seed)
                generated[mode] = image
                image.save(images_dir / f"{index:04d}_{mode}.png")
        target = dataset_image_to_pil(_rgb(row["target_rgb"]))
        _panel([source, target, *(generated[mode] for mode in modes)],
               ["Source", "Target (evaluation only)", *modes]).save(images_dir / f"{index:04d}_panel.png")
        item = {"index": index, "pair": row["pair"], "source": row["source_rgb"],
                "target": row["target_rgb"], "prompts": {"native_detailed": detailed,
                "adapter_short": short}, "oracle_target_geometry": True, "metrics": {}}
        if pose is not None and identity is not None:
            masks = [sample[f"human_mask_{side}"][0].numpy() > .5 for side in ("a", "b")]
            source_masks = [sample[f"source_mask_{side}"][0].numpy() > .5 for side in ("a", "b")]
            boxes = []
            for mask in masks:
                yy, xx = np.nonzero(mask)
                boxes.append(np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]) / 512)
            _, target_detection_count = pose.detect(target, boxes)
            item["target_detected_person_count"] = target_detection_count
            target_kps = [_target_keypoints(row[f"target_dwpose_{side}"], row["native_width"], row["native_height"])
                          for side in ("A", "B")]
            source_features = [identity.feature(source, mask) for mask in source_masks]
            background = ~(masks[0] | masks[1] | source_masks[0] | source_masks[1])
            source_rgb = np.asarray(source).astype(np.float32)
            for mode, image in generated.items():
                detected, detection_count = pose.detect(image, boxes)
                features = [identity.feature(image, mask) for mask in masks]
                similarity = np.array([[float(src @ dst) if src is not None and dst is not None else np.nan
                                        for dst in features] for src in source_features])
                swap = (bool(similarity[0, 1] + similarity[1, 0] > similarity[0, 0] + similarity[1, 1])
                        if detection_count == 2 and all(person is not None for person in detected)
                        and np.isfinite(similarity).all() else None)
                item["metrics"][mode] = {
                    "pose_pck_a": _pose_pck(detected[0], target_kps[0]),
                    "pose_pck_b": _pose_pck(detected[1], target_kps[1]),
                    "detected_person_count": detection_count,
                    "assigned_person_count": sum(person is not None for person in detected),
                    "third_person_detected": detection_count > 2,
                    "two_person_pose_evaluable": all(person is not None for person in detected),
                    "detection_note": ("fewer_than_two_detected_in_target_and_generated"
                                       if detection_count < 2 and target_detection_count < 2 else
                                       "fewer_than_two_detected_in_generated"
                                       if detection_count < 2 else
                                       "more_than_two_detected_in_generated"
                                       if detection_count > 2 else None),
                    "contact_distance_error": _contact_error(detected, target_kps, dual_input.controls["contact_relations"]),
                    "body_identity_a": float(similarity[0, 0]) if np.isfinite(similarity[0, 0]) else None,
                    "body_identity_b": float(similarity[1, 1]) if np.isfinite(similarity[1, 1]) else None,
                    "identity_swap": swap,
                    "background_mae": float(np.abs(np.asarray(image).astype(np.float32) - source_rgb)[background].mean()) if background.any() else None,
                }
        rows.append(item)
        (args.output / "per_sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Hi4D eval {index + 1}/{len(indices)}", flush=True)
    summary = {"checkpoint": str(args.checkpoint), "checkpoint_step": int(saved.get("step", 0)),
               "model_revision": MODEL_REVISION, "pairs": str(args.pairs), "count": len(rows),
               "prompt_version": PROMPT_VERSION, "checkpoint_prompt_version": checkpoint_prompt_version,
               "comparison_modes": modes, "native_has_target_geometry": False,
               "adapter_has_oracle_target_geometry": True,
               "seed": args.seed, "steps": args.steps,
               "mean": _metric_summary(rows, modes) if not args.skip_metrics else None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
