#!/usr/bin/env python3
"""Prepare a fixed, small real-data acceptance set for the full Qwen adapter.

This is an engineering acceptance set, not a training-coverage or quality claim.
Only A frames are passed to ArcFace/DINO. B contributes SMPL-X geometry only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.precompute_iper_face_features_v65 import (
    LocalFaceFeatureExtractor, _extractor_metadata, atomic_save_cache,
    compute_cache_fingerprint, compute_face_box, project_smplx_face72_to_rgb_normalized,
    safe_expanded_crop, sha256_path,
)
from src.pose_control.qwen21_full.data_runtime import SAMPLED, ASSETS, read_pairs, pair_metadata


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_full"))
    parser.add_argument("--sampled-root", type=Path, default=SAMPLED)
    parser.add_argument("--assets-root", type=Path, default=ASSETS)
    parser.add_argument("--train-pairs", type=Path, default=SAMPLED / "splits/dev_train_pairs.jsonl")
    parser.add_argument("--val-pairs", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_face/face_val_pairs.jsonl"))
    parser.add_argument("--cross-pairs", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_face/cross_person_8_pairs.jsonl"))
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--cross-count", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-face-short-side", type=float, default=64.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", type=Path, default=ROOT / "models/dinov2-giant")
    parser.add_argument("--insightface-root", type=Path, default=ROOT / "models/antelopev2")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.train_pairs, "val": args.val_pairs, "cross_person": args.cross_pairs}
    rows = {name: read_pairs(path) for name, path in paths.items()}
    train_apps = {row["appearance"] for row in rows["train"]}
    val_apps = {row["appearance"] for row in rows["val"]}
    if train_apps & val_apps:
        raise RuntimeError("Existing training/validation appearance split overlaps")
    # One frozen extractor, using exactly the previous V6.5 preprocessing.
    # Current ONNX Runtime expects CUDA 13, while this environment has CUDA 12.
    # Run its small detection/recognition models on CPU and DINO on the requested
    # torch device. No environment or existing DeepGen package is changed.
    extractor = LocalFaceFeatureExtractor(insightface_root=args.insightface_root,
                                         dino_model=args.dino_model, device="cpu")
    extractor._device = torch.device(args.device)
    extractor._dino.to(extractor._device)
    geometry = {}
    appearance_features = {}
    source_checks = {}
    audit = []
    exclusions = Counter()

    def get_geometry(app, stem):
        if app not in geometry:
            app_root = args.sampled_root / app
            params_path = app_root / "smplestx/params.pt"
            frame_path = app_root / "smplestx/frame_map.json"
            params = torch.load(params_path, map_location="cpu", weights_only=False)
            frames = json.loads(frame_path.read_text())["frames"]
            geometry[app] = {
                "params": params, "landmarks": project_smplx_face72_to_rgb_normalized(params),
                "indices": {frame["stem"]: i for i, frame in enumerate(frames)},
                "hashes": {"params.pt": sha256_path(params_path), "frame_map.json": sha256_path(frame_path)},
            }
        app_data = geometry[app]
        idx = app_data["indices"][stem]
        landmarks = app_data["landmarks"][idx].float()
        box, valid = compute_face_box(landmarks)
        jaw = torch.as_tensor(app_data["params"]["jaw_pose"][idx], dtype=torch.float32)
        expression = torch.as_tensor(app_data["params"]["expression"][idx], dtype=torch.float32)
        valid = valid and bool(torch.isfinite(jaw).all()) and bool(torch.isfinite(expression).all())
        return {"face_landmarks": landmarks, "face_boxes": box,
                "jaw_pose": jaw, "expression": expression, "geometry_valid": valid}

    def source_features(meta, geom):
        key = (meta["source_appearance"], meta["source_stem"])
        if key in source_checks:
            return source_checks[key]
        with Image.open(meta["source_rgb"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        boxes, landmarks5 = extractor._arcface.app.det_model.detect(np.ascontiguousarray(rgb[..., ::-1]), max_num=0)
        check = {"valid": False, "reason": "no_source_face_detected"}
        if len(boxes):
            detected_index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            box = boxes[detected_index]
            short = float(min(box[2] - box[0], box[3] - box[1]))
            check = {"valid": False, "bbox": box[:4].tolist(), "score": float(box[4]),
                     "short_side": short, "detected_faces": len(boxes)}
            if float(box[4]) < args.min_score:
                check["reason"] = "source_face_score_below_threshold"
            elif short < args.min_face_short_side:
                check["reason"] = "source_face_too_small"
            elif not geom["geometry_valid"]:
                check["reason"] = "invalid_source_face_geometry"
            else:
                try:
                    tight = safe_expanded_crop(rgb, geom["face_boxes"], scale=1.0)
                    expanded = safe_expanded_crop(rgb, geom["face_boxes"], scale=1.35)
                    try:
                        arc, dino = extractor.extract(tight, expanded)
                        check["arcface_preprocess"] = "legacy_tight_crop_redetection"
                    except RuntimeError as exc:
                        if "InsightFace found no face in the tight crop" not in str(exc):
                            raise
                        # The legacy tight crop can remove the detector's needed
                        # context. Reuse the already verified original A detection
                        # and its five alignment points; preserve ArcFace weights
                        # and exactly the original expanded DINO crop.
                        from insightface.utils import face_align
                        aligned = face_align.norm_crop(np.ascontiguousarray(rgb[..., ::-1]),
                                                       landmarks5[detected_index], image_size=112)
                        values = extractor._arcface.app.models["recognition"].get_feat(aligned)
                        arc = F.normalize(torch.as_tensor(values, dtype=torch.float32).reshape(-1), dim=0)
                        with torch.inference_mode():
                            inputs = extractor._processor(images=Image.fromarray(expanded), return_tensors="pt")
                            inputs = {key: value.to(extractor._device) for key, value in inputs.items()}
                            tokens = extractor._dino(**inputs).last_hidden_state
                            if tuple(tokens.shape[1:]) != (257, 1536):
                                raise RuntimeError(f"Unexpected DINO output {tuple(tokens.shape)}")
                            dino = tokens[0, 1:].float().cpu()
                        check["arcface_preprocess"] = "original_A_detection_five_point_alignment_112"
                        check["legacy_failure"] = str(exc)
                        check["alignment_landmarks5"] = landmarks5[detected_index].tolist()
                    if arc.shape != (512,) or dino.shape != (256, 1536):
                        raise ValueError("Unexpected ArcFace/DINO shape")
                    if not torch.isfinite(arc).all() or not torch.isfinite(dino).all():
                        raise ValueError("Non-finite face features")
                    appearance_features[key] = (arc.to(torch.float16), dino.to(torch.float16))
                    check.update(valid=True, reason="accepted")
                except Exception as exc:
                    check["reason"] = "source_feature_extraction_failed"
                    check["error"] = f"{type(exc).__name__}: {exc}"
        source_checks[key] = check
        return check

    def select(name, count):
        selected, used = [], set()
        for original_index, row in enumerate(rows[name]):
            meta = pair_metadata(row, args.sampled_root)
            if name != "cross_person" and meta["source_appearance"] in used:
                exclusions[f"{name}:appearance_already_selected"] += 1
                continue
            record = {"split": name, "original_index": original_index, **meta}
            try:
                if name == "cross_person" and not meta["cross_person"]:
                    raise ValueError("cross-person row has same identity")
                target = get_geometry(meta["target_appearance"], meta["target_stem"])
                if not target["geometry_valid"]:
                    raise ValueError("invalid_target_face_geometry")
                source = get_geometry(meta["source_appearance"], meta["source_stem"])
                check = source_features(meta, source)
                record["source_detection"] = check
                if not check["valid"]:
                    raise ValueError(check["reason"])
                # Catch absent B body geometry assets before freezing the set.
                role = row["target"].get("role", "target")
                for directory in ("normal_512", "mask_512", "part_512", "dwpose_512"):
                    asset = args.assets_root / meta["target_appearance"] / directory / role / f"{meta['target_stem']}.png"
                    if not asset.is_file():
                        raise FileNotFoundError(str(asset))
            except Exception as exc:
                record.update(selected=False, reason=f"{type(exc).__name__}: {exc}")
                exclusions[f"{name}:{record['reason']}"] += 1
                audit.append(record)
                continue
            selected.append({**row, "pair_id": f"{name}_{len(selected):03d}", "original_index": original_index})
            used.add(meta["source_appearance"])
            record.update(selected=True, reason="reliable_source_and_valid_target_geometry")
            audit.append(record)
            print(f"selected {name} {len(selected)}/{count}: {meta['pair_id']}", flush=True)
            if len(selected) == count:
                break
        if len(selected) != count:
            write_jsonl(args.out / "selection_audit.jsonl", audit)
            raise RuntimeError(f"Only {len(selected)} eligible {name} pairs; requested {count}")
        return selected

    selected = {"train": select("train", args.train_count), "val": select("val", args.val_count),
                "cross_person": select("cross_person", args.cross_count)}
    frames = defaultdict(set)
    source_keys = set()
    for group in selected.values():
        for row in group:
            meta = pair_metadata(row, args.sampled_root)
            for side in ("source", "target"):
                frames[meta[f"{side}_appearance"]].add(meta[f"{side}_stem"])
            source_keys.add((meta["source_appearance"], meta["source_stem"]))
    cache_fingerprints = {}
    for app, stems in sorted(frames.items()):
        stems = sorted(stems)
        records = [get_geometry(app, stem) for stem in stems]
        metadata = {"schema_version": 1, "appearance": app, **_extractor_metadata(extractor),
                    "source_hashes": geometry[app]["hashes"], "extraction_failures": [],
                    "identity_policy": "A-only; target-only frames have exact-zero identity and texture",
                    "geometry_valid": [bool(r["geometry_valid"]) for r in records],
                    "preprocess": {"face_box_expansion": 0.15, "dino_crop_scale": 1.35,
                                   "dino_grid": [16, 16], "smplx_face_slice": [65, 137],
                                   "arcface_fallback": "original_A_detection_five_point_alignment_112_if_tight_redetection_fails",
                                   "arcface_provider": "CPUExecutionProvider", "dino_device": args.device,
                                   "per_source": {stem: source_checks[(app, stem)] for stem in stems if (app, stem) in source_keys}}}
        cache = {"metadata": metadata, "frame_stems": stems, "stem_to_idx": {s: i for i, s in enumerate(stems)},
                 "arcface": torch.zeros(len(stems), 512, dtype=torch.float16),
                 "dino_patches": torch.zeros(len(stems), 256, 1536, dtype=torch.float16),
                 "valid": torch.zeros(len(stems), dtype=torch.bool),
                 "geometry_valid": torch.tensor(metadata["geometry_valid"], dtype=torch.bool)}
        for field in ("face_landmarks", "jaw_pose", "expression", "face_boxes"):
            cache[field] = torch.stack([record[field] for record in records]).to(torch.float32)
        for i, stem in enumerate(stems):
            if (app, stem) in source_keys:
                cache["arcface"][i], cache["dino_patches"][i] = appearance_features[(app, stem)]
                cache["valid"][i] = True
        metadata["cache_fingerprint"] = compute_cache_fingerprint(cache)
        cache_fingerprints[app] = metadata["cache_fingerprint"]
        atomic_save_cache(cache, args.out / "face_cache" / f"{app}.pt")
    for name, group in selected.items():
        write_jsonl(args.out / f"{name}_pairs.jsonl", group)
    write_jsonl(args.out / "selection_audit.jsonl", audit)
    manifest = {"purpose": "bounded engineering acceptance, not full-data training",
                "counts": {name: len(group) for name, group in selected.items()},
                "selection": "input order; one valid pair per appearance for train and val; first valid cross-person rows",
                "thresholds": {"source_face_score": args.min_score, "source_face_short_side_original_px": args.min_face_short_side},
                "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                "cache_fingerprints": cache_fingerprints,
                "exclusions": dict(exclusions), "source_feature_frames": len(source_keys),
                "all_cached_frames": sum(len(stems) for stems in frames.values()),
                "test_set_used": False, "target_identity_features_extracted": False}
    (args.out / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    preprocessing = {
        "legacy_issue": "At scale=1.0 and 1.35 the tight face detector can fail despite a valid original-image detection; checked scale=1.8/2.0/3.0 recovers the face.",
        "correction": "When tight-crop redetection fails, use original A SCRFD five points for standard 112-pixel alignment and the same ArcFace recognition weights.",
        "dino_unchanged": "DINOv2-G/14, 256 patches of 1536 dimensions; SMPL-X box expansion 0.15 then crop scale 1.35.",
        "records": {f"{app}/{stem}": source_checks[(app, stem)] for app, stem in sorted(source_keys)},
        "previous_failure_audit": "selection_audit_tight_crop_failure.jsonl",
        "onnx_provider": "CPUExecutionProvider (installed CUDA provider requires unavailable CUDA 13)",
    }
    (args.out / "preprocess_manifest.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
