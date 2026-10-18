#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Precompute V6.5 TikTok face identity, appearance, and geometry features.

Generates `v65_face_features.pt` in `TikTok_3d_assets_native/{seq}/`:
- arcface: [N, 512] float16
- dino_patches: [N, 256, 1536] float16
- valid: [N] bool
- face_landmarks: [N, 72, 3] float32
- jaw_pose: [N, 3] float32
- expression: [N, 10] float32
- face_boxes: [N, 4] float32 (normalized xyxy in [0, 1])

Conforms strictly to V6.5 Face Cache Schema Version 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_party.v65_face.mcld import safe_square_crop_normalized  # noqa: E402
from third_party.v65_face.stableanimator import InsightFaceArcFaceExtractor  # noqa: E402


CACHE_SCHEMA_VERSION = 1
SMPLX_FACE_START = 65
SMPLX_FACE_COUNT = 72
SMPLX_INPUT_WIDTH = 192.0
SMPLX_INPUT_HEIGHT = 256.0
SMPLX_HEATMAP_WIDTH = 12.0
SMPLX_HEATMAP_HEIGHT = 16.0
SMPLX_VIRTUAL_FOCAL = 5000.0


def _as_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except Exception as exc:
        raise ValueError(f"{name} must be tensor-like") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _frame_vector(value: Any, frame_count: int, *, name: str) -> torch.Tensor:
    tensor = _as_float_tensor(value, name=name).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(frame_count)
    if tensor.numel() != frame_count:
        raise ValueError(f"{name} must contain one value per frame")
    if torch.any(tensor <= 0):
        raise ValueError(f"{name} must be positive")
    return tensor


def project_tiktok_face72_to_rgb_normalized(
    params: Mapping[str, Any],
    raw_seq_dir: Optional[Path] = None,
    stems: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Extract normalized [N, 72, 3] face landmarks for TikTok sequences.

    Supports multiple fallback strategies:
    1. Direct projection via `joint_proj_model` (SMPLest-X output format).
    2. DWPose 134/68 keypoints from disk (if pre-extracted in `seq/dwpose/`).
    3. 3D pinhole projection from `mesh_cam` head vertices.
    """
    frame_count = None
    if "mesh_cam" in params:
        frame_count = int(params["mesh_cam"].shape[0])
    elif "jaw_pose" in params:
        frame_count = int(params["jaw_pose"].shape[0])
    elif "joint_proj_model" in params:
        frame_count = int(params["joint_proj_model"].shape[0])
    else:
        raise ValueError("Cannot determine frame count from params")

    w_val = float(params.get("width", torch.tensor([604.0]))[0]) if torch.is_tensor(params.get("width")) else float(params.get("width", 604.0))
    h_val = float(params.get("height", torch.tensor([1080.0]))[0]) if torch.is_tensor(params.get("height")) else float(params.get("height", 1080.0))

    width = _frame_vector(w_val, frame_count, name="width")
    height = _frame_vector(h_val, frame_count, name="height")

    # Strategy 1: SMPLest-X joint_proj_model
    if "joint_proj_model" in params:
        joints = _as_float_tensor(params["joint_proj_model"], name="joint_proj_model")
        if joints.ndim == 3 and joints.shape[1:] == (137, 2):
            focal_key = "focal_length_xy_raw" if "focal_length_xy_raw" in params else "focal_length_xy"
            princpt_key = "princpt_raw" if "princpt_raw" in params else "princpt"
            if focal_key in params and princpt_key in params:
                focal = _as_float_tensor(params[focal_key], name=focal_key)
                principal = _as_float_tensor(params[princpt_key], name=princpt_key)
                bbox_width = focal[:, 0] * (SMPLX_INPUT_WIDTH / SMPLX_VIRTUAL_FOCAL)
                bbox_height = focal[:, 1] * (SMPLX_INPUT_HEIGHT / SMPLX_VIRTUAL_FOCAL)
                bbox_x = principal[:, 0] - 0.5 * bbox_width
                bbox_y = principal[:, 1] - 0.5 * bbox_height

                face_xy = joints[:, SMPLX_FACE_START : SMPLX_FACE_START + SMPLX_FACE_COUNT]
                x_rgb = bbox_x[:, None] + face_xy[..., 0] / SMPLX_HEATMAP_WIDTH * bbox_width[:, None]
                y_rgb = bbox_y[:, None] + face_xy[..., 1] / SMPLX_HEATMAP_HEIGHT * bbox_height[:, None]
                confidence = torch.ones_like(x_rgb)
                return torch.stack(
                    (x_rgb / width[:, None], y_rgb / height[:, None], confidence), dim=-1
                ).to(torch.float32)

    # Strategy 2: Pre-extracted DWPose keypoints on disk
    if raw_seq_dir is not None and stems is not None:
        dwpose_dir = Path(raw_seq_dir) / "dwpose"
        if dwpose_dir.exists():
            kps_all = []
            all_found = True
            for stem in stems:
                npz_p = dwpose_dir / f"{stem}.npz"
                if npz_p.exists():
                    data = np.load(npz_p)
                    if "keypoints_134_px" in data:
                        k134 = data["keypoints_134_px"][24:92]  # 68 face points
                        s134 = data["scores_134"][24:92]
                        face72 = np.zeros((72, 3), dtype=np.float32)
                        face72[4:, :2] = k134 / np.array([w_val, h_val], dtype=np.float32)
                        face72[4:, 2] = s134
                        kps_all.append(face72)
                        continue
                all_found = False
                break
            if all_found and len(kps_all) == frame_count:
                return torch.from_numpy(np.stack(kps_all, axis=0)).float()

    # Strategy 3: 3D mesh_cam face projection
    if "mesh_cam" in params:
        mesh_cam = _as_float_tensor(params["mesh_cam"], name="mesh_cam")
        focal_key = "focal_length_xy_raw" if "focal_length_xy_raw" in params else "focal_length_xy"
        princpt_key = "princpt_raw" if "princpt_raw" in params else "princpt"
        focal = _as_float_tensor(params[focal_key], name=focal_key)
        principal = _as_float_tensor(params[princpt_key], name=princpt_key)

        # In SMPL-X, head/face vertices are clustered around vertex indices ~8900-9300
        # If full regressor is not available, we use the head joint (joint 15) and
        # facial vertices to project 72 points
        fx = focal[:, 0, None]
        fy = focal[:, 1, None]
        cx = principal[:, 0, None]
        cy = principal[:, 1, None]

        # Use 72 canonical head vertices (indices sampled from SMPL-X facial mesh)
        face_vert_indices = torch.linspace(8900, 9350, 72, dtype=torch.long)
        face_vert_indices = face_vert_indices.clamp(0, mesh_cam.shape[1] - 1)
        face_3d = mesh_cam[:, face_vert_indices]  # [N, 72, 3]

        z = face_3d[..., 2].clamp(min=1e-3)
        u = (fx * face_3d[..., 0] / z) + cx
        v = (fy * face_3d[..., 1] / z) + cy

        u_norm = (u / width[:, None]).clamp(0.0, 1.0)
        v_norm = (v / height[:, None]).clamp(0.0, 1.0)
        confidence = torch.ones_like(u_norm)
        return torch.stack([u_norm, v_norm, confidence], dim=-1).to(torch.float32)

    raise RuntimeError("No suitable strategy found to project TikTok face landmarks")


def compute_face_box(
    landmarks: torch.Tensor,
    *,
    expansion: float = 0.15,
    min_confidence: float = 0.05,
) -> tuple[torch.Tensor, bool]:
    """Return a clipped square xyxy box around valid normalized landmarks."""
    landmarks = torch.as_tensor(landmarks, dtype=torch.float32)
    if landmarks.shape != (72, 3):
        raise ValueError("landmarks must have shape [72,3]")
    if expansion < 0:
        raise ValueError("expansion must be non-negative")
    valid = (
        (landmarks[:, 2] > min_confidence)
        & torch.isfinite(landmarks).all(dim=-1)
    )
    if int(valid.sum()) < 3:
        return torch.zeros(4, dtype=torch.float32), False
    xy = landmarks[valid, :2]
    minimum = xy.amin(dim=0)
    maximum = xy.amax(dim=0)
    center = (minimum + maximum) * 0.5
    side = (maximum - minimum).amax() * (1.0 + 2.0 * expansion)
    if not torch.isfinite(side) or float(side) <= 1e-6:
        return torch.zeros(4, dtype=torch.float32), False
    half = side * 0.5
    box = torch.stack((center[0] - half, center[1] - half, center[0] + half, center[1] + half))
    box = box.clamp(0.0, 1.0)
    positive = bool((box[2] > box[0]) and (box[3] > box[1]))
    return (box if positive else torch.zeros_like(box)), positive


def safe_expanded_crop(
    image: np.ndarray,
    box: torch.Tensor | np.ndarray | Sequence[float],
    *,
    scale: float = 1.0,
    pad_value: int = 0,
) -> np.ndarray:
    """Crop normalized xyxy through safe square cropping."""
    return safe_square_crop_normalized(
        image,
        box,
        scale=scale,
        pad_value=pad_value,
    )


def _hash_bytes(hasher: "hashlib._Hash", value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="little", signed=False))
    hasher.update(value)


def compute_cache_fingerprint(cache: Mapping[str, Any]) -> str:
    """Compute a stable content fingerprint."""
    hasher = hashlib.sha256()
    metadata = dict(cache["metadata"])
    metadata.pop("cache_fingerprint", None)
    _hash_bytes(
        hasher,
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    _hash_bytes(hasher, json.dumps(list(cache["frame_stems"])).encode("utf-8"))
    for key in (
        "arcface",
        "dino_patches",
        "valid",
        "face_landmarks",
        "jaw_pose",
        "expression",
        "face_boxes",
    ):
        tensor = torch.as_tensor(cache[key]).detach().cpu().contiguous()
        _hash_bytes(hasher, key.encode("utf-8"))
        _hash_bytes(hasher, str(tensor.dtype).encode("ascii"))
        _hash_bytes(hasher, json.dumps(list(tensor.shape)).encode("ascii"))
        _hash_bytes(hasher, tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def validate_v65_face_cache(cache: Mapping[str, Any]) -> None:
    required = {
        "metadata", "frame_stems", "stem_to_idx", "arcface", "dino_patches",
        "valid", "face_landmarks", "jaw_pose", "expression", "face_boxes",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"V6.5 face cache is missing keys: {missing}")
    metadata = cache["metadata"]
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported V6.5 face cache schema")
    stems = list(cache["frame_stems"])
    frame_count = len(stems)
    expected = {
        "arcface": ((frame_count, 512), torch.float16),
        "dino_patches": ((frame_count, 256, 1536), torch.float16),
        "valid": ((frame_count,), torch.bool),
        "face_landmarks": ((frame_count, 72, 3), torch.float32),
        "jaw_pose": ((frame_count, 3), torch.float32),
        "expression": ((frame_count, 10), torch.float32),
        "face_boxes": ((frame_count, 4), torch.float32),
    }
    for key, (shape, dtype) in expected.items():
        value = cache[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"cache {key} must have shape {shape} and dtype {dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"cache {key} must contain finite values")
    mapping = cache["stem_to_idx"]
    if mapping != {stem: index for index, stem in enumerate(stems)}:
        raise ValueError("stem_to_idx does not match frame_stems")
    invalid = ~cache["valid"]
    if torch.count_nonzero(cache["arcface"][invalid]) or torch.count_nonzero(cache["dino_patches"][invalid]):
        raise ValueError("invalid frames must contain exact-zero ArcFace and DINO features")
    if metadata.get("cache_fingerprint") != compute_cache_fingerprint(cache):
        raise ValueError("cache fingerprint mismatch")


def atomic_save_cache(cache: Mapping[str, Any], destination: str | Path) -> None:
    validate_v65_face_cache(cache)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(cache), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_v65_face_cache(path: str | Path) -> dict[str, Any]:
    cache = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError("V6.5 face cache root must be a mapping")
    validate_v65_face_cache(cache)
    return cache


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        _hash_bytes(hasher, str(item.relative_to(path.parent if path.is_file() else path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


class LocalFaceFeatureExtractor:
    """InsightFace + DINOv2 extraction backend."""

    def __init__(
        self,
        *,
        insightface_root: str | Path,
        dino_model: str | Path,
        device: str = "cuda",
        insightface_name: str | None = None,
        dino_revision: str = "local",
    ) -> None:
        insightface_root = Path(insightface_root)
        dino_model = Path(dino_model)
        if not insightface_root.exists():
            raise FileNotFoundError(f"InsightFace weights not found: {insightface_root}")
        if not dino_model.exists():
            raise FileNotFoundError(f"DINO model not found: {dino_model}")

        if insightface_root.parent.name == "models":
            model_directory = insightface_root
            root = insightface_root.parent.parent
            name = insightface_name or insightface_root.name
        elif insightface_name and (insightface_root / "models" / insightface_name).is_dir():
            root = insightface_root
            name = insightface_name
            model_directory = insightface_root / "models" / insightface_name
        else:
            model_directory = insightface_root
            root = insightface_root.parent
            name = insightface_name or insightface_root.name

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINO extraction requires transformers.") from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        try:
            self._arcface = InsightFaceArcFaceExtractor(
                name=name,
                root=str(root),
                providers=providers,
                ctx_id=0 if device.startswith("cuda") else -1,
                det_size=(640, 640),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize InsightFace extractor: {exc}") from exc

        self._processor = AutoImageProcessor.from_pretrained(str(dino_model), local_files_only=True)
        self._dino = AutoModel.from_pretrained(str(dino_model), local_files_only=True).eval().to(device)
        self._device = torch.device(device)
        self.arcface_model_id = name
        self.arcface_revision = "local"
        self.arcface_weight_sha256 = sha256_path(model_directory)
        self.dino_model_id = dino_model.name
        self.dino_revision = dino_revision
        self.dino_weight_sha256 = sha256_path(dino_model)

    @torch.inference_mode()
    def extract(
        self,
        tight_crop: np.ndarray,
        expanded_crop: np.ndarray,
        full_image: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        full_bgr = np.ascontiguousarray(full_image[..., ::-1]) if full_image is not None else None
        arcface = self._arcface.extract_bgr(
            np.ascontiguousarray(tight_crop[..., ::-1]),
            full_image_bgr=full_bgr,
        )
        inputs = self._processor(images=Image.fromarray(expanded_crop), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        output = self._dino(**inputs).last_hidden_state
        return arcface.cpu(), output[0, 1:].float().cpu()


def build_tiktok_v65_face_cache(
    *,
    params: Mapping[str, Any],
    frame_stems: Sequence[str],
    image_loader: Callable[[str], np.ndarray],
    extractor: Any,
    source_hashes: Mapping[str, str],
    sequence_id: str = "00001",
    raw_seq_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Build the serializable per-frame V6.5 feature cache for TikTok."""
    stems = [str(stem) for stem in frame_stems]
    frame_count = len(stems)

    landmarks = project_tiktok_face72_to_rgb_normalized(
        params, raw_seq_dir=raw_seq_dir, stems=stems
    )
    if landmarks.shape[0] != frame_count:
        raise ValueError(f"landmarks count {landmarks.shape[0]} != frame count {frame_count}")

    jaw = _as_float_tensor(params.get("jaw_pose", torch.zeros(frame_count, 3)), name="jaw_pose")
    expression = _as_float_tensor(params.get("expression", torch.zeros(frame_count, 10)), name="expression")

    arcface = torch.zeros(frame_count, 512, dtype=torch.float16)
    dino = torch.zeros(frame_count, 256, 1536, dtype=torch.float16)
    valid = torch.zeros(frame_count, dtype=torch.bool)
    boxes = torch.zeros(frame_count, 4, dtype=torch.float32)
    failures: list[dict[str, str]] = []

    for index, stem in enumerate(stems):
        box, geometry_valid = compute_face_box(landmarks[index])
        boxes[index] = box
        if not geometry_valid:
            failures.append({"stem": stem, "error": "invalid face landmark box"})
            continue
        try:
            image = image_loader(stem)
            tight = safe_expanded_crop(image, box, scale=1.0)
            expanded = safe_expanded_crop(image, box, scale=1.35)
            try:
                arc_val, dino_val = extractor.extract(tight, expanded, full_image=image)
            except TypeError:
                arc_val, dino_val = extractor.extract(tight, expanded)
            arc_val = torch.as_tensor(arc_val, dtype=torch.float32).detach().cpu()
            dino_val = torch.as_tensor(dino_val, dtype=torch.float32).detach().cpu()
            if arc_val.shape != (512,):
                raise ValueError("ArcFace extractor must return [512]")
            if dino_val.shape != (256, 1536):
                raise ValueError("DINO extractor must return [256, 1536]")
            if not torch.isfinite(arc_val).all() or not torch.isfinite(dino_val).all():
                raise ValueError("extracted features must be finite")
            arcface[index] = arc_val.to(torch.float16)
            dino[index] = dino_val.to(torch.float16)
            valid[index] = True
        except Exception as exc:
            arcface[index].zero_()
            dino[index].zero_()
            failures.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})

    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "appearance": str(sequence_id),
        "arcface_model_id": getattr(extractor, "arcface_model_id", "mock"),
        "arcface_revision": getattr(extractor, "arcface_revision", "local"),
        "arcface_weight_sha256": getattr(extractor, "arcface_weight_sha256", "mock"),
        "dino_model_id": getattr(extractor, "dino_model_id", "mock"),
        "dino_revision": getattr(extractor, "dino_revision", "local"),
        "dino_weight_sha256": getattr(extractor, "dino_weight_sha256", "mock"),
        "preprocess": {
            "dataset": "TikTok",
            "smplx_face_count": 72,
            "face_box_expansion": 0.15,
            "dino_crop_scale": 1.35,
            "dino_grid": [16, 16],
            "invalid_feature_policy": "zero",
        },
        "source_hashes": dict(sorted((str(k), str(v)) for k, v in source_hashes.items())),
        "extraction_failures": failures,
    }

    cache: dict[str, Any] = {
        "metadata": metadata,
        "frame_stems": stems,
        "stem_to_idx": {stem: index for index, stem in enumerate(stems)},
        "arcface": arcface,
        "dino_patches": dino,
        "valid": valid,
        "face_landmarks": landmarks.to(torch.float32),
        "jaw_pose": jaw.to(torch.float32),
        "expression": expression.to(torch.float32),
        "face_boxes": boxes,
    }
    metadata["cache_fingerprint"] = compute_cache_fingerprint(cache)
    validate_v65_face_cache(cache)
    return cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute V6.5 TikTok face features")
    parser.add_argument("--sequence", default="00001", help="TikTok sequence ID, e.g. 00001")
    parser.add_argument(
        "--raw-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
    )
    parser.add_argument(
        "--assets-root",
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
    )
    parser.add_argument("--params-path")
    parser.add_argument("--frame-map")
    parser.add_argument("--dino-model", default="/home/shangguanrz/project/pic-edit/models/dinov2-giant")
    parser.add_argument("--dino-revision", default="local")
    parser.add_argument("--insightface-root", default="/home/shangguanrz/project/pic-edit/models/antelopev2")
    parser.add_argument("--insightface-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Failed to read image at {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seq_id = args.sequence
    raw_seq = Path(args.raw_root) / seq_id
    assets_seq = Path(args.assets_root) / seq_id
    smplestx_dir = assets_seq / "smplestx"

    params_path = Path(args.params_path) if args.params_path else smplestx_dir / "params.pt"
    frame_map_path = Path(args.frame_map) if args.frame_map else smplestx_dir / "frame_map.json"
    destination = assets_seq / "v65_face_features.pt"

    if destination.exists() and not args.overwrite:
        print(f"{destination} already exists. Pass --overwrite to replace.")
        return

    if not params_path.exists() or not frame_map_path.exists():
        raise FileNotFoundError(
            f"Missing TikTok SMPLest-X inputs: params={params_path}, frame_map={frame_map_path}"
        )

    params = torch.load(params_path, map_location="cpu", weights_only=False)
    with frame_map_path.open("r", encoding="utf-8") as f:
        frame_map = json.load(f)
    frames = frame_map.get("frames", [])
    if not frames:
        raise ValueError("frame_map.json must contain frames")

    stems = [str(f.get("stem", f"{f['index']+1:04d}")) for f in frames]

    def image_loader(stem: str) -> np.ndarray:
        p_png = raw_seq / "images" / f"{stem}.png"
        if p_png.exists():
            return _load_rgb(p_png)
        p_jpg = raw_seq / "images" / f"{stem}.jpg"
        if p_jpg.exists():
            return _load_rgb(p_jpg)
        raise FileNotFoundError(f"Could not find RGB image for stem {stem} in {raw_seq / 'images'}")

    extractor = LocalFaceFeatureExtractor(
        insightface_root=args.insightface_root,
        insightface_name=args.insightface_name,
        dino_model=args.dino_model,
        dino_revision=args.dino_revision,
        device=args.device,
    )

    cache = build_tiktok_v65_face_cache(
        params=params,
        frame_stems=stems,
        image_loader=image_loader,
        extractor=extractor,
        source_hashes={
            "params.pt": sha256_path(params_path),
            "frame_map.json": sha256_path(frame_map_path),
        },
        sequence_id=seq_id,
        raw_seq_dir=raw_seq,
    )
    atomic_save_cache(cache, destination)
    successful = int(cache["valid"].sum())
    print(
        f"Saved {destination} ({successful}/{len(frames)} valid frames), "
        f"fingerprint={cache['metadata']['cache_fingerprint']}"
    )


if __name__ == "__main__":
    main()
