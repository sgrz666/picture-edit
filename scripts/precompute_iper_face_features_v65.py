#!/usr/bin/env python3
"""Precompute V6.5 iPER face identity, appearance, and geometry features.

This module deliberately keeps heavyweight model imports behind
``LocalFaceFeatureExtractor`` so its geometry/cache helpers remain CPU-testable.
No model is downloaded: both ArcFace and DINO must already exist locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

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


class FaceFeatureExtractor(Protocol):
    arcface_model_id: str
    arcface_revision: str
    arcface_weight_sha256: str
    dino_model_id: str
    dino_revision: str
    dino_weight_sha256: str

    def extract(
        self, tight_crop: np.ndarray, expanded_crop: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _as_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except Exception as exc:  # pragma: no cover - defensive message path
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


def project_smplx_face72_to_rgb_normalized(
    params: Mapping[str, Any],
) -> torch.Tensor:
    """Invert SMPLest-X crop coordinates into normalized original-image xy.

    ``joint_proj_model`` lives in the 12x16 output heatmap. SMPLest-X stores
    the raw camera used before its 192x256 body crop; this function reconstructs
    that crop and maps the 72 face joints (indices 65:137) back to RGB space.
    """

    if "joint_proj_model" not in params:
        raise ValueError("params is missing joint_proj_model")
    joints = _as_float_tensor(params["joint_proj_model"], name="joint_proj_model")
    if joints.ndim != 3 or joints.shape[1:] != (137, 2):
        raise ValueError("joint_proj_model must have shape [N,137,2]")
    frame_count = int(joints.shape[0])

    raw_keys = ("focal_length_xy_raw", "princpt_raw")
    missing = [key for key in raw_keys if key not in params]
    if missing:
        retargeted = bool(torch.as_tensor(params.get("retargeted_cam", False)).any())
        if retargeted:
            raise ValueError(
                "retargeted_cam=1 requires focal_length_xy_raw and princpt_raw "
                "to recover the raw camera crop"
            )
        raise ValueError(f"raw camera fields are required; missing {missing}")

    focal = _as_float_tensor(params["focal_length_xy_raw"], name="focal_length_xy_raw")
    principal = _as_float_tensor(params["princpt_raw"], name="princpt_raw")
    if focal.shape != (frame_count, 2) or principal.shape != (frame_count, 2):
        raise ValueError("raw camera tensors must have shape [N,2]")
    width = _frame_vector(params.get("width", 1024.0), frame_count, name="width")
    height = _frame_vector(params.get("height", 1024.0), frame_count, name="height")

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
    """Crop normalized xyxy through the pinned MCLD safe-crop adaptation."""

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
    """Compute a stable content fingerprint, excluding the fingerprint itself."""

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


def _extractor_metadata(extractor: FaceFeatureExtractor) -> dict[str, str]:
    return {
        "arcface_model_id": str(extractor.arcface_model_id),
        "arcface_revision": str(extractor.arcface_revision),
        "arcface_weight_sha256": str(extractor.arcface_weight_sha256),
        "dino_model_id": str(extractor.dino_model_id),
        "dino_revision": str(extractor.dino_revision),
        "dino_weight_sha256": str(extractor.dino_weight_sha256),
    }


def build_v65_face_cache(
    *,
    params: Mapping[str, Any],
    frame_stems: Sequence[str],
    image_loader: Callable[[str], np.ndarray],
    extractor: FaceFeatureExtractor,
    source_hashes: Mapping[str, str],
    appearance: str | None = None,
) -> dict[str, Any]:
    """Build the serializable per-frame V6.5 feature cache.

    Extraction is isolated per frame. A detector/model/image failure invalidates
    only that frame and leaves both appearance feature arrays exactly zero.
    """

    landmarks = project_smplx_face72_to_rgb_normalized(params)
    frame_count = int(landmarks.shape[0])
    stems = [str(stem) for stem in frame_stems]
    if len(stems) != frame_count or len(set(stems)) != frame_count:
        raise ValueError("frame_stems must be unique and match the params frame count")
    jaw = _as_float_tensor(params.get("jaw_pose"), name="jaw_pose")
    expression = _as_float_tensor(params.get("expression"), name="expression")
    if jaw.shape != (frame_count, 3) or expression.shape != (frame_count, 10):
        raise ValueError("jaw_pose/expression must have shapes [N,3]/[N,10]")

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
                arc_value, dino_value = extractor.extract(
                    tight, expanded, full_image=image
                )
            except TypeError:
                arc_value, dino_value = extractor.extract(tight, expanded)
            arc_value = torch.as_tensor(arc_value, dtype=torch.float32).detach().cpu()
            dino_value = torch.as_tensor(dino_value, dtype=torch.float32).detach().cpu()
            if arc_value.shape != (512,):
                raise ValueError("ArcFace extractor must return [512]")
            if dino_value.shape != (256, 1536):
                raise ValueError(
                    "DINO extractor must remove CLS and return exactly [256,1536]"
                )
            if not torch.isfinite(arc_value).all() or not torch.isfinite(dino_value).all():
                raise ValueError("extracted features must be finite")
            arcface[index] = arc_value.to(torch.float16)
            dino[index] = dino_value.to(torch.float16)
            valid[index] = True
        except Exception as exc:
            arcface[index].zero_()
            dino[index].zero_()
            failures.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})

    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "appearance": str(appearance) if appearance is not None else "unknown",
        **_extractor_metadata(extractor),
        "preprocess": {
            "smplx_face_slice": [65, 137],
            "smplx_input_wh": [192, 256],
            "smplx_output_heatmap_wh": [12, 16],
            "smplx_virtual_focal": [5000, 5000],
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
    """Validate and atomically replace the destination; never leave a partial cache."""

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
    """Offline-only InsightFace + local DINOv2-G/14 extraction backend."""

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
        # FaceAnalysis always resolves ``root/models/name``. The CLI contract is
        # the exact model directory, normally ``/X/models/antelopev2``; passing
        # its parent directly would silently add one too many path components.
        if insightface_root.parent.name == "models":
            model_directory = insightface_root
            root = insightface_root.parent.parent
            name = insightface_name or insightface_root.name
            if name != insightface_root.name:
                raise ValueError(
                    "--insightface-name must match the exact --insightface-root directory name"
                )
        elif insightface_name and (insightface_root / "models" / insightface_name).is_dir():
            root = insightface_root
            name = insightface_name
            model_directory = insightface_root / "models" / insightface_name
        else:
            raise ValueError(
                "--insightface-root must be the exact standard model directory "
                "'/X/models/<name>' (for example '/X/models/antelopev2'); "
                "alternatively pass the store root '/X' together with --insightface-name"
            )
        if not any(model_directory.glob("*.onnx")):
            raise FileNotFoundError(
                f"no InsightFace ONNX weights found in exact model directory: {model_directory}"
            )
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINO extraction requires a transformers installation with AutoModel support."
            ) from exc
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        try:
            self._arcface = InsightFaceArcFaceExtractor(
                name=name,
                root=str(root),
                providers=providers,
                ctx_id=0 if device.startswith("cuda") else -1,
                det_size=(640, 640),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "InsightFace extraction requires `pip install insightface onnxruntime-gpu` "
                "(or onnxruntime for CPU)."
            ) from exc
        self._processor = AutoImageProcessor.from_pretrained(
            str(dino_model), local_files_only=True
        )
        self._dino = AutoModel.from_pretrained(
            str(dino_model), local_files_only=True
        ).eval().to(device)
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

        full_bgr = (
            np.ascontiguousarray(full_image[..., ::-1])
            if full_image is not None
            else None
        )
        arcface = self._arcface.extract_bgr(
            np.ascontiguousarray(tight_crop[..., ::-1]),
            full_image_bgr=full_bgr,
        )
        inputs = self._processor(
            images=Image.fromarray(expanded_crop), return_tensors="pt"
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        output = self._dino(**inputs).last_hidden_state
        if tuple(output.shape[1:]) != (257, 1536):
            raise RuntimeError(
                f"expected DINOv2-G/14 output [B,257,1536], got {tuple(output.shape)}"
            )
        return arcface.cpu(), output[0, 1:].float().cpu()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute V6.5 iPER face features")
    parser.add_argument("--appearance", required=True)
    parser.add_argument(
        "--sampled-root",
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
    )
    parser.add_argument(
        "--assets-root",
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
    )
    parser.add_argument("--params-path")
    parser.add_argument("--frame-map")
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--dino-revision", default="local")
    parser.add_argument("--insightface-root", required=True)
    parser.add_argument("--insightface-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to read iPER images") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sampled_app = Path(args.sampled_root) / args.appearance
    assets_app = Path(args.assets_root) / args.appearance
    params_path = Path(args.params_path) if args.params_path else sampled_app / "smplestx" / "params.pt"
    frame_map_path = Path(args.frame_map) if args.frame_map else sampled_app / "smplestx" / "frame_map.json"
    destination = assets_app / "v65_face_features.pt"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
    if not params_path.exists() or not frame_map_path.exists():
        raise FileNotFoundError(
            f"missing iPER SMPLest-X inputs: params={params_path}, frame_map={frame_map_path}"
        )
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    with frame_map_path.open("r", encoding="utf-8") as handle:
        frame_map = json.load(handle)
    frames = frame_map.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame_map.json must contain a non-empty frames list")
    frame_by_stem = {str(frame["stem"]): frame for frame in frames}

    def image_loader(stem: str) -> np.ndarray:
        frame = frame_by_stem[stem]
        role = frame.get("role", "target")
        relative = frame.get("rgb_1024", f"rgb_1024/{role}/{stem}.png")
        path = sampled_app / relative
        if not path.exists():
            path = assets_app / "rgb_512" / role / f"{stem}.png"
        return _load_rgb(path)

    extractor = LocalFaceFeatureExtractor(
        insightface_root=args.insightface_root,
        insightface_name=args.insightface_name,
        dino_model=args.dino_model,
        dino_revision=args.dino_revision,
        device=args.device,
    )
    cache = build_v65_face_cache(
        params=params,
        frame_stems=[str(frame["stem"]) for frame in frames],
        image_loader=image_loader,
        extractor=extractor,
        source_hashes={
            "params.pt": sha256_path(params_path),
            "frame_map.json": sha256_path(frame_map_path),
        },
        appearance=args.appearance,
    )
    atomic_save_cache(cache, destination)
    successful = int(cache["valid"].sum())
    print(
        f"saved {destination} ({successful}/{len(frames)} valid frames), "
        f"fingerprint={cache['metadata']['cache_fingerprint']}"
    )


if __name__ == "__main__":
    main()
