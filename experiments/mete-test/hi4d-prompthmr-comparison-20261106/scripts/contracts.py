"""Small, NumPy-only input and coordinate contracts for the Hi4D pilot."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


_REQUIRED = ("id", "image", "image_sha256", "size", "K")
_OPTIONAL = ("split", "category", "camera_id", "frame_id")


def _image_size(value):
    size = np.asarray(value, dtype=np.float64)
    if (size.shape != (2,) or not np.isfinite(size).all()
            or (size <= 0).any() or (size != np.floor(size)).any()):
        raise ValueError("size must be positive integer [width, height]")
    return size


def _intrinsics(value):
    K = np.asarray(value, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("K must be finite 3x3 intrinsics with positive focal lengths")
    return K


def fit_record(row):
    """Whitelist RGB/calibration fields; no GT or other fields reach fitting."""
    for key in _REQUIRED:
        if key not in row:
            raise ValueError(f"missing required field: {key}")
    for key in ("id", "image"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    digest = row["image_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
    _image_size(row["size"])
    _intrinsics(row["K"])
    return {key: row[key] for key in (*_REQUIRED, *_OPTIONAL) if key in row}


def validate_boxes(boxes, image_size):
    """Require exactly two finite, positive-area xyxy boxes inside the image."""
    width, height = _image_size(image_size)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape != (2, 4) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be a finite (2, 4) xyxy array")
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("boxes must have x2>x1 and y2>y1")
    if (boxes < 0).any() or (boxes > [width, height, width, height]).any():
        raise ValueError("boxes must lie inside image bounds; clipping is forbidden")
    return boxes


def project(xyz, K):
    """Project camera-space (..., 3) points into original-image pixels."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim < 1 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite (..., 3) array")
    if (xyz[..., 2] <= 0).any():
        raise ValueError("xyz depth Z must be positive")
    K = _intrinsics(K)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        homogeneous = xyz @ K.T
        xy = homogeneous[..., :2] / homogeneous[..., 2:3]
    if not np.isfinite(xy).all():
        raise ValueError("projection must produce finite pixel coordinates")
    return xy


def model_to_original(xy, scale, offset):
    """Invert model_xy = original_xy * scale + [pad_left, pad_top]."""
    xy = np.asarray(xy, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy must be a finite (..., 2) array")
    if scale.shape != () or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("offset must be finite [pad_left, pad_top]")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        original = (xy - offset) / scale
    if not np.isfinite(original).all():
        raise ValueError("inverse transform must produce finite coordinates")
    return original


def write_json_exclusive(path, payload):
    """Serialize strictly before creating a new file; never overwrite output."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")


def safe_child(root, *components):
    """Resolve a child path and reject unsafe names or symlinks escaping root."""
    if not components:
        raise ValueError("at least one child component is required")
    for component in components:
        if (not isinstance(component, str) or component in ("", ".", "..")
                or any(character in component for character in "/\\:\x00")
                or Path(component).is_absolute()):
            raise ValueError(f"unsafe child component: {component!r}")
    root = Path(root).resolve()
    child = root.joinpath(*components).resolve()
    if child == root or not child.is_relative_to(root):
        raise ValueError("resolved child must stay below root")
    return child


def sha256_file(path):
    """Hash file bytes in bounded-memory chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
