"""Explicit depth semantics for the opt-in Native branch; no legacy loader fallback."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

SCHEMA = "qwen21_native_depth_condition_v1"


def _metadata(records, batch):
    if not isinstance(records, (list, tuple)) or len(records) != batch:
        raise ValueError("Depth provenance requires one metadata record per scene")
    result = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Depth metadata must be a mapping")
        for key in ("source_id", "camera_id", "canvas_id", "units", "scale_status"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Depth metadata requires explicit {key}")
        if record["units"] not in ("unknown", "metre", "model_unit"):
            raise ValueError("Unsupported depth units")
        if record["scale_status"] not in ("unknown", "model_scale", "calibrated"):
            raise ValueError("Unsupported depth scale_status")
        result.append(dict(record))
    return result


def _spatial(value, valid, canvas_valid):
    if (not torch.is_tensor(value) or value.ndim != 4 or not value.is_floating_point()
            or any(size < 1 for size in value.shape)):
        raise ValueError("Depth must be floating [B,P,H,W]")
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != value.shape
            or valid.device != value.device):
        raise ValueError("Depth valid must be bool on the same [B,P,H,W] grid")
    if (not torch.is_tensor(canvas_valid) or canvas_valid.dtype != torch.bool
            or canvas_valid.shape != (value.shape[0], 1, *value.shape[-2:])
            or canvas_valid.device != value.device):
        raise ValueError("canvas_valid must be bool [B,1,H,W] on the depth device")
    if bool((valid & ~canvas_valid).any()):
        raise ValueError("Depth valid extends outside the valid canvas")


@dataclass(frozen=True)
class DepthCondition:
    relative: torch.Tensor
    valid: torch.Tensor
    canvas_valid: torch.Tensor
    available: torch.Tensor
    metadata: tuple[dict, ...]
    camera_z: torch.Tensor | None = None

    def validate(self, *, target_token_hw=None, expected_canvas_ids=None):
        _spatial(self.relative, self.valid, self.canvas_valid)
        if self.relative.shape[1] != 1:
            raise ValueError("Encoded depth must have one composited scene channel")
        b, _, h, w = self.relative.shape
        if h % 16 or w % 16 or (h // 16 * (w // 16)) % 4:
            raise ValueError("Depth canvas must produce an exact Qwen grid in four-token slots")
        if target_token_hw is not None and tuple(target_token_hw) != (h // 16, w // 16):
            raise ValueError("Depth and Qwen target grid differ; no implicit resize")
        if (not torch.isfinite(self.relative).all() or bool(((self.relative < 0) | (self.relative > 1)).any())
                or bool((self.relative.masked_select(~self.valid) != 0).any())):
            raise ValueError("Relative depth must be finite 0..1 and zero outside depth valid")
        if (not torch.is_tensor(self.available) or self.available.dtype != torch.bool
                or self.available.shape != (b,) or self.available.device != self.relative.device
                or not torch.equal(self.available, self.valid.flatten(1).any(1))):
            raise ValueError("Depth available must match actual per-scene depth validity")
        records = _metadata(self.metadata, b)
        for record in records:
            if (record.get("schema") != SCHEMA or record.get("polarity") != "near_one"
                    or record.get("kind") not in ("camera_z", "relative_png")
                    or record.get("normalization_scope") != "scene"):
                raise ValueError("Depth schema/polarity/kind/scene normalization is inconsistent")
            metric = (record["kind"] == "camera_z" and record["units"] == "metre"
                      and record["scale_status"] == "calibrated")
            if record.get("metric") is not metric:
                raise ValueError("Depth metric claim requires camera Z, metres and calibration")
        if expected_canvas_ids is not None and tuple(r["canvas_id"] for r in records) != tuple(expected_canvas_ids):
            raise ValueError("Depth canvas provenance differs")
        if self.camera_z is not None:
            if (self.camera_z.shape != self.relative.shape or self.camera_z.device != self.relative.device
                    or not self.camera_z.is_floating_point() or not torch.isfinite(self.camera_z).all()
                    or bool((self.camera_z[self.valid] <= 0).any())
                    or bool((self.camera_z[~self.valid] != 0).any())):
                raise ValueError("Preserved camera Z must be finite, positive on valid, background zero")
        if any(r["kind"] == "camera_z" for r in records) != (self.camera_z is not None):
            raise ValueError("Camera depth must preserve its raw Z tensor")
        return self

    def to(self, device):
        return replace(self, relative=self.relative.to(device=device, dtype=torch.float32),
                       valid=self.valid.to(device), canvas_valid=self.canvas_valid.to(device),
                       available=self.available.to(device),
                       camera_z=None if self.camera_z is None else self.camera_z.to(device=device, dtype=torch.float32))


def camera_depth_condition(z, valid, canvas_valid, metadata):
    """Composite comparable people by nearest camera Z; normalize once per scene."""
    _spatial(z, valid, canvas_valid)
    if not torch.isfinite(z[valid]).all() or bool((z[valid] <= 0).any()):
        raise ValueError("Valid camera Z must be finite and positive")
    records = _metadata(metadata, z.shape[0])
    z = z.float()
    scene_valid = valid.any(1, keepdim=True)
    raw = torch.where(scene_valid, z.masked_fill(~valid, float("inf")).amin(1, keepdim=True), 0.)
    relative = torch.zeros_like(raw)
    for index, record in enumerate(records):
        values = z[index][valid[index]]
        lo = float(values.amin()) if values.numel() else 0.
        hi = float(values.amax()) if values.numel() else 0.
        normalized = (hi - raw[index]) / (hi - lo) if hi > lo + 1e-8 else torch.full_like(raw[index], .5)
        relative[index] = torch.where(scene_valid[index], normalized.clamp(0, 1), 0.)
        record.update(schema=SCHEMA, kind="camera_z", polarity="near_one", normalization_scope="scene",
                      bounds=[lo, hi], metric=record["units"] == "metre" and record["scale_status"] == "calibrated")
    return DepthCondition(relative, scene_valid, canvas_valid, scene_valid.flatten(1).any(1),
                          tuple(records), raw).validate()


def relative_depth_condition(depth, valid, canvas_valid, metadata, *, polarity):
    _spatial(depth, valid, canvas_valid)
    if depth.shape[1] != 1 or polarity not in ("near_one", "near_zero"):
        raise ValueError("Relative depth requires one channel and explicit near_one/near_zero polarity")
    if not torch.isfinite(depth[valid]).all() or bool(((depth[valid] < 0) | (depth[valid] > 1)).any()):
        raise ValueError("Valid relative depth must be finite in 0..1")
    records = _metadata(metadata, depth.shape[0])
    value = depth.float() if polarity == "near_one" else 1 - depth.float()
    value = torch.where(valid, value, 0.)
    for record in records:
        record.update(schema=SCHEMA, kind="relative_png", polarity="near_one", input_polarity=polarity,
                      normalization_scope="scene", metric=False)
    return DepthCondition(value, valid, canvas_valid, valid.flatten(1).any(1), tuple(records)).validate()


def canvas_identity(canvas: Mapping) -> str:
    facts = {k: list(canvas[k]) for k in ("original_hw", "resized_hw", "canvas_hw", "offset_xy")}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def _onto_canvas(value, valid, canvas):
    """Validity-normalized bilinear sampling, followed by the same declared padding."""
    original, resized, output = (tuple(int(i) for i in canvas[k])
                                  for k in ("original_hw", "resized_hw", "canvas_hw"))
    x, y = (int(i) for i in canvas["offset_xy"])
    if (len(original) != 2 or len(resized) != 2 or len(output) != 2
            or any(i <= 0 for i in (*original, *resized, *output))
            or x < 0 or y < 0 or x + resized[1] > output[1] or y + resized[0] > output[0]):
        raise ValueError("Invalid depth canvas dimensions/offset")
    if tuple(value.shape[-2:]) != original:
        raise ValueError("Depth asset grid differs from the target canvas original grid")
    sy, sx = resized[0] / original[0], resized[1] / original[1]
    if abs(sy - sx) > max(1 / original[0], 1 / original[1]) + 1e-8:
        raise ValueError("Depth canvas must preserve aspect ratio")
    clean = torch.where(valid, value, 0.).float()
    coverage = valid.float()
    if original != resized:
        clean = F.interpolate(clean, size=resized, mode="bilinear", align_corners=False)
        coverage = F.interpolate(coverage, size=resized, mode="bilinear", align_corners=False)
        clean = torch.where(coverage > 1e-8, clean / coverage.clamp_min(1e-8), 0.)
    sampled_valid = coverage > 1e-8
    depth = torch.zeros(1, 1, *output)
    mask = torch.zeros_like(depth, dtype=torch.bool)
    cv = mask.clone()
    depth[..., y:y + resized[0], x:x + resized[1]] = clean
    mask[..., y:y + resized[0], x:x + resized[1]] = sampled_valid
    cv[..., y:y + resized[0], x:x + resized[1]] = True
    return depth, mask, cv


def load_depth_record(record: Mapping, canvas: Mapping) -> DepthCondition:
    """Load exactly the declared asset kind; missing files never choose another codec."""
    if record.get("kind") not in ("camera_z", "relative_png"):
        raise ValueError("Depth record kind must be camera_z or relative_png")
    depth_path, valid_path = Path(record["depth_path"]), Path(record["valid_path"])
    for path in (depth_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if record["kind"] == "camera_z":
        array = np.load(depth_path, allow_pickle=False)
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Camera Z NPY must be floating point")
        array = array.squeeze()
    else:
        with Image.open(depth_path) as image:
            if image.format != "PNG" or image.mode not in ("L", "RGB", "RGBA"):
                raise ValueError("Relative PNG must be an 8bit grayscale-compatible image")
            rgb = np.asarray(image.convert("RGB"))
            if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
                raise ValueError("Relative depth image must have equal grayscale channels")
            array = rgb[..., 0].astype(np.float32) / 255
    with Image.open(valid_path) as image:
        validity = np.asarray(image.convert("L")).copy()
    if array.ndim != 2 or validity.shape != array.shape:
        raise ValueError("Depth and validity assets must share one H,W grid")
    value = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())[None, None]
    valid = torch.from_numpy(validity > 0)[None, None]
    if not torch.isfinite(value[valid]).all() or (record["kind"] == "camera_z" and bool((value[valid] <= 0).any())):
        raise ValueError("Depth asset contains invalid values on its declared support")
    asset_hw = tuple(value.shape[-2:])
    original_hw = tuple(int(x) for x in canvas['original_hw'])
    if 'asset_to_original' in record:
        # Historical iPER conditions were rendered at 512, RGB/camera at 1024.
        # Restore that explicitly declared frame grid before the one target canvas
        # transform. No inferred resize, translation, crop or second camera scaling.
        transform = np.asarray(record['asset_to_original'], dtype=np.float64)
        expected = np.diag([original_hw[1] / asset_hw[1], original_hw[0] / asset_hw[0], 1.])
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or not np.allclose(transform, expected, atol=1e-10, rtol=0):
            raise ValueError('asset_to_original must declare the exact full-frame asset/RGB scale, applied once')
        if asset_hw != original_hw:
            clean = F.interpolate(torch.where(valid, value, 0.), size=original_hw, mode='bilinear', align_corners=False)
            coverage = F.interpolate(valid.float(), size=original_hw, mode='bilinear', align_corners=False)
            value = torch.where(coverage > 1e-8, clean / coverage.clamp_min(1e-8), 0.)
            valid = coverage > 1e-8
    identity = canvas_identity(canvas)
    if "canvas_id" in record and record["canvas_id"] != identity:
        raise ValueError("Depth record canvas_id differs from target canvas")
    meta = dict(record)
    meta['asset_hw'] = list(asset_hw)
    meta.update(canvas_id=identity, source_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                valid_sha256=hashlib.sha256(valid_path.read_bytes()).hexdigest(),
                canvas={k: list(canvas[k]) for k in ("original_hw", "resized_hw", "canvas_hw", "offset_xy")})
    value, valid, cv = _onto_canvas(value, valid, canvas)
    if record["kind"] == "camera_z":
        return camera_depth_condition(value, valid, cv, [meta])
    return relative_depth_condition(value, valid, cv, [meta], polarity=record.get("polarity"))
