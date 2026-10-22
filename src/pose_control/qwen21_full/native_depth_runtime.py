"""Pair-bound depth I/O, component checkpoints and Native training primitives."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from .depth_condition import canvas_identity, load_depth_record
from .native_body import NativeBodyConfig, Qwen21NativeBody
from .native_depth import NativeDepthConfig, Qwen21NativeDepth
from .native_geometry import Qwen21NativeGeometry
from .reasoning import AdapterReasoningConfig

ARCHITECTURE = "qwen21_native_depth_runtime_v1"


class NativeDepthRecords:
    def __init__(self, path, *, mode):
        if mode not in ("camera_z", "relative_png"):
            raise ValueError("Choose an explicit depth mode")
        self.mode, self.path = mode, Path(path)
        self.records = {}
        self.fingerprints = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("pair_id")
            if not isinstance(key, str) or not key or key in self.records:
                raise ValueError("Depth manifest requires unique nonempty pair_id")
            if row.get("kind") != mode or not isinstance(row.get("target_stem"), str):
                raise ValueError("Depth manifest kind/target_stem does not match this run")
            for name in ("depth_path", "valid_path"):
                value = Path(row[name])
                row[name] = str(value if value.is_absolute() else self.path.parent / value)
            row["source_id"] = key
            self.records[key] = row
        if not self.records:
            raise ValueError("Depth manifest is empty")

    def condition(self, sample):
        provenance = sample.get("pair_metadata", {})
        key = provenance.get("pair_id")
        if key not in self.records:
            raise ValueError(f"Depth manifest is missing pair {key!r}")
        row = self.records[key]
        if row["target_stem"] != provenance.get("target_stem"):
            raise ValueError("Depth target frame differs from the geometry pair")
        condition = load_depth_record(row, sample["target_canvas"])
        condition.validate(target_token_hw=sample["target_token_hw"],
                           expected_canvas_ids=(canvas_identity(sample["target_canvas"]),))
        canvas = sample["valid_image_mask"]
        if canvas.shape != condition.canvas_valid.shape[1:] or not torch.equal(canvas.bool(), condition.canvas_valid[0]):
            raise ValueError("Depth and RGB canvas validity differ")
        if not condition.available.all():
            raise ValueError("Enabled depth training requires actual valid depth; no zero-filled fallback")
        meta = condition.metadata[0]
        signature = tuple(meta.get(k) for k in ("source_sha256", "valid_sha256", "canvas_id", "camera_id",
                                               "units", "scale_status", "kind", "input_polarity"))
        if key in self.fingerprints and self.fingerprints[key] != signature:
            raise ValueError(f"Depth asset/provenance changed during this run: {key}")
        self.fingerprints[key] = signature
        return condition


class NativeDepthRuntime(nn.Module):
    def __init__(self, geometry, body, depth=None, *, train_geometry=False):
        super().__init__()
        if (not isinstance(geometry, Qwen21NativeGeometry) or not isinstance(body, Qwen21NativeBody)
                or (depth is not None and not isinstance(depth, Qwen21NativeDepth))):
            raise ValueError("Expected Native geometry/body and optional Native Depth")
        width = geometry.reasoner.config.hidden_dim
        if body.config.geometry_dim != width or (depth is not None and depth.config.geometry_dim != width):
            raise ValueError("Geometry, Depth and Native Body dimensions differ")
        self.geometry, self.body, self.depth = geometry, body, depth
        self.train_geometry = bool(train_geometry)
        self.geometry.requires_grad_(self.train_geometry)
        self.geometry.train(self.train_geometry)

    def train(self, mode=True):
        super().train(mode)
        self.geometry.train(mode and self.train_geometry)
        return self

    def encode(self, batch, *, target_token_hw, depth=None, depth_strength=1.):
        device = next(self.body.parameters()).device
        def value(key):
            return batch[key].to(device=device, dtype=torch.float32)
        bundle = self.geometry.condition_injector(
            normal_a=value("normal"), pose_heatmap_a=value("pose_heatmap"),
            part_onehot_a=value("part_onehot"), human_mask_a=value("human_mask"),
            smplx_global_a=value("smplx_global"), task_id=batch["task_id"].to(device=device, dtype=torch.long))
        features = self.geometry(bundle, target_token_hw=target_token_hw)
        if self.depth is None:
            if depth is not None:
                raise ValueError("Depth was supplied to an off runtime")
            return features
        return self.depth(features, None if depth is None else depth.to(device), strength=depth_strength)

    def checkpoint_payload(self, *, metadata):
        return {"architecture": ARCHITECTURE, "geometry_config": self.geometry.get_extra_state(),
                "body_config": self.body.config.to_dict(),
                "depth_config": None if self.depth is None else asdict(self.depth.config),
                "state_dict": self.state_dict(), "metadata": dict(metadata)}

    @classmethod
    def from_native_checkpoint(cls, saved, *, train_geometry=False):
        if not isinstance(saved, Mapping) or saved.get("architecture") != ARCHITECTURE:
            raise ValueError("Expected Native Depth runtime checkpoint")
        config = saved["geometry_config"]["reasoning_config"]
        body_config = dict(saved["body_config"])
        body_config["injection_layers"] = tuple(body_config["injection_layers"])
        with torch.device("cpu"):
            geometry = Qwen21NativeGeometry(reasoning_config=AdapterReasoningConfig(**config)).float()
            body = Qwen21NativeBody(NativeBodyConfig(**body_config)).float()
            depth = None if saved["depth_config"] is None else Qwen21NativeDepth(NativeDepthConfig(**saved["depth_config"])).float()
        result = cls(geometry, body, depth, train_geometry=train_geometry)
        result.load_native_weights(saved, include_depth=True)
        return result

    def load_native_weights(self, saved, *, include_depth):
        """Explicit warm start, never optimizer/RNG resume or silent architecture conversion."""
        if (not isinstance(saved, Mapping) or saved.get("architecture") != ARCHITECTURE
                or saved.get("geometry_config") != self.geometry.get_extra_state()
                or saved.get("body_config") != self.body.config.to_dict()):
            raise ValueError("Native checkpoint architecture/geometry/body config differs")
        if include_depth and saved.get("depth_config") != (None if self.depth is None else asdict(self.depth.config)):
            raise ValueError("Native checkpoint depth config differs")
        source = saved.get("state_dict")
        if not isinstance(source, Mapping):
            raise ValueError("Missing Native state_dict")
        prefixes = ("geometry.", "body.", "depth.") if include_depth else ("geometry.", "body.")
        expected = {k: v for k, v in self.state_dict().items() if k.startswith(prefixes)}
        selected = {k: v for k, v in source.items() if k.startswith(prefixes)}
        if selected.keys() != expected.keys():
            raise ValueError("Native checkpoint component keys differ")
        for key, value in selected.items():
            template = expected[key]
            if torch.is_tensor(template):
                if (not torch.is_tensor(value) or value.shape != template.shape
                        or value.dtype != template.dtype or value.is_meta
                        or (value.is_floating_point() and not torch.isfinite(value).all())):
                    raise ValueError(f"Invalid Native checkpoint tensor: {key}")
            elif value != template:
                raise ValueError(f"Native component schema/config differs: {key}")
        self.load_state_dict({**self.state_dict(), **selected}, strict=True)
        return {"included_depth": include_depth, "loaded_keys": len(selected), "optimizer_resumed": False}


def native_velocity_loss(prediction, target, coverage):
    if (prediction.shape != target.shape or prediction.ndim != 3
            or coverage.shape != prediction.shape[:2] or not torch.isfinite(coverage).all()
            or bool(((coverage < 0) | (coverage > 1)).any()) or not bool(coverage.sum() > 0)):
        raise ValueError("Flow loss requires matching [B,N,C] and nonempty 0..1 canvas coverage")
    squared = (prediction.float() - target.float()).square()
    return (squared * coverage[..., None]).sum() / (coverage.sum() * prediction.shape[-1])
