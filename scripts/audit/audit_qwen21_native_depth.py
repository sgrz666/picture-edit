#!/usr/bin/env python3
"""CPU audit of official CHAMP active weights and Native 512-grid depth fusion."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from src.pose_control.qwen21_full.depth_condition import relative_depth_condition
from src.pose_control.qwen21_full.native_depth import NativeDepthConfig, Qwen21NativeDepth
from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures


def upstream_attention(source_root):
    """Execute only the reviewed class definitions, after verifying pinned source hashes."""
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
    from diffusers.models.attention import AdaLayerNorm, Attention, FeedForward
    from einops import rearrange, repeat
    manifest = json.loads((source_root.parent.parent / "upstream_sources.json").read_text())
    hashes = {r["upstream_path"]: r["sha256"] for r in manifest["files"]}
    namespace = dict(torch=torch, nn=nn, Optional=Optional, dataclass=dataclass, BaseOutput=BaseOutput,
                     ModelMixin=ModelMixin, ConfigMixin=ConfigMixin, register_to_config=register_to_config,
                     AdaLayerNorm=AdaLayerNorm, Attention=Attention, FeedForward=FeedForward,
                     rearrange=rearrange, repeat=repeat)
    for filename, names in (("models/attention.py", {"TemporalBasicTransformerBlock"}),
                            ("models/transformer_3d.py", {"Transformer3DModelOutput", "Transformer3DModel"})):
        path = source_root / filename
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[filename]:
            raise ValueError(f"Pinned CHAMP reference source changed: {filename}")
        tree = ast.parse(data, filename=str(path))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        if {node.name for node in classes} != names:
            raise ValueError("Expected reviewed CHAMP classes are missing")
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Transformer3DModel"](
        8, 32, 256, norm_num_groups=32, unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--champ-checkpoint", type=Path, required=True)
    p.add_argument("--champ-source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(23)
    state = torch.load(args.champ_checkpoint, map_location="cpu", weights_only=True)
    model = Qwen21NativeDepth()
    inheritance = model.load_champ_state_dict(state)
    reference = upstream_attention(args.champ_source_root)
    reference.load_state_dict({k[len("attentions.5."):]: v for k, v in state.items()
                               if k.startswith("attentions.5.")}, strict=True)
    value = torch.randn(1, 256, 4, 6)
    with torch.no_grad():
        expected = reference(value[:, :, None]).sample[:, :, 0]
        actual = model.encoder.attention(value)
    error = float((expected-actual).abs().max())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)
    h, w = 32, 64
    relative = torch.rand(1, 1, h, w)
    valid = torch.ones_like(relative, dtype=torch.bool)
    depth = relative_depth_condition(relative, valid, valid,
        [{"source_id": "audit", "camera_id": "synthetic-camera", "canvas_id": "synthetic-canvas",
          "units": "unknown", "scale_status": "unknown"}], polarity="near_one")
    features = NativeGeometryFeatures(torch.randn(1, 8, 512), torch.zeros(1, 8, 512), (2, 4), torch.tensor([[True, False]]))
    initial = model(features, depth)
    assert torch.equal(initial.geometry, features.geometry)
    initial.geometry.square().mean().backward()
    head_grad = float(model.projection.weight.grad.norm())
    assert head_grad > 0
    model.zero_grad(set_to_none=True)
    nn.init.normal_(model.projection.weight, std=.01)
    model(features, depth).geometry.square().mean().backward()
    encoder_grad = float(model.encoder.conv_in.weight.grad.norm())
    assert encoder_grad > 0
    saved = model.state_dict()
    restored = Qwen21NativeDepth()
    restored.load_state_dict(saved, strict=True)
    assert torch.equal(model(features, depth).geometry, restored(features, depth).geometry)
    result = {"schema": "qwen21_native_depth_cpu_audit_v1", "torch": torch.__version__,
              "checkpoint_sha256": hashlib.sha256(args.champ_checkpoint.read_bytes()).hexdigest(),
              "inheritance": inheritance, "spatial_attention_reference_max_abs": error,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_geometry_max_abs": float((initial.geometry-features.geometry).detach().abs().max()),
              "initial_depth_head_grad": head_grad, "active_encoder_grad": encoder_grad,
              "state_roundtrip_exact": True, "target_token_hw": [2, 4], "quality_claim": False}
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inheritance"}))


if __name__ == "__main__":
    main()
