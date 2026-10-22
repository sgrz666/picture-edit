"""CHAMP-style, checkpoint-audited depth residual on the Native geometry grid."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from third_party.champ_guidance.spatial_attention import ChampSpatialAttention
from .depth_condition import DepthCondition
from .native_geometry import NativeGeometryFeatures


@dataclass(frozen=True)
class NativeDepthConfig:
    geometry_dim: int = 512
    attention_scale: int = 16

    def __post_init__(self):
        if type(self.geometry_dim) is not int or self.geometry_dim <= 0:
            raise ValueError("Depth geometry_dim must be positive")
        if type(self.attention_scale) is not int or self.attention_scale not in (8, 16):
            raise ValueError("Depth attention_scale must be 8 (upstream) or 16 (Qwen adapted)")


class ChampDepthEncoder(nn.Module):
    def __init__(self, attention_scale=16):
        super().__init__()
        self.attention_scale = attention_scale
        self.conv_in = nn.Conv2d(2, 16, 3, padding=1)
        blocks = []
        for a, b in zip((16, 32, 96), (32, 96, 256)):
            blocks.extend((nn.Conv2d(a, a, 3, padding=1), nn.Conv2d(a, b, 3, stride=2, padding=1)))
        self.blocks = nn.ModuleList(blocks)
        self.attention = ChampSpatialAttention()
        self.downsample = nn.Conv2d(256, 256, 3, stride=2, padding=1)

    def forward(self, relative, valid):
        value = F.silu(self.conv_in(torch.cat((relative, valid.float()), dim=1)))
        for block in self.blocks:
            value = F.silu(block(value))
        if self.attention_scale == 16:
            value = F.silu(self.downsample(value))
        support = F.avg_pool2d(valid.float(), self.attention_scale).flatten(1) > 0
        value = self.attention(value, support)
        if self.attention_scale == 8:
            value = F.silu(self.downsample(value))
        return value


class _ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, delta):
        return torch.where(delta == 0, value, value + delta)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class Qwen21NativeDepth(nn.Module):
    def __init__(self, config: NativeDepthConfig | None = None):
        super().__init__()
        self.config = config or NativeDepthConfig()
        if not isinstance(self.config, NativeDepthConfig):
            raise ValueError("Expected NativeDepthConfig")
        self.encoder = ChampDepthEncoder(self.config.attention_scale)
        self.projection = nn.Conv2d(256, self.config.geometry_dim, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.inheritance_report = {}

    def get_extra_state(self):
        return {"schema": "qwen21_native_depth_v1", "config": asdict(self.config)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Native Depth schema/config differs")

    def _require_fp32(self):
        if any(p.dtype != torch.float32 or p.is_meta for p in self.parameters()):
            raise ValueError("Native Depth parameters must be materialized FP32")

    def forward(self, features: NativeGeometryFeatures, depth: DepthCondition | None, *, enabled=True, strength=1.):
        self._require_fp32()
        if type(enabled) is not bool or type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError("Depth requires bool enabled and finite nonnegative strength")
        if not isinstance(features, NativeGeometryFeatures):
            raise ValueError("Expected NativeGeometryFeatures")
        value, grid = features.geometry, features.target_token_hw
        if (not isinstance(grid, tuple) or len(grid) != 2
                or any(type(size) is not int or size <= 0 for size in grid) or math.prod(grid) % 4):
            raise ValueError("Native Depth requires a positive Qwen target grid")
        if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
                or not torch.isfinite(value).all() or value.shape[1:] != (math.prod(grid), self.config.geometry_dim)):
            raise ValueError("Depth fusion requires finite geometry on its exact target grid")
        if not enabled or strength == 0:
            return features
        if not isinstance(depth, DepthCondition):
            raise ValueError("Enabled Native Depth requires typed DepthCondition")
        depth.validate(target_token_hw=grid)
        if depth.relative.shape[0] != value.shape[0] or depth.relative.device != value.device:
            raise ValueError("Depth and geometry batch/device differ")
        active = depth.available.nonzero(as_tuple=True)[0]
        if not active.numel():
            return features
        with torch.autocast(device_type=value.device.type, enabled=False):
            encoded = self.encoder(depth.relative[active].float(), depth.valid[active])
            if tuple(encoded.shape[-2:]) != grid:
                raise ValueError("Depth encoder produced the wrong target grid")
            residual = self.projection(encoded).flatten(2).transpose(1, 2)
            support = F.avg_pool2d(depth.valid[active].float(), 16).flatten(1) > 0
            residual = residual * support[..., None] * strength
            delta = torch.zeros_like(value, dtype=torch.float32).index_copy(0, active, residual)
        return replace(features, geometry=_ResidualAdd.apply(value, delta.to(value.dtype)))

    def load_champ_state_dict(self, state: Mapping):
        """Atomic active-path import; all unconsumed keys must be recognized upstream keys."""
        self._require_fp32()
        if not isinstance(state, Mapping) or any(not isinstance(k, str) for k in state):
            raise ValueError("Expected a CHAMP tensor state dict")
        expected = {}
        for name, value in self.encoder.state_dict().items():
            if name.startswith("downsample."):
                continue
            old = "attentions.5." + name[len("attention."):] if name.startswith("attention.") else name
            expected[old] = (name, tuple(value.shape))
        expected["conv_in.weight"] = ("conv_in.weight", (16, 3, 3, 3))
        ignored = {"conv_out.weight": (320, 256, 3, 3), "conv_out.bias": (320,)}
        for index, channels in enumerate((16, 16, 32, 32, 96, 96)):
            for suffix in ("weight", "bias"):
                ignored[f"norms.{index}.{suffix}"] = (channels,)
        with torch.device("meta"):
            for prefix, channels in [(f"attentions.{i}.", c) for i, c in enumerate((16, 32, 32, 96, 96))] + [("guidance_attention.", 256)]:
                module = ChampSpatialAttention(channels, norm_num_groups=1 if channels == 16 else 32)
                ignored.update({prefix+k: tuple(v.shape) for k, v in module.state_dict().items()})
        missing = expected.keys() - state.keys()
        extra = state.keys() - expected.keys() - ignored.keys()
        if missing or extra:
            raise ValueError(f"CHAMP keys differ: missing={sorted(missing)}, unknown={sorted(extra)}")
        selected = {}
        for key, value in state.items():
            shape = expected[key][1] if key in expected else ignored[key]
            if (not torch.is_tensor(value) or tuple(value.shape) != shape or not value.is_floating_point()
                    or value.is_meta or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid CHAMP tensor: {key}")
            if key in expected:
                selected[expected[key][0]] = value.detach().float().clone()
        w = selected["conv_in.weight"]
        selected["conv_in.weight"] = torch.cat((w.sum(1, keepdim=True), torch.zeros_like(w[:, :1])), dim=1)
        self.encoder.load_state_dict({**self.encoder.state_dict(), **selected}, strict=True)
        self.inheritance_report = {
            "upstream_commit": "4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140",
            "loaded_tensor_count": len(selected), "source_elements": sum(state[k].numel() for k in expected),
            "loaded_elements": sum(v.numel() for v in selected.values()),
            "ignored_keys": sorted(state.keys() - expected.keys()), "discarded_sd_output_head": True,
            "attention_relocated": self.config.attention_scale == 16,
            "input_conversion": "sum RGB kernels for depth; valid kernel zero",
        }
        return dict(self.inheritance_report)
