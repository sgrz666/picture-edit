"""Lightweight Full-derived control that reads the current Qwen target hidden."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .native_geometry import NativeGeometryFeatures


@dataclass(frozen=True)
class NativeBodyConfig:
    hidden_dim: int = 4096
    geometry_dim: int = 512
    width: int = 512
    num_layers: int = 32
    num_groups: int = 6
    injection_layers: tuple[int, ...] = tuple(range(32))

    def __post_init__(self):
        for name in ("hidden_dim", "geometry_dim", "width", "num_layers", "num_groups"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_groups > self.num_layers:
            raise ValueError("num_groups cannot exceed num_layers")
        if (not isinstance(self.injection_layers, tuple) or not self.injection_layers
                or len(set(self.injection_layers)) != len(self.injection_layers)
                or any(type(i) is not int or not 0 <= i < self.num_layers for i in self.injection_layers)):
            raise ValueError("injection_layers must contain unique valid layer indices")

    def to_dict(self):
        values = asdict(self)
        values["injection_layers"] = list(self.injection_layers)
        return values


@dataclass(frozen=True)
class PreparedNativeGeometry:
    tokens: torch.Tensor
    target_token_hw: tuple[int, int]


class _IdentitySafeAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, delta):
        # Preserve the original bits (including -0) while the output head is zero.
        return torch.where(delta == 0, hidden, hidden + delta)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class Qwen21NativeBody(nn.Module):
    """Shared 512-wide fusion with six independent zero output heads.

    Geometry is prepared once per transformer.forward, then each layer reads its
    own target hidden. Trainable storage/compute is FP32; BF16 hidden is allowed.
    Body geometry is consumed here; the separate interaction output is reserved.
    """

    def __init__(self, config: NativeBodyConfig | None = None):
        super().__init__()
        self.config = config or NativeBodyConfig()
        if not isinstance(self.config, NativeBodyConfig):
            raise ValueError("config must be NativeBodyConfig")
        cfg = self.config
        self.query_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.query_projection = nn.Linear(cfg.hidden_dim, cfg.width)
        self.geometry_norm = nn.LayerNorm(cfg.geometry_dim, elementwise_affine=False, eps=1e-6)
        self.geometry_projection = nn.Linear(cfg.geometry_dim, cfg.width)
        self.local_geometry = nn.Conv2d(cfg.width, cfg.width, 3, padding=1, groups=cfg.width)
        self.geometry_modulation = nn.Linear(cfg.width, cfg.width)
        self.layer_embedding = nn.Embedding(cfg.num_layers, cfg.width)
        self.time_projection = nn.Sequential(nn.Linear(1, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.fusion = nn.Sequential(
            nn.LayerNorm(cfg.width, elementwise_affine=False, eps=1e-6),
            nn.Linear(cfg.width, cfg.width * 2), nn.SiLU(), nn.Linear(cfg.width * 2, cfg.width),
        )
        self.output_heads = nn.ModuleList(nn.Linear(cfg.width, cfg.hidden_dim, bias=False) for _ in range(cfg.num_groups))
        for head in self.output_heads:
            nn.init.zeros_(head.weight)

    def get_extra_state(self):
        return {"schema": "qwen21_full_native_body_v1", "config": self.config.to_dict()}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Native Body schema/config does not match this model")

    def _require_fp32(self):
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("Native Body parameters must remain FP32; convert the backbone dtype before installation")

    def prepare_geometry(self, features: NativeGeometryFeatures, *,
                         conv_input_valid: torch.Tensor | None = None) -> PreparedNativeGeometry:
        self._require_fp32()
        if not isinstance(features, NativeGeometryFeatures):
            raise ValueError("Expected NativeGeometryFeatures")
        grid = features.target_token_hw
        if (not isinstance(grid, tuple) or len(grid) != 2
                or any(type(i) is not int or i <= 0 for i in grid) or grid[0] * grid[1] % 4):
            raise ValueError("Geometry requires a positive target grid with four-token Qwen slots")
        value = features.geometry
        if (not isinstance(value, torch.Tensor) or value.ndim != 3
                or value.shape[1:] != (grid[0] * grid[1], self.config.geometry_dim)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("Geometry must be finite [B,N,geometry_dim] on the exact target grid")
        valid = features.person_valid
        if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != (value.shape[0], 2) or not bool(valid[:, 0].all())):
            raise ValueError("Geometry person_valid must be bool [B,2] with person A valid")
        if conv_input_valid is not None:
            # The gate has to be applied after projection, not to the raw features: a norm layer
            # turns an all-zero input row into a non-zero one (its bias still lands), so masking
            # upstream would not stop padding tokens reaching their neighbours.
            if (not isinstance(conv_input_valid, torch.Tensor)
                    or conv_input_valid.shape != value.shape[:2]
                    or not (conv_input_valid.dtype == torch.bool or conv_input_valid.is_floating_point())
                    or (conv_input_valid.is_floating_point() and (
                        not bool(torch.isfinite(conv_input_valid).all())
                        or bool((conv_input_valid < 0).any()) or bool((conv_input_valid > 1).any())))):
                raise ValueError(
                    "conv_input_valid must be bool or finite 0..1 [B,N] matching the target grid")
        with torch.autocast(device_type=value.device.type, enabled=False):
            projected = self.geometry_projection(self.geometry_norm(value.float()))
            if conv_input_valid is not None:
                projected = projected * conv_input_valid.to(device=projected.device,
                                                             dtype=projected.dtype)[..., None]
            spatial = projected.transpose(1, 2).reshape(value.shape[0], self.config.width, *grid)
            tokens = projected + self.local_geometry(spatial).flatten(2).transpose(1, 2)
        return PreparedNativeGeometry(tokens, grid)

    def forward(self, hidden, *, prepared: PreparedNativeGeometry, layer: int, sigma,
                target_mask, valid_mask=None, strength: float = 1.0, return_delta: bool = False):
        self._require_fp32()
        cfg = self.config
        if type(layer) is not int or not 0 <= layer < cfg.num_layers:
            raise ValueError("Invalid Native Body layer index")
        if layer not in cfg.injection_layers:
            return hidden[:,target_mask].new_zeros(hidden.shape[0],int(target_mask.sum()),cfg.hidden_dim,dtype=torch.float32) if return_delta else hidden
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != cfg.hidden_dim or not hidden.is_floating_point()):
            raise ValueError("Hidden must be floating [B,L,hidden_dim]")
        if (not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool
                or target_mask.ndim != 1 or target_mask.numel() != hidden.shape[1]):
            raise ValueError("target_mask must be bool [L]")
        b, n = hidden.shape[0], int(target_mask.sum())
        if (not isinstance(prepared, PreparedNativeGeometry) or prepared.tokens.shape != (b, n, cfg.width)
                or prepared.target_token_hw[0] * prepared.target_token_hw[1] != n):
            raise ValueError("Prepared geometry must match this target layout")
        if (not isinstance(sigma, torch.Tensor) or sigma.shape != (b,)
                or not bool(torch.isfinite(sigma).all()) or bool(((sigma < 0) | (sigma > 1)).any())):
            raise ValueError("sigma must be finite [B] in [0,1]")
        if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be a finite nonnegative scalar")
        if valid_mask is not None:
            # A boundary token of a padded canvas covers only part of the real image, so a bool
            # mask would hand it the same full residual as a fully valid token. Accept the 0..1
            # fractional area coverage that valid_image_tokens() already computes.
            if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (b, n)
                    or not (valid_mask.dtype == torch.bool or valid_mask.is_floating_point())
                    or (valid_mask.is_floating_point() and (
                        not bool(torch.isfinite(valid_mask).all())
                        or bool((valid_mask < 0).any()) or bool((valid_mask > 1).any())))):
                raise ValueError(
                    "valid_mask must be bool or 0..1 area coverage [B,N] over the target canvas")
        target_mask = target_mask.to(hidden.device)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden[:, target_mask].float()))
            geometry = prepared.tokens
            time = self.time_projection(sigma.to(device=hidden.device, dtype=torch.float32)[:, None])[:, None]
            fused = query + geometry + query * self.geometry_modulation(geometry).tanh()
            fused = self.fusion(fused + self.layer_embedding.weight[layer][None, None] + time)
            delta = self.output_heads[cfg.num_groups * layer // cfg.num_layers](fused) * strength
            if valid_mask is not None:
                delta = delta * valid_mask.to(device=hidden.device, dtype=delta.dtype)[..., None]
        if return_delta:
            return delta
        result = hidden.clone()
        result[:, target_mask] = _IdentitySafeAdd.apply(hidden[:, target_mask], delta.to(hidden.dtype))
        return result
