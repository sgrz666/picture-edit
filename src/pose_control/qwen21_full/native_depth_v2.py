"""CHAMP + lightweight geometry detail + channel-wise gated Native residual.

Architecture implementation only. Runtime/gradient/quality acceptance is deferred.
"""
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .depth_condition_v2 import DepthConditionV2, DETAIL_CHANNELS, GATE_CHANNELS, FEATURE_SCHEMA
from .depth_geometry import DepthGeometryConfig, gate_context
from .native_depth import NativeDepthConfig, Qwen21NativeDepth, _ResidualAdd
from .native_geometry import NativeGeometryFeatures

DEPTH_SCHEMA = 'qwen21_native_depth_v2'


@dataclass(frozen=True)
class NativeDepthV2Config:
    geometry_dim: int = 512
    attention_scale: int = 16
    body_jump_threshold: float = .02
    relative_jump_threshold: float = .05
    epsilon: float = 1e-6

    def __post_init__(self):
        NativeDepthConfig(self.geometry_dim, self.attention_scale)
        self.geometry_config()

    def geometry_config(self):
        return DepthGeometryConfig(self.body_jump_threshold, self.relative_jump_threshold, self.epsilon)


class GeometryDetailBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(32, out_channels // 4)
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                                  nn.GroupNorm(groups, out_channels), nn.SiLU(),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                  nn.GroupNorm(groups, out_channels))
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, value):
        return F.silu(self.main(value) + self.skip(value))


class GeometryDetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(len(DETAIL_CHANNELS), 16, 3, padding=1),
                                  nn.GroupNorm(4, 16), nn.SiLU())
        self.blocks = nn.ModuleList(GeometryDetailBlock(a, b) for a, b in zip((16, 32, 64, 128), (32, 64, 128, 256)))

    def forward(self, detail, valid):
        value = self.stem(detail * valid.float())
        support = valid
        value = value * support
        for block in self.blocks:
            support = F.max_pool2d(support.float(), 2, 2) > 0
            value = block(value) * support
        return value


class GeometryAwareDepthGate(nn.Module):
    def __init__(self, geometry_dim):
        super().__init__()
        self.geometry = nn.Conv2d(geometry_dim, 64, 1)
        self.depth = nn.Conv2d(256, 64, 1)
        self.context = nn.Conv2d(len(GATE_CHANNELS), 32, 1)
        self.network = nn.Sequential(nn.Conv2d(160, 128, 1), nn.SiLU(), nn.Conv2d(128, geometry_dim, 1), nn.Sigmoid())

    def forward(self, geometry, depth, context):
        return self.network(torch.cat((self.geometry(geometry), self.depth(depth), self.context(context)), 1))


class Qwen21NativeDepthV2(Qwen21NativeDepth):
    """Reuse the existing audited CHAMP loader without changing its input/states."""
    def __init__(self, config: NativeDepthV2Config | None = None):
        config = config or NativeDepthV2Config()
        if not isinstance(config, NativeDepthV2Config):
            raise ValueError('Expected NativeDepthV2Config')
        super().__init__(NativeDepthConfig(config.geometry_dim, config.attention_scale))
        self.config = config
        self.detail_encoder = GeometryDetailEncoder()
        self.mixer = nn.Sequential(nn.Conv2d(512, 256, 1), nn.GroupNorm(32, 256), nn.SiLU())
        self.gate = GeometryAwareDepthGate(config.geometry_dim)
        # super constructed the only zero-initialized output projection. Gate
        # weights retain ordinary nonzero initialization, and sigmoid is not zero.

    def get_extra_state(self):
        return {'schema': DEPTH_SCHEMA, 'config': asdict(self.config), 'feature_schema': FEATURE_SCHEMA,
                'detail_channels': list(DETAIL_CHANNELS), 'gate_channels': list(GATE_CHANNELS),
                'normal_basis': 'camera_cv', 'geometry_source': 'paired_target_frame',
                'initialization': 'only_final_depth_projection_zero'}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Native Depth V2 schema/config/channel order differs')

    def reset_new_fusion(self):
        """Explicit V1 upgrade resets every new branch, never CHAMP parameters."""
        for branch in (self.detail_encoder, self.mixer, self.gate):
            for module in branch.modules():
                reset = getattr(module, 'reset_parameters', None)
                if reset is not None:
                    reset()
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, depth, *, enabled=True, strength=1., use_detail=True):
        import math
        self._require_fp32()
        if type(enabled) is not bool or type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError('Depth V2 requires bool enabled and finite nonnegative strength')
        if not isinstance(features, NativeGeometryFeatures):
            raise ValueError('Expected NativeGeometryFeatures')
        value, grid = features.geometry, features.target_token_hw
        if (not isinstance(grid, tuple) or len(grid) != 2 or any(type(x) is not int or x <= 0 for x in grid)
                or grid[0] * grid[1] % 4 or value.ndim != 3
                or value.shape[1:] != (grid[0] * grid[1], self.config.geometry_dim)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('Depth V2 requires finite exact-grid geometry')
        if not enabled or strength == 0:
            return features
        if (features.person_valid.dtype != torch.bool or features.person_valid.shape != (value.shape[0], 2)
                or not features.person_valid[:, 0].all() or features.person_valid[:, 1].any()):
            raise ValueError('Depth V2 supports one valid person only; use off/V1 for the original other paths')
        if not isinstance(depth, DepthConditionV2):
            raise ValueError('Enabled Depth V2 requires DepthConditionV2; no V1/PNG guessing')
        depth.validate(target_token_hw=grid)
        if depth.relative.shape[0] != value.shape[0] or depth.relative.device != value.device:
            raise ValueError('Depth V2 and Geometry batch/device differ')
        if any(m.get('geometry_config') != asdict(self.config.geometry_config()) for m in depth.metadata):
            raise ValueError('Preprocessed geometry thresholds differ from checkpoint configuration')
        active = depth.available.nonzero(as_tuple=True)[0]
        if not active.numel():
            return features
        selected = depth.index_select(active)
        with torch.autocast(device_type=value.device.type, enabled=False):
            champ = self.encoder(selected.relative, selected.valid)
            detail = self.detail_encoder(selected.detail_features, selected.valid) if use_detail else torch.zeros_like(champ)
            if tuple(champ.shape[-2:]) != grid or tuple(detail.shape[-2:]) != grid:
                raise ValueError('CHAMP/detail must land directly on the target 1/16 grid')
            mixed = self.mixer(torch.cat((champ, detail), 1))
            geometry = value[active].float().transpose(1, 2).reshape(active.numel(), self.config.geometry_dim, *grid)
            gate = self.gate(geometry, mixed, gate_context(selected))
            support = F.avg_pool2d(selected.valid.float(), 16) > 0
            delta = (self.projection(mixed) * gate * support * strength).flatten(2).transpose(1, 2)
            delta = torch.zeros_like(value, dtype=torch.float32).index_copy(0, active, delta)
        return replace(features, geometry=_ResidualAdd.apply(value, delta.to(value.dtype)))
