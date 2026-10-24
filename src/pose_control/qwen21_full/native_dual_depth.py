"""Scene CHAMP once, shared person details, structured relations and zero output fusions."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .depth_geometry import gate_context
from .native_depth_v2 import Qwen21NativeDepthV2, NativeDepthV2Config, GeometryDetailBlock
from .dual_depth_condition import RELATION_CHANNELS, INTERACTION_CHANNELS
from .native_dual_identity import ContextDelta

DEPTH_SCHEMA = 'native_dual_depth_v3'


@dataclass(frozen=True)
class NativeDualConfig:
    attention_scale: int = 16
    body_jump_threshold: float = .02
    relative_jump_threshold: float = .05
    epsilon: float = 1e-6
    owner_tolerance: float = 1e-4

    def __post_init__(self):
        self.depth_config()
        if not math.isfinite(self.owner_tolerance) or self.owner_tolerance <= 0:
            raise ValueError('Owner tolerance must be finite positive')

    def depth_config(self):
        return NativeDepthV2Config(attention_scale=self.attention_scale, body_jump_threshold=self.body_jump_threshold,
                                  relative_jump_threshold=self.relative_jump_threshold, epsilon=self.epsilon)


@dataclass(frozen=True)
class DualControlOptions:
    depth_mode: str = 'full_geometry'   # off / scene_only / full_geometry
    detail: bool = True
    interaction: bool = True
    raster: bool = True
    part_contact: bool = True
    identity: bool = True
    depth_strength: float = 1.
    relation_strength: float = 1.
    interaction_strength: float = 1.
    identity_strength: float = 1.

    def __post_init__(self):
        if self.depth_mode not in ('off', 'scene_only', 'full_geometry'):
            raise ValueError('Unknown explicit dual depth mode')
        if any(type(getattr(self, k)) is not bool for k in ('detail', 'interaction', 'raster', 'part_contact', 'identity')):
            raise ValueError('Control switches must be bool')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for k, v in asdict(self).items() if k.endswith('_strength')):
            raise ValueError('Control strengths must be finite nonnegative')
        if (self.raster or self.part_contact) and (not self.interaction or self.depth_mode != 'full_geometry'):
            raise ValueError('Physical relations require full geometry AND an enabled Interaction fusion')
        if self.detail and self.depth_mode != 'full_geometry':
            raise ValueError('Person Detail requires full_geometry')


def interaction_context(condition):
    def pool(x):
        return F.avg_pool2d(x.float(), 16)
    b, _, h, w = condition.scene_depth.shape
    available = condition.camera_available[:, None, None, None].expand(b, 1, h, w)
    distance = condition.part_distance_valid.flatten(1).any(1)[:, None, None, None].expand_as(available)
    maps = [condition.projected[:, p] for p in range(2)] + [condition.visible[:, p] for p in range(2)]
    maps += [condition.person_depth_valid[:, p] for p in range(2)]
    maps += [condition.overlap, condition.front[:, 0], condition.front[:, 1], condition.tie, available, distance]
    return torch.cat([pool(x) for x in maps], 1)


class NativeDualDepth(Qwen21NativeDepthV2):
    """Keep exact V2 Encoder/Detail/Mixer/Gate state names for strict inheritance."""
    def get_extra_state(self):
        state = super().get_extra_state()
        return {**state, 'schema': DEPTH_SCHEMA, 'scene_encoder_calls': 1, 'person_detail_weights': 'shared'}

    def forward(self, geometry, condition, *, use_detail=True):
        self._require_fp32()
        b, n, _ = geometry.shape
        grid = (condition.scene_depth.shape[-2]//16, condition.scene_depth.shape[-1]//16)
        active = condition.scene_valid.flatten(1).any(1).nonzero(as_tuple=True)[0]
        result = geometry.new_zeros(b, n, 512)
        if not active.numel():
            return result
        c = condition.index_select(active)
        scene = self.encoder(c.scene_depth, c.scene_valid)
        g = geometry[active].transpose(1, 2).reshape(active.numel(), 512, *grid)
        delta = torch.zeros_like(g)
        for p, person in enumerate((c.person_a, c.person_b)):
            detail = self.detail_encoder(person.detail_features, person.valid) if use_detail else torch.zeros_like(scene)
            mixed = self.mixer(torch.cat((scene, detail), 1))
            update = self.projection(mixed) * self.gate(g, mixed, gate_context(person))
            weight = F.avg_pool2d(c.owner_weights[:, p], 16)
            delta = delta + weight * update
        return result.index_copy(0, active, delta.flatten(2).transpose(1, 2))


class RelationRasterEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(len(RELATION_CHANNELS), 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.blocks = nn.ModuleList(GeometryDetailBlock(a, b) for a, b in zip((16, 32, 64, 128), (32, 64, 128, 256)))
        self.projection = nn.Conv2d(256, 512, 1)  # ordinary init: the final Interaction head is zero

    def forward(self, condition):
        support = condition.projected.any(1).float()
        value = self.stem(condition.relation_raster) * support
        for block in self.blocks:
            support = F.max_pool2d(support, 2, 2)
            value = block(value) * support
        physical = condition.camera_available[:, None, None, None]
        return (self.projection(value) * support * physical).flatten(2).transpose(1, 2)


class PartRelationEncoder(nn.Module):
    """196 pair tokens -> 28 row/column summaries -> spatial attention delta."""
    def __init__(self):
        super().__init__()
        self.numeric = nn.Sequential(nn.Linear(9, 256), nn.SiLU(), nn.Linear(256, 256))
        self.part_a = nn.Embedding(14, 256)
        self.part_b = nn.Embedding(14, 256)
        self.summary_queries = nn.Parameter(torch.randn(28, 256) * .02)
        self.summary = ContextDelta(256, 8)
        self.projection = nn.Linear(256, 512)
        self.spatial = ContextDelta(512, 8)

    def forward(self, geometry, interaction, condition):
        b = geometry.shape[0]
        scale = condition.body_height.mean(1)[:, None, None, None]
        displacement = (condition.closest_points[..., 1, :] - condition.closest_points[..., 0, :]) / scale
        displacement = displacement * condition.closest_points_valid[..., None]
        numeric = torch.cat((torch.asinh(condition.part_distance[..., None]), torch.asinh(displacement),
                             condition.part_contact, condition.closest_points_valid[..., None].float()), -1)
        tokens = self.numeric(numeric) + self.part_a.weight[None, :, None] + self.part_b.weight[None, None]
        summaries, masks = [], []
        for person in range(2):
            for part in range(14):
                values = tokens[:, part] if person == 0 else tokens[:, :, part]
                valid = condition.part_distance_valid[:, part] if person == 0 else condition.part_distance_valid[:, :, part]
                q = self.summary_queries[person*14+part][None, None].expand(b, 1, -1)
                summaries.append(q + self.summary(q, values, valid, valid.any(1, keepdim=True)))
                masks.append(valid.any(1))
        context = self.projection(torch.cat(summaries, 1))
        valid = torch.stack(masks, 1)
        grid = condition.scene_depth.shape[-2]//16, condition.scene_depth.shape[-1]//16
        part_coverage = F.avg_pool2d(condition.person_part.flatten(0, 1), 16).reshape(b, 28, *grid).flatten(2)
        bias = part_coverage.transpose(1, 2).clamp_min(1e-6).log()
        query_valid = condition.projected.any(1).float()
        query_valid = F.max_pool2d(query_valid, 16).flatten(1).bool()
        return self.spatial(geometry + interaction, context, valid, query_valid, bias=bias)


class GatedNativeResidual(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.base = nn.Linear(512, 64)
        self.update = nn.Linear(512, 64)
        self.context = nn.Linear(context_dim, 32)
        self.gate = nn.Sequential(nn.Linear(160, 128), nn.SiLU(), nn.Linear(128, 512), nn.Sigmoid())
        self.output = nn.Linear(512, 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, update, context, support):
        gate = self.gate(torch.cat((self.base(base), self.update(update), self.context(context)), -1))
        return self.output(gate * update) * support[..., None]
