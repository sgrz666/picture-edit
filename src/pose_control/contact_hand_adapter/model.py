"""Hierarchical hand geometry memory and Geometry-RGB co-attention."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .conditions import JOINT_PARTS, UNKNOWN
from .geometry import ContactGraph, SpatialEncoder, build_contact_graph, pixel_geometry


@dataclass(frozen=True)
class AdapterConfig:
    width: int = 768
    heads: int = 12
    base_width: int = 4096
    base_heads: int = 32
    layers: tuple[int, ...] = tuple(range(3, 32, 4))
    intra_layers: int = 3
    inter_layers: int = 4
    surface_tokens: int = 64
    roi_size: int = 256
    rank: int = 32
    image_feedback: bool = True
    contact_level: str = "surface"
    hand_lora_rank: int = 0

    def __post_init__(self):
        if (min(self.width, self.heads, self.base_width, self.base_heads, self.rank) < 1
                or self.width % self.heads or self.base_width % self.base_heads
                or min(self.intra_layers, self.inter_layers) < 1):
            raise ValueError("Invalid Adapter attention dimensions")
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers or self.layers[0] < 0:
            raise ValueError("Injection layers must be sorted, unique nonnegative indices")
        if self.surface_tokens not in (32, 64) or self.roi_size < 32 or self.roi_size % 16:
            raise ValueError("Expected 32/64 surface tokens and a stride-aligned geometry ROI")
        if self.contact_level not in ("hand", "part", "surface") or not 0 <= self.hand_lora_rank <= self.base_width:
            raise ValueError("Invalid contact level or native hand LoRA rank")


@dataclass(frozen=True)
class HandMemory:
    tokens: torch.Tensor
    slots: torch.Tensor
    xy: torch.Tensor  # token anchors in Qwen target-grid coordinates
    support: torch.Tensor  # [4, N] pixel support pooled to the target grid
    roi: torch.Tensor  # [R] target-grid raster indices used to read and write Qwen
    spatial: torch.Tensor  # [R, D] aligned features in the target ROI
    graph: ContactGraph
    hierarchy_counts: tuple[int, ...]
    token_hw: tuple[int, int]


class GraphEncoder(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.layers = nn.ModuleList(nn.Linear(width * 2, width) for _ in range(3))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))

    def forward(self, value, valid, edges):
        current = self.input(value) * valid[..., None]
        sender, receiver = torch.cat((edges, edges.flip(0)), dim=1)
        keep = valid[:, sender] & valid[:, receiver]
        degree = current.new_zeros(valid.shape).index_add(1, receiver, keep.to(current.dtype)).clamp_min(1)
        for layer, norm in zip(self.layers, self.norms):
            messages = current[:, sender] * keep[..., None]
            aggregate = torch.zeros_like(current).index_add(1, receiver, messages) / degree[..., None]
            current = norm(current + F.silu(layer(torch.cat((current, aggregate), -1)))) * valid[..., None]
        return current


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, query, context, bias=None):
        def split(value):
            return value.unflatten(-1, (self.heads, -1)).transpose(0, 1)

        update = F.scaled_dot_product_attention(split(self.query(query)), split(self.key(context)),
                                                split(self.value(context)), attn_mask=bias)
        return self.out(update.transpose(0, 1).flatten(1))


class HierarchyBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                    nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, value, bias):
        normalized = self.input_norm(value)
        value = value + self.attention(normalized, normalized, bias)
        return value + self.output(value)


class GeometryFeedback(nn.Module):
    def __init__(self, width, heads, base_width):
        super().__init__()
        self.image_projection = nn.Linear(base_width, width, bias=False)
        self.image_attention = Attention(width, heads)
        self.image_query = nn.Linear(width, width)
        self.geometry_attention = Attention(width, heads)
        self.state_norm = nn.LayerNorm(width)
        self.context = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                    nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, state, memory, roi_hidden, sigma_and_layer, routing):
        image = self.image_projection(roi_hidden) + memory.spatial
        state = state + self.image_attention(self.image_query(self.state_norm(state)), image, routing.T)
        query = self.state_norm(state) + self.context(sigma_and_layer)
        slots = memory.slots
        anchors = (slots[:, None] == slots[None, :]) | memory.graph.neighbors[slots[:, None], slots[None, :]]
        state = state + self.geometry_attention(query, memory.tokens, anchors)
        return state + self.update(state)


class SharedOutput(nn.Module):
    """A common output projection with one low-rank update per injection layer."""

    def __init__(self, config):
        super().__init__()
        self.shared = nn.Linear(config.width, config.base_width, bias=False)
        nn.init.zeros_(self.shared.weight)
        self.down = nn.ModuleList(nn.Linear(config.width, config.rank, bias=False) for _ in config.layers)
        self.up = nn.ModuleList(nn.Linear(config.rank, config.base_width, bias=False) for _ in config.layers)
        for module in self.up:
            nn.init.zeros_(module.weight)

    def forward(self, value, layer):
        return self.shared(value) + self.up[layer](self.down[layer](value))


class NativeOutputLoRA(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.down(value))


def masked_mean(value, mask):
    return (value * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[..., None]


class ContactHandAdapter(nn.Module):
    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        self.config = config or AdapterConfig()
        cfg = self.config
        width, heads = cfg.width, cfg.heads
        self.joint_graph = GraphEncoder(5, width)
        self.mesh_graph = GraphEncoder(3, width)
        joint_edges = [(0 if i % 4 == 1 else i - 1, i) for i in range(1, 21)]
        self.register_buffer("joint_edges", torch.tensor(joint_edges).T, persistent=False)
        self.slot_embedding = nn.Embedding(4, width)
        self.part_embedding = nn.Embedding(16, width)
        self.kind_embedding = nn.Embedding(5, width)
        self.position = nn.Linear(2, width)
        self.spatial_encoder = SpatialEncoder(width, cfg.roi_size)
        self.appearance = nn.Sequential(nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.SiLU(),
                                        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                                        nn.Conv2d(128, width, 3, stride=2, padding=1), nn.SiLU(),
                                        nn.AdaptiveAvgPool2d((4, 4)))
        self.appearance_patch = nn.Parameter(torch.zeros(16, width))
        self.intra = nn.ModuleList(HierarchyBlock(width, heads) for _ in range(cfg.intra_layers))
        self.inter = nn.ModuleList(HierarchyBlock(width, heads) for _ in range(cfg.inter_layers))
        self.hand_bias = nn.Linear(13, heads, bias=False)
        self.part_bias = nn.Linear(4, heads, bias=False)
        self.surface_bias = nn.Linear(5, heads, bias=False)
        self.feedback = GeometryFeedback(width, heads, cfg.base_width)
        self.native_keys = nn.Linear(width, cfg.base_width, bias=False)
        self.native_values = nn.Linear(width, cfg.base_width, bias=False)
        self.key_norm = nn.RMSNorm(cfg.base_width // cfg.base_heads)
        self.compress = nn.Linear(cfg.base_width, width, bias=False)
        self.gate = nn.Sequential(nn.Linear(width * 2 + 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.output = SharedOutput(cfg)
        self.native_lora = nn.ModuleList(NativeOutputLoRA(cfg.base_width, cfg.hand_lora_rank)
                                         for _ in cfg.layers) if cfg.hand_lora_rank else nn.ModuleList()

    def prepare(self, condition):
        condition.validate()
        heatmaps, support_pixels = pixel_geometry(condition)
        graph = build_contact_graph(condition, support_pixels)
        if self.config.contact_level == "hand":
            graph = replace(graph, part_state=torch.full_like(graph.part_state, UNKNOWN),
                            part_confidence=torch.zeros_like(graph.part_confidence),
                            occlusion=torch.zeros_like(graph.occlusion))
        if self.config.contact_level != "surface":
            graph = replace(graph, surface_edges=graph.surface_edges[:0], surface_features=graph.surface_features[:0])
        stride = condition.canvas.stride
        support = F.avg_pool2d(support_pixels, stride).flatten(1)
        support = support / support.sum(0, keepdim=True).clamp_min(1)
        roi = torch.where(support.any(0))[0]
        spatial = self.spatial_encoder(condition, heatmaps, support_pixels, graph) if roi.numel() else None
        valid = condition.active_joints() & support.any(1)[:, None]
        th, tw = condition.canvas.token_hw
        dimensions = support.new_tensor([tw, th])
        xy = torch.where(valid[..., None], condition.joints[..., :2].float() / stride, 0)
        origin = torch.where(valid[:, :1], xy[:, 0], masked_mean(xy, valid))
        local = (xy - origin[:, None]) / ((xy - origin[:, None]).norm(dim=-1) * valid).amax(1).clamp_min(1)[:, None, None]
        confidence = torch.where(valid, condition.joints[..., 2].float(), 0)
        joint = self.joint_graph(torch.cat((local, xy / dimensions, confidence[..., None]), -1), valid, self.joint_edges)
        hand = masked_mean(joint, valid)[:, None]
        part_ranges = (slice(0, 1), *(slice(i, i + 4) for i in range(1, 21, 4)))
        fingers = torch.stack([masked_mean(joint[:, part], valid[:, part]) for part in part_ranges], 1)
        finger_xy = torch.stack([masked_mean(xy[:, part], valid[:, part]) for part in part_ranges], 1)
        finger_valid = torch.stack([valid[:, part].any(1) for part in part_ranges], 1)
        mesh = condition.mesh
        surf_count = self.config.surface_tokens
        anchor_group = 64 // surf_count
        anchor_ids = torch.arange(0, 64, anchor_group, device=support.device)
        surface = joint.new_zeros(4, surf_count, self.config.width)
        surface_xy = xy.new_zeros(4, surf_count, 2)
        surface_valid = valid.new_zeros(4, surf_count)
        if mesh is not None:
            mesh_active = mesh.valid & valid.any(1)
            vertices = torch.where(mesh_active[:, None, None], mesh.vertices.float(), 0)
            vertices = (vertices - vertices.mean(1, keepdim=True)) / .10
            mesh_tokens = self.mesh_graph(vertices, mesh_active[:, None].expand(-1, 778), mesh.edges)
            surface = mesh_tokens[:, mesh.anchors].mean(2).unflatten(1, (surf_count, anchor_group)).mean(2)
            projected = mesh.anchor_xy.float().unflatten(1, (surf_count, anchor_group)).mean(2)
            surface_xy = torch.where(mesh_active[:, None, None], projected / stride, 0)
            surface_valid = mesh_active[:, None].expand(-1, surf_count)
        appearance = joint.new_zeros(4, 16, self.config.width)
        appearance_valid = valid.new_zeros(4, 16)
        if condition.appearance is not None:
            slots = condition.appearance_valid & valid.any(1)
            if slots.any():
                appearance[slots] = self.appearance(condition.appearance[slots].float()).flatten(2).transpose(1, 2)
                appearance_valid = slots[:, None].expand(-1, 16)
        lengths = (1, 6, 21, surf_count, 16)
        values = torch.cat((hand, fingers, joint, surface, appearance + self.appearance_patch), 1)
        present = torch.cat((valid.any(1)[:, None], finger_valid, valid, surface_valid, appearance_valid), 1)
        positions = torch.cat((origin[:, None], finger_xy, xy, surface_xy, origin[:, None].expand(-1, 16, -1)), 1)
        kinds = torch.repeat_interleave(torch.arange(5, device=support.device), torch.tensor(lengths, device=support.device))
        parts = torch.cat((kinds.new_zeros(1), kinds.new_tensor([0, 1, 4, 7, 10, 13]),
                           kinds.new_tensor(JOINT_PARTS), anchor_ids // 4, kinds.new_zeros(16)))
        anchors = torch.cat((kinds.new_full((28,), -1), anchor_ids, kinds.new_full((16,), -1)))
        slots, token = torch.where(present)
        kinds, parts, anchors = kinds[token], parts[token], anchors[token]
        positions = positions[present]
        values = (values[present] + self.slot_embedding(slots) + self.kind_embedding(kinds) + self.part_embedding(parts)
                  + self.position(positions / dimensions))
        if roi.numel():
            spatial = (spatial[:, roi] * support[:, roi, None]).sum(0) / support[:, roi].sum(0)[:, None]
            values = self.encode_hierarchy(values, slots, kinds, parts, anchors, positions, graph, spatial, support, roi)
        else:
            spatial = values.new_empty(0, self.config.width)
        return HandMemory(values, slots, positions, support, roi, spatial, graph, lengths, (th, tw))

    def encode_hierarchy(self, values, slots, kinds, parts, anchors, positions, graph, spatial, support, roi):
        first, second = slots[:, None], slots[None, :]
        a, b = parts[:, None], parts[None, :]
        type_a, type_b = kinds[:, None], kinds[None, :]
        same = first == second
        global_token = (type_a == 0) | (type_b == 0)
        same_digit = ((a == 0) & (b == 0)) | ((a > 0) & (b > 0) & ((a - 1) // 3 == (b - 1) // 3))
        allowed = same & (global_token | same_digit | (type_a == 4) | (type_b == 4))
        intra_bias = torch.zeros_like(allowed, dtype=values.dtype).masked_fill(~allowed, -torch.inf)
        # Link the static token and raster paths before the first DiT layer.
        route = support[slots][:, roi]
        values = values + (route @ spatial) / route.sum(1).clamp_min(1e-6)[:, None]
        for block in self.intra:
            values = block(values, intra_bias)
        hand = self.hand_bias(graph.hand_features[first, second])
        states = graph.part_state[first, second, a, b]
        confidence = graph.part_confidence[first, second, a, b]
        occlusion = graph.occlusion[first, second, a, b]
        part_data = torch.stack(((states == 1) * confidence, (states == 0) * confidence,
                                 (states != UNKNOWN) * confidence, occlusion), -1)
        hand = hand + self.part_bias(part_data) * (~global_token & (type_a != 4) & (type_b != 4))[..., None]
        if graph.surface_edges.numel():
            hand = self.add_surface_bias(hand, slots, anchors, graph)
        peers = graph.neighbors[first, second]
        compatible = ((type_a == 0) & (type_b == 0)) | ((type_a == 1) & (type_b == 1))
        compatible |= (type_a >= 2) & (type_a <= 3) & (type_b >= 2) & (type_b <= 3)
        inter_allowed = allowed | (peers & compatible)
        inter_bias = hand.masked_fill(~inter_allowed[..., None], -torch.inf).permute(2, 0, 1)
        for block in self.inter:
            values = block(values, inter_bias)
        return values

    def add_surface_bias(self, bias, slots, anchors, graph):
        lookup = torch.full((4, 64), -1, device=slots.device, dtype=torch.long)
        surface = anchors >= 0
        lookup[slots[surface], anchors[surface]] = torch.where(surface)[0]
        hand_a, hand_b, anchor_a, anchor_b = graph.surface_edges.T
        group = 64 // self.config.surface_tokens
        a, b = lookup[hand_a, anchor_a // group * group], lookup[hand_b, anchor_b // group * group]
        keep = (a >= 0) & (b >= 0)
        distance, normal, state, confidence = graph.surface_features[keep].T
        features = torch.stack((distance / .10, normal, (state == 1) * confidence,
                                (state == 0) * confidence, (state != UNKNOWN) * confidence), -1)
        linear = a[keep] * bias.shape[1] + b[keep]
        update = torch.zeros_like(bias.flatten(0, 1)).index_add(0, linear, self.surface_bias(features))
        count = update.new_zeros(len(update)).index_add(0, linear, torch.ones_like(linear, dtype=update.dtype))
        return bias + (update / count.clamp_min(1)[:, None]).reshape_as(bias)

    def update_attention(self, state, memory, layer, sigma, roi_hidden, query, rope):
        """Returns a branch update for native attention and the next dynamic hand state."""
        width = memory.token_hw[1]
        roi_xy = torch.stack((memory.roi % width, memory.roi // width), -1).float() + .5
        distances = (roi_xy[:, None] - memory.xy[None]).square().sum(-1)
        coverage = memory.support[memory.slots][:, memory.roi].T
        routing = (coverage.clamp_min(1e-6).log() - .05 * distances).masked_fill(coverage == 0, -torch.inf)
        level = roi_hidden.new_tensor(layer / max(1, len(self.config.layers) - 1))
        step = torch.stack((sigma.reshape(()).float(), level.float()))
        if self.config.image_feedback:
            state = self.feedback(state, memory, roi_hidden, step, routing)
        shape = (self.config.base_heads, -1)
        keys = self.key_norm(self.native_keys(state).unflatten(-1, shape))
        if rope is not None:
            keys = rope(keys)
        values = self.native_values(state).unflatten(-1, shape)
        attended = F.scaled_dot_product_attention(query.transpose(0, 1), keys.transpose(0, 1),
                                                  values.transpose(0, 1), attn_mask=routing)
        feature = self.compress(attended.transpose(0, 1).flatten(1))
        support = memory.support[:, memory.roi].sum(0)[:, None]
        gate_input = torch.cat((feature, memory.spatial, step.expand(len(feature), -1), support), -1)
        return self.output(feature, layer) * self.gate(gate_input).sigmoid() * support, state
