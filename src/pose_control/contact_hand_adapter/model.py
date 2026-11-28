"""Hierarchical hand geometry memory and Geometry-RGB co-attention."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .conditions import JOINT_PARENTS, JOINT_PARTS, UNKNOWN
from .geometry import ContactGraph, SpatialEncoder, build_contact_graph, pixel_geometry
from .routing import (APPEARANCE, CONTACT, HAND, JOINT, SURFACE, InstanceSpatialAttention,
                      TypedDenoiseGate, contact_evidence, enhance_support, spatial_evidence)


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
    architecture_version: int = 3
    spatial_mode: str = "instance"
    gate_mode: str = "typed"
    residual_mode: str = "compressed"
    appearance_mode: str = "part"
    appearance_dino: bool = True

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
        if self.architecture_version not in (2, 21, 3):
            raise ValueError("Unsupported Contact Hand Adapter architecture version")
        if self.spatial_mode not in ("instance", "average") or self.gate_mode not in ("typed", "uniform"):
            raise ValueError("Unsupported hand spatial or gate mode")
        if self.residual_mode not in ("compressed", "low_rank", "direct"):
            raise ValueError("Unsupported hand residual projection")
        if self.appearance_mode not in ("part", "global", "off") or type(self.appearance_dino) is not bool:
            raise ValueError("Unsupported source appearance mode")


@dataclass(frozen=True)
class HandMemory:
    tokens: torch.Tensor
    slots: torch.Tensor
    xy: torch.Tensor  # token anchors in Qwen target-grid coordinates
    support: torch.Tensor  # [4, N] pixel support pooled to the target grid
    roi: torch.Tensor  # [R] target-grid raster indices used to read and write Qwen
    spatial: torch.Tensor  # [R, 4, D] per-instance, or [R, D] in the average ablation
    graph: ContactGraph
    hierarchy_counts: tuple[int, ...]
    token_hw: tuple[int, int]
    anchor_strength: torch.Tensor | None = None
    spatial_metadata: torch.Tensor | None = None  # [R,4,5] renderer and graph evidence
    token_types: torch.Tensor | None = None


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
    def __init__(self, width, heads, base_width, *, legacy=False):
        super().__init__()
        self.image_projection = nn.Linear(base_width, width, bias=False)
        self.image_attention = Attention(width, heads)
        self.image_query = nn.Linear(width, width)
        self.geometry_attention = Attention(width, heads)
        if not legacy:
            peer_width = max(heads, width // (4 * heads) * heads)
            self.peer_input = nn.Linear(width, peer_width, bias=False)
            self.peer_attention = Attention(peer_width, heads)
            self.peer_output = nn.Linear(peer_width, width, bias=False)
        else:
            self.peer_attention = None
        self.state_norm = nn.LayerNorm(width)
        self.context = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                    nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, state, memory, roi_hidden, sigma_and_layer, routing, spatial=None):
        image = self.image_projection(roi_hidden) + (memory.spatial if spatial is None else spatial)
        state = state + self.image_attention(self.image_query(self.state_norm(state)), image, routing.T)
        query = self.state_norm(state) + self.context(sigma_and_layer)
        slots = memory.slots
        peers = memory.graph.neighbors[slots[:, None], slots[None, :]]
        if self.peer_attention is not None:
            active = torch.where(peers.any(1))[0]
            if active.numel():
                exchanged = self.peer_input(query)
                strength = memory.graph.peer_strength[slots[active, None], slots[None, :]]
                weights = strength.clamp_min(1e-6).log().masked_fill(~peers[active], -torch.inf)
                delta = self.peer_attention(exchanged[active], exchanged, weights)
                state = state.index_add(0, active, self.peer_output(delta) * strength.amax(1, keepdim=True))
                query = self.state_norm(state) + self.context(sigma_and_layer)
        anchors = (slots[:, None] == slots[None, :]) | peers
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


class NativeWidthOutput(nn.Module):
    """Project the native-width attention result directly, or via an explicit low rank."""

    def __init__(self, config):
        super().__init__()
        width = config.base_width
        if config.residual_mode == "direct":
            self.first = nn.Identity()
            self.last = nn.Linear(width, width, bias=False)
        else:
            self.first = nn.Linear(width, config.rank, bias=False)
            self.last = nn.Linear(config.rank, width, bias=False)
        nn.init.zeros_(self.last.weight)
        self.layer_scale = nn.Parameter(torch.ones(len(config.layers)))

    def forward(self, value, layer):
        return self.last(self.first(value)) * self.layer_scale[layer]


def masked_mean(value, mask):
    return (value * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[..., None]


class ContactHandAdapter(nn.Module):
    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        self.config = config or AdapterConfig()
        cfg = self.config
        width, heads = cfg.width, cfg.heads
        legacy = cfg.architecture_version == 2
        v3 = cfg.architecture_version == 3
        self.joint_graph = GraphEncoder(5 if legacy else 19, width)
        self.mesh_graph = GraphEncoder(3, width)
        self.register_buffer("joint_edges", torch.tensor([JOINT_PARENTS[1:], tuple(range(1, 21))]), persistent=False)
        if legacy:
            self.slot_embedding = nn.Embedding(4, width)
        else:
            self.person_embedding = nn.Embedding(2, width)
            self.side_embedding = nn.Embedding(2, width)
            self.joint_validity = nn.Linear(5, width, bias=False)
            self.mano_encoder = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.LayerNorm(width))
        self.part_embedding = nn.Embedding(16, width)
        self.kind_embedding = nn.Embedding(5 if legacy else 7 if v3 else 6, width)
        self.position = nn.Linear(2, width)
        self.spatial_encoder = SpatialEncoder(width, cfg.roi_size, legacy=legacy)
        if v3:
            from .appearance import AppearanceEncoder

            self.appearance = AppearanceEncoder(width)
        else:
            self.appearance = nn.Sequential(nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.SiLU(),
                                            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                                            nn.Conv2d(128, width, 3, stride=2, padding=1), nn.SiLU(),
                                            nn.AdaptiveAvgPool2d((4, 4)))
            self.appearance_patch = nn.Parameter(torch.zeros(16, width))
        self.intra = nn.ModuleList(HierarchyBlock(width, heads) for _ in range(cfg.intra_layers))
        self.inter = nn.ModuleList(HierarchyBlock(width, heads) for _ in range(cfg.inter_layers))
        self.hand_bias = nn.Linear(13, heads, bias=False)
        self.part_bias = nn.Linear(4, heads, bias=False)
        self.surface_bias = nn.Linear(5 if legacy else 6, heads, bias=False)
        self.feedback = GeometryFeedback(width, heads, cfg.base_width, legacy=legacy)
        self.instance_router = (InstanceSpatialAttention(cfg.base_width, width)
                                if v3 and cfg.spatial_mode == "instance" else None)
        self.typed_gate = TypedDenoiseGate(width, len(cfg.layers)) if v3 and cfg.gate_mode == "typed" else None
        self.contact_token = nn.Linear(6, width) if v3 else None
        self.native_keys = nn.Linear(width, cfg.base_width, bias=False)
        self.native_values = nn.Linear(width, cfg.base_width, bias=False)
        self.key_norm = nn.RMSNorm(cfg.base_width // cfg.base_heads)
        self.compress = nn.Linear(cfg.base_width, width, bias=False)
        self.gate = nn.Sequential(nn.Linear(width * 2 + 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.output = (NativeWidthOutput(cfg) if v3 and cfg.residual_mode != "compressed" else SharedOutput(cfg))
        self.native_lora = nn.ModuleList(NativeOutputLoRA(cfg.base_width, cfg.hand_lora_rank)
                                         for _ in cfg.layers) if cfg.hand_lora_rank else nn.ModuleList()

    def prepare(self, condition):
        condition.validate()
        legacy = self.config.architecture_version == 2
        v3 = self.config.architecture_version == 3
        heatmaps, support_pixels = pixel_geometry(condition, legacy=legacy, renderer_support=v3)
        if v3:
            support_pixels = enhance_support(condition, heatmaps, support_pixels)
        graph = build_contact_graph(condition, support_pixels, legacy=legacy,
                                    contact_level=self.config.contact_level if v3 else "surface")
        if not v3 and self.config.contact_level == "hand":
            graph = replace(graph, part_state=torch.full_like(graph.part_state, UNKNOWN),
                            part_confidence=torch.zeros_like(graph.part_confidence),
                            occlusion=torch.zeros_like(graph.occlusion))
        if not v3 and self.config.contact_level != "surface":
            graph = replace(graph, surface_edges=graph.surface_edges[:0], surface_features=graph.surface_features[:0],
                            surface_probability=graph.surface_probability[:0])
        stride = condition.canvas.stride
        support = F.avg_pool2d(support_pixels, stride).flatten(1)
        support = support / support.sum(0, keepdim=True).clamp_min(1)
        roi = torch.where(support.any(0))[0]
        projected = condition.active_joints() & support.any(1)[:, None] if legacy else condition.projected_joints()
        valid = projected if legacy else condition.anatomy_joints()
        th, tw = condition.canvas.token_hw
        dimensions = support.new_tensor([tw, th])
        xy = torch.where(projected[..., None], condition.joints[..., :2].float() / stride, 0)
        origin = torch.where(projected[:, :1], xy[:, 0], masked_mean(xy, projected))
        if not legacy:
            y, x = torch.meshgrid(torch.arange(th, device=xy.device), torch.arange(tw, device=xy.device), indexing="ij")
            center = support @ (torch.stack((x, y), -1).reshape(-1, 2).float() + .5)
            center = center / support.sum(1).clamp_min(1e-6)[:, None]
            origin = torch.where(projected.any(1)[:, None], origin, center)
        local = xy - origin[:, None]
        local = local / (local.norm(dim=-1) * projected).amax(1).clamp_min(1)[:, None, None]
        confidence = torch.where(projected, condition.joints[..., 2].float(), 0)
        if legacy:
            features = torch.cat((local, xy / dimensions, confidence[..., None]), -1)
            joint = self.joint_graph(features, valid, self.joint_edges)
        else:
            features, validity = self.joint_features(condition, xy, local, dimensions, projected, confidence)
            joint = self.joint_graph(features, valid, self.joint_edges) + self.joint_validity(validity)
        hand = masked_mean(joint, valid)[:, None]
        part_ranges = (slice(0, 1), *(slice(i, i + 4) for i in range(1, 21, 4)))
        fingers = torch.stack([masked_mean(joint[:, part], valid[:, part]) for part in part_ranges], 1)
        finger_xy = torch.stack([masked_mean(xy[:, part], projected[:, part]) for part in part_ranges], 1)
        finger_valid = torch.stack([valid[:, part].any(1) for part in part_ranges], 1)
        mesh = condition.mesh
        surf_count = self.config.surface_tokens
        anchor_group = 64 // surf_count
        anchor_ids = torch.arange(0, 64, anchor_group, device=support.device)
        surface = joint.new_zeros(4, surf_count, self.config.width)
        surface_xy = xy.new_zeros(4, surf_count, 2)
        surface_valid = valid.new_zeros(4, surf_count)
        if mesh is not None:
            mesh_active = mesh.valid & (valid.any(1) if legacy else condition.present_hands())
            vertices = torch.where(mesh_active[:, None, None], mesh.vertices.float(), 0)
            vertices = (vertices - vertices.mean(1, keepdim=True)) / .10
            mesh_tokens = self.mesh_graph(vertices, mesh_active[:, None].expand(-1, 778), mesh.edges)
            surface = mesh_tokens[:, mesh.anchors].mean(2).unflatten(1, (surf_count, anchor_group)).mean(2)
            anchor_pixels = mesh.anchor_xy.float().unflatten(1, (surf_count, anchor_group)).mean(2)
            surface_xy = torch.where(mesh_active[:, None, None], anchor_pixels / stride, 0)
            surface_valid = mesh_active[:, None].expand(-1, surf_count)
        appearance = joint.new_zeros(4, 16, self.config.width)
        appearance_valid = valid.new_zeros(4, 16)
        appearance_semantic = torch.zeros_like(appearance_valid)
        if condition.appearance is not None and (not v3 or self.config.appearance_mode != "off"):
            slots = condition.appearance_valid & valid.any(1)
            if slots.any():
                if v3:
                    def available(value):
                        return None if value is None else value[slots]

                    part_map = available(condition.appearance_parts) if self.config.appearance_mode == "part" else None
                    descriptors = available(condition.appearance_dino) if self.config.appearance_dino else None
                    source = self.appearance(condition.appearance[slots].float(), part_map, descriptors,
                                             mask=available(condition.appearance_mask),
                                             global_valid=available(condition.appearance_dino_valid))
                    appearance[slots] = source.tokens
                    appearance_semantic[slots] = source.part_valid
                    appearance_valid = appearance_semantic.clone()
                    appearance_valid[:, 0] |= slots & ~appearance_semantic.any(1)
                else:
                    appearance[slots] = self.appearance(condition.appearance[slots].float()).flatten(2).transpose(1, 2)
                    appearance_valid = slots[:, None].expand(-1, 16)
        if not v3:
            appearance = appearance + self.appearance_patch
        lengths = (1, 6, 21, surf_count, 16)
        values = (hand, fingers, joint, surface, appearance)
        geometry_hands = valid.any(1) if legacy else valid.any(1) | surface_valid.any(1) | support.any(1)
        present = (geometry_hands[:, None], finger_valid, valid, surface_valid, appearance_valid)
        positions = (origin[:, None], finger_xy, xy, surface_xy, origin[:, None].expand(-1, 16, -1))
        anchor_strength = None
        if not legacy:
            mano_valid = valid.new_zeros(4) if condition.mano_valid is None else condition.mano_valid
            mano = joint.new_zeros(4, self.config.width)
            if mano_valid.any():
                params = torch.cat((condition.mano_shape[mano_valid], condition.mano_wrist_rotation_6d[mano_valid],
                                    condition.mano_palm_scale[mano_valid, None] / .10), -1).float()
                mano[mano_valid] = self.mano_encoder(params)
            values += (mano[:, None],)
            present += (mano_valid[:, None],)
            lengths += (1,)
            strengths = condition.spatial_joint_confidence()
            finger_strength = torch.stack([strengths[:, part].amax(1) for part in part_ranges], 1)
            surface_strength = self.surface_strength(surface_xy, surface_valid, support, condition)
            global_strength = support.any(1).float()[:, None]
            appearance_xy = origin[:, None].expand(-1, 16, -1)
            appearance_strength = global_strength.expand(-1, 16)
            if v3 and appearance_semantic.any():
                anchors_xy = torch.cat((xy, surface_xy), 1)
                strength = torch.cat((strengths, surface_strength), 1)
                part_ids = torch.cat((anchor_ids.new_tensor(JOINT_PARTS), anchor_ids // 4))
                weights = strength[:, :, None] * F.one_hot(part_ids, 16)
                mass = weights.sum(1)
                aligned = weights.transpose(1, 2) @ anchors_xy / mass.clamp_min(1e-6)[..., None]
                known = appearance_semantic & (mass > 0)
                appearance_xy = torch.where(known[..., None], aligned, appearance_xy)
                appearance_strength = torch.where(known, weights.amax(1), appearance_strength * .35)
            anchor_strength = torch.cat((global_strength, finger_strength, strengths,
                                         surface_strength, appearance_strength, global_strength), 1)
            positions = (origin[:, None], torch.where(finger_strength[..., None] > 0, finger_xy, origin[:, None]),
                         torch.where(projected[..., None], xy, origin[:, None]), surface_xy, appearance_xy, origin[:, None])
            if v3:
                contacts = contact_evidence(graph)
                valid_contact = graph.neighbors & ((contacts[..., 2] > 0) | (contacts[..., 4] > 0)
                                                    | (contacts[..., 5] > 0))
                values += (self.contact_token(contacts),)
                present += (valid_contact,)
                lengths += (4,)
                positions += (origin[:, None].expand(-1, 4, -1),)
                anchor_strength = torch.cat((anchor_strength, graph.peer_strength * global_strength), 1)
        values, present, positions = (torch.cat(items, 1) for items in (values, present, positions))
        kinds = torch.repeat_interleave(torch.arange(len(lengths), device=support.device),
                                        torch.tensor(lengths, device=support.device))
        parts = torch.cat((kinds.new_zeros(1), kinds.new_tensor([0, 1, 4, 7, 10, 13]),
                           kinds.new_tensor(JOINT_PARTS), anchor_ids // 4,
                           torch.arange(16, device=kinds.device) if v3 else kinds.new_zeros(16),
                           kinds.new_zeros(sum(lengths[5:]))))
        anchors = torch.cat((kinds.new_full((28,), -1), anchor_ids, kinds.new_full((sum(lengths[4:]),), -1)))
        slots, token = torch.where(present)
        kinds, parts, anchors = kinds[token], parts[token], anchors[token]
        positions = positions[present]
        identity = self.slot_embedding(slots) if legacy else self.person_embedding(slots // 2) + self.side_embedding(slots % 2)
        values = (values[present] + identity + self.kind_embedding(kinds) + self.part_embedding(parts)
                  + self.position(positions / dimensions))
        anchor_strength = None if legacy else anchor_strength[present]
        metadata = spatial_evidence(condition, graph, roi) if self.instance_router is not None else None
        if roi.numel():
            spatial, token_context = self.spatial_encoder(
                condition, heatmaps, support_pixels, graph, tokens=None if legacy else values,
                token_slots=slots, token_xy=positions, anchor_strength=anchor_strength)
            spatial = spatial[:, roi]
            if self.instance_router is None:
                spatial = (spatial * support[:, roi, None]).sum(0) / support[:, roi].sum(0)[:, None]
            else:
                spatial = spatial.transpose(0, 1)
            if legacy:
                route = support[slots][:, roi]
                values = values + (route @ spatial) / route.sum(1).clamp_min(1e-6)[:, None]
            else:
                values = values + token_context
            values = self.encode_hierarchy(values, slots, kinds, parts, anchors, graph)
        else:
            empty_shape = (0, 4, self.config.width) if self.instance_router is not None else (0, self.config.width)
            spatial = values.new_empty(empty_shape)
        types = kinds.new_tensor([HAND, JOINT, JOINT, SURFACE, APPEARANCE, HAND, CONTACT])[kinds] if v3 else None
        return HandMemory(values, slots, positions, support, roi, spatial, graph, lengths, (th, tw),
                          anchor_strength, metadata, types)

    def joint_features(self, condition, xy, local, dimensions, projected, confidence):
        valid_3d = condition.joint_3d_valid
        xyz = xy.new_zeros(4, 21, 3) if valid_3d is None else torch.where(valid_3d[..., None], condition.joints_3d.float(), 0)
        if valid_3d is None:
            valid_3d = torch.zeros_like(projected)
        relative_valid = valid_3d & valid_3d[:, :1]
        relative = torch.where(relative_valid[..., None], (xyz - xyz[:, :1]) / .10, 0.)
        parents = torch.cat((self.joint_edges.new_zeros(1), self.joint_edges[0]))
        bone = xyz - xyz[:, parents]
        length = bone.norm(dim=-1)
        bone_valid = valid_3d & valid_3d[:, parents] & (length > 1e-6)
        direction = torch.where(bone_valid[..., None], bone / length.clamp_min(1e-6)[..., None], 0.)
        length = torch.where(bone_valid, length / .10, 0.)
        rotation_valid = condition.joint_rotation_valid
        rotation = xy.new_zeros(4, 21, 6) if rotation_valid is None else torch.where(
            rotation_valid[..., None], condition.joint_rotation_6d.float(), 0.)
        if rotation_valid is None:
            rotation_valid = torch.zeros_like(projected)
        occlusion = torch.zeros_like(projected) if condition.joint_occluded is None else condition.joint_occluded
        features = torch.cat((torch.where(projected[..., None], local, 0.), xy / dimensions,
                              relative, direction, length[..., None], rotation, condition.visible[..., None],
                              confidence[..., None]), -1)
        validity = torch.stack((projected, relative_valid, bone_valid, rotation_valid, occlusion), -1).float()
        return features, validity

    @staticmethod
    def surface_strength(xy, valid, support, condition):
        height, width = condition.canvas.token_hw
        x0, y0, x1, y1 = (edge / condition.canvas.stride for edge in condition.canvas.valid_rect)
        x, y = xy.unbind(-1)
        in_frame = valid & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        flat = y.long().clamp(0, height - 1) * width + x.long().clamp(0, width - 1)
        observed = support.gather(1, flat) > 0
        return in_frame.float() * torch.where(observed, 1., .20)

    def encode_hierarchy(self, values, slots, kinds, parts, anchors, graph):
        first, second = slots[:, None], slots[None, :]
        a, b = parts[:, None], parts[None, :]
        type_a, type_b = kinds[:, None], kinds[None, :]
        same = first == second
        global_a = (type_a == 0) | (type_a >= 5)
        global_b = (type_b == 0) | (type_b >= 5)
        global_token = global_a | global_b
        same_digit = ((a == 0) & (b == 0)) | ((a > 0) & (b > 0) & ((a - 1) // 3 == (b - 1) // 3))
        allowed = same & (global_token | same_digit | (type_a == 4) | (type_b == 4))
        intra_bias = torch.zeros_like(allowed, dtype=values.dtype).masked_fill(~allowed, -torch.inf)
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
        if self.config.architecture_version != 2:
            penalty = graph.peer_strength[first, second].clamp_min(1e-6).log()
            hand = hand + (penalty * peers)[..., None]
        compatible = (global_a & global_b) | ((type_a == 1) & (type_b == 1))
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
        if self.config.architecture_version != 2:
            distance, normal = distance * confidence, normal * confidence
        features = torch.stack((distance / .10, normal, (state == 1) * confidence,
                                (state == 0) * confidence, (state != UNKNOWN) * confidence), -1)
        if self.config.architecture_version != 2:
            features = torch.cat((features, graph.surface_probability[keep, None]), -1)
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
        routing = (-.05 * distances).masked_fill(coverage == 0, -torch.inf)
        if memory.anchor_strength is not None:
            strength = memory.anchor_strength[None]
            routing = routing + (.35 + .65 * strength).log() - .20 * strength * distances
        level = roi_hidden.new_tensor(layer / max(1, len(self.config.layers) - 1))
        step = torch.stack((sigma.reshape(()).float(), level.float()))
        spatial = memory.spatial
        slot_support = memory.support[:, memory.roi].T
        if self.instance_router is not None:
            spatial, instance_weights = self.instance_router(roi_hidden, spatial, slot_support, memory.spatial_metadata)
            # The same target query's slot selection also gates its access to structured keys.
            routing = routing + instance_weights[:, memory.slots].clamp_min(1e-6).log()
        else:
            routing = routing + coverage.clamp_min(1e-6).log()
        if self.config.image_feedback:
            state = self.feedback(state, memory, roi_hidden, step, routing, spatial)
        magnitude = None
        if self.typed_gate is not None:
            typed = self.typed_gate(self.feedback.image_projection(roi_hidden), spatial, slot_support, sigma, layer)
            token_gate = typed[:, memory.token_types]
            magnitude = (routing.softmax(-1) * token_gate).sum(-1, keepdim=True)
            routing = routing + token_gate.clamp_min(1e-6).log()
        shape = (self.config.base_heads, -1)
        keys = self.key_norm(self.native_keys(state).unflatten(-1, shape))
        if rope is not None:
            keys = rope(keys)
        values = self.native_values(state).unflatten(-1, shape)
        attended = F.scaled_dot_product_attention(query.transpose(0, 1), keys.transpose(0, 1),
                                                  values.transpose(0, 1), attn_mask=routing)
        native = attended.transpose(0, 1).flatten(1)
        feature = self.compress(native)
        support = slot_support.sum(1, keepdim=True)
        gate_input = torch.cat((feature, spatial, step.expand(len(feature), -1), support), -1)
        residual = native if isinstance(self.output, NativeWidthOutput) else feature
        update = self.output(residual, layer) * self.gate(gate_input).sigmoid() * support
        return (update if magnitude is None else update * magnitude), state
