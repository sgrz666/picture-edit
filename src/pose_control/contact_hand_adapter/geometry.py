"""Canvas-space routing, spatial pyramid and multi-scale contact geometry."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .conditions import JOINT_PARTS, UNKNOWN, invert_relative_pose


@dataclass(frozen=True)
class ContactGraph:
    neighbors: torch.Tensor  # [4,4]
    hand_features: torch.Tensor  # [4,4,13] 2D, 3D pose/validity, touching/separate/known evidence
    part_state: torch.Tensor  # [4,4,16,16]
    part_confidence: torch.Tensor
    occlusion: torch.Tensor
    surface_edges: torch.Tensor  # [E,4], sender hand, receiver hand, sender anchor, receiver anchor
    surface_features: torch.Tensor
    surface_probability: torch.Tensor
    peer_strength: torch.Tensor


def pixel_geometry(condition, *, legacy=False, renderer_support=True):
    """Rasterize wrist and five digit chains, with support even for keypoints-only input."""
    h, w = condition.canvas.canvas_hw
    valid = condition.active_joints() if legacy else condition.projected_joints()
    joints = torch.where(valid[..., None], condition.joints.float(), 0)
    if not legacy:
        joints[..., 2] = condition.spatial_joint_confidence()
    y, x = torch.meshgrid(torch.arange(h, device=joints.device), torch.arange(w, device=joints.device), indexing="ij")
    pixels = torch.stack((x, y), dim=-1).float()
    spread = joint_spread(joints[..., :2], valid)
    heatmaps = []
    wrist = joints[:, 0, :2]
    point_distance = (pixels - wrist[:, None, None]).square().sum(-1)
    heatmaps.append(torch.exp(-point_distance / (2 * spread[:, None, None].square()))
                    * joints[:, 0, 2, None, None])
    for finger in range(5):
        start = 1 + 4 * finger
        key = joints[:, start:start + 4]
        parent = torch.cat((joints[:, :1], key[:, :-1]), 1)
        offset = pixels - parent[..., None, None, :2]
        vector = (key[..., :2] - parent[..., :2])[..., None, None, :]
        along = ((offset * vector).sum(-1) / vector.square().sum(-1).clamp_min(1)).clamp(0, 1)
        distance = (offset - along[..., None] * vector).square().sum(-1)
        at_joint = (pixels - key[..., None, None, :2]).square().sum(-1)
        distance = torch.where(parent[..., 2, None, None] > 0, distance, at_joint)
        heatmaps.append((torch.exp(-distance / (2 * spread[:, None, None, None].square()))
                         * key[..., 2, None, None]).amax(1))
    heatmaps = torch.stack(heatmaps, 1)
    coverage = condition.canvas.valid_mask(device=joints.device)
    peak = joints[..., 2].amax(1)[:, None, None]
    support = ((heatmaps.amax(1) > .12 * peak) * peak if condition.support is None else condition.support.float())
    if renderer_support and not legacy and condition.map_valid is not None:
        owned = condition.map_valid[:, [0, 4]].any(1)
        support = torch.maximum(support, owned.to(support.dtype))
    has_hand = valid.any(1) if legacy else condition.present_hands()
    support = support * coverage * has_hand[:, None, None]
    return heatmaps * coverage, support


def joint_spread(xy, valid):
    low = torch.where(valid[..., None], xy, float("inf")).amin(1)
    high = torch.where(valid[..., None], xy, -float("inf")).amax(1)
    size = torch.where(valid.any(1), (high - low).amax(-1), 0)
    return (size * .085).clamp(2, 12)


def build_contact_graph(condition, support, *, legacy=False, contact_level="surface"):
    device = support.device
    active = support.flatten(1).any(-1)
    present = active if legacy else condition.present_hands()
    part_state = torch.full((4, 4, 16, 16), UNKNOWN, dtype=torch.int8, device=device)
    part_confidence = support.new_zeros(4, 4, 16, 16)
    occlusion = torch.zeros_like(part_state)
    hand_features = support.new_zeros(4, 4, 13)
    height, width = condition.canvas.canvas_hw
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    xy = torch.stack((x / width, y / height), -1)
    center = (support[..., None] * xy).sum((1, 2)) / support.sum((1, 2)).clamp_min(1)[:, None]
    offset = center[None, :, :] - center[:, None, :]
    distance = offset.norm(dim=-1)
    hand_features[..., :2] = offset
    hand_features[..., 2] = distance
    expanded = F.max_pool2d(support[:, None], 17, stride=1, padding=8)[:, 0].flatten(1) > 0
    close = (expanded.float() @ (support.flatten(1) > 0).float().T) > 0
    neighbors = (close | close.T) & (active[:, None] & active[None, :])
    neighbors.fill_diagonal_(False)
    peer_strength = neighbors.float() * .65
    sparse_indices = []
    sparse_features = []
    sparse_probability = []
    for pair in condition.pairs:
        first, second = pair.slots
        if not (present[first] and present[second]):
            continue
        state = pair.part_state
        confidence = (state != UNKNOWN).float() if pair.part_confidence is None else pair.part_confidence
        touched = (confidence * (state == 1)).amax()
        separate = (confidence * (state == 0)).amax()
        summary = torch.stack((touched, separate, torch.maximum(touched, separate)))
        hand_features[first, second, 10:] = hand_features[second, first, 10:] = summary
        if contact_level != "hand":
            part_state[first, second], part_state[second, first] = state, state.T
            part_confidence[first, second] = confidence * (state != UNKNOWN)
            part_confidence[second, first] = part_confidence[first, second].T
        nearby = bool(neighbors[first, second]) or bool(touched > 0 if legacy else touched >= .5)
        strength = torch.maximum(peer_strength[first, second], touched * (touched >= .5))
        if pair.occlusion is not None and contact_level != "hand":
            occlusion[first, second], occlusion[second, first] = pair.occlusion, -pair.occlusion.T
            nearby |= bool(pair.occlusion.any())
            strength = torch.maximum(strength, pair.occlusion.any().float() * .75)
        if pair.relative_pose is not None:
            pose = pair.relative_pose.float()
            hand_features[first, second, 3:9] = pose
            hand_features[second, first, 3:9] = invert_relative_pose(pose)
            hand_features[first, second, 9] = hand_features[second, first, 9] = 1
            near_pose = pose[:3].norm() < .25
            nearby |= bool(near_pose)
            strength = torch.maximum(strength, near_pose.float() * .6)
        if pair.surface_indices is not None and contact_level == "surface":
            indices = pair.surface_indices
            prefix = indices.new_tensor([first, second]).expand(len(indices), 2)
            edge = torch.cat((prefix, indices), -1)
            sparse_indices.extend((edge, edge[:, [1, 0, 3, 2]]))
            distance, _, state, quality = pair.surface_features.unbind(-1)
            if pair.surface_quality is not None:
                quality = quality * pair.surface_quality.prod(1)
            weighted = torch.cat((pair.surface_features[:, :3], quality[:, None]), 1)
            features = pair.surface_features if legacy else weighted
            sparse_features.extend((features, features))
            probability = (-.5 * (distance / .008).square()).exp() * quality
            sparse_probability.extend((probability, probability))
            if legacy:
                nearby |= bool((((distance < .025) | (state == 1)) & (quality > 0)).any())
            else:
                connection = torch.maximum(probability, quality * ((state == 1) & (quality >= .5)))
                nearby |= bool((connection >= .05).any())
                if connection.numel():
                    strength = torch.maximum(strength, connection.max())
        neighbors[first, second] = neighbors[second, first] = nearby
        peer_strength[first, second] = peer_strength[second, first] = strength if nearby else 0.
    edges = torch.cat(sparse_indices) if sparse_indices else torch.empty(0, 4, device=device, dtype=torch.long)
    features = torch.cat(sparse_features).float() if sparse_features else support.new_empty(0, 4)
    probability = torch.cat(sparse_probability).float() if sparse_probability else support.new_empty(0)
    return ContactGraph(neighbors, hand_features, part_state, part_confidence, occlusion, edges, features,
                        probability, peer_strength)


def support_box(support):
    y, x = torch.where(support > 0)
    if not x.numel():
        raise ValueError("No eligible hand pixels for the requested ROI")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def roi_boxes(support, graph):
    """Share a square frame along contact-connected components, preserving each instance."""
    pending = set(torch.where(support.flatten(1).any(-1))[0].tolist())
    while pending:
        component = {min(pending)}
        frontier = list(component)
        while frontier:
            for neighbor in torch.where(graph.neighbors[frontier.pop()])[0].tolist():
                if neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        pending -= component
        slots = sorted(component)
        x0, y0, x1, y1 = support_box(support[slots].amax(0))
        size = max(x1 - x0, y1 - y0) * 1.35
        yield slots, ((x0 + x1 - size) / 2, (y0 + y1 - size) / 2, size)


class DeformableHandSampler(nn.Module):
    """Each target anchor reads four learned offsets from each ROI pyramid level."""

    points = 4
    levels = 4

    def __init__(self, width):
        super().__init__()
        self.offsets = nn.Linear(width, self.levels * self.points * 2)
        self.logits = nn.Linear(width, self.levels * self.points)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        with torch.no_grad():
            self.offsets.bias.copy_(torch.tensor([0., 0., .5, 0., 0., .5, -.5, 0.]).repeat(self.levels))

    def forward(self, pyramid, anchors, tokens):
        offsets = self.offsets(tokens).reshape(-1, self.levels, self.points, 2).tanh()
        weights = self.logits(tokens).softmax(-1).reshape(-1, self.levels, self.points)
        result = torch.zeros_like(tokens)
        for level, feature in enumerate(pyramid):
            h, w = feature.shape[-2:]
            step = offsets.new_tensor([2 / w, 2 / h])
            grid = anchors[:, None] + offsets[:, level] * step
            sampled = F.grid_sample(feature[None], grid[None], align_corners=False)[0].permute(1, 2, 0)
            result = result + (sampled * weights[:, level, :, None]).sum(1)
        return result


class SpatialEncoder(nn.Module):
    def __init__(self, width, roi_size, *, legacy=False):
        super().__init__()
        self.roi_size = roi_size
        channels = (min(64, width), min(128, width), min(256, width), width)
        self.stages = nn.ModuleList()
        self.laterals = nn.ModuleList()
        self.token_sampler = None if legacy else DeformableHandSampler(width)
        # V2.1 uses 21 joints, five bone chains, support/contact, depth/normal values and masks, 16 parts.
        previous = 18 if legacy else 52
        for count in channels:
            self.stages.append(nn.Sequential(nn.Conv2d(previous, count, 3, stride=2, padding=1),
                                             nn.GroupNorm(1, count), nn.SiLU()))
            self.laterals.append(nn.Conv2d(count, width, 1))
            previous = count

    def forward(self, condition, heatmaps, support, graph, *, tokens=None, token_slots=None,
                token_xy=None, anchor_strength=None):
        canvas = condition.canvas
        h, w = canvas.canvas_hw
        th, tw = canvas.token_hw
        dense = support.new_zeros(4, 5, h, w)
        mask = dense
        if condition.geometry_maps is not None:
            mask = condition.map_valid.float() * canvas.valid_mask(device=support.device)
            dense = torch.where(mask.bool(), condition.geometry_maps.float(), 0)
        hand_contact = ((graph.part_state == 1) * graph.part_confidence).amax((1, 3))
        fingers = torch.as_tensor(JOINT_PARTS[1:], device=support.device).reshape(5, 4)[:, 0]
        contact = torch.cat((hand_contact[:, :1], hand_contact[:, fingers[:, None] + torch.arange(3, device=support.device)].amax(-1)), 1)
        contact_map = (heatmaps * contact[..., None, None]).amax(1, keepdim=True)
        values = torch.cat((heatmaps, support[:, None], contact_map, dense, mask), dim=1)
        total = support.new_zeros(4, th * tw, self.laterals[0].out_channels)
        token_context = None if tokens is None else torch.zeros_like(tokens)
        joint_valid = condition.projected_joints()
        spread = joint_spread(condition.joints[..., :2], joint_valid)
        confidence = condition.spatial_joint_confidence()
        xy = torch.where(joint_valid[..., None], condition.joints[..., :2].float(), 0)
        coordinates = torch.arange(self.roi_size, device=support.device, dtype=support.dtype) + .5
        target_y, target_x = torch.meshgrid(
            (torch.arange(th, device=support.device) + .5) * canvas.stride,
            (torch.arange(tw, device=support.device) + .5) * canvas.stride, indexing="ij")
        for slots, (x0, y0, size) in roi_boxes(support, graph):
            y, x = torch.meshgrid(y0 + coordinates * size / self.roi_size,
                                  x0 + coordinates * size / self.roi_size, indexing="ij")
            grid = torch.stack((2 * x / w - 1, 2 * y / h - 1), -1).expand(len(slots), -1, -1, -1)
            feature = F.grid_sample(values[slots], grid, align_corners=False)
            if self.token_sampler is not None:
                coords = torch.stack((x, y), -1)
                squared = (coords - xy[slots, :, None, None]).square().sum(-1)
                points = (-squared / (2 * spread[slots, None, None, None].square())).exp()
                points = points * confidence[slots, :, None, None]
                # Resample discrete semantics independently, before one-hot encoding.
                parts = F.grid_sample(dense[slots, 4:5], grid, mode="nearest", align_corners=False)[:, 0]
                part_valid = F.grid_sample(mask[slots, 4:5], grid, mode="nearest", align_corners=False).bool()
                onehot = F.one_hot((parts * 15).round().long().clamp(0, 15), 16).movedim(-1, 1)
                feature = torch.cat((points, feature[:, 1:12], feature[:, 13:17], onehot * part_valid), 1)
            levels = []
            for stage, lateral in zip(self.stages, self.laterals):
                feature = stage(feature)
                levels.append(lateral(feature))
            fused = levels[-1]
            for level in reversed(levels[:-1]):
                fused = level + F.interpolate(fused, size=level.shape[-2:], mode="bilinear", align_corners=False)
            inverse = torch.stack((2 * (target_x - x0) / size - 1, 2 * (target_y - y0) / size - 1), -1)
            aligned = F.grid_sample(fused, inverse.expand(len(slots), -1, -1, -1), align_corners=False)
            total[slots] = aligned.flatten(2).transpose(1, 2).float()
            if token_context is not None:
                for index, slot in enumerate(slots):
                    chosen = torch.where((token_slots == slot) & (anchor_strength > 0))[0]
                    if chosen.numel():
                        position = token_xy[chosen] * canvas.stride
                        anchors = 2 * (position - position.new_tensor([x0, y0])) / size - 1
                        sampled = self.token_sampler([level[index] for level in levels], anchors, tokens[chosen])
                        token_context[chosen] = sampled * anchor_strength[chosen, None]
        return total, token_context
