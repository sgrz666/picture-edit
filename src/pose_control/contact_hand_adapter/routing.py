"""Instance spatial attention and trainable denoising controls for V3."""

import math

import torch
from torch import nn
from torch.nn import functional as F


# Type indices follow the public gate order.
HAND, JOINT, SURFACE, CONTACT, APPEARANCE = range(5)


def enhance_support(condition, heatmaps, support):
    """Add confidence-weighted wrist and boundary support on the valid canvas."""
    wrist = heatmaps[:, 0]
    support = torch.maximum(support, .35 * torch.where(wrist > .08, wrist, 0.))
    halo = F.max_pool2d(support[:, None], 5, stride=1, padding=2)[:, 0]
    valid = condition.canvas.valid_mask(device=support.device)
    return torch.maximum(support, .25 * halo) * valid * condition.present_hands()[:, None, None]


def contact_evidence(graph):
    """Summarize each directed pair without converting a soft candidate into an annotation."""
    confidence = graph.hand_features[..., 10:13]
    direction = graph.occlusion.float()
    known = direction.ne(0)
    order = direction.sum((-1, -2)) / known.sum((-1, -2)).clamp_min(1)
    surface = graph.peer_strength.new_zeros(16)
    if graph.surface_probability.numel():
        sender, receiver = graph.surface_edges[:, :2].T
        surface.scatter_reduce_(0, sender * 4 + receiver, graph.surface_probability, reduce="amax")
    surface = surface.reshape(4, 4)
    return torch.cat((confidence, order[..., None], known.any(-1).any(-1)[..., None],
                      surface[..., None]), dim=-1)


def spatial_evidence(condition, graph, roi):
    """Depth comparisons are local to a target token and verified common cameras."""
    dtype = graph.peer_strength.dtype
    owned = graph.peer_strength.new_zeros(4, roi.numel())
    depth_known = torch.zeros_like(owned)
    frontness = torch.zeros_like(owned)
    if condition.map_valid is not None:
        stride = condition.canvas.stride
        valid = condition.canvas.valid_mask(device=owned.device).bool()

        def pool(value):
            return F.avg_pool2d(value.to(dtype), stride).flatten(1)[:, roi]

        mask = condition.map_valid[:, 0] & valid
        owned = pool((condition.map_valid[:, 4] | mask) & valid)
        depth_known = pool(mask)
        if condition.mesh is not None:
            camera_ids = condition.mesh.camera_ids
            common = owned.new_tensor([[bool(a) and a == b for b in camera_ids] for a in camera_ids]).bool()
            common &= condition.mesh.valid[:, None] & condition.mesh.valid[None, :]
            common.fill_diagonal_(False)
            depth = pool(torch.where(mask, condition.geometry_maps[:, 0], 0.)) / depth_known.clamp_min(1e-6)
            both = (depth_known[:, None] > 0) & (depth_known[None, :] > 0) & common[..., None]
            difference = (depth[None, :] - depth[:, None]) / .04
            frontness = (difference.tanh() * both).sum(1) / both.sum(1).clamp_min(1)
    evidence = contact_evidence(graph)
    order = (evidence[..., 3] * graph.neighbors).sum(1) / graph.neighbors.sum(1).clamp_min(1)
    touch = torch.maximum(evidence[..., 0], evidence[..., 5]).amax(1)
    return torch.stack((owned, frontness, depth_known, order[:, None].expand_as(owned),
                        touch[:, None].expand_as(owned)), dim=-1).transpose(0, 1)


class InstanceSpatialAttention(nn.Module):
    def __init__(self, hidden_width, width):
        super().__init__()
        self.query = nn.Linear(hidden_width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.metadata = nn.Linear(5, 1, bias=False)
        self.scale = 1 / math.sqrt(width)

    def forward(self, hidden, instances, support, metadata):
        query = self.query(hidden)[:, None]
        logits = (query * self.key(instances)).sum(-1) * self.scale
        logits = logits + self.metadata(metadata).squeeze(-1) + support.clamp_min(1e-6).log()
        valid = support > 0
        weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1) * valid
        selected = (self.value(instances) * weights[..., None]).sum(1)
        return selected, weights


class TypedDenoiseGate(nn.Module):
    """Five gates learn both a phase curve and a correction from the current denoising state."""

    def __init__(self, width, layers):
        super().__init__()
        self.layer_embedding = nn.Embedding(layers, width)
        # Knots are sigma=0 (late), .5 (mid), 1 (early), and remain trainable.
        self.phase_logits = nn.Parameter(torch.tensor([
            [-.5, .8, 1.8, .5, 2.0],
            [1.0, 2.0, .8, 2.0, .6],
            [2.0, .9, -.5, .6, -.6],
        ]))
        self.controller = nn.Sequential(nn.Linear(width * 3 + 2, width), nn.SiLU(), nn.Linear(width, 5))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, feature, spatial, support, sigma, layer):
        time = sigma.reshape(()).float()
        basis = (1 - (time * 2 - time.new_tensor([0., 1., 2.])).abs()).clamp_min(0)
        phase = basis @ self.phase_logits
        embedding = self.layer_embedding.weight[layer].expand(len(feature), -1)
        support = support.reshape(len(feature), -1).mean(-1, keepdim=True)
        inputs = torch.cat((feature, spatial, embedding, support, time.expand(len(feature), 1)), dim=-1)
        return (phase + self.controller(inputs)).sigmoid()
