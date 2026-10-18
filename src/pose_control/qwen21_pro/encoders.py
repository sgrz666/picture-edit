"""Typed geometry 与成对接触编码；无绝对人物编号 embedding。"""
from __future__ import annotations

import torch
from torch import nn

from .contracts import ProAdapterConfig, SPATIAL_CHANNELS, TaskCondition


class ContactEncoder(nn.Module):
    """按 query 人物给两端加 self/other 标记，再保留顺序 concat。"""

    def __init__(self, config: ProAdapterConfig):
        super().__init__()
        self.width, self.max_parts = config.width, config.max_parts
        self.part_embedding = nn.Embedding(config.max_parts, config.width)
        self.endpoint_role = nn.Embedding(2, config.width)
        self.pair_projection = nn.Sequential(nn.Linear(2 * config.width + 6, config.width), nn.SiLU(), nn.Linear(config.width, config.width))

    def forward(self, conditions: TaskCondition):
        relation = conditions.relations
        b, p = conditions.person_valid.shape
        if relation.valid.shape[1] == 0:
            return self.part_embedding.weight.new_zeros(b, p, self.width)
        src_part = self.part_embedding(relation.src_part.clamp(0, self.max_parts - 1))
        dst_part = self.part_embedding(relation.dst_part.clamp(0, self.max_parts - 1))
        result = []
        for person in range(p):
            src_self = relation.src_person == person
            dst_self = relation.dst_person == person
            src = src_part + self.endpoint_role(src_self.long())
            dst = dst_part + self.endpoint_role(dst_self.long())
            paired = self.pair_projection(torch.cat((src, dst, relation.relative.float()), dim=-1))
            weight = (relation.valid & (src_self | dst_self)).float() * relation.confidence.float()
            encoded = (paired * weight[..., None]).sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp_min(1)
            result.append(encoded * conditions.person_valid[:, person, None])
        return torch.stack(result, dim=1)


class GeometryEncoder(nn.Module):
    """各模态有独立投影，part 用离散 embedding；支持 amodal mask。"""

    def __init__(self, config: ProAdapterConfig):
        super().__init__()
        self.width, self.max_parts = config.width, config.max_parts
        self.projections = nn.ModuleDict({name: nn.Linear(channels, config.width) for name, channels in SPATIAL_CHANNELS.items() if name != "part"})
        self.part_embedding = nn.Embedding(config.max_parts, config.width)
        self.roi_projection = nn.Linear(3, config.width, bias=False)
        self.contacts = ContactEncoder(config)

    def forward(self, conditions: TaskCondition):
        b, p, n = conditions.person_mask.shape
        encoded = self.part_embedding.weight.new_zeros(b, p, n, self.width)
        for name, value in conditions.spatial.items():
            flat = value.flatten(-2).transpose(-2, -1)
            if name == "part":
                labels = flat[..., 0]
                signal = self.part_embedding(labels.clamp_min(0)) * (labels >= 0)[..., None]
            else:
                signal = self.projections[name](flat.float())
            encoded = encoded + signal * conditions.spatial_valid[name][..., None, None]
        roi = torch.cat((conditions.face_mask[..., None], conditions.hand_mask.transpose(-2, -1)), dim=-1)
        encoded = encoded + self.roi_projection(roi.float()) + self.contacts(conditions)[:, :, None]
        return encoded * conditions.person_mask.float()[..., None] * conditions.person_valid[..., None, None]
