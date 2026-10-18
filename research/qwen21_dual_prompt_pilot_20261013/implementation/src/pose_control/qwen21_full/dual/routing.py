"""Slot-symmetric routing on the RGB grid before area reduction to tokens."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .conditions import GeometryCondition


def compute_routing(geometry: GeometryCondition, target_hw: tuple[int, int]):
    mask = geometry.amodal_mask
    valid = geometry.person_valid[..., None, None, None]
    active = (mask > 0) & valid
    a, b = active[:, 0], active[:, 1]
    da, db = geometry.depth[:, 0], geometry.depth[:, 1]
    overlap = a & b
    # Equal depth is shared, with no slot-index tie breaker.
    tie = overlap & torch.isclose(da, db, atol=1e-5, rtol=1e-5)
    owner_a = ((a & ~b) | (overlap & (da < db) & ~tie)).to(mask.dtype) + .5*tie
    owner_b = ((b & ~a) | (overlap & (db < da) & ~tie)).to(mask.dtype) + .5*tie
    owner = torch.stack((owner_a, owner_b), dim=1)
    batch, persons, _, height, width = owner.shape
    # Dilation radius is eight original RGB pixels, independent of token resolution.
    expanded = F.max_pool2d(owner.flatten(0,1), kernel_size=17, stride=1, padding=8).reshape(batch,persons,1,height,width)
    expanded = expanded * valid
    # Compete with the closest visible camera surface within each dilation.
    # A symmetric tie is shared. Appearance retains the original owner below.
    local_depth = torch.where(owner>0,geometry.depth,torch.full_like(geometry.depth,float('inf')))
    local_depth = -F.max_pool2d(-local_depth.flatten(0,1),17,1,8).reshape_as(local_depth)
    nearest = local_depth.amin(1,keepdim=True)
    winners = (expanded>0) & torch.isfinite(local_depth) & torch.isclose(local_depth,nearest,atol=1e-5,rtol=0)
    expanded = winners.to(mask.dtype) / winners.sum(1,keepdim=True).clamp_min(1)
    coverage = expanded.sum(1)
    has_contact = geometry.contact_valid.any(-1)[:,None,None,None]
    contact = torch.minimum(geometry.contact_map, coverage) * has_contact
    geo = expanded / coverage.clamp_min(1e-8)[:,None] * (coverage-contact)[:,None]
    app = geometry.visible_owner * valid

    def persons_to_tokens(value):
        reduced = F.adaptive_avg_pool2d(value.flatten(0,1),target_hw)
        return reduced.flatten(2).transpose(1,2).reshape(batch,2,-1,1)

    contact = F.adaptive_avg_pool2d(contact,target_hw).flatten(2).transpose(1,2)
    return persons_to_tokens(geo), persons_to_tokens(app), contact
