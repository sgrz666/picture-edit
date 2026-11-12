"""Support-normalized global and sparse Pose sampling in exact Canvas coordinates."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from ..native_v6.poselink import _align, _budget_grid, _flat, _grid, _map, _pool


@dataclass(frozen=True)
class PoseKeySetV61:
    hidden: torch.Tensor
    xy: torch.Tensor
    owner: torch.Tensor
    contact: torch.Tensor
    geometry: torch.Tensor
    person_tokens: torch.Tensor
    person_weights: torch.Tensor
    interaction: torch.Tensor | None
    weight: torch.Tensor
    valid: torch.Tensor
    person_ids: torch.Tensor
    type_ids: torch.Tensor
    reference_ids: torch.Tensor


def _pixel_to_tokens(value, hw):
    return _flat(F.interpolate(value, size=hw, mode='area'))


def _sample(values, weight, hw, xy):
    """Normalize every bilinear footprint before pooling; invalid cells cannot bleed."""
    numerator = torch.where(weight[..., None] > 0, values, 0) * weight[..., None]
    grid = (xy * 2 - 1)[:, :, None]
    top = F.grid_sample(_map(numerator, hw), grid, align_corners=False).squeeze(-1).transpose(1, 2)
    bottom = F.grid_sample(_map(weight[..., None], hw), grid, align_corners=False).squeeze(-1).transpose(1, 2)
    return top / bottom.clamp_min(torch.finfo(top.dtype).eps), bottom[..., 0]


def _shares(total, people):
    return {person: total // len(people) + (rank < total % len(people))
            for rank, person in enumerate(people)} if people else {}


def _sparse_coordinates(spatial, geometry, people, kind, budget):
    rows = []
    for row in range(geometry.batch_size):
        active = [person for person in people if geometry.person_valid[row, person]]
        points, ids = [], []
        for person, quota in _shares(budget, active).items():
            if kind == 1:
                indices = spatial.joint_valid[row, person].nonzero().flatten()[:quota]
                points.extend(spatial.joint_uv[row, person, index] for index in indices)
                ids.extend([person] * len(indices))
                continue
            regions = spatial.local_valid[row, person].nonzero().flatten().tolist()
            if not regions or not quota:
                continue
            # Fixed anatomical order, round-robin region allocation; both people
            # receive their own quota before any region receives another sample.
            allocations = [quota // len(regions) + (i < quota % len(regions)) for i in range(len(regions))]
            banks = []
            for region, count in zip(regions, allocations):
                side = max(1, math.ceil(math.sqrt(count)))
                unit = _grid((side, side), 1, spatial.local_boxes)[0, :count]
                box = spatial.local_boxes[row, person, region]
                banks.append(box[:2] + unit * (box[2:] - box[:2]))
            for step in range(max(allocations, default=0)):
                for bank in banks:
                    if step < len(bank):
                        point = bank[step]
                        if bool(((point >= 0) & (point < 1)).all()):
                            points.append(point); ids.append(person)
        rows.append((points, ids))
    count = max((len(points) for points, _ in rows), default=0)
    xy = spatial.joint_uv.new_zeros(geometry.batch_size, count, 2)
    person_ids = torch.full((geometry.batch_size, count), -1, device=xy.device, dtype=torch.long)
    valid = torch.zeros(geometry.batch_size, count, device=xy.device, dtype=torch.bool)
    for row, (points, ids) in enumerate(rows):
        if points:
            xy[row, :len(points)] = torch.stack(points)
            person_ids[row, :len(points)] = torch.tensor(ids, device=xy.device)
            valid[row, :len(points)] = True
    return xy, person_ids, valid


def sample_pose_keys(joint, layout, geometry, spatial, config):
    """Raw per-reference samples, bounded independently of image resolution."""
    b = geometry.batch_size
    hw = geometry.target_token_hw
    pieces = []
    for ref in layout.references:
        person = 0 if ref.role == 'person_0_pose' else 1 if ref.role == 'person_1_pose' else None
        people = (person,) if person is not None else (0, 1)
        active = geometry.person_valid.to(joint.dtype)
        pixels = spatial.key_support if config.geometry_key_mask else spatial.valid_pixels.expand(-1, 2, -1, -1)
        person_pixels = pixels.to(joint.dtype) * active[:, :, None, None]
        reference_pixels = person_pixels[:, list(people)].amax(1, keepdim=True)
        target_weight = _pixel_to_tokens(reference_pixels, hw)[..., 0]
        pose_weight = _pixel_to_tokens(reference_pixels, ref.token_hw)[..., 0]
        pose = joint.index_select(1, ref.indices.to(joint.device))

        def align(value):
            return _align(value.to(joint.dtype), target_weight, hw, ref.token_hw)

        raw_fields = dict(geometry=geometry.geometry, owner=geometry.owner, contact=geometry.contact,
                          projected=geometry.projected,
                          person_0=geometry.person_tokens[:, 0], person_1=geometry.person_tokens[:, 1])
        if geometry.interaction is not None:
            raw_fields['interaction'] = geometry.interaction
        aligned = {name: align(value) for name, value in raw_fields.items()}

        def append(hidden, xy, weights, valid, ids, type_id, gather):
            geo = gather('geometry')
            owner = gather('owner') * active[:, None]
            contact = gather('contact') * active[:, None]
            support = gather('projected') * active[:, None]
            per_person = torch.stack([gather('person_'+str(p)) for p in range(2)], 2)
            person_weights = support / support.sum(-1, keepdim=True).clamp_min(1.)
            bound = ids >= 0
            safe = ids.clamp_min(0)
            onehot = F.one_hot(safe, 2).to(owner.dtype)
            person_weights = torch.where(bound[..., None], onehot*support, person_weights)
            owner = torch.where(bound[..., None], onehot, owner)
            contact = torch.where(bound[..., None], contact * onehot, contact)
            interaction = None if geometry.interaction is None else gather('interaction')
            valid = valid & (weights > 0)
            pieces.append(PoseKeySetV61(hidden, xy, owner, contact, geo, per_person, person_weights, interaction,
                weights, valid, ids, torch.full_like(ids, type_id), torch.full_like(ids, ref.image_id)))

        budget = config.global_token_budget if config.multiscale_tokens else config.max_pose_tokens_per_reference
        if budget:
            pool_hw = _budget_grid(ref.token_hw, budget)
            weights = _flat(F.adaptive_avg_pool2d(_map(pose_weight[..., None], ref.token_hw), pool_hw))[..., 0]
            def pooled(value): return _pool(value, pose_weight, ref.token_hw, pool_hw)
            hidden = pooled(pose)
            xy = pooled(_grid(ref.token_hw, b, joint))
            ids = torch.full(weights.shape, -1 if person is None else person, device=joint.device, dtype=torch.long)
            append(hidden, xy, weights, weights > 0, ids, 0, lambda name: pooled(aligned[name]))

        if config.multiscale_tokens:
            for kind, budget in ((1, config.joint_token_budget), (2, config.local_token_budget)):
                if not budget: continue
                xy, ids, valid = _sparse_coordinates(spatial, geometry, people, kind, budget)
                if not xy.shape[1]: continue
                hidden = joint.new_zeros(b, xy.shape[1], joint.shape[-1])
                weights = joint.new_zeros(b, xy.shape[1])
                sampled = {name: joint.new_zeros(b, xy.shape[1], value.shape[-1])
                           for name, value in raw_fields.items()}
                # Person binding must precede every bilinear footprint, including
                # target->reference geometry alignment. Post-filtering a union
                # sample would still retain the other person's values/gradients.
                for p in people:
                    pixels_p = person_pixels[:, p:p+1]
                    target_p = _pixel_to_tokens(pixels_p, hw)[..., 0]
                    pose_p = _pixel_to_tokens(pixels_p, ref.token_hw)[..., 0]
                    chosen = ids == p
                    value_p, weight_p = _sample(pose, pose_p, ref.token_hw, xy)
                    hidden = torch.where(chosen[..., None], value_p, hidden)
                    weights = torch.where(chosen, weight_p, weights)
                    for name, raw_value in raw_fields.items():
                        aligned_p = _align(raw_value.to(joint.dtype), target_p, hw, ref.token_hw)
                        value_p = _sample(aligned_p, pose_p, ref.token_hw, xy)[0]
                        sampled[name] = torch.where(chosen[..., None], value_p, sampled[name])
                append(hidden, xy, weights, valid, ids, kind, lambda name: sampled[name])

    if not pieces:
        # A dummy invalid key keeps empty-row softmax and its zero gradients finite.
        zero = joint.new_zeros(b, 1, 1)
        ids = torch.full((b, 1), -1, device=joint.device, dtype=torch.long)
        return PoseKeySetV61(zero.expand(-1, -1, config.hidden_dim), zero.expand(-1, -1, 2),
            zero.expand(-1, -1, 2), zero.expand(-1, -1, 2), zero.expand(-1, -1, 512),
            zero[..., None].expand(-1, -1, 2, 512), zero.expand(-1, -1, 2),
            None if geometry.interaction is None else zero.expand(-1, -1, 512),
            zero[..., 0], torch.zeros_like(ids, dtype=torch.bool), ids, torch.zeros_like(ids), ids)
    values = {}
    for name in PoseKeySetV61.__dataclass_fields__:
        items = [getattr(piece, name) for piece in pieces]
        values[name] = None if items[0] is None else torch.cat(items, dim=1)
    return PoseKeySetV61(**values)
