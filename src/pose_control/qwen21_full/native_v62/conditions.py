"""Build V6.2 permissions from existing typed geometry and stable identities."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from ..depth_condition_v2 import DepthConditionV2
from ..dual_depth_condition import DualDepthConditionV3
from ..native_local_condition import NativeLocalConditionV4
from ..native_v5.source_bank import NativeSourceBankV5
from ..native_v6.contracts import PoseGeometry
from ..native_v61.contracts import PoseSpatialConditionV61
from ..native_v61.spatial import _rows, validate_repair_condition
from .contracts import (PoseLinkConfigV62, RoutingConditionV62, BranchSpatialGatesV62,
    ROUTING_SCHEMA, GATE_SCHEMA)


def _validate_inputs(geometry, spatial, config):
    if (not isinstance(geometry, PoseGeometry) or not isinstance(spatial, PoseSpatialConditionV61)
            or not isinstance(config, PoseLinkConfigV62)):
        raise ValueError('V6.2 builders require typed geometry, spatial and config')
    spatial.validate(geometry)


def _local_binding(batch, geometry, *, required):
    local = batch.get('local_condition_v4')
    if local is None:
        if required:
            raise ValueError('person routing requires typed local_condition_v4 stable person IDs')
        return None
    if not isinstance(local, NativeLocalConditionV4):
        raise ValueError('local_condition_v4 requires original typed raw condition')
    local = local.to(geometry.geometry.device).validate()
    if not torch.equal(local.person_valid, geometry.person_valid):
        raise ValueError('local and geometry person_valid differ')
    if local.region_masks.shape[-2:] != tuple(v * 16 for v in geometry.target_token_hw):
        raise ValueError('local and geometry pixel grids differ')
    for row, meta in enumerate(local.metadata):
        if meta['target_canvas_id'] != geometry.canvas_ids[row]:
            raise ValueError('local target_canvas_id differs from geometry canvas')
        target, source = meta['target_person_ids'], meta['source_person_ids']
        active_ids = []
        for person in range(2):
            if not local.person_valid[row, person]:
                continue
            if (not isinstance(target[person], str) or not target[person] or target[person] == '__inactive__'
                    or source[int(local.source_indices[row, person])] != target[person]):
                raise ValueError('local source/target stable person mapping differs')
            active_ids.append(target[person])
        if len(active_ids) != len(set(active_ids)):
            raise ValueError('active target_person_ids must be distinct')
    bank = batch.get('source_bank_v5')
    if bank is not None:
        if not isinstance(bank, NativeSourceBankV5):
            raise ValueError('source_bank_v5 requires typed original provenance')
        bank.validate()
        if len(bank.target_person_ids) != geometry.batch_size or not torch.equal(bank.source_indices.cpu(), local.source_indices.cpu()):
            raise ValueError('source bank and local source image mapping differ')
        slots = [s.source_slot_id for s in bank.sources]
        if local.source_slot_ids.cpu().tolist() != [slots for _ in geometry.canvas_ids]:
            raise ValueError('source bank and local source logical slots differ')
        for row, meta in enumerate(local.metadata):
            expected = {'source_person_ids': [s.person_ids[0] for s in bank.sources],
                'target_person_ids': list(bank.target_person_ids[row]),
                'source_canvas_id': CanvasSpec.from_record(bank.sources[0].canvas).fingerprint(),
                'source_rgb_sha256': bank.sources[0].rgb_sha256}
            if 'source_asset_sha256' in meta:
                expected['source_asset_sha256'] = bank.sources[0].geometry_sha256
            for name, wanted in expected.items():
                actual = meta.get(name)
                if isinstance(actual, tuple):
                    actual = list(actual)
                if actual != wanted:
                    raise ValueError('source bank and local stable source binding differs: ' + name)
    for person, suffix in enumerate(('', '_b')):
        provenance = batch.get('pose_provenance' + suffix)
        if provenance is None:
            continue
        for row, meta in enumerate(_rows(provenance, geometry.batch_size, 'pose_provenance' + suffix)):
            if not geometry.person_valid[row, person]:
                continue
            expected = {'person_id': local.metadata[row]['target_person_ids'][person],
                'canvas_id': geometry.canvas_ids[row], 'record_sha256': local.metadata[row]['driver_record_sha256']}
            if any(meta.get(name) != wanted for name, wanted in expected.items()):
                raise ValueError('raw BodyPose Canvas/person/driver provenance differs')
    return local


def _repair(batch, geometry):
    repair = batch.get('pose_repair_v61')
    if repair is None:
        return None
    _local_binding(batch, geometry, required=True)
    validate_repair_condition(repair, batch)
    mask = repair['mask'].to(device=geometry.geometry.device)
    if mask.shape != (geometry.batch_size, 2, *(x * 16 for x in geometry.target_token_hw)):
        raise ValueError('repair and geometry pixel grids differ')
    return mask


def _parts_contacts(batch, geometry, local, spatial):
    b, n = geometry.coverage.shape
    device = geometry.geometry.device
    parts = torch.zeros(b, 2, n, 14, device=device, dtype=torch.float32)
    positive = torch.zeros(b, 14, 14, device=device, dtype=torch.bool)
    forbidden = torch.zeros_like(positive)
    ignored = [0] * b
    dual = batch.get('dual_depth')
    single = batch.get('depth_condition_v2')
    if isinstance(single, list):
        single = DepthConditionV2.stack(single)
    if dual is not None and single is not None:
        raise ValueError('declare either dual_depth or depth_condition_v2')
    if dual is not None:
        if not isinstance(dual, DualDepthConditionV3):
            raise ValueError('dual_depth requires original typed DualDepthConditionV3')
        dual = dual.to(device).validate(target_token_hw=geometry.target_token_hw)
        if dual.person_part.shape != (b, 2, 14, *(v * 16 for v in geometry.target_token_hw)):
            raise ValueError('dual depth and geometry grids differ')
        for row, meta in enumerate(dual.metadata):
            if meta['canvas_id'] != geometry.canvas_ids[row]:
                raise ValueError('dual depth Canvas differs from geometry')
            if local is None or list(meta['person_ids']) != list(local.metadata[row]['target_person_ids']):
                raise ValueError('dual depth stable person_ids differ from local target mapping')
            if meta['record_sha256'] != local.metadata[row]['driver_record_sha256']:
                raise ValueError('dual depth driver_record_sha256 differs from local provenance')
            for child in (dual.person_a, dual.person_b):
                if child.metadata[row]['pose_record_sha256'] != local.metadata[row]['driver_record_sha256']:
                    raise ValueError('dual child part-map driver geometry differs from local provenance')
        known_parts = torch.stack((dual.person_a.context.part_valid, dual.person_b.context.part_valid), 1)
        raw = dual.person_part * known_parts * spatial.valid_pixels[:, None]
        parts = F.avg_pool2d(raw.flatten(0, 1), 16).flatten(2).transpose(1, 2).reshape(b, 2, n, 14)
        flags = dual.part_contact
        positive = (flags[..., 0] > 0) & (flags[..., 1] > 0)
        forbidden = ((flags[..., 1] > 0) & ~positive) | ((flags[..., 2] > 0) & (flags[..., 3] > 0))
        positive = positive & ~forbidden
        # Raw vertex indices have no stable 14-part correspondence in this input.
        # They remain diagnostic records and never relax a part permission.
        ignored = dual.contacts.valid.sum(1).cpu().tolist()
    elif single is not None:
        if not isinstance(single, DepthConditionV2):
            raise ValueError('depth_condition_v2 requires original typed DepthConditionV2')
        single = single.to(device).validate(target_token_hw=geometry.target_token_hw)
        if geometry.person_valid[:, 1].any():
            raise ValueError('single depth cannot supply an active person B')
        if single.relative.shape[0] != b:
            raise ValueError('single depth and geometry batch differs')
        for row, meta in enumerate(single.metadata):
            if meta['canvas_id'] != geometry.canvas_ids[row] or meta['pose_canvas_id'] != geometry.canvas_ids[row]:
                raise ValueError('single depth Canvas differs from geometry')
            if meta.get('person_id') is not None and (local is None or meta['person_id'] != local.metadata[row]['target_person_ids'][0]):
                raise ValueError('single depth person_id differs from local target mapping')
            if local is None or meta['pose_record_sha256'] != local.metadata[row]['driver_record_sha256']:
                raise ValueError('single depth driver geometry differs from local provenance')
        first = F.avg_pool2d(single.context.part_onehot * single.context.part_valid * spatial.valid_pixels, 16).flatten(2).transpose(1, 2)
        parts = torch.stack((first, torch.zeros_like(first)), 1)
    parts = parts * geometry.person_valid[:, :, None, None]
    return parts, positive, forbidden, ignored


def build_routing_conditions(batch, geometry: PoseGeometry, spatial: PoseSpatialConditionV61,
        config: PoseLinkConfigV62) -> RoutingConditionV62:
    _validate_inputs(geometry, spatial, config)
    local = _local_binding(batch, geometry, required=True) if config.person_routing else None
    repair = _repair(batch, geometry) if config.person_routing else None
    if config.person_routing:
        parts, positive, forbidden, ignored = _parts_contacts(batch, geometry, local, spatial)
    else:
        # Disabled routing must not demand extra part/identity bindings on an
        # otherwise valid legacy Body-only call. Nothing consumes this evidence.
        b, n = geometry.coverage.shape
        parts = geometry.geometry.new_zeros(b, 2, n, 14, dtype=torch.float32)
        positive = torch.zeros(b, 14, 14, device=parts.device, dtype=torch.bool)
        forbidden = torch.zeros_like(positive)
        ignored = [0] * b
    person_mask = geometry.person_valid[:, None]
    owner = geometry.owner.float() * person_mask
    projected = geometry.projected.float() * person_mask
    support = F.avg_pool2d(spatial.key_support, 16).flatten(2).transpose(1, 2) * person_mask
    evidence = owner
    sources = torch.zeros_like(geometry.coverage, dtype=torch.long)
    for index, candidate in enumerate((projected, support), 1):
        missing = evidence.sum(-1) <= config.routing_epsilon
        take = missing & (candidate.sum(-1) > config.routing_epsilon)
        evidence = torch.where(missing[..., None], candidate, evidence)
        sources = torch.where(take, index, sources)
    if repair is not None:
        repair_support = F.avg_pool2d(repair * spatial.valid_pixels, 16).flatten(2).transpose(1, 2) * person_mask
        missing = evidence.sum(-1) <= config.routing_epsilon
        sources = torch.where(missing & (repair_support.sum(-1) > config.routing_epsilon), 3, sources)
        evidence = torch.where(missing[..., None], repair_support, evidence)
    total = evidence.sum(-1)
    probability = evidence / total[..., None].clamp_min(config.routing_epsilon)
    strength, labels = probability.max(-1)
    hard = (total > config.routing_epsilon) & (strength >= config.owner_threshold) & (geometry.coverage > 0)
    if not config.person_routing:
        hard = torch.zeros_like(hard)
    labels = torch.where(hard, labels, -1).long()
    metadata, person_ids = [], []
    for row, canvas in enumerate(geometry.canvas_ids):
        ids = ('', '') if local is None else tuple(local.metadata[row]['target_person_ids'])
        person_ids.append(ids)
        metadata.append({'schema': ROUTING_SCHEMA, 'target_canvas_id': canvas,
            'routing_enabled': config.person_routing,
            'person_ids_status': ('missing_routing_disabled' if batch.get('local_condition_v4') is None
                else 'not_consumed_routing_disabled') if local is None else 'raw_local_binding',
            'local_spatial_policy': 'inherited_owned_roi', 'ignored_unmapped_vertex_contacts': int(ignored[row]),
            'hard_query_count': int(hard[row].sum()), 'soft_query_count': int((~hard[row]).sum()),
            'owner_source_counts': {name: int(((sources[row] == i) & hard[row]).sum())
                for i, name in enumerate(('visible', 'projected', 'spatial', 'repair'))}})
    return RoutingConditionV62(labels, parts, positive, forbidden, geometry.person_valid,
        geometry.target_token_hw, geometry.canvas_ids, tuple(person_ids), tuple(metadata)).validate(geometry)


def build_branch_gates(batch, geometry: PoseGeometry, spatial: PoseSpatialConditionV61,
        config: PoseLinkConfigV62) -> BranchSpatialGatesV62:
    _validate_inputs(geometry, spatial, config)
    # Even an explicit empty mask is a supplied repair; malformed records never
    # silently switch to the missing-repair compatibility path.
    repair = _repair(batch, geometry) if config.body_spatial_gate else None
    body = torch.ones_like(geometry.coverage, dtype=torch.float32)
    body_reason = 'disabled'
    if config.body_spatial_gate:
        if repair is None:
            body_reason = 'missing_repair_legacy_coverage'
        else:
            body, body_reason = spatial.edit_gate, 'explicit_repair'
    interaction = torch.ones_like(body)
    if config.interaction_spatial_gate:
        union = spatial.key_support.amax(1, keepdim=True)
        numerator = F.avg_pool2d(union, 16).flatten(1)
        coverage = geometry.coverage.float()
        interaction = torch.where(coverage > 0, numerator / coverage.clamp_min(config.routing_epsilon), 0.).clamp(0, 1)
        interaction = interaction * geometry.person_valid.all(1)[:, None]
    metadata = tuple({'schema': GATE_SCHEMA, 'target_canvas_id': canvas, 'body_gate_reason': body_reason,
        'interaction_gate_reason': 'body_support_union' if config.interaction_spatial_gate else 'disabled',
        'local_spatial_policy': 'inherited_owned_roi', 'relation_valid_applied_externally': True}
        for canvas in geometry.canvas_ids)
    return BranchSpatialGatesV62(body, interaction, geometry.target_token_hw, geometry.canvas_ids, metadata).validate(geometry)


def routing_permissions(query_ids, query_parts, key_person_ids, key_parts, positive, forbidden,
        key_valid, epsilon=1e-6):
    """Return one query chunk's boolean permissions, with no persistent matrix.

    Part relations are stored in person0 -> person1 order. Reverse reads use
    their transpose. An uncertain query keeps every valid, labeled key available.
    """
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon requires a finite positive number')
    if (not torch.is_tensor(query_ids) or query_ids.ndim != 2 or query_ids.dtype != torch.long
            or not torch.is_tensor(key_person_ids) or key_person_ids.ndim != 2 or key_person_ids.dtype != torch.long):
        raise ValueError('query/key person IDs require int64 [B,Q/K]')
    b, q = query_ids.shape
    k = key_person_ids.shape[1]
    device = query_ids.device
    for name, value, shape, dtype in (('key_person_ids', key_person_ids, (b, k), torch.long),
            ('key_valid', key_valid, (b, k), torch.bool), ('positive', positive, (b, 14, 14), torch.bool),
            ('forbidden', forbidden, (b, 14, 14), torch.bool)):
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
            raise ValueError(name + ' has incompatible routing shape/dtype/device')
    for value in (query_ids, key_person_ids):
        if bool(((value < -1) | (value > 1)).any()):
            raise ValueError('routing person IDs must be -1,0,1')
    if torch.is_tensor(query_parts) and tuple(query_parts.shape) == (b, q, 2, 14):
        query_parts = query_parts.gather(2, query_ids.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 14)).squeeze(2)
    for name, value, shape in (('query_parts', query_parts, (b, q, 14)), ('key_parts', key_parts, (b, k, 14))):
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.device != device
                or not value.is_floating_point() or not bool(torch.isfinite(value).all()) or bool((value < 0).any())):
            raise ValueError(name + ' requires finite nonnegative part fractions')
    qp, kp = (query_parts > epsilon).float(), (key_parts > epsilon).float()
    def matches(matrix):
        forward = (qp @ matrix.float()) @ kp.transpose(1, 2) > 0
        reverse = (qp @ matrix.transpose(1, 2).float()) @ kp.transpose(1, 2) > 0
        return torch.where(query_ids[:, :, None] == 0, forward, reverse)
    cross = matches(positive) & ~matches(forbidden)
    same = query_ids[:, :, None] == key_person_ids[:, None]
    uncertain = query_ids[:, :, None] == -1
    return (same | uncertain | cross) & (key_person_ids[:, None] >= 0) & key_valid[:, None]
