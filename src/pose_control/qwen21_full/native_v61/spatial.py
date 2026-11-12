"""Build finite target-Canvas spatial support from existing raw geometry only."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from ..body_pose import JOINT_NAMES
from ..native_local_condition import NativeLocalConditionV4
from ..native_v5.source_bank import NativeSourceBankV5
from ..native_v6.contracts import PoseGeometry
from .contracts import PoseLinkConfigV61, PoseSpatialConditionV61, SPATIAL_SCHEMA

REPAIR_SCHEMA = 'native_pose_repair_v61'
LOCAL_NAMES = ('left_hand', 'right_hand', 'left_foot', 'right_foot', 'face')


def _raw_tensor(value, shape, name, *, boolean=False, fp32=False):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} requires shape {tuple(shape)}')
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(name + ' requires bool')
    elif (not value.is_floating_point() or (fp32 and value.dtype != torch.float32)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' requires finite ' + ('FP32' if fp32 else 'floating values'))
    return value


def _rows(value, b, name):
    if isinstance(value, Mapping) and b == 1:
        return (value,)
    if not isinstance(value, (tuple, list)) or len(value) != b or any(not isinstance(v, Mapping) for v in value):
        raise ValueError(name + ' requires one mapping per sample')
    return tuple(value)


def _valid_pixels(batch, geometry):
    b = geometry.batch_size
    h, w = (x * 16 for x in geometry.target_token_hw)
    valid = _raw_tensor(batch.get('valid_image_mask'), (b, 1, h, w), 'valid_image_mask')
    if bool(((valid < 0) | (valid > 1)).any()):
        raise ValueError('valid_image_mask must be in [0,1]')
    valid = valid.to(device=geometry.geometry.device, dtype=torch.float32)
    coverage = F.avg_pool2d(valid, 16).flatten(1)
    if not torch.allclose(coverage, geometry.coverage.float(), atol=1e-6, rtol=1e-6):
        raise ValueError('valid_image_mask and geometry coverage differ')
    if batch.get('target_canvas') is not None:
        rows = _rows(batch['target_canvas'], b, 'target_canvas')
        for row, raw in enumerate(rows):
            canvas = CanvasSpec.from_record(raw)
            if canvas.canvas_hw != (h, w) or canvas.token_hw != geometry.target_token_hw:
                raise ValueError('target_canvas and geometry pixel/token grids differ')
            if canvas.fingerprint() != geometry.canvas_ids[row]:
                raise ValueError('target_canvas and geometry canvas_ids differ')
            if not torch.equal(valid[row], canvas.valid_mask(device=valid.device)):
                raise ValueError('valid_image_mask differs from target_canvas valid pixels')
    return valid


def _canonical_joints(batch, geometry, h, w):
    b, device = geometry.batch_size, geometry.geometry.device
    uv = torch.zeros(b, 2, 21, 2, device=device)
    valid = torch.zeros(b, 2, 21, dtype=torch.bool, device=device)
    reasons = []
    for person, suffix in enumerate(('', '_b')):
        raw_uv, raw_valid = batch.get('joint_uv' + suffix), batch.get('projection_valid' + suffix)
        if raw_uv is None and raw_valid is None:
            reasons.append('canonical_joint_fields_missing')
            continue
        if raw_uv is None or raw_valid is None:
            raise ValueError('joint_uv and projection_valid must be provided together' + suffix)
        # Invalid canonical projections may be outside the Canvas; they are never clamped into valid observations.
        _raw_tensor(raw_uv, (b, 21, 2), 'joint_uv' + suffix)
        _raw_tensor(raw_valid, (b, 21), 'projection_valid' + suffix, boolean=True)
        pixels = raw_uv.to(device=device, dtype=torch.float32)
        projected = raw_valid.to(device) & ((pixels >= 0) & (pixels < pixels.new_tensor([w, h]))).all(-1)
        fk = batch.get('fk_valid' + suffix)
        if fk is not None:
            _raw_tensor(fk, (b, 22), 'fk_valid' + suffix, boolean=True)
            projected = projected & fk.to(device)[:, 1:]
        projected = projected & geometry.person_valid[:, person, None]
        uv[:, person] = torch.where(projected[..., None], pixels / pixels.new_tensor([w, h]), 0.)
        valid[:, person] = projected
        reasons.append('canonical_joint_fields')
    return uv, valid, reasons


def _local_boxes(batch, geometry, uv, joint_valid, h, w):
    b, device = geometry.batch_size, geometry.geometry.device
    boxes = torch.zeros(b, 2, 5, 4, device=device)
    valid = torch.zeros(b, 2, 5, dtype=torch.bool, device=device)
    local = batch.get('local_condition_v4')
    if local is not None:
        if not isinstance(local, NativeLocalConditionV4):
            raise ValueError('local_condition_v4 must be the original typed raw condition')
        local = local.to(device).validate()
        if local.person_valid.shape != geometry.person_valid.shape or not torch.equal(local.person_valid, geometry.person_valid):
            raise ValueError('local and geometry person_valid differ')
        if local.region_masks.shape[-2:] != (h, w):
            raise ValueError('local and geometry pixel grids differ')
        for row, meta in enumerate(local.metadata):
            if meta['target_canvas_id'] != geometry.canvas_ids[row]:
                raise ValueError('local target_canvas_id differs from geometry canvas')
        boxes[:, :, :2] = local.hand_boxes.clamp(0, 1)
        valid[:, :, :2] = local.hand_roi_valid
        boxes[:, :, 4] = local.face_boxes.clamp(0, 1)
        valid[:, :, 4] = local.face_roi_valid
    # A missing face ROI can use a canonical head, while missing hands stay missing.
    fallback_names = (('left_foot', 'left_ankle'), ('right_foot', 'right_ankle'), ('head',))
    for slot, names in zip((2, 3, 4), fallback_names):
        available = torch.zeros(b, 2, dtype=torch.bool, device=device)
        center = torch.zeros(b, 2, 2, device=device)
        for name in names:
            joint = JOINT_NAMES.index(name)
            chosen = joint_valid[:, :, joint] & ~available
            center = torch.where(chosen[..., None], uv[:, :, joint], center)
            available = available | chosen
        use = available & ~valid[:, :, slot]
        radius = center.new_tensor([16 / w, 16 / h])
        candidate = torch.cat((center - radius, center + radius), -1).clamp(0, 1)
        boxes[:, :, slot] = torch.where(use[..., None], candidate, boxes[:, :, slot])
        valid[:, :, slot] = valid[:, :, slot] | use
    valid = valid & geometry.person_valid[..., None] & (boxes[..., 2:] > boxes[..., :2]).all(-1)
    boxes = torch.where(valid[..., None], boxes, 0.)
    return boxes, valid, local


def _raster_boxes(boxes, valid, h, w):
    # Iterate the small anatomical dimension rather than materialize [B,2,K,H,W].
    yy = (torch.arange(h, device=boxes.device, dtype=torch.float32) + .5) / h
    xx = (torch.arange(w, device=boxes.device, dtype=torch.float32) + .5) / w
    support = boxes.new_zeros(boxes.shape[0], 2, h, w)
    for slot in range(boxes.shape[2]):
        box = boxes[:, :, slot]
        x = (xx[None, None] >= box[..., 0, None]) & (xx[None, None] < box[..., 2, None])
        y = (yy[None, None] >= box[..., 1, None]) & (yy[None, None] < box[..., 3, None])
        mask = x[..., None, :] & y[..., :, None] & valid[:, :, slot, None, None]
        support = torch.maximum(support, mask.float())
    return support


def _expand_internal_feather(mask, expansion, feather, valid):
    """Finite dilation followed by feather only INSIDE its expanded support."""
    mask = mask * valid
    if expansion:
        mask = F.max_pool2d(mask, 2 * expansion + 1, stride=1, padding=expansion)
    mask = mask * valid
    if feather:
        mean = F.avg_pool2d(mask, 2 * feather + 1, stride=1, padding=feather)
        mask = torch.minimum(mask, mean)
    return mask * valid


def _expected_repair_metadata(batch, canvas_ids, local):
    if local is None:
        raise ValueError('pose_repair_v61 requires explicit local_condition_v4 provenance')
    bank = batch.get('source_bank_v5')
    if bank is not None:
        if not isinstance(bank, NativeSourceBankV5):
            raise ValueError('pose_repair_v61 requires typed source_bank_v5 provenance')
        bank.validate()
        source_shas = [source.rgb_sha256 for source in bank.sources]
        if len(bank.target_person_ids) != len(canvas_ids):
            raise ValueError('source bank repair provenance batch differs')
        if not torch.equal(bank.source_indices.cpu(), local.source_indices.cpu()):
            raise ValueError('source bank and local source image mapping differ')
        slots = [source.source_slot_id for source in bank.sources]
        if local.source_slot_ids.cpu().tolist() != [slots for _ in canvas_ids]:
            raise ValueError('source bank and local source logical slots differ')
    expected = []
    for row, meta in enumerate(local.metadata):
        shas = source_shas if bank is not None else [meta.get('source_rgb_sha256')]
        ids = meta.get('target_person_ids')
        driver = meta.get('driver_record_sha256')
        if (any(not isinstance(sha, str) or not sha for sha in shas) or not driver
                or not isinstance(ids, (tuple, list)) or len(ids) != 2):
            raise ValueError('pose_repair_v61 requires exact source_rgb_sha256s/target_person_ids/driver_record_sha256')
        source_ids = meta['source_person_ids']
        for person in range(2):
            if local.person_valid[row, person]:
                source_index = int(local.source_indices[row, person])
                if source_ids[source_index] != ids[person]:
                    raise ValueError('local source person mapping differs from target_person_ids')
        if bank is not None and list(bank.target_person_ids[row]) != list(ids):
            raise ValueError('source bank and local target_person_ids differ')
        if bank is not None:
            primary = bank.sources[0]
            source_bindings = {'source_canvas_id': CanvasSpec.from_record(primary.canvas).fingerprint(),
                'source_rgb_sha256': primary.rgb_sha256,
                'source_person_ids': [source.person_ids[0] for source in bank.sources]}
            if 'source_asset_sha256' in meta:
                source_bindings['source_asset_sha256'] = primary.geometry_sha256
            for key, value in source_bindings.items():
                if meta.get(key) != value:
                    raise ValueError('source bank and local source binding differ: ' + key)
        expected.append({'target_canvas_id': canvas_ids[row], 'source_rgb_sha256s': list(shas),
            'target_person_ids': list(ids), 'driver_record_sha256': driver})
    return expected


def validate_repair_condition(repair, batch):
    """Verify an optional raw repair before constructing any learned modules.

    Return the original mapping unchanged, preserving the raw mask and autograd.
    Both an uncollated B1 sample and a collated batch are supported. NativeLocalV4
    supplies the exact Canvas, ordered people and Driver geometry fingerprint;
    a two-source bank supplies its two RGB fingerprints in Qwen image order.
    """
    if (not isinstance(repair, Mapping) or repair.get('schema') != REPAIR_SCHEMA
            or repair.get('role') != 'source_repair'):
        raise ValueError('pose_repair_v61 requires native_pose_repair_v61 schema and source_repair role')
    local = batch.get('local_condition_v4')
    if not isinstance(local, NativeLocalConditionV4):
        raise ValueError('pose_repair_v61 requires explicit typed local_condition_v4 provenance')
    local.validate()
    b = local.person_valid.shape[0]
    h, w = local.region_masks.shape[-2:]
    canvas_ids = tuple(meta['target_canvas_id'] for meta in local.metadata)
    target_records = batch.get('target_canvas')
    if target_records is not None:
        for row, raw_canvas in enumerate(_rows(target_records, b, 'target_canvas')):
            canvas = CanvasSpec.from_record(raw_canvas)
            if canvas.canvas_hw != (h, w) or canvas.fingerprint() != canvas_ids[row]:
                raise ValueError('pose_repair_v61 target_canvas_id/grid differs from local condition')
    source_records = batch.get('source_canvas')
    if source_records is not None:
        for row, raw_canvas in enumerate(_rows(source_records, b, 'source_canvas')):
            if CanvasSpec.from_record(raw_canvas).fingerprint() != local.metadata[row]['source_canvas_id']:
                raise ValueError('pose_repair_v61 source_canvas differs from local source binding')
    valid_pixels = batch.get('valid_image_mask')
    if valid_pixels is not None:
        if b == 1 and torch.is_tensor(valid_pixels) and valid_pixels.ndim == 3:
            valid_pixels = valid_pixels[None]
        _raw_tensor(valid_pixels, (b, 1, h, w), 'valid_image_mask')
        if bool(((valid_pixels < 0) | (valid_pixels > 1)).any()):
            raise ValueError('valid_image_mask must be in [0,1]')
        if target_records is not None:
            for row, raw_canvas in enumerate(_rows(target_records, b, 'target_canvas')):
                canvas = CanvasSpec.from_record(raw_canvas)
                if not torch.equal(valid_pixels[row], canvas.valid_mask(device=valid_pixels.device)):
                    raise ValueError('pose_repair_v61 valid_image_mask differs from exact target Canvas')
    shape = (b, 2, h, w)
    mask = _raw_tensor(repair.get('mask'), shape, 'pose_repair_v61.mask', fp32=True)
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError('pose_repair_v61.mask must be in [0,1]')
    expected = _expected_repair_metadata(batch, canvas_ids, local)
    metadata = _rows(repair.get('metadata'), b, 'pose_repair_v61.metadata')
    for row, (actual, wanted) in enumerate(zip(metadata, expected)):
        for key, value in wanted.items():
            candidate = actual.get(key)
            if key in ('source_rgb_sha256s', 'target_person_ids') and isinstance(candidate, (tuple, list)):
                candidate = list(candidate)
            if candidate != value:
                raise ValueError('pose_repair_v61 provenance differs: ' + key)
        if bool((mask[row] * ~local.person_valid[row, :, None, None].to(mask.device)).any()):
            raise ValueError('pose_repair_v61 inactive target_person_ids cannot have repair pixels')
    for person, suffix in enumerate(('', '_b')):
        provenance = batch.get('pose_provenance' + suffix)
        if provenance is None:
            continue
        for row, actual in enumerate(_rows(provenance, b, 'pose_provenance' + suffix)):
            if not local.person_valid[row, person]:
                continue
            expected_body = {'canvas_id': canvas_ids[row],
                'person_id': local.metadata[row]['target_person_ids'][person],
                'record_sha256': local.metadata[row]['driver_record_sha256']}
            if any(actual.get(key) != value for key, value in expected_body.items()):
                raise ValueError('pose_repair_v61 raw BodyPose Canvas/person/driver_record_sha256 differs')
    return repair


def _repair(batch, geometry, local, valid_pixels):
    raw = batch.get('pose_repair_v61')
    if raw is None:
        return None, None
    validate_repair_condition(raw, batch)
    mask = raw['mask']
    metadata = _rows(raw['metadata'], geometry.batch_size, 'pose_repair_v61.metadata')
    return mask.to(valid_pixels.device) * valid_pixels, tuple(deepcopy(m) for m in metadata)


def build_spatial_condition(batch, geometry: PoseGeometry, config: PoseLinkConfigV61) -> PoseSpatialConditionV61:
    """Build from canonical BodyPose UV, raw local ROIs and optional aligned repair.

    No landmark estimator, source-to-target remapping, RGB or training sidecar is read.
    Minimal synthetic batches can omit raw joint/local fields; provenance is then
    explicitly missing and such batches cannot supply an aligned source repair.
    """
    if not isinstance(geometry, PoseGeometry) or not isinstance(config, PoseLinkConfigV61):
        raise ValueError('V6.1 builder requires typed PoseGeometry and PoseLinkConfigV61')
    geometry.validate()
    valid_pixels = _valid_pixels(batch, geometry)
    h, w = valid_pixels.shape[-2:]
    uv, joint_valid, joint_reasons = _canonical_joints(batch, geometry, h, w)
    rows = torch.arange(geometry.batch_size, device=uv.device)[:, None, None]
    px = (uv[..., 0] * w).long().clamp(0, w - 1)
    py = (uv[..., 1] * h).long().clamp(0, h - 1)
    joint_valid = joint_valid & (valid_pixels[:, 0][rows, py, px] > 0)
    uv = torch.where(joint_valid[..., None], uv, 0.)
    boxes, local_valid, local = _local_boxes(batch, geometry, uv, joint_valid, h, w)
    # Validate explicit raw BodyPose provenance whenever it exists.
    for person, suffix in enumerate(('', '_b')):
        provenance = batch.get('pose_provenance' + suffix)
        if provenance is None:
            continue
        rows = _rows(provenance, geometry.batch_size, 'pose_provenance' + suffix)
        for row, meta in enumerate(rows):
            if meta.get('canvas_id') != geometry.canvas_ids[row]:
                raise ValueError('pose_provenance canvas_id differs from spatial canvas')
            if local is not None and geometry.person_valid[row, person]:
                wanted = local.metadata[row]
                if meta.get('person_id') != wanted['target_person_ids'][person]:
                    raise ValueError('pose_provenance target_person_ids differ')
                if meta.get('record_sha256') != wanted['driver_record_sha256']:
                    raise ValueError('pose_provenance driver_record_sha256 differs')
    projected = geometry.projected.float().transpose(1, 2).reshape(geometry.batch_size, 2, *geometry.target_token_hw)
    projected = F.interpolate(projected, size=(h, w), mode='nearest')
    radius = uv.new_tensor([4 / w, 4 / h])
    joint_boxes = torch.cat((uv - radius, uv + radius), -1).clamp(0, 1)
    base_support = torch.maximum(projected, _raster_boxes(joint_boxes, joint_valid, h, w))
    base_support = torch.maximum(base_support, _raster_boxes(boxes, local_valid, h, w))
    base_support = base_support * geometry.person_valid[..., None, None] * valid_pixels
    key_support = _expand_internal_feather(base_support, config.support_expand_px, config.feather_px, valid_pixels)
    repair, repair_metadata = _repair(batch, geometry, local, valid_pixels)
    edit_support = base_support if repair is None else torch.maximum(base_support, repair)
    edit_pixels = _expand_internal_feather(edit_support.amax(1, keepdim=True), config.edit_expand_px,
        config.feather_px, valid_pixels)
    coverage = F.avg_pool2d(valid_pixels, 16).flatten(1)
    numerator = F.avg_pool2d(edit_pixels, 16).flatten(1)
    gate = torch.where(coverage > 0, numerator / coverage.clamp_min(1e-12), 0.).clamp(0, 1)
    metadata = []
    for row in range(geometry.batch_size):
        empty = not bool(key_support[row].any())
        meta = {'schema': SPATIAL_SCHEMA, 'target_canvas_id': geometry.canvas_ids[row],
            'key_support_empty': empty, 'key_support_reason': 'no_valid_geometry' if empty else 'available',
            'source_repair': 'missing' if repair is None else 'aligned',
            'joint_sources': list(joint_reasons), 'joint_valid_counts': joint_valid[row].sum(-1).tolist(),
            'local_names': list(LOCAL_NAMES), 'local_valid_counts': local_valid[row].sum(-1).tolist(),
            'local_source': 'missing' if local is None else 'local_condition_v4'}
        if local is not None:
            meta.update({key: deepcopy(local.metadata[row][key]) for key in
                ('driver_record_sha256', 'target_person_ids')})
        if repair_metadata is not None:
            meta['repair_provenance'] = deepcopy(repair_metadata[row])
        metadata.append(meta)
    return PoseSpatialConditionV61(key_support, valid_pixels, gate, uv, joint_valid, boxes,
        local_valid, geometry.canvas_ids, tuple(metadata)).validate(geometry)
