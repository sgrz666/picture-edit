"""Offline, exclusive Source preservation sidecars; no RGB inference or model loading.

An index is JSON with ``schema``, ``evaluator`` (absolute path + SHA256), and
``entries`` mapping ``source_id + '\\0' + pair_id`` to record references.
``read_index`` returns ``(index_dict, verified_index_ref)`` and verifies every
record, while leaving evaluator loading and configuration matching to DataPlan.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec

SCHEMA = 'native_v7_preservation_sample_v1'
INDEX_SCHEMA = 'native_v7_preservation_index_v1'
BACKGROUND_IMPLEMENTATION = 'source_mask_union_target_projected_maxpool_avgpool16_v1'


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' requires a lowercase SHA256')
    return value


def _ids(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not isinstance(v, str) or not v for v in value) or value[0] == value[1]):
        raise ValueError(name + ' requires two distinct person IDs')
    return list(value)


def _grid(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError(name + ' requires positive integer token HW')
    return list(value)


def _mapping(value):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.long or tuple(value.shape) not in ((2,), (1, 2)):
            raise ValueError('source_indices must be int64 [2] or B1 [1,2]')
        value = value.detach().cpu().reshape(2).tolist()
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int for v in value) or sorted(value) != [0, 1]):
        raise ValueError('source_indices must be a permutation of [0,1]')
    return list(value)


def _canvases(sample):
    source = CanvasSpec.from_record(sample['source_canvas'])
    target = CanvasSpec.from_record(sample['target_canvas'])
    for name, canvas in (('source', source), ('target', target)):
        if canvas.stride != 16 or _grid(sample[name + '_token_hw'], name) != list(canvas.token_hw):
            raise ValueError(name + ' token grid differs from its Canvas')
        declared = sample[name + '_canvas'].get('token_hw')
        if declared is not None and _grid(declared, name) != list(canvas.token_hw):
            raise ValueError(name + ' Canvas record token grid differs')
    return source, target


def sample_binding(sample, evaluator_sha):
    """Bind a raw B1 4Ref sample to its Source, slot mapping, geometry and model."""
    if sample.get('source_bank_v5') is not None:
        raise ValueError('Independent source_bank preservation requires per-Source bindings')
    source, target = _canvases(sample)
    identity = sample['identity_metadata']
    local = sample['local_condition_v4']
    if not isinstance(identity, dict) or len(local.metadata) != 1:
        raise ValueError('Explicit raw B1 identity/local metadata required')
    metadata = local.metadata[0]
    source_ids = _ids(identity.get('source_person_ids'), 'source_person_ids')
    target_ids = _ids(metadata.get('target_person_ids'), 'target_person_ids')
    indices = _mapping(sample['source_indices'])
    if _mapping(identity.get('source_indices')) != indices or _mapping(local.source_indices) != indices:
        raise ValueError('Source identity/local source_indices differ')
    if [source_ids[i] for i in indices] != target_ids:
        raise ValueError('Source mapping differs from target person IDs')
    source_sha = _sha(identity.get('source_rgb_sha256'), 'source_rgb_sha256')
    driver_sha = _sha(metadata.get('driver_record_sha256'), 'driver_record_sha256')
    expected = dict(source_rgb_sha256=source_sha, source_canvas_id=source.fingerprint(),
        target_canvas_id=target.fingerprint(), source_person_ids=source_ids, target_person_ids=target_ids)
    for record in (identity, metadata):
        for key, value in expected.items():
            if key in record and record[key] != value:
                raise ValueError('Source/local binding differs: ' + key)
    for required in ('source_rgb_sha256', 'source_canvas_id', 'target_canvas_id',
                     'source_person_ids', 'target_person_ids'):
        if metadata.get(required) != expected[required]:
            raise ValueError('Local metadata missing or differs: ' + required)
    if identity.get('source_canvas_id') != expected['source_canvas_id']:
        raise ValueError('Identity Source Canvas differs')
    pair = sample['pair_metadata']
    pair_id = pair.get('pair_id')
    if not isinstance(pair_id, str) or not pair_id or '\0' in pair_id:
        raise ValueError('Explicit nonempty pair_id required')
    if pair.get('source_rgb_sha256') is not None and pair['source_rgb_sha256'] != source_sha:
        raise ValueError('Pair Source RGB SHA differs')
    for index, key in enumerate(('pose_provenance', 'pose_provenance_b')):
        provenance = sample.get(key)
        if provenance is not None and (provenance.get('record_sha256') != driver_sha
                or provenance.get('canvas_id') != expected['target_canvas_id']
                or provenance.get('person_id') != target_ids[index]):
            raise ValueError('Driver record/Canvas/person provenance differs')
    return {**expected, 'pair_id': pair_id, 'source_indices': indices,
        'driver_record_sha256': driver_sha, 'identity_evaluator_sha256': _sha(evaluator_sha, 'evaluator'),
        'source_token_hw': list(source.token_hw), 'target_token_hw': list(target.token_hw)}


def _margin(value):
    if type(value) is not int or value < 0:
        raise ValueError('background_margin_px must be a nonnegative integer')
    return value


def background_support(sample, margin_px=16):
    """Average-pool valid background after dilating the union of both-frame people.

    The returned FP32 [1,N] already contains partial padding coverage. It must
    not be multiplied by token_valid_coverage a second time.
    """
    margin_px = _margin(margin_px)
    if (sample.get('source_bank_v5') is not None or sample.get('native_scene_kind') != 'dual'
            or sample.get('qwen_condition_profile', 'dual_4ref') != 'dual_4ref'):
        raise ValueError('Background preservation requires single-scene Source dual_4ref')
    source, target = _canvases(sample)
    if source != target:
        raise ValueError('Background requires exactly the same Source/target Canvas')
    h, w = target.canvas_hw
    source_mask = sample['source_person_mask']
    projected = sample['dual_depth'].projected
    valid = sample['valid_image_mask']
    if (not isinstance(source_mask, torch.Tensor) or tuple(source_mask.shape) != (2, 1, h, w)
            or not isinstance(projected, torch.Tensor) or tuple(projected.shape) != (1, 2, 1, h, w)
            or not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (1, h, w)):
        raise ValueError('Background masks must use the exact B1 target Canvas grid')
    for tensor in (source_mask, projected, valid):
        if not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all():
            raise ValueError('Background geometry/valid masks must be binary')
    valid = valid.detach().to(device='cpu', dtype=torch.float32)
    if not torch.equal(valid, target.valid_mask()):
        raise ValueError('Background valid mask differs from exact target Canvas padding')
    foreground = (source_mask.detach().cpu().bool().any(0)
        | projected.detach().cpu().bool().any(1)[0]).float()[None]
    if margin_px:
        foreground = F.max_pool2d(foreground, 2 * margin_px + 1, stride=1, padding=margin_px)
    support = (1 - foreground) * valid[None]
    return F.avg_pool2d(support, 16, stride=16).flatten(1).contiguous()


def _plain(value):
    """Permit only finite JSON primitives in metadata, never serialized objects."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for entry in value: _plain(entry)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for entry in value.values(): _plain(entry)
        return
    raise ValueError('Preservation metadata must contain finite primitives only')


def _validate_record(record):
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        raise ValueError('Explicit preservation sample schema required')
    features, valid, support = (record.get(key) for key in ('source_features', 'source_valid', 'background_support'))
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
            or tuple(features.shape) != (2, 512) or not torch.isfinite(features).all()):
        raise ValueError('source_features must be finite FP32 [2,512]')
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or tuple(valid.shape) != (2,):
        raise ValueError('source_valid must be bool [2]')
    if features.device != valid.device or features[~valid].count_nonzero() or (features[valid].norm(dim=-1) <= 0).any():
        raise ValueError('Missing Source features must be zero; valid Source features must be nonzero')
    metadata = record.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Explicit preservation metadata required')
    _plain(metadata)
    for key in ('source_rgb_sha256', 'source_canvas_id', 'target_canvas_id',
                'driver_record_sha256', 'identity_evaluator_sha256'):
        _sha(metadata.get(key), key)
    for key in ('source_person_ids', 'target_person_ids'):
        _ids(metadata.get(key), key)
    indices = _mapping(metadata.get('source_indices'))
    if [metadata['source_person_ids'][i] for i in indices] != metadata['target_person_ids']:
        raise ValueError('Preservation source_indices/person IDs differ')
    if not isinstance(metadata.get('pair_id'), str) or not metadata['pair_id'] or '\0' in metadata['pair_id']:
        raise ValueError('Preservation pair_id required')
    _grid(metadata.get('source_token_hw'), 'source_token_hw')
    th, tw = _grid(metadata.get('target_token_hw'), 'target_token_hw')
    _margin(metadata.get('background_margin_px'))
    if metadata.get('background_implementation') != BACKGROUND_IMPLEMENTATION:
        raise ValueError('Explicit background implementation evidence required')
    if (not isinstance(support, torch.Tensor) or support.dtype != torch.float32
            or tuple(support.shape) != (1, th * tw) or not torch.isfinite(support).all()
            or (support < 0).any() or (support > 1).any()):
        raise ValueError('background_support must be finite FP32 [1,N] in [0,1]')
    return record


def write_record(path, features, valid, support, metadata):
    """Create one record exclusively and return its absolute path/SHA reference."""
    record = _validate_record(dict(schema=SCHEMA, source_features=features, source_valid=valid,
        background_support=support, metadata=deepcopy(metadata)))
    for key in ('source_features', 'source_valid', 'background_support'):
        record[key] = record[key].detach().to('cpu').contiguous().clone()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or not isinstance(ref['path'], str):
        raise ValueError('Preservation reference requires path + SHA256 only')
    path = Path(ref['path'])
    if not path.is_absolute():
        raise ValueError('Preservation reference path must be absolute')
    _sha(ref['sha256'], 'reference SHA')
    return path


def _verified_bytes(ref):
    path = _ref(ref)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('Preservation asset SHA differs: ' + str(path))
    return data


def _load_record(ref):
    # Hash and deserialize the same byte buffer, avoiding a file-change race.
    return _validate_record(torch.load(io.BytesIO(_verified_bytes(ref)), map_location='cpu', weights_only=True))


def read_record(ref, sample, evaluator_sha):
    """Verify the record SHA, complete current sample binding and geometric support."""
    record = _load_record(ref)
    binding = sample_binding(sample, evaluator_sha)
    for key, value in binding.items():
        if record['metadata'].get(key) != value:
            raise ValueError('Preservation sample binding differs: ' + key)
    source, target = _canvases(sample)
    support = record['background_support']
    if source == target:
        expected = background_support(sample, record['metadata']['background_margin_px'])
        if not torch.equal(support, expected):
            raise ValueError('Preservation background support/margin/geometry differs')
    elif support.count_nonzero():
        raise ValueError('Preservation background cannot use different Source/target Canvas')
    return record


def read_index(ref):
    """Return (validated index dict, verified ref), verifying the record SHA closure.

    The evaluator ref is syntax checked only. Its fixed SHA must also match the
    configured evaluator; no model file or runtime is opened by this reader.
    """
    index = json.loads(_verified_bytes(ref).decode('utf8'))
    if (not isinstance(index, dict) or index.get('schema') != INDEX_SCHEMA
            or not isinstance(index.get('entries'), dict) or not index['entries']):
        raise ValueError('Explicit nonempty preservation index required')
    _ref(index.get('evaluator'))
    for key, entry in index['entries'].items():
        if not isinstance(key, str) or key.count('\0') != 1 or not all(key.split('\0')):
            raise ValueError('Preservation index key requires source_id + NUL + pair_id')
        record = _load_record(entry)
        if record['metadata']['identity_evaluator_sha256'] != index['evaluator']['sha256']:
            raise ValueError('Preservation index/record evaluator SHA differs')
        if record['metadata']['pair_id'] != key.split('\0')[1]:
            raise ValueError('Preservation index/record pair_id differs')
    return index, {'path': str(_ref(ref)), 'sha256': ref['sha256']}
