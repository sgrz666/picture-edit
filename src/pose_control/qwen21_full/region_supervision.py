"""Immutable, GT/Canvas/person-bound training ROI sidecars; no inference input."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from .region_config import FAMILIES
from .region_objective import region_eligibility
from .native_v5.training.io import verify, file_sha, read_json_ref

SCHEMA = 'native_region_supervision_v1'
INDEX_SCHEMA = 'native_region_index_v1'
HAND_SLOTS = ['A-left', 'A-right', 'B-left', 'B-right']


def _check(arrays, metadata):
    if metadata.get('schema') != SCHEMA or metadata.get('hand_slot_order') != HAND_SLOTS:
        raise ValueError('Regional schema or anatomical hand order differs')
    spec = CanvasSpec.from_record(metadata['canvas'])
    if metadata.get('canvas_id') != spec.fingerprint():
        raise ValueError('Regional Canvas fingerprint differs')
    for key in ('target_rgb_sha256', 'annotation_sha256'):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Regional provenance requires a SHA256: ' + key)
    people = metadata.get('person_ids')
    if not isinstance(people, list) or len(people) != 2 or not people[0] or (people[1] is not None and people[1] == people[0]):
        raise ValueError('Stable target person IDs required')
    valid = arrays['valid']
    if valid.shape != spec.canvas_hw or not np.array_equal(valid, spec.valid_mask()[0].numpy()):
        raise ValueError('Regional valid differs from actual Canvas')
    for name, slots in FAMILIES.items():
        support, quality = arrays[name], arrays[name + '_quality']
        if support.shape != (slots, *spec.canvas_hw) or quality.shape != (slots,):
            raise ValueError('Regional pixel shape or slots differ: ' + name)
        for value in (support, quality):
            if value.dtype.kind not in 'fbu' or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError('Finite 0..1 regional arrays required')
        if (support > valid[None] + 1e-6).any():
            raise ValueError('Regional support includes padding')
        if people[1] is None:
            invalid = (1,) if name in ('person', 'joint') else (2, 3) if name == 'hands' else ()
            if any(support[i].any() or quality[i] for i in invalid):
                raise ValueError('Regional support belongs to an absent person')
    return spec


def write_region_sidecar(path, arrays, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite regional supervision: ' + str(path))
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    _check(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {name: value.astype(np.float16) for name, value in arrays.items()}
    stored['valid'] = arrays['valid'].astype(np.uint8)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **stored, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    return {'path': str(path.resolve()), 'sha256': file_sha(path)}


@lru_cache(maxsize=512)
def _read_tokens(path, sha):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].item())
        arrays = {k: archive[k] for k in archive.files if k != 'metadata'}
    spec = _check(arrays, metadata)
    hw = tuple(spec.metadata()['token_hw'])
    valid = torch.from_numpy(arrays['valid'].astype(np.float32))
    coverage = F.interpolate(valid[None, None], size=hw, mode='area').flatten(1)
    supports = {name: F.interpolate(torch.from_numpy(arrays[name].astype(np.float32))[None],
                                   size=hw, mode='area').flatten(2)[0] for name in FAMILIES}
    quality = {name: torch.from_numpy(arrays[name + '_quality'].astype(np.float32)) for name in FAMILIES}
    return {'coverage': coverage, 'supports': supports, 'quality': quality, 'metadata': metadata,
            'target_token_hw': hw}


def read_region_sidecar(ref, *, target_rgb_sha256, canvas, person_ids):
    bound = verify(ref)  # Recheck bytes even when pooled tensors are cached.
    payload = _read_tokens(bound['path'], bound['sha256'])
    metadata = payload['metadata']
    if (metadata['target_rgb_sha256'] != target_rgb_sha256 or metadata['canvas'] != canvas or
            metadata['person_ids'] != list(person_ids)):
        raise ValueError('Regional sidecar belongs to another GT, Canvas or target person')
    return payload


def read_region_index(ref):
    value, bound = read_json_ref(ref)
    if value.get('schema') != INDEX_SCHEMA or not isinstance(value.get('entries'), dict):
        raise ValueError('Explicit regional index required')
    for key, entry in value['entries'].items():
        metadata = entry.get('metadata', {})
        if key != metadata.get('target_rgb_sha256', '') + ':' + metadata.get('canvas_id', ''):
            raise ValueError('Regional index GT/Canvas key differs')
        file = entry.get('file', {})
        if not file.get('path') or not file.get('sha256'):
            raise ValueError('Regional index requires pinned files')
    return value, bound


def attach_region(sample, entry):
    meta = sample['local_condition_v4'].metadata[0]
    payload = read_region_sidecar(entry['file'], target_rgb_sha256=meta['driver_rgb_sha256'],
        canvas=sample['target_canvas'], person_ids=meta['target_person_ids'])
    if payload['metadata'] != entry['metadata']:
        raise ValueError('Regional index metadata differs from sidecar')
    sample['spatial_weights_v5'] = {**sample.get('spatial_weights_v5', {}), 'regional': payload}
    return sample


def attach_region_status(sample, status):
    regional = sample.get('spatial_weights_v5', {}).get('regional')
    return status if regional is None else {**status, 'region_eligible': region_eligibility(regional)}
