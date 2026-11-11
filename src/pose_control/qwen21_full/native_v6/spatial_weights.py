"""Offline GT-bound Canvas components and normalized spatial Flow Matching loss.

Components contain no Adapter features. Fuse and cap pixels before area pooling;
the resulting weights already contain padding coverage and must not be masked twice.
"""
from collections.abc import Mapping
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from src.data.canvas_contract import CanvasSpec
from .training.io import read_json_ref, verify

COMPONENT_SCHEMA = 'native_spatial_components_v1'
INDEX_SCHEMA = 'native_spatial_index_v1'
COMPONENTS = ('body', 'joint', 'hand')


def _coefficients(coefficients, max_weight):
    if isinstance(coefficients, Mapping):
        if set(coefficients) != set(COMPONENTS):
            raise ValueError('Spatial coefficients require body/joint/hand')
        coefficients = tuple(coefficients[key] for key in COMPONENTS)
    try:
        values = tuple(coefficients)
    except TypeError as error:
        raise ValueError('Three spatial coefficients required') from error
    if len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Spatial coefficients must be finite nonnegative numbers')
    if type(max_weight) not in (int, float) or not math.isfinite(max_weight) or max_weight < 1:
        raise ValueError('Spatial maximum must be finite and at least one')
    return values


def _pixels(value, name):
    if not torch.is_tensor(value):
        raise ValueError('Spatial ' + name + ' requires a tensor')
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError('Spatial ' + name + ' requires [B,H,W] or [B,1,H,W]')
    if not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('Spatial ' + name + ' must be finite in [0,1]')
    return value.float()


def spatial_token_weights(components, valid, target_token_hw, coefficients, max_weight=2.5):
    """Return row-major [B,N] weights on the actual target token grid.

    ``valid`` includes background within the content rectangle, excludes padding.
    Inputs can be [H,W], [B,H,W], or [B,1,H,W]; no fixed VAE stride is assumed.
    """
    values = _coefficients(coefficients, max_weight)
    valid = _pixels(valid, 'valid')
    if (not isinstance(target_token_hw, (tuple, list)) or len(target_token_hw) != 2 or
            any(type(v) != int or v < 1 for v in target_token_hw) or
            any(v > size for v, size in zip(target_token_hw, valid.shape[-2:]))):
        raise ValueError('Spatial target_token_hw must be an actual positive pooling grid')
    if not isinstance(components, Mapping) or not set(COMPONENTS) <= set(components):
        raise ValueError('Spatial body/joint/hand components required')
    fused = torch.ones_like(valid)
    for name, coefficient in zip(COMPONENTS, values):
        pixels = _pixels(components[name], name)
        if pixels.shape != valid.shape or pixels.device != valid.device:
            raise ValueError('Spatial component/valid Canvas or batch differs')
        fused = fused + coefficient * pixels
    weights = valid * fused.clamp(1., float(max_weight))
    pooled = F.interpolate(weights, size=tuple(target_token_hw), mode='area').flatten(1)
    if not bool((pooled.sum(1) > 0).all()):
        raise ValueError('Spatial valid Canvas must be nonempty for every sample')
    return pooled


def spatial_velocity_loss(prediction, target, weights):
    """Normalize sum(w * squared error) per sample over C * sum(w), then mean."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or min(prediction.shape) < 1 or
            weights.shape != prediction.shape[:2] or prediction.device != target.device or
            weights.device != prediction.device or not torch.isfinite(weights).all() or
            bool((weights < 0).any()) or not bool((weights.sum(1) > 0).all())):
        raise ValueError('Spatial Flow loss requires matching [B,N,C] and nonempty finite weights per sample')
    squared = (prediction.float() - target.float()).square()
    numerator = (squared * weights.float()[..., None]).sum((1, 2))
    denominator = weights.float().sum(1) * prediction.shape[-1]
    return (numerator / denominator).mean()


def _pinned(ref):
    sha = ref.get('sha256') if isinstance(ref, dict) else None
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Spatial artifacts require a pinned SHA256')
    return verify(ref)


def _metadata(metadata, *, gt_rgb_sha256=None, target_canvas_id=None, canvas=None):
    if not isinstance(metadata, dict) or metadata.get('schema') != COMPONENT_SCHEMA:
        raise ValueError('Spatial components schema differs')
    gt = metadata.get('gt_rgb_sha256')
    if not isinstance(gt, str) or len(gt) != 64 or any(c not in '0123456789abcdef' for c in gt):
        raise ValueError('Spatial GT SHA256 required')
    spec = CanvasSpec.from_record(metadata.get('canvas', {}))
    if metadata['canvas'] != spec.metadata() or metadata.get('target_canvas_id') != spec.fingerprint():
        raise ValueError('Spatial target Canvas metadata/fingerprint differs')
    if ((gt_rgb_sha256 is not None and gt != gt_rgb_sha256) or
            (target_canvas_id is not None and metadata['target_canvas_id'] != target_canvas_id)):
        raise ValueError('Spatial components belong to another GT RGB or target Canvas')
    if canvas is not None:
        expected = canvas if isinstance(canvas, CanvasSpec) else CanvasSpec.from_record(canvas)
        if expected.metadata() != spec.metadata():
            raise ValueError('Spatial components target Canvas differs')
    for name in ('frame_id', 'geometry_method'):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise ValueError('Spatial provenance requires ' + name)
    return spec


def read_spatial_components(ref, *, gt_rgb_sha256, target_canvas_id, canvas=None):
    """Read a pinned NPZ, rejecting stale GT, Canvas, shape, range or valid mask."""
    bound = _pinned(ref)
    with np.load(bound['path'], allow_pickle=False) as archive:
        if not set((*COMPONENTS, 'valid', 'metadata')) <= set(archive.files):
            raise ValueError('Spatial NPZ omits components or metadata')
        raw = archive['metadata']
        if raw.shape != () or raw.dtype.kind not in ('U', 'S'):
            raise ValueError('Spatial metadata must be a scalar JSON string')
        metadata = json.loads(raw.item())
        spec = _metadata(metadata, gt_rgb_sha256=gt_rgb_sha256, target_canvas_id=target_canvas_id, canvas=canvas)
        result = {}
        for name in (*COMPONENTS, 'valid'):
            value = archive[name]
            if value.dtype not in (np.dtype('float16'), np.dtype('uint8')) or value.shape != spec.canvas_hw:
                raise ValueError('Spatial ' + name + ' must be float16/uint8 on target Canvas')
            if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError('Spatial ' + name + ' must be finite in [0,1]')
            result[name] = torch.from_numpy(value.astype(np.float32, copy=True))[None]
    if not torch.equal(result['valid'], spec.valid_mask()):
        raise ValueError('Spatial valid must include all content/background and exclude padding')
    result['metadata'] = metadata
    return result


def read_spatial_index(ref):
    """Verify the index and its file references; NPZ payloads are checked on use."""
    bound = _pinned(ref)
    index, _ = read_json_ref(bound)
    if not isinstance(index, dict) or index.get('schema') != INDEX_SCHEMA or not isinstance(index.get('entries'), dict):
        raise ValueError('Spatial index schema/entries required')
    for key, entry in index['entries'].items():
        if not isinstance(entry, dict):
            raise ValueError('Spatial index entry must declare file and metadata')
        metadata = entry.get('metadata')
        _metadata(metadata)
        if key != metadata['gt_rgb_sha256'] + ':' + metadata['target_canvas_id']:
            raise ValueError('Spatial index key differs from GT/Canvas metadata')
        _pinned(entry.get('file'))
    return index, bound
