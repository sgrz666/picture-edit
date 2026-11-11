"""Bind frozen V5 multi-reference tensors to V6's explicit Canvas provenance."""
from src.data.canvas_contract import CanvasSpec
from ..native_v5.conditioning import (
    ROLES, CACHE_SCHEMA, primary_source, read_condition, validate_pair as validate_v5_pair,
    encode_pair as encode_v5_pair,
)


def bind_pair_provenance(pair, spec, *, condition_sha256=None):
    validate_v5_pair(pair)
    roles = tuple(image['role'] for image in spec['images'])
    hashes = [image['file']['sha256'] for image in spec['images']]
    canvases = tuple(image['canvas_id'] for image in spec['images'])
    grids = tuple(CanvasSpec.from_record(image['canvas']).token_hw for image in spec['images'])
    target_grids = {grid for grid, role in zip(grids, roles) if role not in ('source', 'source_a', 'source_b')}
    if (pair['qwen_condition_profile'] != spec['profile'] or tuple(pair['reference_roles']) != roles
            or pair.get('prompt_sha256') != spec['prompt']['sha256']
            or pair.get('reference_sha256s') != hashes
            or tuple(map(tuple, pair['reference_token_hws'])) != grids
            or target_grids != {tuple(pair['target_token_hw'])}):
        raise ValueError('V6 cached Pose/Source RGB, roles, Prompt or Canvas grid differ from locked condition')
    for key, value in (('reference_canvas_ids', canvases), ('target_canvas_id', spec['target_canvas_id']),
                       ('condition_sha256', condition_sha256)):
        if key in pair and (tuple(pair[key]) if key == 'reference_canvas_ids' else pair[key]) != value:
            raise ValueError('V6 frozen condition provenance differs: ' + key)
    result = dict(pair, reference_canvas_ids=canvases, target_canvas_id=spec['target_canvas_id'])
    if condition_sha256 is not None:
        result['condition_sha256'] = condition_sha256
    return validate_pair(result)


def validate_pair(pair):
    validate_v5_pair(pair)
    ids = tuple(pair.get('reference_canvas_ids', ()))
    target = pair.get('target_canvas_id')
    if (not isinstance(target, str) or not target or len(ids) != len(pair['reference_roles'])
            or any(not isinstance(value, str) or not value for value in ids)):
        raise ValueError('V6 references require explicit target/reference Canvas IDs')
    if any(value != target for value, role in zip(ids, pair['reference_roles']) if role not in ('source', 'source_a', 'source_b')):
        raise ValueError('V6 Pose Canvas differs from target geometry Canvas')
    return pair


def encode_pair(pipe, sample, spec, *, include_target=True):
    pair = encode_v5_pair(pipe, sample, spec, include_target=include_target)
    return bind_pair_provenance(pair, spec)
