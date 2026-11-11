"""Offline V6 source-bank records; frozen Qwen latents are bound only at runtime.

The NPZ stores a primitive dictionary with ``sources`` (two source dictionaries),
``target_qwen_image_indices``, ``target_person_indices``, and ``target_person_ids``.
Each source has the NativeAppearanceSourceV5 fields except ``latent``. Its frozen
``references`` use the existing V4 typed record; ``detail`` is a field dictionary
or None. This schema cannot persist a prepared Adapter or live Qwen feature.
"""
from dataclasses import fields, replace
from pathlib import Path

import torch
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from ..native_full_assets import read_record, write_record, sha256
from ..native_local_condition import NativeLocalReferencesV4
from ..native_v5.contracts import NativeDetailReferencesV5
from ..native_v5.source_bank import NativeAppearanceSourceV5, NativeSourceBankV5

SOURCE_BANK_SCHEMA = 'native_v6_source_bank_v1'
_BANK_FIELDS = {'sources', 'target_qwen_image_indices', 'target_person_indices', 'target_person_ids'}
_SOURCE_FIELDS = tuple(f.name for f in fields(NativeAppearanceSourceV5) if f.name != 'latent')


def _verified(ref):
    path = Path(ref['path'])
    if not path.is_absolute() or sha256(path) != ref['sha256']:
        raise ValueError('V6 Source reference requires an absolute path and exact SHA')
    return path


def validate_offline_bank(bank, *, forbidden_rgb_sha256=None):
    """Verify independent source assets without requiring frozen runtime latents."""
    if not isinstance(bank, NativeSourceBankV5):
        raise ValueError('A typed independent Source bank is required')
    bank.validate()
    for source in bank.sources:
        if source.latent is not None:
            raise ValueError('Offline source banks cannot contain Qwen latents')
        canvas = CanvasSpec.from_record(source.canvas)
        for metadata in source.references.metadata:
            count = source.references.arcface.shape[2]
            if metadata.get('reference_count') != count:
                raise ValueError('Declare the exact frozen Source reference count')
            if count > 1 and not metadata.get('explicit_multi_reference'):
                raise ValueError('Multiple frozen Source references require an explicit declaration')
            for index, reference in enumerate(metadata['reference_sources']):
                if reference['rgb']['sha256'] == forbidden_rgb_sha256:
                    raise ValueError('Supervision target RGB cannot supply Source appearance')
                own_canvas = CanvasSpec.from_record(reference['canvas'])
                if own_canvas.fingerprint() != reference['canvas_id']:
                    raise ValueError('Offline Source reference Canvas fingerprint differs')
                if not reference.get('geometry_sha256'):
                    raise ValueError('Every frozen reference requires its own geometry SHA')
                if index == 0 and (reference['rgb']['sha256'] != source.rgb_sha256
                        or reference['canvas_id'] != canvas.fingerprint()):
                    raise ValueError('Primary Source RGB/Canvas differs from its frozen references')
                with Image.open(_verified(reference['rgb'])) as image:
                    if (image.height, image.width) != own_canvas.original_hw:
                        raise ValueError('Offline Source reference dimensions differ from its Canvas')
        if source.rgb_sha256 == forbidden_rgb_sha256:
            raise ValueError('Supervision target RGB cannot be an appearance source')
    return bank


def write_source_bank(path, bank):
    """Write an exclusive no-pickle NPZ; never save actual Qwen source latents."""
    validate_offline_bank(bank)
    sources = []
    for source in bank.sources:
        record = {name: getattr(source, name) for name in _SOURCE_FIELDS}
        record['detail'] = None if source.detail is None else {
            f.name: getattr(source.detail, f.name) for f in fields(source.detail)}
        sources.append(record)
    return write_record(path, dict(sources=tuple(sources),
        target_qwen_image_indices=bank.target_qwen_image_indices,
        target_person_indices=bank.target_person_indices,
        target_person_ids=bank.target_person_ids), schema=SOURCE_BANK_SCHEMA)


def read_source_bank(ref, *, forbidden_rgb_sha256=None):
    """Restore existing typed bank classes and verify every frozen RGB reference."""
    _verified(ref)
    raw = read_record(ref, schema=SOURCE_BANK_SCHEMA)
    if not isinstance(raw, dict) or set(raw) != _BANK_FIELDS:
        raise ValueError('Undeclared V6 source-bank fields')
    sources = []
    for value in raw['sources']:
        if not isinstance(value, dict) or set(value) != set(_SOURCE_FIELDS):
            raise ValueError('Undeclared V6 source fields; latents are runtime-only')
        value = dict(value)
        if not isinstance(value['references'], NativeLocalReferencesV4):
            raise ValueError('Frozen typed V4 references are required')
        detail = value['detail']
        if detail is not None:
            if not isinstance(detail, dict) or set(detail) != {f.name for f in fields(NativeDetailReferencesV5)}:
                raise ValueError('Explicit raw Source detail fields required')
            detail = NativeDetailReferencesV5(**detail).validate()
            detail = replace(detail, metadata=tuple({**m, 'container_sha256': ref['sha256']} for m in detail.metadata))
        value['detail'] = detail
        value['person_ids'] = tuple(value['person_ids'])
        sources.append(NativeAppearanceSourceV5(**value))
    bank = NativeSourceBankV5(tuple(sources), raw['target_qwen_image_indices'],
        raw['target_person_indices'], tuple(tuple(ids) for ids in raw['target_person_ids']))
    return validate_offline_bank(bank, forbidden_rgb_sha256=forbidden_rgb_sha256)


def validate_target_sample(bank, sample):
    """Offline equivalent of V5 validate_target, with complete pose provenance."""
    bank.validate()
    if sample['native_scene_kind'] != 'dual' or len(bank.target_person_ids) != 1:
        raise ValueError('Independent V6 source banks require a dual B1 sample')
    local = sample['local_condition_v4'].validate()
    if local.person_valid.shape != (1, 2) or not local.person_valid.all():
        raise ValueError('Independent source-bank target requires both actual people')
    if not torch.equal(local.source_indices.cpu(), bank.source_indices.cpu()):
        raise ValueError('Local/Qwen source image mapping differs')
    if (not torch.is_tensor(sample['source_indices']) or sample['source_indices'].dtype != torch.long
            or not torch.equal(sample['source_indices'].cpu(), bank.source_indices[0].cpu())):
        raise ValueError('Body/Qwen source image mapping differs')
    if local.source_slot_ids.tolist() != [[s.source_slot_id for s in bank.sources]]:
        raise ValueError('Local logical source slots differ from the bank')
    source = CanvasSpec.from_record(sample['source_canvas'])
    target = CanvasSpec.from_record(sample['target_canvas'])
    primary = bank.sources[0]
    metadata = local.metadata[0]
    expected = dict(source_canvas_id=source.fingerprint(), target_canvas_id=target.fingerprint(),
        source_rgb_sha256=primary.rgb_sha256, source_person_ids=[s.person_ids[0] for s in bank.sources],
        target_person_ids=list(bank.target_person_ids[0]))
    if source.fingerprint() != CanvasSpec.from_record(primary.canvas).fingerprint():
        raise ValueError('Sample Source Canvas differs from Qwen source0')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('Local/bank provenance differs: ' + key)
    if metadata.get('source_asset_sha256', primary.geometry_sha256) != primary.geometry_sha256:
        raise ValueError('Local Source geometry differs from source0')
    for key, canvas in (('source', source), ('target', target)):
        if tuple(sample[key + '_token_hw']) != canvas.token_hw:
            raise ValueError('Raw sample token grid differs from its ' + key + ' Canvas')
    if local.region_masks.shape[-2:] != target.canvas_hw:
        raise ValueError('Local geometry must use the exact target Canvas')
    for index, key in enumerate(('pose_provenance', 'pose_provenance_b')):
        pose = sample[key]
        if any(pose.get(k) != v for k, v in dict(person_id=bank.target_person_ids[0][index],
                record_sha256=metadata['driver_record_sha256'], canvas_id=target.fingerprint()).items()):
            raise ValueError('Bank/body target pose provenance differs')
    identity = sample.get('identity_metadata')
    if identity is not None:
        for key, value in {**expected, 'source_geometry_sha256': primary.geometry_sha256,
                'source_indices': bank.source_indices[0].tolist()}.items():
            if key == 'target_canvas_id':
                continue
            if identity.get(key) != value:
                raise ValueError('Body Source identity provenance differs: ' + key)
    for key, mask in (('source_person_mask', primary.person_mask), ('source_head_mask', primary.head_mask)):
        if key in sample and not torch.equal(sample[key].cpu(), mask[0].cpu()):
            raise ValueError('Raw Source mask differs from independent source0')
    return bank
