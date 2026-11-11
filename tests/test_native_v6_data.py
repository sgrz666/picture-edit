"""CPU manifest contracts for independent five-reference appearance sources."""
from dataclasses import fields, replace
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_full_assets import read_record, sha256, write_record
from src.pose_control.qwen21_full.native_local_condition import (
    LOCAL_SCHEMA, REFERENCE_SCHEMA, NativeLocalConditionV4, NativeLocalReferencesV4,
)
from src.pose_control.qwen21_full.native_v5.contracts import (
    DETAIL_SCHEMA, SUPERVISION_SCHEMA, NativeDetailReferencesV5, NativeTrainingSupervisionV5,
)
from src.pose_control.qwen21_full.native_v5.source_bank import NativeAppearanceSourceV5, NativeSourceBankV5


def _image(path, spec, color):
    Image.new('RGB', spec.original_hw[::-1], color).save(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


def _references(rgb, canvas, person, geometry):
    ids = [person, '__inactive__']
    metadata = dict(schema=REFERENCE_SCHEMA, role='source_only', source_rgb_sha256=rgb['sha256'],
        source_canvas_id=canvas.fingerprint(), source_person_ids=ids, reference_count=1,
        reference_sources=[dict(rgb=rgb, canvas=canvas.metadata(), canvas_id=canvas.fingerprint(),
            geometry_sha256=geometry, person_ids=ids)])
    return NativeLocalReferencesV4(torch.ones(1, 2, 1, 512), torch.tensor([[[True], [False]]]),
        torch.zeros(1, 2, 1, 256, 1536), torch.zeros(1, 2, 1, dtype=torch.bool),
        torch.zeros(1, 2, 2, 1, 256, 1536), torch.zeros(1, 2, 2, 1, dtype=torch.bool),
        torch.tensor([[[0., 0., 1., 1.], [0., 0., 0., 0.]]]), torch.zeros(1, 2, 2, 4),
        (metadata,)).validate()


def _bank(tmp_path):
    sources = []
    for i, hw in enumerate(((32, 64), (64, 32))):
        spec = native_pad(hw)
        rgb = _image(tmp_path / f'source{i}.png', spec, ('red', 'blue')[i])
        person = f'person{i}'
        refs = _references(rgb, spec, person, f'geometry{i}')
        support = torch.zeros(1, 6, *hw, dtype=torch.bool)
        support[:, 0] = True
        valid = torch.tensor([[True, False, False, False, False, False]])
        detail = NativeDetailReferencesV5(torch.tensor([[[0., 0., 1., 1.]] * 6]), support,
            valid, torch.tensor([[0, -1]]), ({'schema': DETAIL_SCHEMA, 'role': 'primary_source_only',
                'source_rgb_sha256': rgb['sha256'], 'source_canvas_id': spec.fingerprint(),
                'source_person_ids': [person, '__inactive__']},)).validate()
        sources.append(NativeAppearanceSourceV5(i, (person,), i + 3, spec.metadata(), rgb['sha256'],
            f'geometry{i}', torch.ones(1, 1, 1, *hw, dtype=torch.bool),
            torch.ones(1, 1, 1, *hw, dtype=torch.bool), refs, detail))
    return NativeSourceBankV5(tuple(sources), torch.tensor([[0, 1]]), torch.tensor([[0, 0]]),
        (('person0', 'person1'),)).validate()


def _local(bank, target, target_rgb):
    z = lambda *shape: torch.zeros(1, *shape)
    m = lambda *shape: torch.zeros(1, *shape, dtype=torch.bool)
    h, w = target.canvas_hw
    return NativeLocalConditionV4(torch.ones(1, 2, dtype=torch.bool), bank.source_indices.clone(),
        torch.tensor([[3, 4]]), z(2, 72, 3), m(2, 72), z(2, 3), m(2), z(2, 10), m(2),
        z(2, 4), m(2), z(2, 2, 45), m(2, 2), z(2, 2, 21, 3), m(2, 2, 21),
        z(2, 2, 1024), m(2, 2), z(2, 2, 4), m(2, 2), z(2, 2), z(6, h, w), z(6, h, w),
        z(6, h, w), m(6, h, w), m(6, h, w), z(2, h, w), z(1, h, w), m(1, h, w),
        torch.ones(1, 2) * 1.7, ({'schema': LOCAL_SCHEMA, 'expression_mode': 'off',
            'target_canvas_id': target.fingerprint(), 'source_canvas_id': bank.sources[0].record()['canvas_id'],
            'source_rgb_sha256': bank.sources[0].rgb_sha256, 'source_asset_sha256': 'geometry0',
            'driver_record_sha256': 'target_geometry', 'driver_rgb_sha256': target_rgb['sha256'],
            'camera_id': 'camera', 'scale_group_id': 'scale', 'normal_space': 'camera_cv',
            'source_person_ids': ['person0', 'person1'], 'target_person_ids': list(bank.target_person_ids[0]),
            'hand_pose_convention': 'unavailable'},)).validate()


def _manifest(tmp_path, *, sample_mutation=None, bank_mutation=None, supervision_mutation=None):
    from src.pose_control.qwen21_full.native_v6.source_bank import write_source_bank
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA, text_sha
    bank = _bank(tmp_path)
    if bank_mutation:
        bank = bank_mutation(bank)
    target = native_pad((64, 64))
    target_rgb = _image(tmp_path / 'target.png', target, 'green')
    local = _local(bank, target, target_rgb)
    primary = bank.sources[0]
    sample = dict(source_canvas=primary.canvas, target_canvas=target.metadata(),
        source_token_hw=primary.token_hw, target_token_hw=target.token_hw,
        valid_image_mask=target.valid_mask(), source_indices=bank.source_indices[0].clone(),
        source_person_mask=primary.person_mask[0], source_head_mask=primary.head_mask[0],
        pair_metadata={'pair_id': 'pair', 'cross_person': False, 'source_rgb_sha256': primary.rgb_sha256,
            'target_rgb_sha256': target_rgb['sha256']},
        pose_provenance={'record_sha256': 'target_geometry', 'canvas_id': target.fingerprint(), 'person_id': bank.target_person_ids[0][0]},
        pose_provenance_b={'record_sha256': 'target_geometry', 'canvas_id': target.fingerprint(), 'person_id': bank.target_person_ids[0][1]},
        identity_metadata={'source_canvas_id': primary.record()['canvas_id'], 'source_rgb_sha256': primary.rgb_sha256,
            'source_geometry_sha256': primary.geometry_sha256, 'source_person_ids': ['person0', 'person1'],
            'target_person_ids': list(bank.target_person_ids[0]), 'source_indices': bank.source_indices[0].tolist()},
        local_condition_v4=local)
    if sample_mutation:
        sample_mutation(sample)
    supervision = NativeTrainingSupervisionV5(torch.zeros(1, 6, 64, 64), torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, 5, 2), torch.tensor([True]), ({'schema': SUPERVISION_SCHEMA, 'role': 'training_only',
            'driver_record_sha256': 'target_geometry', 'driver_rgb_sha256': target_rgb['sha256'],
            'target_canvas_id': target.fingerprint(), 'target_person_ids': list(bank.target_person_ids[0])},)).validate()
    if supervision_mutation:
        supervision = supervision_mutation(supervision)
    images = []
    for i, role in enumerate(('source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose')):
        canvas = native_pad((32, 64) if i == 0 else (64, 32)) if i < 2 else target
        rgb = bank.sources[i].references.metadata[0]['reference_sources'][0]['rgb'] if i < 2 else _image(
            tmp_path / f'pose{i}.png', canvas, (i * 20, 40, 60))
        images.append(dict(role=role, file=rgb, canvas=canvas.metadata(), canvas_id=canvas.fingerprint(),
            space='original' if i < 2 else 'canvas'))
    prompt_path = tmp_path / 'prompt.txt'
    prompt_path.write_text('Transfer both poses.', encoding='utf8')
    condition = dict(schema=SCHEMA, profile='dual_source_5ref', pair_id='pair', primary_source_index=0,
        appearance_source_indices=[0, 1], target_canvas_id=target.fingerprint(), images=images,
        prompt=dict(text='Transfer both poses.', sha256=text_sha('Transfer both poses.'),
            field='compiled_prompt_inference', bank={'path': str(prompt_path.resolve()), 'sha256': sha256(prompt_path)}))
    condition_path = tmp_path / 'condition.json'
    condition_path.write_text(json.dumps(condition), encoding='utf8')
    row = dict(schema='native_full_pair_v6', pair_id='pair', scene_kind='dual', data_profile='full_smplx',
        sample=write_record(tmp_path / 'sample.npz', sample, schema='native_full_sample_v6'),
        source_bank=write_source_bank(tmp_path / 'bank.npz', bank),
        supervision=write_record(tmp_path / 'supervision.npz', {f.name: getattr(supervision, f.name)
            for f in fields(supervision)}, schema=SUPERVISION_SCHEMA),
        qwen_condition={'path': str(condition_path.resolve()), 'sha256': sha256(condition_path)},
        source_rgb=images[0]['file'], target_rgb=target_rgb)
    manifest = tmp_path / 'pairs.jsonl'
    manifest.write_text(json.dumps(row) + '\n', encoding='utf8')
    return manifest, row, bank


def test_source_bank_roundtrip_retains_rectangular_canvases_without_latents(tmp_path):
    from src.pose_control.qwen21_full.native_v6.source_bank import read_source_bank, write_source_bank
    original = _bank(tmp_path)
    ref = write_source_bank(tmp_path / 'bank.npz', original)
    restored = read_source_bank(ref)
    assert isinstance(restored, NativeSourceBankV5)
    assert [s.token_hw for s in restored.sources] == [(2, 4), (4, 2)]
    assert all(s.latent is None for s in restored.sources)
    assert torch.equal(restored.sources[1].detail.support, original.sources[1].detail.support)
    raw = read_record(ref, schema='native_v6_source_bank_v1')
    assert all(isinstance(s, dict) and 'latent' not in s and isinstance(s['detail'], dict) for s in raw['sources'])


def test_source_bank_serialization_rejects_live_qwen_latents(tmp_path):
    from src.pose_control.qwen21_full.native_v6.source_bank import write_source_bank
    bank = _bank(tmp_path)
    source = replace(bank.sources[0], latent=torch.ones(1, 8, 64))
    with pytest.raises(ValueError, match='latent'):
        write_source_bank(tmp_path / 'bank.npz', replace(bank, sources=(source, bank.sources[1])))


def test_inference_reads_five_ref_bank_without_opening_target_or_supervision(tmp_path, monkeypatch):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, row, _ = _manifest(tmp_path)
    original = Path.open
    denied = {Path(row['target_rgb']['path']), Path(row['supervision']['path'])}
    def guarded(path, *args, **kwargs):
        assert path not in denied, f'Inference opened training-only asset: {path}'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    sample = NativeFullDatasetV6(manifest, load_target=False)[0]
    assert sample['native_scene_kind'] == 'dual' and not sample['supervised_target_rgb']
    assert 'tgt_image' not in sample and 'training_supervision_v5' not in sample
    assert sample['src_image'].shape == (3, 32, 64)
    assert isinstance(sample['source_bank_v5'], NativeSourceBankV5)
    assert sample['local_references_v4'].arcface_valid[0, :, 0].all()


def test_training_loads_exact_target_and_supervision(tmp_path):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6, collate_native_full_v6
    manifest, _, _ = _manifest(tmp_path)
    sample = NativeFullDatasetV6(manifest, load_target=True)[0]
    assert sample['tgt_image'].shape == (3, 64, 64)
    assert sample['training_supervision_v5'].paired.all()
    batch = collate_native_full_v6([sample])
    assert batch['source_bank_v5'] is sample['source_bank_v5']
    assert batch['source_bank_v5'].sources[1].person_mask.shape == (1, 1, 1, 64, 32)
    assert batch['src_image'].shape == (1, 3, 32, 64)


@pytest.mark.parametrize('mutation', [
    lambda s: s['local_condition_v4'].metadata[0].update(source_person_ids=['other', 'person1']),
    lambda s: s['local_condition_v4'].metadata[0].update(target_person_ids=['person1', 'person0']),
    lambda s: s['local_condition_v4'].source_indices.copy_(torch.tensor([[1, 0]])),
    lambda s: s['local_condition_v4'].source_slot_ids.copy_(torch.tensor([[4, 3]])),
    lambda s: s['pose_provenance_b'].update(person_id='other'),
    lambda s: s['pose_provenance_b'].update(record_sha256='other'),
    lambda s: s['pose_provenance_b'].update(canvas_id='other'),
    lambda s: s.update(source_canvas=native_pad((64, 32)).metadata()),
    lambda s: s.update(target_token_hw=(2, 8)),
    lambda s: s['pair_metadata'].update(pair_id='other'),
    lambda s: s['identity_metadata'].update(source_geometry_sha256='other'),
])
def test_loader_rejects_geometry_identity_or_canvas_rebinding(tmp_path, mutation):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, _, _ = _manifest(tmp_path, sample_mutation=mutation)
    with pytest.raises(ValueError):
        NativeFullDatasetV6(manifest)[0]


def test_source_bank_rejects_auxiliary_target_rgb_and_wrong_primary_canvas(tmp_path):
    from src.pose_control.qwen21_full.native_v6.source_bank import read_source_bank, write_source_bank
    bank = _bank(tmp_path)
    reference = bank.sources[0].references.metadata[0]['reference_sources'][0]
    ref = write_source_bank(tmp_path / 'bank.npz', bank)
    with pytest.raises(ValueError, match='target|supervision'):
        read_source_bank(ref, forbidden_rgb_sha256=reference['rgb']['sha256'])
    source = bank.sources[0]
    altered = deepcopy(source.references.metadata[0])
    altered['reference_sources'][0]['canvas'] = bank.sources[1].canvas
    bad = replace(source, references=replace(source.references, metadata=(altered,)))
    with pytest.raises(ValueError, match='Canvas|canvas'):
        write_source_bank(tmp_path / 'bad.npz', replace(bank, sources=(bad, bank.sources[1])))


def test_bank_collation_rejects_multirow_and_mixed_bank_batches(tmp_path):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6, collate_native_full_v6
    manifest, _, _ = _manifest(tmp_path)
    sample = NativeFullDatasetV6(manifest)[0]
    with pytest.raises(ValueError, match='B1|batch|bank'):
        collate_native_full_v6([sample, sample])
    plain = {k: v for k, v in sample.items() if k != 'source_bank_v5'}
    with pytest.raises(ValueError, match='B1|batch|bank'):
        collate_native_full_v6([sample, plain])


@pytest.mark.parametrize('scene_kind', ['single', 'dual'])
def test_legacy_v5_rows_use_existing_reader_without_temporary_manifests(tmp_path, scene_kind):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, row, bank = _manifest(tmp_path)
    raw = read_record(row['sample'], schema='native_full_sample_v6')
    local = raw.pop('local_condition_v4')
    if scene_kind == 'single':
        local = replace(local, person_valid=torch.tensor([[True, False]]), source_indices=torch.tensor([[0, -1]]),
            metadata=({**local.metadata[0], 'target_person_ids': ['person0', '__inactive__']},))
        raw['source_indices'] = local.source_indices[0]
        raw['identity_metadata']['target_person_ids'] = ['person0', '__inactive__']
        raw['identity_metadata']['source_indices'] = [0, -1]
        supervision = read_record(row['supervision'], schema=SUPERVISION_SCHEMA)
        supervision['metadata'][0]['target_person_ids'] = ['person0', '__inactive__']
        row['supervision'] = write_record(tmp_path / 'single_supervision.npz', supervision, schema=SUPERVISION_SCHEMA)
    refs = bank.sources[0].references
    metadata = deepcopy(refs.metadata[0])
    metadata['source_person_ids'] = ['person0', 'person1']
    metadata['reference_sources'][0]['person_ids'] = ['person0', 'person1']
    refs = replace(refs, metadata=(metadata,))
    detail = replace(bank.sources[0].detail, source_indices=local.source_indices,
        metadata=({**bank.sources[0].detail.metadata[0], 'source_person_ids': ['person0', 'person1']},))
    v4row = dict(schema='native_full_pair_v4', pair_id='pair', scene_kind=scene_kind,
        body=write_record(tmp_path / 'body.npz', raw, schema='native_full_body_record_v4'),
        local=write_record(tmp_path / 'local.npz', local, schema=LOCAL_SCHEMA),
        references=write_record(tmp_path / 'references.npz', refs, schema=REFERENCE_SCHEMA),
        source_rgb=row['source_rgb'], target_rgb=row['target_rgb'])
    v4path = tmp_path / 'v4.jsonl'
    v4path.write_text(json.dumps(v4row) + '\n', encoding='utf8')
    v5row = dict(schema='native_full_pair_v5', pair_id='pair',
        v4_manifest={'path': str(v4path.resolve()), 'sha256': sha256(v4path)},
        detail=write_record(tmp_path / 'detail.npz', {f.name: getattr(detail, f.name) for f in fields(detail)}, schema=DETAIL_SCHEMA),
        supervision=row['supervision'])
    manifest.write_text(json.dumps(v5row) + '\n', encoding='utf8')
    before = set(tmp_path.iterdir())
    loaded = NativeFullDatasetV6(manifest, load_target=True)[0]
    assert 'source_bank_v5' not in loaded and loaded['training_supervision_v5'].paired.all()
    assert loaded['native_scene_kind'] == scene_kind
    assert set(tmp_path.iterdir()) == before


def test_swapped_target_slots_keep_source_array_order_and_bind_actual_qwen_latents(tmp_path):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6, collate_native_full_v6
    def swapped(bank):
        return replace(bank, target_qwen_image_indices=torch.tensor([[1, 0]]),
            target_person_ids=(('person1', 'person0'),))
    manifest, _, _ = _manifest(tmp_path, bank_mutation=swapped)
    sample = NativeFullDatasetV6(manifest)[0]
    grids = [s.token_hw for s in sample['source_bank_v5'].sources] + [(4, 4)] * 3
    latents = torch.cat([torch.full((1, h * w, 64), float(i + 1)) for i, (h, w) in enumerate(grids)], 1)
    count = latents.shape[1]
    pair = dict(qwen_condition_profile='dual_source_5ref',
        reference_roles=['source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose'],
        reference_token_hws=grids, reference_sha256s=[s.rgb_sha256 for s in sample['source_bank_v5'].sources] + ['pose'] * 3,
        reference_latents=latents, source_latent=latents[:, :8], source_token_hw=grids[0], target_token_hw=(4, 4),
        prompt_embeds=torch.zeros(1, count // 4 + 2, 8), prompt_mask=torch.ones(1, count // 4 + 2, dtype=torch.bool),
        img_mask=torch.tensor([[False, False] + [True] * (count // 4 + 4)]))
    bank = sample['source_bank_v5'].with_pair(pair)
    bank.validate_target(collate_native_full_v6([sample]))
    assert bank.sources[0].latent.data_ptr() == latents.data_ptr()
    assert bank.sources[1].latent.storage_offset() == 8 * 64
    assert bank.source_indices.tolist() == [[1, 0]]


@pytest.mark.parametrize('key', ['sample', 'source_bank', 'qwen_condition', 'source_rgb'])
def test_dataset_rejects_changed_input_sha(tmp_path, key):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, row, _ = _manifest(tmp_path)
    row[key]['sha256'] = 'changed'
    manifest.write_text(json.dumps(row) + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='SHA|fingerprint|Source|RGB declaration'):
        NativeFullDatasetV6(manifest)[0]


def test_inference_allows_omitting_target_reference_and_source_masks(tmp_path):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    def omit_masks(sample):
        sample.pop('source_person_mask')
        sample.pop('source_head_mask')
        sample.pop('identity_metadata')
    manifest, row, _ = _manifest(tmp_path, sample_mutation=omit_masks)
    row.pop('target_rgb')
    manifest.write_text(json.dumps(row) + '\n', encoding='utf8')
    sample = NativeFullDatasetV6(manifest)[0]
    assert sample['source_person_mask'].shape == (1, 1, 32, 64)
    with pytest.raises(ValueError, match='references'):
        NativeFullDatasetV6(manifest, load_target=True)


@pytest.mark.parametrize('mutation', [
    lambda s: replace(s, metadata=({**s.metadata[0], 'driver_record_sha256': 'wrong'},)),
    lambda s: replace(s, paired=torch.tensor([False])),
])
def test_training_rejects_wrong_or_unpaired_supervision(tmp_path, mutation):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, _, _ = _manifest(tmp_path, supervision_mutation=mutation)
    with pytest.raises(ValueError, match='supervision|paired'):
        NativeFullDatasetV6(manifest, load_target=True)[0]


def test_auxiliary_frozen_source_reference_cannot_use_training_target(tmp_path):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    from src.pose_control.qwen21_full.native_v6.source_bank import write_source_bank
    manifest, row, bank = _manifest(tmp_path)
    source = bank.sources[1]
    refs = source.references
    metadata = deepcopy(refs.metadata[0])
    metadata.update(reference_count=2, explicit_multi_reference=True)
    metadata['reference_sources'].append({**metadata['reference_sources'][0], 'rgb': row['target_rgb'],
        'canvas': native_pad((64, 64)).metadata(), 'canvas_id': native_pad((64, 64)).fingerprint()})
    extra = {}
    for name in ('arcface', 'arcface_valid', 'face_dino', 'face_dino_valid', 'hand_dino', 'hand_dino_valid'):
        dim = 3 if name.startswith('hand_') else 2
        extra[name] = torch.cat([getattr(refs, name)] * 2, dim=dim)
    source = replace(source, references=replace(refs, **extra, metadata=(metadata,)))
    row['source_bank'] = write_source_bank(tmp_path / 'leaking_bank.npz', replace(bank, sources=(bank.sources[0], source)))
    manifest.write_text(json.dumps(row) + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='target|Supervision'):
        NativeFullDatasetV6(manifest, load_target=True)[0]


def test_auxiliary_reference_canvas_must_match_actual_image_dimensions(tmp_path):
    from src.pose_control.qwen21_full.native_v6.source_bank import write_source_bank
    bank = _bank(tmp_path)
    source = bank.sources[0]
    refs = source.references
    metadata = deepcopy(refs.metadata[0])
    metadata.update(reference_count=2, explicit_multi_reference=True)
    other_canvas = native_pad((64, 32))
    metadata['reference_sources'].append({**metadata['reference_sources'][0],
        'canvas': other_canvas.metadata(), 'canvas_id': other_canvas.fingerprint()})
    extra = {}
    for name in ('arcface', 'arcface_valid', 'face_dino', 'face_dino_valid', 'hand_dino', 'hand_dino_valid'):
        extra[name] = torch.cat([getattr(refs, name)] * 2, dim=3 if name.startswith('hand_') else 2)
    altered = replace(source, references=replace(refs, **extra, metadata=(metadata,)))
    with pytest.raises(ValueError, match='Canvas|dimensions'):
        write_source_bank(tmp_path / 'bad_canvas.npz', replace(bank, sources=(altered, bank.sources[1])))


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(native_scene_kind='single'),
    lambda s: s.update(data_profile='hybrid_hi4d'),
    lambda s: s.update(source_indices=s['source_indices'].float()),
])
def test_loader_rejects_conflicting_raw_scene_profile_or_mapping_dtype(tmp_path, mutation):
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, _, _ = _manifest(tmp_path, sample_mutation=mutation)
    with pytest.raises(ValueError):
        NativeFullDatasetV6(manifest)[0]
