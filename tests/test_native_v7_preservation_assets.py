"""CPU contracts for SHA-bound Source identity and background sidecars."""
from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from src.data.canvas_contract import native_pad


MODULE = 'src.pose_control.qwen21_full.native_v7.preservation_assets'
EVALUATOR_SHA = 'a' * 64


def _assets():
    return importlib.import_module(MODULE)


def _sample(hw=(48, 64)):
    canvas = native_pad(hw)
    h, w = canvas.canvas_hw
    shared = dict(source_canvas_id=canvas.fingerprint(), target_canvas_id=canvas.fingerprint(),
        source_rgb_sha256='b' * 64, source_person_ids=['alice', 'bob'],
        target_person_ids=['bob', 'alice'], driver_record_sha256='c' * 64)
    sample = dict(native_scene_kind='dual', source_canvas=canvas.metadata(), target_canvas=canvas.metadata(),
        source_token_hw=canvas.token_hw, target_token_hw=canvas.token_hw,
        source_indices=torch.tensor([1, 0]), valid_image_mask=canvas.valid_mask(),
        source_person_mask=torch.zeros(2, 1, h, w, dtype=torch.bool),
        dual_depth=SimpleNamespace(projected=torch.zeros(1, 2, 1, h, w, dtype=torch.bool)),
        identity_metadata={**shared, 'source_indices': [1, 0]},
        local_condition_v4=SimpleNamespace(metadata=(shared,), source_indices=torch.tensor([[1, 0]])),
        pair_metadata=dict(pair_id='pair', source_rgb_sha256='b' * 64),
        pose_provenance=dict(record_sha256='c' * 64, canvas_id=canvas.fingerprint(), person_id='bob'),
        pose_provenance_b=dict(record_sha256='c' * 64, canvas_id=canvas.fingerprint(), person_id='alice'))
    return sample


def _record(tmp_path, sample=None, *, margin=16):
    api = _assets()
    sample = _sample() if sample is None else sample
    features = torch.ones(2, 512)
    features[1].zero_()
    valid = torch.tensor([True, False])
    support = api.background_support(sample, margin_px=margin)
    metadata = {**api.sample_binding(sample, EVALUATOR_SHA), 'background_margin_px': margin,
        'background_implementation': api.BACKGROUND_IMPLEMENTATION}
    ref = api.write_record(tmp_path / 'preservation.pt', features, valid, support, metadata)
    return sample, ref, metadata


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(tmp_path, ref, evaluator_sha=EVALUATOR_SHA):
    path = tmp_path / 'index.json'
    record = dict(schema='native_v7_preservation_index_v1',
        evaluator=dict(path=str((tmp_path / 'fixed-evaluator.pt').resolve()), sha256=evaluator_sha),
        entries={'train\0pair': ref})
    path.write_text(json.dumps(record), encoding='utf8')
    return path, dict(path=str(path.resolve()), sha256=_sha(path)), record


def test_record_roundtrip_is_detached_cpu_fp32_and_exclusive(tmp_path):
    api = _assets()
    sample, ref, metadata = _record(tmp_path)
    restored = api.read_record(ref, sample, EVALUATOR_SHA)
    assert restored['schema'] == 'native_v7_preservation_sample_v1'
    assert restored['metadata'] == metadata
    assert restored['source_features'].shape == (2, 512)
    assert restored['source_features'].dtype == torch.float32
    assert restored['source_valid'].dtype == torch.bool
    assert restored['background_support'].shape == (1, 16)
    assert all(not restored[key].requires_grad and restored[key].device.type == 'cpu'
        for key in ('source_features', 'source_valid', 'background_support'))
    with pytest.raises(FileExistsError):
        api.write_record(ref['path'], restored['source_features'], restored['source_valid'],
            restored['background_support'], metadata)


@pytest.mark.parametrize('mutation', [
    lambda s: s['identity_metadata'].update(source_rgb_sha256='d' * 64),
    lambda s: s.update(source_canvas=native_pad((32, 64)).metadata()),
    lambda s: s.update(target_canvas=native_pad((64, 32)).metadata()),
    lambda s: s['local_condition_v4'].metadata[0].update(target_person_ids=['alice', 'bob']),
    lambda s: s['identity_metadata'].update(source_person_ids=['other', 'bob']),
    lambda s: s.update(source_indices=torch.tensor([0, 1])),
    lambda s: s['local_condition_v4'].source_indices.copy_(torch.tensor([[0, 1]])),
    lambda s: s['pose_provenance'].update(record_sha256='d' * 64),
    lambda s: s.update(target_token_hw=(2, 8)),
    lambda s: s['pair_metadata'].update(pair_id='other'),
])
def test_record_rejects_source_canvas_person_driver_mapping_or_grid_rebinding(tmp_path, mutation):
    api = _assets()
    sample, ref, _ = _record(tmp_path)
    mutation(sample)
    with pytest.raises(ValueError):
        api.read_record(ref, sample, EVALUATOR_SHA)


def test_record_rejects_another_embedding_space_and_changed_file(tmp_path):
    api = _assets()
    sample, ref, _ = _record(tmp_path)
    with pytest.raises(ValueError, match='evaluator'):
        api.read_record(ref, sample, 'd' * 64)
    with open(ref['path'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        api.read_record(ref, sample, EVALUATOR_SHA)


def test_background_excludes_both_source_and_target_people_and_padding_without_target_rgb():
    api = _assets()
    sample = _sample((40, 64))
    sample['source_person_mask'][0, 0, 0:16, 0:16] = True
    sample['dual_depth'].projected[0, 1, 0, 16:32, 32:48] = True
    support = api.background_support(sample, margin_px=0).view(4, 4)
    assert support[0, 0] == 0 and support[1, 2] == 0
    assert support[0, 3] == 1 and support[1, 0] == 1
    assert support[2, 0] == .5 and support[3].sum() == 0
    assert api.background_support(sample, margin_px=16).sum() < support.sum()


def test_background_union_dilates_by_exact_pixel_margin_and_empty_support_is_valid(tmp_path):
    api = _assets()
    sample = _sample((64, 64))
    sample['source_person_mask'][0, 0, 16, 16] = True
    mask = torch.ones(64, 64)
    mask[15:18, 15:18] = 0
    expected = torch.nn.functional.avg_pool2d(mask[None, None], 16, 16).flatten(1)
    assert torch.equal(api.background_support(sample, margin_px=1), expected)
    sample['source_person_mask'].fill_(True)
    _, ref, _ = _record(tmp_path, sample)
    assert api.read_record(ref, sample, EVALUATOR_SHA)['background_support'].sum() == 0


@pytest.mark.parametrize('change', ['bank', 'canvas', 'scene', 'grid', 'valid', 'margin'])
def test_background_refuses_unsupported_or_inconsistent_geometry(change):
    api = _assets()
    sample = _sample()
    margin = 16
    if change == 'bank':
        sample['source_bank_v5'] = object()
    elif change == 'canvas':
        sample['source_canvas'] = native_pad((32, 64)).metadata()
    elif change == 'scene':
        sample['native_scene_kind'] = 'single'
    elif change == 'grid':
        sample['target_token_hw'] = (2, 8)
    elif change == 'valid':
        sample['valid_image_mask'].fill_(1)
    else:
        margin = -1
    with pytest.raises(ValueError):
        api.background_support(sample, margin_px=margin)


def test_record_rejects_wrong_margin_or_repeated_padding_coverage(tmp_path):
    api = _assets()
    sample, ref, metadata = _record(tmp_path, _sample((40, 64)), margin=0)
    restored = api.read_record(ref, sample, EVALUATOR_SHA)
    coverage = native_pad((40, 64)).token_valid_coverage().flatten()[None]
    wrong = api.write_record(tmp_path / 'double-coverage.pt', restored['source_features'],
        restored['source_valid'], restored['background_support'] * coverage, metadata)
    with pytest.raises(ValueError, match='background'):
        api.read_record(wrong, sample, EVALUATOR_SHA)
    sample['source_person_mask'][0, 0, 16, 16] = True
    supported = api.background_support(sample, margin_px=0)
    wrong = api.write_record(tmp_path / 'wrong-margin.pt', restored['source_features'], restored['source_valid'],
        supported, {**metadata, 'background_margin_px': 16})
    with pytest.raises(ValueError, match='background'):
        api.read_record(wrong, sample, EVALUATOR_SHA)


@pytest.mark.parametrize('change', ['feature_shape', 'dtype', 'nonfinite', 'invalid_feature', 'support_range'])
def test_write_rejects_malformed_tensor_contract(tmp_path, change):
    api = _assets()
    sample, _, metadata = _record(tmp_path)
    features, valid, support = torch.ones(2, 512), torch.ones(2, dtype=torch.bool), api.background_support(sample)
    if change == 'feature_shape': features = features[:, :511]
    elif change == 'dtype': features = features.half()
    elif change == 'nonfinite': features[0, 0] = float('nan')
    elif change == 'invalid_feature': valid[0] = False
    else: support[0, 0] = 1.1
    with pytest.raises(ValueError):
        api.write_record(tmp_path / 'bad.pt', features, valid, support, metadata)


def test_index_verifies_its_sha_and_all_record_closure_without_loading_evaluator(tmp_path):
    api = _assets()
    _, ref, _ = _record(tmp_path)
    path, bound, original = _index(tmp_path, ref)
    index, verified = api.read_index(bound)
    assert index == original and verified == bound
    with pytest.raises(ValueError, match='SHA'):
        api.read_index({**bound, 'sha256': 'f' * 64})
    with open(ref['path'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        api.read_index(bound)
    assert path.exists()


@pytest.mark.parametrize('change', ['evaluator', 'entry_key', 'record_sha', 'index_schema'])
def test_index_rejects_cross_model_or_invalid_reference_closure(tmp_path, change):
    api = _assets()
    _, ref, _ = _record(tmp_path)
    path, _, record = _index(tmp_path, ref)
    if change == 'evaluator': record['evaluator']['sha256'] = 'd' * 64
    elif change == 'entry_key': record['entries'] = {'missing-separator': ref}
    elif change == 'record_sha': record['entries']['train\0pair']['sha256'] = 'f' * 64
    else: record['schema'] = 'other'
    path.write_text(json.dumps(record), encoding='utf8')
    with pytest.raises(ValueError):
        api.read_index(dict(path=str(path.resolve()), sha256=_sha(path)))
