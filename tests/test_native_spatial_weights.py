"""CPU contracts for offline Canvas masks and configurable weighted FM."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.data.canvas_contract import native_pad
from src.pose_control.qwen21_full.native_v5.training.config import LossSpec, RouteSpec, SourceSpec, StageSpec, TrainingConfig
from src.pose_control.qwen21_full.native_v5.training.io import file_sha


def kernels():
    from src.pose_control.qwen21_full.native_v5 import spatial_weights
    return spatial_weights


def components(shape=(1, 4, 8)):
    return {key: torch.zeros(shape) for key in ('body', 'joint', 'hand')}


def test_pixel_fusion_caps_before_pool_and_keeps_row_major():
    maps = components()
    maps['body'][:, :2, :4] = 1
    maps['joint'][:, :2, :4] = 1
    maps['hand'][:, 2:, 4:] = 1
    actual = kernels().spatial_token_weights(maps, torch.ones(1, 4, 8), (2, 2), (1., 1., .5))
    torch.testing.assert_close(actual, torch.tensor([[2.5, 1., 1., 1.5]]))


def test_partial_padding_applies_valid_once_and_background_retains_one():
    maps = components((1, 4, 4))
    valid = torch.zeros(1, 4, 4)
    valid[:, :3] = 1
    maps['body'][:, :2, :2] = 1
    weights = kernels().spatial_token_weights(maps, valid, (2, 2), {'body': 1., 'joint': 0., 'hand': 0.})
    torch.testing.assert_close(weights, torch.tensor([[2., 1., .5, .5]]))


def test_area_pool_uses_actual_grid_without_fixed_stride_assumption():
    maps = components((1, 6, 10))
    maps['joint'][:, :3, 5:] = 1
    value = kernels().spatial_token_weights(maps, torch.ones(1, 6, 10), (2, 2), (0., .75, 0.))
    torch.testing.assert_close(value, torch.tensor([[1., 1.75, 1., 1.]]))


def test_spatial_velocity_averages_each_sample_before_batch():
    prediction = torch.tensor([[[1., 1.], [3., 3.]], [[2., 2.], [4., 4.]]], requires_grad=True)
    weight = torch.tensor([[1., 0.], [1., 3.]])
    loss = kernels().spatial_velocity_loss(prediction, torch.zeros_like(prediction), weight)
    torch.testing.assert_close(loss, torch.tensor(7.))
    loss.backward()
    assert prediction.grad[0, 1].count_nonzero() == 0
    torch.testing.assert_close(prediction.grad[1, 1, 0], torch.tensor(1.5))


def test_uniform_weight_scaling_leaves_velocity_and_gradient_unchanged():
    prediction = torch.tensor([[[1.], [3.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    weight = torch.tensor([[.5, 1.]])
    base = kernels().spatial_velocity_loss(prediction, target, weight)
    scaled = kernels().spatial_velocity_loss(prediction, target, weight * 2)
    torch.testing.assert_close(base, scaled)
    torch.testing.assert_close(torch.autograd.grad(base, prediction)[0], torch.autograd.grad(scaled, prediction)[0])


def test_hand_boost_increases_relative_hand_gradient():
    maps = components((1, 2, 4))
    maps['hand'][:, :, 2:] = 1
    weights = kernels().spatial_token_weights(maps, torch.ones(1, 2, 4), (1, 2), (0., 0., 1.))
    prediction = torch.ones(1, 2, 1, requires_grad=True)
    kernels().spatial_velocity_loss(prediction, torch.zeros_like(prediction), weights).backward()
    torch.testing.assert_close(prediction.grad[0, 1] / prediction.grad[0, 0], torch.tensor([2.]))


@pytest.mark.parametrize('coefficients,max_weight', [((-1., 0., 0.), 2.5), ((0., float('nan'), 0.), 2.5), ((0., 0., 0.), .9), ((0., 0., 0.), float('inf'))])
def test_invalid_spatial_coefficients_fail(coefficients, max_weight):
    with pytest.raises(ValueError):
        kernels().spatial_token_weights(components(), torch.ones(1, 4, 8), (2, 2), coefficients, max_weight=max_weight)


@pytest.mark.parametrize('bad', [float('nan'), -1., 1.01])
def test_invalid_component_range_fails(bad):
    maps = components()
    maps['body'][0, 0, 0] = bad
    with pytest.raises(ValueError):
        kernels().spatial_token_weights(maps, torch.ones(1, 4, 8), (2, 2), (1., 0., 0.))


def test_loss_rejects_empty_sample_and_mismatched_token_count():
    for weight in (torch.tensor([[1., 0.], [0., 0.]]), torch.ones(2, 3)):
        with pytest.raises(ValueError):
            kernels().spatial_velocity_loss(torch.ones(2, 2, 3), torch.zeros(2, 2, 3), weight)


def test_loss_spec_defaults_and_conflicting_legacy_local_loss():
    default = LossSpec()
    assert default.spatial_body == default.spatial_joint == default.spatial_hand == 0.
    assert default.spatial_max == 2.5
    assert replace(default, spatial_hand=1.).spatial_coefficients == (0., 0., 1.)
    for kwargs in ({'spatial_body': -1.}, {'spatial_joint': float('nan')}, {'spatial_max': .5},
                   {'spatial_max': float('inf')}, {'spatial_body': .5, 'face': .5}, {'spatial_hand': .5, 'hand': .5}):
        with pytest.raises(ValueError):
            LossSpec(**kwargs)


def artifact(path):
    return {'path': str(path.resolve()), 'sha256': file_sha(path)}


def write_components(tmp_path, *, canvas=None, metadata_updates=None, array_updates=None, name='components.npz'):
    canvas = canvas or native_pad((29, 30))
    metadata = {'schema': 'native_spatial_components_v1', 'gt_rgb_sha256': 'a' * 64,
                'target_canvas_id': canvas.fingerprint(), 'canvas': canvas.metadata(),
                'frame_id': 'frame001', 'geometry_method': 'trusted_geometry_fixture'}
    metadata.update(metadata_updates or {})
    arrays = {name: np.zeros(canvas.canvas_hw, dtype=np.float16) for name in ('body', 'joint', 'hand')}
    arrays['valid'] = canvas.valid_mask()[0].numpy().astype(np.uint8)
    arrays.update(array_updates or {})
    path = tmp_path / name
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return artifact(path), metadata, canvas


def write_index(tmp_path, ref, metadata, *, name='index.json'):
    key = metadata['gt_rgb_sha256'] + ':' + metadata['target_canvas_id']
    path = tmp_path / name
    path.write_text(json.dumps({'schema': 'native_spatial_index_v1', 'entries': {key: {'file': ref, 'metadata': metadata}}}), encoding='utf8')
    return artifact(path)


def test_read_components_verifies_gt_canvas_and_valid_background(tmp_path):
    ref, metadata, canvas = write_components(tmp_path)
    actual = kernels().read_spatial_components(ref, gt_rgb_sha256='a' * 64, target_canvas_id=canvas.fingerprint(), canvas=canvas.metadata())
    assert actual['metadata'] == metadata
    assert actual['body'].shape == (1, *canvas.canvas_hw)
    torch.testing.assert_close(actual['valid'], canvas.valid_mask())
    assert int(actual['valid'].sum()) == 29 * 30


@pytest.mark.parametrize('change', ['sha', 'gt', 'canvas', 'valid', 'dtype', 'nonfinite', 'schema'])
def test_read_components_refuses_stale_or_malformed_assets(tmp_path, change):
    canvas = native_pad((29, 30))
    updates = {}
    metadata_updates = {}
    if change == 'valid': updates['valid'] = np.ones(canvas.canvas_hw, dtype=np.uint8)
    if change == 'dtype': updates['body'] = np.zeros(canvas.canvas_hw, dtype=np.float32)
    if change == 'nonfinite': updates['joint'] = np.full(canvas.canvas_hw, np.nan, dtype=np.float16)
    if change == 'schema': metadata_updates['schema'] = 'old'
    ref, metadata, _ = write_components(tmp_path, array_updates=updates, metadata_updates=metadata_updates)
    if change == 'sha': ref['sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        kernels().read_spatial_components(ref, gt_rgb_sha256=('b' if change == 'gt' else 'a') * 64,
                                         target_canvas_id='b' * 64 if change == 'canvas' else canvas.fingerprint(), canvas=canvas.metadata())


def test_index_is_hash_pinned_and_source_reference_roundtrips(tmp_path):
    ref, metadata, _ = write_components(tmp_path)
    index_ref = write_index(tmp_path, ref, metadata)
    index, bound = kernels().read_spatial_index(index_ref)
    assert bound == index_ref and len(index['entries']) == 1
    source = SourceSpec('train', spatial_weights=index_ref)
    config = TrainingConfig(model_path=str(tmp_path), sources=(source,))
    assert TrainingConfig.from_dict(config.record()).sources[0].spatial_weights == index_ref
    with pytest.raises(ValueError):
        SourceSpec('train', spatial_weights={'path': index_ref['path']})
    with pytest.raises(ValueError):
        kernels().read_spatial_index({'path': index_ref['path']})


def test_prepare_missing_bindings_reports_spatial_cache_for_selected_sources(tmp_path):
    route = RouteSpec('body', 'body', source_ids=('train',), loss=LossSpec(spatial_body=.5))
    config = TrainingConfig(model_path=str(tmp_path), stage=StageSpec(routes=(route,)),
        sources=(SourceSpec('train', weight=1.), SourceSpec('unused', weight=1.), SourceSpec('val', role='validation')))
    missing = config.missing()
    assert 'train.spatial_weights' in missing and 'val.spatial_weights' in missing
    assert 'unused.spatial_weights' not in missing


def tiny_supervision():
    from src.pose_control.qwen21_full.native_v5.contracts import NativeTrainingSupervisionV5, SUPERVISION_SCHEMA
    metadata = {'schema': SUPERVISION_SCHEMA, 'role': 'training_only', 'driver_record_sha256': 'a' * 64,
                'driver_rgb_sha256': 'b' * 64, 'target_canvas_id': 'c' * 64, 'target_person_ids': ['person', None]}
    return NativeTrainingSupervisionV5(torch.zeros(1, 6, 16, 32), torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, 5, 2), torch.tensor([True]), (metadata,))


def micro_loss(spec, spatial=None, coverage=None):
    from src.pose_control.qwen21_full.native_v5.training.objective import compute_micro_loss
    prediction = torch.tensor([[[1.], [3.]]], requires_grad=True)
    loss, parts = compute_micro_loss(None, prediction, torch.zeros_like(prediction),
        torch.ones(1, 2) if coverage is None else coverage, tiny_supervision(), torch.zeros_like(prediction),
        torch.tensor([.5]), torch.zeros(1, 3, 16, 32), spec,
        {'velocity': 1, 'face': 0, 'hand': 0, 'identity': 0, 'perceptual': 0}, 1, (None, None), spatial=spatial)
    return loss, parts


def test_zero_coefficients_keep_legacy_objective_even_if_spatial_passed():
    loss, parts = micro_loss(LossSpec(), spatial={'deliberately': 'unread'})
    torch.testing.assert_close(loss, torch.tensor(5.))
    assert 'spatial_weight_sum' not in parts


def test_zero_coefficients_exactly_match_legacy_qwen64_with_partial_padding():
    from src.pose_control.qwen21_full.native_v5.losses import LocalLossConfig, local_velocity_loss
    from src.pose_control.qwen21_full.native_v5.training.objective import compute_micro_loss
    prediction = torch.linspace(-1., 1., 128).reshape(1, 2, 64).requires_grad_()
    target = torch.full_like(prediction, .25)
    coverage = torch.tensor([[1., .5]])
    supervision = tiny_supervision()
    expected, expected_parts = local_velocity_loss(prediction, target, coverage, supervision, LocalLossConfig())
    actual, parts = compute_micro_loss(None, prediction, target, coverage, supervision, torch.zeros_like(prediction),
        torch.tensor([.5]), torch.zeros(1, 3, 16, 32), LossSpec(),
        {'velocity': 1, 'face': 0, 'hand': 0, 'identity': 0, 'perceptual': 0}, 1, (None, None))
    assert torch.equal(actual, expected)
    for key, value in expected_parts.items():
        assert torch.equal(parts[key], value)
    assert torch.equal(torch.autograd.grad(actual, prediction, retain_graph=True)[0], torch.autograd.grad(expected, prediction)[0])


def test_qwen64_weighted_loss_matches_channel_normalized_reference():
    prediction = torch.linspace(-1., 2., 256).reshape(2, 2, 64)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([[1., .5], [0., 2.5]])
    reference = torch.stack([(prediction[row].square() * weights[row, :, None]).sum() / (64 * weights[row].sum()) for row in range(2)]).mean()
    torch.testing.assert_close(kernels().spatial_velocity_loss(prediction, target, weights), reference)


def test_objective_uses_spatial_weights_with_no_second_coverage_multiply():
    maps = components((1, 16, 32))
    maps['hand'][:, :, 16:] = 1
    valid = torch.ones(1, 16, 32)
    valid[:, 8:, 16:] = 0
    spatial = {'components': maps, 'valid': valid, 'target_token_hw': (1, 2)}
    loss, parts = micro_loss(LossSpec(spatial_hand=1.), spatial, torch.tensor([[1., .5]]))
    torch.testing.assert_close(loss, torch.tensor(5.))
    torch.testing.assert_close(parts['spatial_weight_sum'], torch.tensor(2.))
    torch.testing.assert_close(parts['velocity_denominator'], torch.tensor(2.))


def test_objective_requires_spatial_maps_and_matching_coverage():
    with pytest.raises(ValueError, match='[Ss]patial'):
        micro_loss(LossSpec(spatial_body=1.))
    spatial = {'components': components((1, 16, 32)), 'valid': torch.ones(1, 16, 32), 'target_token_hw': (1, 2)}
    with pytest.raises(ValueError, match='coverage'):
        micro_loss(LossSpec(spatial_body=1.), spatial, torch.tensor([[1., .5]]))


def data_fixture(tmp_path, monkeypatch, *, spatial=True, missing_entry=False):
    from src.pose_control.qwen21_full.native_v5.training import data
    canvas = native_pad((29, 30))
    samples = {}
    sources = []
    for role, sha in (('train', 'a' * 64), ('validation', 'b' * 64)):
        ref, metadata, _ = write_components(tmp_path, canvas=canvas, metadata_updates={'gt_rgb_sha256': sha}, name=role + '.npz')
        index = write_index(tmp_path, ref, metadata, name=role + '_index.json')
        if missing_entry and role == 'train':
            path = tmp_path / 'empty.json'
            path.write_text(json.dumps({'schema': 'native_spatial_index_v1', 'entries': {}}), encoding='utf8')
            index = artifact(path)
        manifest = tmp_path / (role + '.jsonl')
        manifest.write_text('{}', encoding='utf8')
        groups = tmp_path / (role + '_groups.json')
        groups.write_text(json.dumps({'pair': role + '-group'}), encoding='utf8')
        sources.append(SourceSpec(role, role=role, manifest=artifact(manifest), groups=artifact(groups), spatial_weights=index if spatial else None))
        samples[str(manifest)] = {'local_condition_v4': SimpleNamespace(metadata=({'driver_rgb_sha256': sha, 'target_canvas_id': canvas.fingerprint()},)),
            'target_canvas': canvas.metadata(), 'valid_image_mask': canvas.valid_mask(), 'target_token_hw': canvas.token_hw,
            'native_scene_kind': 'single'}
    class Dataset:
        def __init__(self, manifest, **kwargs):
            self.sample = samples[str(manifest)]
            self.rows = [{'pair_id': 'pair'}]
        def __len__(self): return 1
        def __getitem__(self, index): return deepcopy(self.sample)
    monkeypatch.setattr(data, 'NativeFullDatasetV5', Dataset)
    monkeypatch.setattr(data, 'sample_status', lambda *args: {'eligible': True, 'reasons': [], 'face_area': 0, 'hand_area': 0, 'auxiliary_faces': 0})
    route = RouteSpec('body', 'body', loss=LossSpec(spatial_body=1.))
    config = TrainingConfig(model_path=str(tmp_path), stage=StageSpec(routes=(route,)), sources=tuple(sources))
    return data, config


def test_data_plan_binds_index_and_component_files_in_resume_identity(tmp_path, monkeypatch):
    data, config = data_fixture(tmp_path, monkeypatch)
    plan = data.DataPlan(config)
    bound = plan.identity['sources']['train']['spatial_weights']
    assert bound['index'] == config.sources[0].spatial_weights
    assert len(bound['entries']) == 1
    assert len(plan.records) == 2 and plan.report()['all_rows_retained']
    assert plan.spatial[('train', 0)]['file']['sha256'] == file_sha(tmp_path / 'train.npz')
    sample = plan.sample('train', 0)
    assert 'spatial_weights_v5' in sample
    torch.testing.assert_close(sample['spatial_weights_v5']['valid'], sample['valid_image_mask'])


@pytest.mark.parametrize('case', ['missing_index', 'missing_entry', 'stale_after_preflight'])
def test_data_plan_fails_missing_or_changed_maps_without_filtering(tmp_path, monkeypatch, case):
    data, config = data_fixture(tmp_path, monkeypatch, spatial=case != 'missing_index', missing_entry=case == 'missing_entry')
    if case == 'stale_after_preflight':
        plan = data.DataPlan(config)
        with (tmp_path / 'train.npz').open('ab') as handle: handle.write(b'changed')
        with pytest.raises(ValueError, match='SHA'):
            plan.sample('train', 0)
    else:
        with pytest.raises(ValueError, match='[Ss]patial'):
            data.DataPlan(config)


@pytest.mark.parametrize('has_spatial', [True, False])
def test_backend_pair_preserves_spatial_payload_and_actual_cached_grid(tmp_path, has_spatial):
    from src.pose_control.qwen21_full.native_v5.training.backend import NativeBackend
    meta = {'source_rgb_sha256': 'a' * 64, 'driver_rgb_sha256': 'b' * 64,
            'source_canvas_id': 'c' * 64, 'target_canvas_id': 'd' * 64}
    sample = {'pair_metadata': {'pair_id': 'pair'}, 'local_condition_v4': SimpleNamespace(metadata=(meta,)),
              'source_token_hw': (2, 2), 'target_token_hw': (2, 4), 'valid_image_mask': torch.ones(1, 32, 64)}
    if has_spatial:
        sample['spatial_weights_v5'] = {'components': components((1, 32, 64)), 'valid': torch.ones(1, 32, 64),
                                        'target_token_hw': (2, 4)}
    path = tmp_path / 'pair.pt'
    torch.save({'source_latent': torch.zeros(1, 4, 64), 'target_latent': torch.zeros(1, 8, 64),
                'source_token_hw': (2, 2), 'target_token_hw': (2, 4)}, path)
    backend = NativeBackend.__new__(NativeBackend)
    backend.data = SimpleNamespace(sample=lambda *args: deepcopy(sample))
    backend.config = SimpleNamespace(cache_root=str(tmp_path), qwen_condition_profile='legacy_single')
    backend.cache = {'entries': {'train\0pair': {'file': artifact(path), **meta}}}
    backend.device = torch.device('cpu')
    returned, batch, pair = backend.pair('train', 0)
    if has_spatial:
        assert isinstance(batch['spatial_weights_v5'], dict)
        assert batch['spatial_weights_v5']['target_token_hw'] == pair['target_token_hw']
        assert batch['spatial_weights_v5']['components']['body'].shape == (1, 32, 64)
    else:
        assert 'spatial_weights_v5' not in returned and 'spatial_weights_v5' not in batch


def test_unselected_train_source_does_not_require_spatial_index(tmp_path, monkeypatch):
    data, config = data_fixture(tmp_path, monkeypatch)
    extra = replace(config.sources[0], source_id='unused', spatial_weights=None, weight=1.)
    selected = replace(config.sources[0], weight=1.)
    route = replace(config.stage.routes[0], source_ids=('train',))
    config = replace(config, sources=(selected, config.sources[1], extra), stage=replace(config.stage, routes=(route,)))
    plan = data.DataPlan(config)
    assert ('unused', 0) not in plan.spatial
    assert len(plan.records) == 3


def test_index_metadata_cannot_disagree_with_npz(tmp_path, monkeypatch):
    data, config = data_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'train_index.json'
    raw = json.loads(path.read_text(encoding='utf8'))
    next(iter(raw['entries'].values()))['metadata']['frame_id'] = 'wrong-frame'
    path.write_text(json.dumps(raw), encoding='utf8')
    source = replace(config.sources[0], spatial_weights=artifact(path))
    with pytest.raises(ValueError, match='metadata'):
        data.DataPlan(replace(config, sources=(source, config.sources[1])))
