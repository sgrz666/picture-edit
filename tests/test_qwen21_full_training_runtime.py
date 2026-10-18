"""CPU checks for multi-pair diagnostics and exact training continuation."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full import execution
from train_iper_qwen21_full import arguments, state_digest


class PairDataset:
    def __init__(self, count=6):
        self.samples = [
            {'appearance': f'{index:03d}_actor',
             'source': {'stem': f'source_{index}', 'rgb_1024': f'source_{index}.png'},
             'target': {'stem': f'target_{index}', 'rgb_1024': f'target_{index}.png'}}
            for index in range(count)
        ]

    def __len__(self):
        return len(self.samples)


@pytest.fixture
def fixed_loss(monkeypatch):
    calls = []

    def run_loss(pipe, adapter, cache, dataset, store, index, branch, *, seed,
                 backward, fixed_sigma, **kwargs):
        assert not backward
        assert not torch.is_grad_enabled()
        # Exercise global RNG restoration while noise itself uses its own seed.
        torch.rand(3)
        random.random()
        np.random.random()
        noise = torch.randn(16, generator=torch.Generator().manual_seed(seed))
        row = {'pair_index': index, 'appearance': dataset.samples[index]['appearance'],
               'source': dataset.samples[index]['source']['stem'],
               'target': dataset.samples[index]['target']['stem'],
               'sigma': fixed_sigma, 'flow': float(noise.square().mean()),
               'loss': float(noise.square().mean()) + index, 'noise': noise.tolist()}
        calls.append((index, seed, fixed_sigma, kwargs))
        return row

    monkeypatch.setattr(execution, 'run_loss', run_loss)
    return calls


def test_four_pair_diagnostics_reuse_32_independent_noise_draws_and_preserve_rng(fixed_loss):
    adapter = nn.Linear(1, 1)
    dataset = PairDataset()
    indices = [5, 1, 4, 0]
    execution.seed_all(73)
    rng_before = state_digest(execution.capture_rng())
    first = execution.diagnostic_probe(None, adapter, None, dataset, None, 'face', step=0,
                                       pair_indices=indices, face_flow_loss='roi')
    assert state_digest(execution.capture_rng()) == rng_before
    second = execution.diagnostic_probe(None, adapter, None, dataset, None, 'face', step=300,
                                        pair_indices=indices, face_flow_loss='roi')
    assert state_digest(execution.capture_rng()) == rng_before
    assert first['rows'] == second['rows']
    assert len(first['rows']) == 32
    assert first['pair_indices'] == indices
    assert first['fixed_noise_seeds'] == list(range(41042, 41074))
    assert len({tuple(row['noise']) for row in first['rows']}) == 32
    assert {row['appearance'] for row in first['rows']} == {
        dataset.samples[index]['appearance'] for index in indices}
    for offset, index in enumerate(indices):
        rows = first['rows'][offset * 8:(offset + 1) * 8]
        assert [row['pair_index'] for row in rows] == [index] * 8
        assert [row['fixed_sigma'] for row in rows] == first['sigmas']
        assert [row['noise_seed'] for row in rows] == list(range(41042 + offset * 8, 41050 + offset * 8))
    assert all(call[3] == {'face_flow_loss': 'roi'} for call in fixed_loss)


def test_default_diagnostic_indices_and_seeds_preserve_historical_single_pair(fixed_loss):
    args = arguments(['--branch', 'body'])
    assert args.diagnostic_pair_indices == [0]
    adapter = nn.Linear(1, 1)
    result = execution.diagnostic_probe(None, adapter, None, PairDataset(), None, 'body', step=20)
    assert result['pair_indices'] == [0]
    assert result['fixed_noise_seeds'] == list(range(41042, 41050))
    assert result['sigmas'] == [.05, .15, .25, .35, .50, .65, .80, .95]
    assert all(row['pair_index'] == 0 for row in result['rows'])


def test_before_after_probe_uses_all_explicit_pairs_and_preserves_single_row_api(fixed_loss):
    adapter = nn.Linear(1, 1)
    dataset = PairDataset()
    multi = execution.probe(None, adapter, None, dataset, None, 'face', pair_indices=[3, 1])
    assert multi['pair_indices'] == [3, 1]
    assert multi['fixed_noise_seeds'] == [24042, 24043]
    assert multi['loss'] == pytest.approx(np.mean([row['loss'] for row in multi['rows']]))
    single = execution.probe(None, adapter, None, dataset, None, 'face')
    assert single['pair_index'] == 0
    assert single['noise_seed'] == 24042
    assert single['sigma'] == .2


@pytest.mark.parametrize('indices', [[], [1, 1], [-1], [6], [0, 6], [True], [1.5]])
def test_probe_indices_reject_empty_duplicate_and_out_of_range_before_forward(indices, fixed_loss):
    with pytest.raises(ValueError, match='diagnostic-pair-indices'):
        execution.diagnostic_probe(None, nn.Linear(1, 1), None, PairDataset(), None,
                                   'face', step=0, pair_indices=indices)
    assert not fixed_loss


def _resume_configuration(tmp_path):
    train = tmp_path / 'train.jsonl'
    val = tmp_path / 'val.jsonl'
    train.write_text('train manifest\n')
    val.write_text('val manifest\n')
    args = arguments(['--branch', 'face', '--train-pairs', str(train), '--val-pairs', str(val)])
    saved = {'resolution': args.resolution, 'branch': args.branch,
             'args': {key: str(value) if hasattr(value, '__fspath__') else value
                      for key, value in vars(args).items()},
             'train_pairs_sha256': execution.file_sha256(train),
             'val_pairs_sha256': execution.file_sha256(val)}
    return args, saved


@pytest.mark.parametrize(('key', 'value'), [
    ('seed', 99), ('warmup_steps', 200), ('face_aux_alignment', 'five_point'),
    ('face_aux_every', 0), ('face_flow_loss', 'roi'), ('face_evaluator_range', 'normalized'),
    ('lr_scale', .1), ('diagnostic_probes', True), ('diagnostic_pair_indices', [0, 2]),
    ('require_active_gradients', True), ('dataset_type', 'iper'), ('align_mode', 'different'),
    ('features', '/different/features'), ('face_supervision', '/different/supervision.json'),
    ('cache', '/different/native_cache'),
])
def test_resume_rejects_changes_to_training_and_diagnostic_configuration(tmp_path, key, value):
    args, saved = _resume_configuration(tmp_path)
    execution.validate_resume_configuration(saved, args)
    setattr(args, key, value)
    with pytest.raises(ValueError, match=key):
        execution.validate_resume_configuration(saved, args)


def test_resume_new_fields_use_historical_defaults_and_manifest_fingerprint_fallback(tmp_path):
    args, saved = _resume_configuration(tmp_path)
    for key in ('diagnostic_pair_indices', 'diagnostic_probes', 'require_active_gradients',
                'face_aux_alignment', 'face_flow_loss', 'face_evaluator_range', 'lr_scale'):
        saved['args'].pop(key)
    previous_manifest = {key: saved.pop(key) for key in ('train_pairs_sha256', 'val_pairs_sha256')}
    execution.validate_resume_configuration(saved, args, previous_manifest)
    args.val_pairs.write_text('changed manifest with unchanged path\n')
    with pytest.raises(ValueError, match='fingerprint.*val_pairs'):
        execution.validate_resume_configuration(saved, args, previous_manifest)


def test_resume_allows_extending_step_limit_but_requires_dataset_fingerprints(tmp_path):
    args, saved = _resume_configuration(tmp_path)
    args.max_steps = 300
    args.save_steps = [0, 30, 300]
    execution.validate_resume_configuration(saved, args)
    saved.pop('train_pairs_sha256')
    with pytest.raises(ValueError, match='fingerprint.*train_pairs'):
        execution.validate_resume_configuration(saved, args)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_bridges = nn.ModuleDict({'face': nn.Linear(2, 2), 'geometry': nn.Linear(2, 2)})
        self.legacy = nn.Module()
        self.legacy.control_interface = nn.Module()
        self.legacy.control_interface.face_adapter = nn.Linear(2, 2)
        self.unused = nn.Linear(2, 2)
        self.loading_report = {'test_model': True}

    def forward(self, value):
        return self.output_bridges['face'](self.legacy.control_interface.face_adapter(value))


def test_face_warmup_updates_only_output_bridge_then_unfreezes_face_core():
    adapter = TinyAdapter()
    groups, _ = execution.parameter_groups(adapter, 'face')
    optimizer = torch.optim.AdamW(groups)
    frozen_before = execution.frozen_digest(adapter, 'face')
    execution.set_training_phase(adapter, 'face', warmup=True)
    assert {name for name, parameter in adapter.named_parameters() if parameter.requires_grad} == {
        'output_bridges.face.weight', 'output_bridges.face.bias'}
    adapter(torch.ones(1, 2)).square().mean().backward()
    execution.assert_frozen_gradients(adapter)
    optimizer.step()
    execution.assert_fp32_training_state(adapter, optimizer)
    assert execution.frozen_digest(adapter, 'face') == frozen_before
    optimizer.zero_grad(set_to_none=True)
    execution.set_training_phase(adapter, 'face', warmup=False)
    adapter(torch.ones(1, 2)).square().mean().backward()
    assert adapter.legacy.control_interface.face_adapter.weight.grad.abs().sum() > 0
    execution.assert_frozen_gradients(adapter)
    assert not adapter.output_bridges['geometry'].weight.requires_grad
    assert not adapter.unused.weight.requires_grad


def test_frozen_gradient_and_adam_precision_checks_reject_invalid_state():
    adapter = TinyAdapter()
    execution.set_training_phase(adapter, 'face', warmup=True)
    optimizer = torch.optim.AdamW(execution.parameter_groups(adapter, 'face')[0])
    adapter.unused.weight.grad = torch.zeros_like(adapter.unused.weight)
    with pytest.raises(AssertionError, match='Frozen parameters'):
        execution.assert_frozen_gradients(adapter)
    adapter.unused.weight.grad = None
    adapter(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    state = next(iter(optimizer.state.values()))
    state['exp_avg'] = state['exp_avg'].bfloat16()
    with pytest.raises(AssertionError, match='Adam state'):
        execution.assert_fp32_training_state(adapter, optimizer)


def test_init_checkpoint_loads_weights_into_fresh_step_zero_without_optimizer_state(tmp_path):
    args, _ = _resume_configuration(tmp_path)
    source = TinyAdapter()
    source_optimizer = torch.optim.AdamW(source.parameters())
    source(torch.ones(1, 2)).square().mean().backward()
    source_optimizer.step()
    source_path = tmp_path / 'source' / 'checkpoint_last.pt'
    execution.save_checkpoint(source_path, source, source_optimizer, step=200, branch='body',
                              args=args, sample_order=[0, 1])
    manifest_path = source_path.parent / 'run_manifest.json'
    execution.json_write(manifest_path, {'name': 'source body run'})
    destination = TinyAdapter()
    loaded = execution.load_adapter_checkpoint(destination, source_path)
    destination.initialization_provenance = {
        'mode': 'init_checkpoint', 'optimizer_state_restored': False,
        **execution.checkpoint_provenance(source_path, loaded),
    }
    fresh_optimizer = torch.optim.AdamW(execution.parameter_groups(destination, 'face')[0])
    assert not fresh_optimizer.state
    assert loaded['optimizer_state_dict']['state']
    for name, value in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)
    step_zero = tmp_path / 'new_run' / 'checkpoint_step_000000.pt'
    execution.save_checkpoint(step_zero, destination, fresh_optimizer, step=0, branch='face',
                              args=args, sample_order=[])
    saved = torch.load(step_zero, weights_only=False)
    assert saved['step'] == 0
    assert saved['sample_order'] == []
    assert saved['optimizer_state_dict']['state'] == {}
    provenance = saved['initialization_provenance']
    assert not provenance['optimizer_state_restored']
    assert provenance['checkpoint_step'] == 200
    assert provenance['checkpoint_sha256'] == execution.file_sha256(source_path)
    assert provenance['source_run_manifest']['sha256'] == execution.file_sha256(manifest_path)
    assert provenance['source_run_manifest']['content'] == {'name': 'source body run'}
    assert saved['args']['diagnostic_pair_indices'] == [0]


def test_restore_rng_reproduces_next_cpu_draws_and_accepts_historical_checkpoint(tmp_path):
    execution.seed_all(29)
    saved = execution.capture_rng()
    expected = (torch.rand(4), random.random(), np.random.random(4))
    execution.seed_all(1)
    execution.restore_rng(saved)
    torch.testing.assert_close(torch.rand(4), expected[0], rtol=0, atol=0)
    assert random.random() == expected[1]
    np.testing.assert_array_equal(np.random.random(4), expected[2])
    adapter = TinyAdapter()
    checkpoint = tmp_path / 'old.pt'
    torch.save({'architecture': execution.ARCHITECTURE,
                'adapter_state_dict': copy.deepcopy(adapter.state_dict())}, checkpoint)
    assert 'initialization_provenance' not in execution.load_adapter_checkpoint(TinyAdapter(), checkpoint)
