"""Synthetic CPU checkpoint/log checks for the bounded pilot resume gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run/run_qwen21_joint_pilot.py'
    spec = importlib.util.spec_from_file_location('joint_pilot_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, value):
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


def test_preprocessing_hold_prevents_phase_dispatch(tmp_path, monkeypatch):
    import sys
    module = _runner()
    (tmp_path / 'data').mkdir()
    _json(tmp_path / 'data/preprocessing_review_required.json', {
        'reason': 'User requires original pixel dimensions, not a fixed 512 square.'})
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(tmp_path), '--phase', 'Qwen-preflight'])
    monkeypatch.setattr(module, 'train', lambda *args, **kwargs: pytest.fail('Held pilot dispatched training'))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def _rows(delta=0.):
    return [dict(step=i + 1, pair_index=i, pair_id=f'pair{i}', noise_seed=42 + i, sigma=.2 + .1 * i,
                 face_active=i < 2, auxiliary_requested=i == 0, auxiliary_used=i == 0,
                 auxiliary_eligible_faces=int(i == 0), source_stem='source', target_stem=f'target{i}',
                 loss=1. + (delta if i >= 2 else 0.), body_flow=.5 + (delta if i >= 2 else 0.),
                 face_flow=.25, identity=0., perceptual=0., elapsed_seconds=float(i), peak_cuda_gib=1.)
            for i in range(4)]


def _save_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _state(step=4, delta=0.):
    return {
        'architecture': 'joint', 'model_revision': 'revision', 'branch': 'joint', 'resolution': 512,
        'step': step, 'sample_order': list(range(step)), 'joint_signature': {'schema': 'pilot', 'seed': 42},
        'joint_initialization_sha256': 'initial',
        'adapter_state_dict': {'bridge': torch.tensor([0., 2., 3.], dtype=torch.float64) + delta},
        'optimizer_state_dict': {'state': {0: {'step': torch.tensor(float(step)),
            'exp_avg': torch.tensor([0., .2, .3], dtype=torch.float64) + delta,
            'exp_avg_sq': torch.tensor([.01, .02, .03], dtype=torch.float64)}},
            'param_groups': [{'params': [0], 'lr': .001, 'betas': (.9, .999)}]},
        'torch_rng_state': torch.tensor([1, 2, 3], dtype=torch.uint8),
        'cuda_rng_state': [torch.tensor([4, 5], dtype=torch.uint8)],
        'python_rng_state': (3, (1, 2, 3), None),
        'numpy_rng_state': ('MT19937', np.array([6, 7], dtype=np.uint32), 1, 0, 0.),
    }


def _direct(run):
    direct = run / 'preflight/resume_check/direct'
    direct.mkdir(parents=True)
    torch.save(_state(2), direct / 'checkpoint_step_000002.pt')
    torch.save(_state(), direct / 'checkpoint_last.pt')
    _save_rows(direct / 'metrics.jsonl', _rows())
    _json(direct / 'run_manifest.json', {'joint_signature': _state()['joint_signature'],
        'controls': {'body': 1, 'face': 1, 'hand': 0, 'interaction': 0}, 'loss': {'body_flow': 1},
        'optimizer_parameters': {'body': ['bridge'], 'face': ['face']}, 'parameter_dtype': 'float32'})
    _json(direct / 'summary.json', {'completed': True, 'steps': 4,
        'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True})
    return direct


def _finish(folder, delta=0.):
    source_hash = hashlib.sha256((folder / 'checkpoint_step_000002.pt').read_bytes()).hexdigest()
    # Match the trainer's atomic replacement, preserving shared step2 hardlinks.
    temporary = folder / 'checkpoint_last.tmp'
    torch.save(_state(delta=delta), temporary)
    temporary.replace(folder / 'checkpoint_last.pt')
    _save_rows(folder / 'metrics.jsonl', _rows(delta))
    _json(folder / 'resume_state_step_000002.json', {
        'adapter_restored_exact': True, 'optimizer_restored_exact': True, 'rng_restored_exact': True,
        'checkpoint_step': 2, 'checkpoint_sha256': source_hash, 'joint_signature': _state()['joint_signature']})
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    _json(folder / 'resume_manifest.json', manifest)
    _json(folder / 'summary.json', {'completed': True, 'steps': 4,
        'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True})


def _experiment(run, deltas=(0., 0., 0.)):
    module = _runner()
    _direct(run)
    assert hasattr(module, 'fork_resume_checkpoint'), 'Resume gate must fork the exact direct step2 checkpoint'
    module.fork_resume_checkpoint(run)
    for name, delta in zip(('resume', 'repeat1', 'repeat2'), deltas):
        _finish(run / 'preflight/resume_check' / name, delta)
    return module


def test_fork_preserves_exact_checkpoint_bytes_and_two_metric_rows(tmp_path):
    module = _runner()
    source = _direct(tmp_path)
    assert hasattr(module, 'fork_resume_checkpoint'), 'Missing shared-step2 fork'
    module.fork_resume_checkpoint(tmp_path)
    prefix = b''.join((source / 'metrics.jsonl').read_bytes().splitlines(keepends=True)[:2])
    for name in ('resume', 'repeat1', 'repeat2'):
        folder = source.parent / name
        assert (folder / 'checkpoint_last.pt').read_bytes() == (source / 'checkpoint_step_000002.pt').read_bytes()
        assert (folder / 'checkpoint_step_000002.pt').read_bytes() == (source / 'checkpoint_step_000002.pt').read_bytes()
        assert (folder / 'metrics.jsonl').read_bytes() == prefix
        assert (folder / 'run_manifest.json').read_bytes() == (source / 'run_manifest.json').read_bytes()


def test_resume_phase_runs_only_ten_updates_from_one_shared_fork(tmp_path, monkeypatch):
    module = _runner()
    calls = []

    def train(run, arm, steps, *, resume=False, private=None, **kwargs):
        calls.append((arm, steps, resume, private))
        if private == 'direct':
            _direct(run)
        else:
            folder = run / 'preflight/resume_check' / private
            assert len((folder / 'metrics.jsonl').read_text().splitlines()) == 2
            assert torch.load(folder / 'checkpoint_last.pt', weights_only=False)['step'] == 2
            _finish(folder)

    monkeypatch.setattr(module, 'train', train)
    assert hasattr(module, 'run_resume_check'), 'Missing bounded shared-fork orchestration'
    report = module.run_resume_check(tmp_path)
    assert calls == [('J', 4, False, 'direct'), ('J', 4, True, 'resume'),
                     ('J', 4, True, 'repeat1'), ('J', 4, True, 'repeat2')]
    assert report['passed'] and report['private_optimizer_updates'] == 10


def test_bit_exact_four_trajectories_pass_and_record_ten_actual_updates(tmp_path):
    report = _experiment(tmp_path).compare_resume(tmp_path)
    assert report['passed'] and not report['review_required']
    assert report['mandatory_exact_gate']['passed']
    assert report['exact_parameters_optimizer_and_losses']
    assert report['private_optimizer_updates'] == 10
    assert set(report['comparisons']) == {'direct_vs_resume', 'direct_vs_repeat1', 'direct_vs_repeat2', 'repeat1_vs_repeat2'}
    assert json.loads((tmp_path / 'preflight/resume_check/result.json').read_text()) == report


def test_numerical_drift_inside_independent_observed_envelope_is_reported(tmp_path):
    report = _experiment(tmp_path, (.05, .1, -.1)).compare_resume(tmp_path)
    assert report['passed'] and not report['review_required']
    assert not report['exact_parameters_optimizer_and_losses']
    stats = report['comparisons']['direct_vs_resume']['adapter']
    assert stats['max_abs'] == pytest.approx(.05)
    assert stats['mean_abs'] == pytest.approx(.05)
    assert stats['relative_l2'] > 0
    assert report['observed_envelope']['adapter']['max_abs'] == pytest.approx(.2)
    assert report['observed_envelope']['losses']['loss']['max_abs'] == pytest.approx(.2)
    assert report['numerical_diagnostics']['within_observed_envelope']
    assert not report['numerical_review_required']


def test_drift_outside_observed_envelope_requires_review_without_a_statistical_hard_failure(tmp_path):
    report = _experiment(tmp_path, (.4, .1, -.1)).compare_resume(tmp_path)
    assert report['passed'] and report['review_required'] and report['numerical_review_required']
    assert report['mandatory_exact_gate']['passed']
    assert not report['numerical_diagnostics']['within_observed_envelope']
    assert any(row['path'] == 'adapter.max_abs' for row in report['numerical_diagnostics']['outside_envelope'])


def test_zero_repeat_envelope_does_not_admit_a_fixed_arbitrary_tolerance(tmp_path):
    report = _experiment(tmp_path, (1e-18, 0., 0.)).compare_resume(tmp_path)
    assert report['passed'] and report['numerical_review_required']
    assert report['observed_envelope']['adapter']['max_abs'] == 0.


@pytest.mark.parametrize('damage', ['prefix', 'pair_id', 'noise_seed', 'sigma', 'length', 'signature',
    'sample_order', 'rng', 'adapter_restore', 'optimizer_restore', 'rng_restore', 'source_sha',
    'optimizer_metadata', 'optimizer_step', 'manifest', 'frozen_summary'])
def test_exact_metadata_or_restore_failures_cannot_be_excused_by_numeric_envelope(tmp_path, damage):
    module = _experiment(tmp_path, (0., .1, -.1))
    folder = tmp_path / 'preflight/resume_check/resume'
    if damage in ('prefix', 'pair_id', 'noise_seed', 'sigma', 'length'):
        rows = _rows()
        if damage == 'prefix':
            rows[0]['loss'] += .01
        elif damage == 'length':
            rows.pop()
        else:
            rows[2][damage] = 'wrong' if damage == 'pair_id' else 99
        _save_rows(folder / 'metrics.jsonl', rows)
    elif damage in ('signature', 'sample_order', 'rng', 'optimizer_metadata', 'optimizer_step'):
        state = _state()
        if damage == 'signature':
            state['joint_signature']['seed'] = 99
        elif damage == 'sample_order':
            state['sample_order'] = [0, 1, 3, 2]
        elif damage == 'rng':
            state['numpy_rng_state'][1][0] = 99
        elif damage == 'optimizer_step':
            state['optimizer_state_dict']['state'][0]['step'].add_(1)
        else:
            state['optimizer_state_dict']['param_groups'][0]['lr'] = .1
        torch.save(state, folder / 'checkpoint_last.pt')
    elif damage in ('adapter_restore', 'optimizer_restore', 'rng_restore', 'source_sha'):
        path = folder / 'resume_state_step_000002.json'
        proof = json.loads(path.read_text())
        proof['checkpoint_sha256' if damage == 'source_sha' else damage.removesuffix('_restore') + '_restored_exact'] = 'wrong' if damage == 'source_sha' else False
        _json(path, proof)
    elif damage == 'manifest':
        path = folder / 'resume_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['controls']['hand'] = 1
        _json(path, manifest)
    else:
        path = folder / 'summary.json'
        summary = json.loads(path.read_text())
        summary['frozen_transformer_unchanged'] = False
        _json(path, summary)
    report = module.compare_resume(tmp_path)
    assert not report['passed'] and report['review_required']
    assert not report['mandatory_exact_gate']['passed']
    assert report['mandatory_exact_gate']['errors']


@pytest.mark.parametrize('damage', ['tensor_shape', 'tensor_dtype', 'tensor_nan', 'loss_nan'])
def test_invalid_numeric_state_is_recorded_as_failure_without_nonfinite_json(tmp_path, damage):
    module = _experiment(tmp_path)
    folder = tmp_path / 'preflight/resume_check/resume'
    state = _state()
    if damage == 'tensor_shape':
        state['adapter_state_dict']['bridge'] = torch.ones(4)
    elif damage == 'tensor_dtype':
        state['adapter_state_dict']['bridge'] = state['adapter_state_dict']['bridge'].float()
    elif damage == 'tensor_nan':
        state['adapter_state_dict']['bridge'][0] = float('nan')
    else:
        rows = _rows()
        rows[3]['loss'] = float('nan')
        _save_rows(folder / 'metrics.jsonl', rows)
    torch.save(state, folder / 'checkpoint_last.pt')
    report = module.compare_resume(tmp_path)
    assert not report['passed']
    json.dumps(report, allow_nan=False)


def test_fork_refuses_existing_destination_before_overwriting_any_artifact(tmp_path):
    module = _runner()
    _direct(tmp_path)
    folder = tmp_path / 'preflight/resume_check/repeat2'
    folder.mkdir()
    (folder / 'do_not_overwrite').write_text('prior experiment')
    assert hasattr(module, 'fork_resume_checkpoint'), 'Missing shared-step2 fork'
    with pytest.raises((ValueError, FileExistsError)):
        module.fork_resume_checkpoint(tmp_path)
    assert not (folder.parent / 'resume').exists()
    assert (folder / 'do_not_overwrite').read_text() == 'prior experiment'


def test_both_evaluation_datasets_use_the_confirmed_complete_face_cache(tmp_path, monkeypatch):
    module = _runner()
    (tmp_path / 'data').mkdir()
    _json(tmp_path / 'data/lock.json', {})
    calls = []
    monkeypatch.setattr(module, 'invoke', lambda script, values: calls.append((script, list(values))))
    module.evaluate(tmp_path, 'init', Path('initial.pt'), 'initial')
    eval_calls = [values for script, values in calls if '--features' in values]
    assert len(eval_calls) == 2
    assert all(values[values.index('--features') + 1] == module.OLD / 'data/face_cache' for values in eval_calls)
