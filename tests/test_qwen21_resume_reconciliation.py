"""CPU continuation checks for interrupted, unsaved optimizer update tails."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full import execution
from train_iper_qwen21_full import arguments, state_digest


def _rows(count):
    return [{'step': step, 'pair_index': step * 5 % 16, 'elapsed_seconds': step * .25,
             'grad_norm': 1., 'note': '原始记录'} for step in range(1, count + 1)]


def _raw(rows):
    # Mixed whitespace/newlines make byte preservation stricter than rewriting
    # parsed JSON. Real metrics are ordinary LF JSONL.
    return b''.join((json.dumps(row, ensure_ascii=False) + ('\r\n' if i % 2 else '\n')).encode()
                    for i, row in enumerate(rows))


def _saved(rows, step):
    return {'step': step, 'sample_order': [row['pair_index'] for row in rows[:step]]}


def _reconcile(path, saved):
    return execution.reconcile_resume_metrics(path, saved, checkpoint_path=path.parent / 'checkpoint_last.pt',
                                              checkpoint_sha256='locked-checkpoint-sha')


def test_240_rows_rollback_to_21_preserves_raw_evidence_and_attempt_accounting(tmp_path):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    original = _raw(rows)
    path.write_bytes(original)
    execution.seed_all(63)
    rng = state_digest(execution.capture_rng())
    ledger = _reconcile(path, _saved(rows, 21))
    assert state_digest(execution.capture_rng()) == rng
    assert path.read_bytes() == _raw(rows[:21])
    assert ledger['discarded_update_attempts'] == 219
    assert ledger['discarded_attempt_seconds'] == 219 * .25
    assert ledger['last_resume']['original_last_step'] == 240
    assert ledger['last_resume']['retained_last_step'] == 21
    assert ledger['last_resume']['rolled_back_update_attempts'] == 219
    event = ledger['events'][0]
    assert event['status'] == 'completed'
    assert event['checkpoint_sha256'] == 'locked-checkpoint-sha'
    archive = tmp_path / event['archive_path']
    assert archive.read_bytes() == b''.join(original.splitlines(keepends=True)[21:])
    assert path.read_bytes() + archive.read_bytes() == original
    assert event['archive_sha256'] == execution.file_sha256(archive)
    assert sum(event['discarded_training_pair_uses'].values()) == 219
    assert ledger == json.loads((tmp_path / 'resume_reconciliation.json').read_text())
    report = execution.training_attempt_accounting(path, step=21, sample_order=_saved(rows, 21)['sample_order'],
                                                   seconds=21 * .25)
    assert report['optimizer_steps'] == report['training_pair_uses'] == 21
    assert report['nonzero_gradient_steps'] == 21
    assert report['total_update_attempts'] == 240
    assert report['training_seconds_including_discarded_attempts'] == 240 * .25


@pytest.mark.parametrize('final_step', [250, 500, 1000, 2000])
def test_successful_stage_resumes_keep_history_and_final_step_limit(tmp_path, final_step):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    path.write_bytes(_raw(rows))
    first = _reconcile(path, _saved(rows, 21))
    final_rows = _rows(final_step)
    with path.open('ab') as stream:
        stream.write(_raw(final_rows[21:]))
    before = path.read_bytes()
    completed = _reconcile(path, _saved(final_rows, final_step))
    assert path.read_bytes() == before
    assert completed['events'] == first['events']
    assert completed['discarded_update_attempts'] == 219
    assert completed['last_resume']['rolled_back_update_attempts'] == 0
    assert len(list(tmp_path.glob('train_metrics.discarded*.jsonl'))) == 1
    report = execution.training_attempt_accounting(
        path, step=final_step, sample_order=_saved(final_rows, final_step)['sample_order'], seconds=final_step * .25)
    assert report['optimizer_steps'] == report['training_pair_uses'] == final_step
    assert report['total_update_attempts'] == final_step + 219
    assert report['repeated_training_pair_uses'] == final_step - 16


@pytest.mark.parametrize('problem', [
    'missing_step', 'duplicate_committed', 'duplicate_tail', 'prefix_pair',
    'short_order', 'missing_committed', 'bad_json', 'blank_line', 'bool_step', 'missing_newline',
])
def test_invalid_history_fails_before_archive_or_log_mutation(tmp_path, problem):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    saved = _saved(rows, 21)
    if problem == 'missing_step':
        del rows[9]
    elif problem == 'duplicate_committed':
        rows[10]['step'] = 10
    elif problem == 'duplicate_tail':
        rows.append(dict(rows[-1]))
    elif problem == 'prefix_pair':
        rows[20]['pair_index'] += 1
    elif problem == 'short_order':
        saved['sample_order'].pop()
    elif problem == 'missing_committed':
        rows = rows[:20]
    elif problem == 'bool_step':
        rows[0]['step'] = True
    elif problem == 'missing_newline':
        rows = rows[:21]
    original = _raw(rows)
    if problem == 'bad_json':
        original += b'{"step": 241'
    elif problem == 'blank_line':
        original += b'\n'
    elif problem == 'missing_newline':
        original = original.rstrip(b'\r\n')
    path.write_bytes(original)
    with pytest.raises(ValueError, match='Resume'):
        _reconcile(path, saved)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('train_metrics.discarded*.jsonl'))
    assert not (tmp_path / 'resume_reconciliation.json').exists()


def test_missing_committed_log_rejected_and_empty_step_zero_allowed(tmp_path):
    path = tmp_path / 'train_metrics.jsonl'
    with pytest.raises(ValueError, match='missing committed steps'):
        _reconcile(path, {'step': 1, 'sample_order': [0]})
    result = _reconcile(path, {'step': 0, 'sample_order': []})
    assert result['events'] == []
    assert not path.exists()


@pytest.mark.parametrize('interruption', ['before_replace', 'after_replace'])
def test_interrupted_reconciliation_completes_once_on_retry(tmp_path, monkeypatch, interruption):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    path.write_bytes(_raw(rows))
    original_replace = execution._atomic_replace_bytes

    def interrupted_replace(destination, data):
        if destination == path:
            if interruption == 'after_replace':
                original_replace(destination, data)
            raise RuntimeError('simulated interruption during metrics replacement')
        return original_replace(destination, data)

    monkeypatch.setattr(execution, '_atomic_replace_bytes', interrupted_replace)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        _reconcile(path, _saved(rows, 21))
    pending = json.loads((tmp_path / 'resume_reconciliation.json').read_text())
    assert pending['events'][0]['status'] == 'pending'
    monkeypatch.setattr(execution, '_atomic_replace_bytes', original_replace)
    completed = _reconcile(path, _saved(rows, 21))
    assert completed['events'][0]['status'] == 'completed'
    assert completed['discarded_update_attempts'] == 219
    assert len(completed['events']) == 1
    assert len(list(tmp_path.glob('train_metrics.discarded*.jsonl'))) == 1
    assert path.read_bytes() == _raw(rows[:21])
    repeated = _reconcile(path, _saved(rows, 21))
    assert repeated['events'] == completed['events']
    assert repeated['discarded_update_attempts'] == 219


def test_second_interruption_accumulates_discarded_consumption_without_duplicates(tmp_path):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    path.write_bytes(_raw(rows))
    first = _reconcile(path, _saved(rows, 21))
    # A later process reaches 480, but its newest committed checkpoint is 250.
    later = _rows(480)
    with path.open('ab') as stream:
        stream.write(_raw(later[21:]))
    second = _reconcile(path, _saved(later, 250))
    assert second['events'][0] == first['events'][0]
    assert len(second['events']) == 2
    assert len(set(second['discarded_archive_paths'])) == 2
    assert second['discarded_update_attempts'] == 219 + 230
    assert second['discarded_attempt_seconds'] == (219 + 230) * .25


def test_tampered_archive_and_uncommitted_summary_are_rejected(tmp_path):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(240)
    path.write_bytes(_raw(rows))
    with pytest.raises(ValueError, match='uncommitted'):
        execution.training_attempt_accounting(path, step=21, sample_order=_saved(rows, 21)['sample_order'], seconds=10)
    ledger = _reconcile(path, _saved(rows, 21))
    (tmp_path / ledger['events'][0]['archive_path']).write_bytes(b'tampered evidence\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='archive is missing or changed'):
        _reconcile(path, _saved(rows, 21))
    assert path.read_bytes() == before


def test_historical_missing_time_is_reported_unknown_not_zero(tmp_path):
    path = tmp_path / 'train_metrics.jsonl'
    rows = _rows(30)
    del rows[-1]['elapsed_seconds']
    path.write_bytes(_raw(rows))
    ledger = _reconcile(path, _saved(rows, 21))
    assert ledger['discarded_update_attempts'] == 9
    assert ledger['discarded_attempt_seconds'] is None
    report = execution.training_attempt_accounting(path, step=21, sample_order=_saved(rows, 21)['sample_order'], seconds=5)
    assert report['total_update_attempts'] == 30
    assert report['training_seconds_including_discarded_attempts'] is None


def test_cpu_adam_rng_resume_21_after_240_attempts_matches_uninterrupted_250(tmp_path):
    """Exercise actual checkpoint, Adam moments, dropout, and all three RNGs."""
    train_pairs = tmp_path / 'train.jsonl'
    val_pairs = tmp_path / 'val.jsonl'
    train_pairs.write_text('locked train manifest\n')
    val_pairs.write_text('locked val manifest\n')
    args = arguments(['--branch', 'body', '--train-pairs', str(train_pairs), '--val-pairs', str(val_pairs)])
    checkpoint_path = tmp_path / 'checkpoint_last.pt'
    metrics_path = tmp_path / 'train_metrics.jsonl'

    def new_training_state():
        adapter = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 2)).train()
        adapter.loading_report = {'cpu_reference': True}
        return adapter, torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    def update(adapter, optimizer, step):
        epoch, offset = divmod(step, 16)
        index = int(torch.randperm(16, generator=torch.Generator().manual_seed(42 + epoch))[offset])
        value = torch.randn(2, 3) + index / 16
        target = torch.randn(2, 2) + random.random() + np.random.random()
        optimizer.zero_grad(set_to_none=True)
        loss = (adapter(value) - target).square().mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
        optimizer.step()
        return {'step': step + 1, 'pair_index': index, 'loss': float(loss.detach()),
                'elapsed_seconds': (step + 1) * .25, 'grad_norm': norm}

    execution.seed_all(63)
    continuous, continuous_optimizer = new_training_state()
    baseline_rows = [update(continuous, continuous_optimizer, step) for step in range(250)]
    expected_model = state_digest(continuous.state_dict())
    expected_optimizer = state_digest(continuous_optimizer.state_dict())
    expected_rng = state_digest(execution.capture_rng())

    execution.seed_all(63)
    interrupted, interrupted_optimizer = new_training_state()
    order = []
    interrupted_rows = []
    for step in range(240):
        row = update(interrupted, interrupted_optimizer, step)
        interrupted_rows.append(row)
        order.append(row['pair_index'])
        if step == 20:
            execution.save_checkpoint(checkpoint_path, interrupted, interrupted_optimizer, step=21,
                                      branch='body', args=args, sample_order=order)
    original = _raw(interrupted_rows)
    metrics_path.write_bytes(original)
    restored_adapter, restored_optimizer = new_training_state()
    saved = execution.load_adapter_checkpoint(restored_adapter, checkpoint_path)
    execution.validate_resume_configuration(saved, args)
    ledger = execution.reconcile_resume_metrics(metrics_path, saved, checkpoint_path=checkpoint_path,
                                               checkpoint_sha256=execution.file_sha256(checkpoint_path))
    restored_optimizer.load_state_dict(saved['optimizer_state_dict'])
    execution.restore_rng(saved)
    final_order = list(saved['sample_order'])
    with metrics_path.open('ab') as stream:
        for step in range(21, 250):
            row = update(restored_adapter, restored_optimizer, step)
            final_order.append(row['pair_index'])
            stream.write(_raw([row]))
    assert state_digest(restored_adapter.state_dict()) == expected_model
    assert state_digest(restored_optimizer.state_dict()) == expected_optimizer
    assert state_digest(execution.capture_rng()) == expected_rng
    final_rows = [json.loads(line) for line in metrics_path.read_bytes().splitlines()]
    assert final_rows == baseline_rows
    assert [row['step'] for row in final_rows] == list(range(1, 251))
    archived = (tmp_path / ledger['events'][0]['archive_path']).read_bytes()
    assert _raw(interrupted_rows[:21]) + archived == original
    report = execution.training_attempt_accounting(metrics_path, step=250, sample_order=final_order, seconds=62.5)
    assert report['training_pair_uses'] == report['optimizer_steps'] == report['nonzero_gradient_steps'] == 250
    assert report['discarded_update_attempts'] == 219
    assert report['total_update_attempts'] == 469
    assert report['training_seconds_including_discarded_attempts'] == 117.25
