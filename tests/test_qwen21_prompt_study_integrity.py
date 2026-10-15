from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run.run_qwen21_pei2i_study as runner
from scripts.reports.build_qwen21_pei2i_study_report import build_study_report
from src.pose_control.qwen21_full.prompt_study.results import aggregate_pair_metrics, package_export


@pytest.mark.parametrize('phase', runner.PHASES)
def test_dry_run_is_read_only_for_every_phase(tmp_path, monkeypatch, phase):
    output = tmp_path / 'absent'
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(output), '--phase', phase, '--dry-run'])
    monkeypatch.setattr(runner, 'invoke', lambda *a, **k: pytest.fail('dry run dispatched'))
    runner.main()
    assert not output.exists()


def test_dry_run_does_not_import_torch(tmp_path):
    script = 'import sys; import scripts.run.run_qwen21_pei2i_study as r; r.main(); assert "torch" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', script, '--run', str(tmp_path), '--phase', 'initial', '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('argument,value', [('--max-updates', '256'), ('--eval-every', '256')])
def test_runner_rejects_budget_changes(tmp_path, monkeypatch, argument, value):
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(tmp_path), '--phase', 'train', argument, value, '--dry-run'])
    with pytest.raises(SystemExit):
        runner.main()


def test_initial_refuses_missing_selfcheck_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(tmp_path), '--phase', 'initial'])
    monkeypatch.setattr(runner, 'invoke', lambda *a, **k: pytest.fail('dispatched without prerequisite'))
    with pytest.raises(RuntimeError, match='selfcheck'):
        runner.main()


def test_train_preserves_later_checkpoint_when_earlier_boundary_is_missing(tmp_path, monkeypatch):
    directory = tmp_path / 'training/qwen_p0'
    directory.mkdir(parents=True)
    (directory / 'checkpoint_step_000000.pt').write_bytes(b'initial')
    (directory / 'checkpoint_last.pt').write_bytes(b'committed512')
    monkeypatch.setattr(runner, 'require_preflight', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'locked_assets', lambda *a: {'model': {'path': 'model'}})
    monkeypatch.setattr(runner, 'checkpoint_step', lambda p, *a: 512 if p.name == 'checkpoint_last.pt' else 0)
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(tmp_path), '--phase', 'train', '--resume', '--arms', 'qwen_p0'])
    dispatched = []
    monkeypatch.setattr(runner, 'invoke', lambda *a, **k: dispatched.append(a))
    monkeypatch.setattr(runner, 'dispatch_eval', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='Missing scheduled checkpoint'):
        runner.main()
    assert dispatched == []
    assert (directory / 'checkpoint_last.pt').read_bytes() == b'committed512'


def test_preflight_rejects_resume_result_bound_to_other_lock(tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/lock.json').write_text('{}')
    (tmp_path / 'preflight').mkdir()
    (tmp_path / 'preflight/selfcheck.json').write_text('{"passed":true}')
    for arm in runner.ARMS:
        directory = tmp_path / f'preflight/{arm}'
        directory.mkdir()
        (directory / 'gradient_only.json').write_text(json.dumps({'passed': True, 'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True, 'joint_signature': {}}))
    (tmp_path / 'preflight/resume_check').mkdir()
    (tmp_path / 'preflight/resume_check/result.json').write_text(json.dumps({'mandatory_exact_gate': {'passed': True}, 'private_optimizer_updates': 10, 'study_lock_sha256': 'f' * 64, 'joint_signature': {}}))
    monkeypatch.setattr(runner, 'run_selfcheck', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'validate_gradient_artifact', lambda *a, **k: None, raising=False)
    with pytest.raises(RuntimeError, match='resume'):
        runner.require_preflight(tmp_path)


def test_report_rejects_missing_png_and_unproven_prompts(tmp_path):
    edir = tmp_path / 'evaluation/qwen_pg/step_000000/val/seed_42'
    edir.mkdir(parents=True)
    (edir / 'config.json').write_text(json.dumps({'arm': 'qwen_pg', 'study_step': 0, 'study_split': 'val', 'seed': 42}))
    (edir / 'per_sample.jsonl').write_text(json.dumps({'pair_index': 0, 'pair_id': 'p0', 'status': 'complete', 'image': 'missing.png'}) + '\n')
    prompt_dir = tmp_path / 'prompts/pg'
    prompt_dir.mkdir(parents=True)
    (prompt_dir / 'val.jsonl').write_text(json.dumps({'pair_id': 'p0', 'protocol': 'pg', 'status': 'complete', 'compiled_prompt': 'fake'}) + '\n')
    report = build_study_report(tmp_path)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0
    assert summary['pe_success'] == 0
    assert summary['provenance_errors_count'] > 0


def test_report_does_not_count_summary_only_training_updates(tmp_path):
    for arm in runner.ARMS:
        directory = tmp_path / f'training/{arm}'
        directory.mkdir(parents=True)
        (directory / 'summary.json').write_text('{"steps":1024,"completed":true}')
    report = build_study_report(tmp_path)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['train_updates'] == 0
    assert not summary['is_completed']


def test_aggregation_keeps_expected_missing_pairs_and_video_groups():
    rows = [{'split': 'val', 'pair_id': 'a', 'source_video_group': 'video', 'group': 'appearance', 'seed': 42, 'metrics': {'m': 2}}]
    result = aggregate_pair_metrics(rows, ('m',), expected_pairs=[{'split': 'val', 'pair_id': 'a', 'source_video_group': 'video'}, {'split': 'val', 'pair_id': 'b', 'source_video_group': 'video'}])
    assert result['expected_n'] == 2
    assert result['valid_counts']['m'] == 1
    assert result['pairs'][1]['metrics']['m'] is None
    assert set(result['group_means']) == {'video'}


def test_aggregation_requires_scheduled_seed_coverage():
    rows = [{'split': 'val', 'pair_id': 'a', 'seed': 42, 'metrics': {'m': 2}}]
    result = aggregate_pair_metrics(rows, ('m',), expected_seeds=(42, 43))
    assert result['valid_counts']['m'] == 0
    assert result['pairs'][0]['valid_seed_counts']['m'] == 1
    assert result['pairs'][0]['metrics']['m'] is None


def test_aggregation_averages_finite_metrics_after_all_seed_attempts_exist():
    rows = [{'split': 'val', 'pair_id': 'a', 'seed': 42, 'metrics': {'m': 2}}, {'split': 'val', 'pair_id': 'a', 'seed': 43, 'metrics': {'m': None}}]
    result = aggregate_pair_metrics(rows, ('m',), expected_seeds=(42, 43))
    assert result['pairs'][0]['metrics']['m'] == 2
    assert result['pairs'][0]['valid_seed_counts']['m'] == 1
    assert result['pairs'][0]['invalid_seed_counts']['m'] == 1
    assert result['pairs'][0]['missing_seed_count'] == 0


def test_export_rejects_report_link_to_excluded_checkpoint(tmp_path):
    (tmp_path / 'training').mkdir()
    (tmp_path / 'training/model.pt').write_bytes(b'weights')
    (tmp_path / 'report').mkdir()
    (tmp_path / 'report/index.html').write_text('<a href="../training/model.pt">weights</a>')
    with pytest.raises(RuntimeError, match='links'):
        package_export(tmp_path)


def test_export_rejects_symlink_to_internal_cache(tmp_path):
    (tmp_path / 'cache').mkdir()
    source = tmp_path / 'cache/secret.json'
    source.write_text('{}')
    (tmp_path / 'data').mkdir()
    try:
        (tmp_path / 'data/alias.json').symlink_to(source)
    except OSError:
        pytest.skip('symlink permission unavailable')
    with pytest.raises(ValueError, match='symlink|cache'):
        package_export(tmp_path)
