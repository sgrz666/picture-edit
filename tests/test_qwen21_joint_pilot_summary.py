"""Saved JSON contracts matching the pilot eval/score outputs; CPU only."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def summary_module():
    path = ROOT / 'scripts/reports/summarize_qwen21_joint_pilot.py'
    assert path.is_file(), 'Missing pilot summary implementation'
    spec = importlib.util.spec_from_file_location('pilot_summary', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def experiment(tmp_path):
    tt = [{'pair_id': f't{i}', 'sequence_id': f'clip{i}', 'source_video_group': 'video_a' if i < 2 else 'video_b',
           'source_stem': 'a', 'target_stem': f'b{i}'} for i in range(3)]
    ip = [{'pair_id': 'i0', 'appearance': '007_2', 'source': {'stem': 'a'}, 'target': {'stem': 'b'}}]
    jsonl(tmp_path / 'data/val24_pairs.jsonl', tt)
    jsonl(tmp_path / 'data/iper12_pairs.jsonl', ip)
    (tmp_path / 'data/lock.json').write_text(json.dumps({'val_tiktok': 3, 'val_iper': 1,
        'monitor_tiktok_indices': [0], 'monitor_iper_indices': [0]}))
    return tmp_path, {'tiktok_v3': tt, 'iper': ip}


def write_eval(experiment, dataset, stage, variant, *, indices=None, identities=None, absent_scores=(), failures=()):
    root, manifests = experiment
    selected = list(range(len(manifests[dataset]))) if indices is None else indices
    run = root / 'evaluations' / dataset / f'{stage}_{variant}'
    run.mkdir(parents=True)
    config = {'dataset_type': dataset, 'stage': stage, 'variant': variant, 'indices': selected,
              'label': f'{stage}_{variant}', 'seed': 42, 'steps': 30, 'resolution': 512,
              'align_mode': 'letterbox' if dataset == 'tiktok_v3' else 'resize', 'pairs_sha256': dataset,
              'prompt': 'fixed'}
    (run / 'config.json').write_text(json.dumps(config))
    jsonl(run / 'selected_pairs.jsonl', [manifests[dataset][i] for i in selected])
    rows, scores = [], []
    for i in selected:
        status = 'failed' if i in failures else 'complete'
        rows.append({'pair_index': i, 'status': status, 'metadata': {'pair_id': manifests[dataset][i]['pair_id']},
                     'failure': 'generation failed' if i in failures else None, 'face_valid': True})
        if i in absent_scores:
            continue
        identity = (identities or {}).get(i, .5)
        scores.append({'pair_index': i, 'status': status, 'identity_A': identity if status == 'complete' else None,
            'source_face_count': 1, 'face_count': 1 if identity is not None and status == 'complete' else 0,
            'pose_detected': status == 'complete', 'pose_pck_002': .6 if status == 'complete' else None,
            'pose_pck_005': .8 if status == 'complete' else None})
    jsonl(run / 'per_sample.jsonl', rows)
    (run / 'scores.json').write_text(json.dumps({'per_sample': scores}))
    return run


def test_group_means_use_original_video_groups_and_actor_prefix(experiment):
    module = summary_module()
    write_eval(experiment, 'tiktok_v3', 'final', 'Qwen', identities={0: .9, 1: .7, 2: .1})
    write_eval(experiment, 'iper', 'final', 'Qwen', identities={0: .6})
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']
    assert stats['expected_n'] == 4
    metric = stats['by_dataset']['tiktok_v3']['metrics']['identity_A']
    assert metric['valid_n'] == 3
    assert metric['mean'] == pytest.approx((.9 + .7 + .1) / 3)
    assert metric['group_macro_mean'] == pytest.approx((.8 + .1) / 2)
    assert set(stats['by_dataset']['tiktok_v3']['groups']) == {'video_a', 'video_b'}
    assert set(stats['by_dataset']['iper']['groups']) == {'007'}
    assert (experiment[0] / 'report/summary.json').is_file()
    assert (experiment[0] / 'report/summary.md').is_file()


def test_missing_scores_and_failed_outputs_remain_in_expected_denominator(experiment):
    module = summary_module()
    write_eval(experiment, 'tiktok_v3', 'final', 'Qwen', absent_scores=(1,), failures=(2,))
    write_eval(experiment, 'iper', 'final', 'Qwen', identities={0: None})
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']
    assert stats['expected_n'] == 4
    assert stats['metrics']['identity_A']['valid_n'] == 1
    assert stats['metrics']['identity_A']['missing_n'] == 3
    assert stats['status_counts']['score_missing'] == 1
    assert stats['status_counts']['generation_failed'] == 1
    assert stats['detections']['output_no_face_n'] == 1
    assert len(stats['cases']) == 4
    assert any('scores' in error['message'] for error in result['errors'])


def test_final_paired_difference_uses_shared_valid_identity_intersection(experiment):
    module = summary_module()
    for dataset in ('tiktok_v3', 'iper'):
        write_eval(experiment, dataset, 'final', 'Qwen', identities={0: .5, 1: .4, 2: .2})
        write_eval(experiment, dataset, 'final', 'BFSBody', identities={0: .8, 1: None, 2: .1})
    result = module.summarize(experiment[0])
    paired = result['paired_final']['BFSBody']['metrics']['identity_A']
    assert paired['expected_n'] == 4
    assert paired['valid_pair_n'] == 3
    assert paired['mean_difference'] == pytest.approx((.3 - .1 + .3) / 3)
    assert len(paired['per_case_differences']) == 3
    assert result['paired_final']['BFSBody']['metrics']['pose_pck_002']['valid_pair_n'] == 4


def test_hotload_and_final_comparison_uses_only_locked_monitor_subset(experiment):
    module = summary_module()
    write_eval(experiment, 'tiktok_v3', 'hotload', 'BFSBody', indices=[0], identities={0: .2})
    write_eval(experiment, 'iper', 'hotload', 'BFSBody', indices=[0], identities={0: .3})
    write_eval(experiment, 'tiktok_v3', 'final', 'BFSBody', identities={0: .5, 1: .99, 2: .99})
    write_eval(experiment, 'iper', 'final', 'BFSBody', identities={0: .5})
    result = module.summarize(experiment[0])
    assert result['cohorts']['full36']['stages']['final']['BFSBody']['expected_n'] == 4
    assert result['cohorts']['fixed12']['stages']['hotload']['BFSBody']['expected_n'] == 2
    paired = result['hotload_vs_final']['BFSBody']['metrics']['identity_A']
    assert paired['expected_n'] == 2
    assert paired['valid_pair_n'] == 2
    assert paired['mean_difference'] == pytest.approx(.25)


def test_common_j_mid_is_reported_on_the_locked_monitor_subset(experiment):
    module = summary_module()
    write_eval(experiment, 'tiktok_v3', 'mid', 'J', indices=[0], identities={0: .7})
    write_eval(experiment, 'iper', 'mid', 'J', indices=[0], identities={0: .6})

    result = module.summarize(experiment[0])

    mid = result['cohorts']['fixed12']['stages']['mid']['J']
    assert mid['expected_n'] == 2
    assert mid['status_counts']['complete'] == 2
    assert mid['metrics']['identity_A']['valid_n'] == 2
    assert 'J' not in result['cohorts']['fixed12']['stages']['hotload']
    assert '| mid/J | 2 | 2 |' in (experiment[0] / 'report/summary.md').read_text(encoding='utf-8')


def test_formal_training_only_and_errors_are_retained(experiment):
    module = summary_module()
    root = experiment[0]
    jsonl(root / 'training/J/metrics.jsonl', [
        {'step': 1, 'loss': 1., 'face_active': True, 'auxiliary_requested': True, 'auxiliary_used': True, 'auxiliary_eligible_faces': 1},
        {'step': 2, 'loss': .9, 'face_active': False, 'auxiliary_requested': False, 'auxiliary_used': False, 'auxiliary_eligible_faces': 0},
        {'step': 3, 'error': 'nonfinite gradient'},
    ])
    jsonl(root / 'private_preflight_updates/training/J/metrics.jsonl', [{'step': 99, 'loss': 1., 'face_active': True}])
    with (root / 'training/J/metrics.jsonl').open('a') as stream:
        stream.write('{"step":4, broken\n')
    result = module.summarize(root)
    assert result['training']['J']['formal_steps'] == 2
    assert result['training']['J']['last_logged_step'] == 2
    assert result['training']['J']['face_active_steps'] == 1
    assert result['training']['J']['auxiliary_used_steps'] == 1
    assert result['training']['total_formal_steps'] == 2
    assert any('nonfinite gradient' in str(error) for error in result['errors'])
    assert any(error['kind'] == 'malformed_json' for error in result['errors'])


def test_absent_runs_have_null_metrics_without_identity_imputation(experiment):
    module = summary_module()
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['initial']['init']
    assert stats['expected_n'] == 4
    assert stats['metrics']['identity_A']['mean'] is None
    assert stats['metrics']['identity_A']['valid_n'] == 0
    assert stats['status_counts']['missing_run'] == 4
    markdown = (experiment[0] / 'report/summary.md').read_text(encoding='utf-8')
    assert '不填补' in markdown
    assert '不报告置信区间' in markdown


def test_partial_scoring_failure_preserves_independently_valid_identity(experiment):
    module = summary_module()
    run = write_eval(experiment, 'tiktok_v3', 'final', 'Qwen', identities={0: .8})
    payload = json.loads((run / 'scores.json').read_text())
    payload['per_sample'][0].update(score_failure='pose evaluator failed', pose_detected=None,
                                    pose_pck_002=None, pose_pck_005=None)
    (run / 'scores.json').write_text(json.dumps(payload))
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']['by_dataset']['tiktok_v3']
    assert stats['metrics']['identity_A']['valid_n'] == 3
    assert stats['metrics']['pose_pck_002']['valid_n'] == 2
    assert stats['status_counts']['score_failed'] == 1
    assert any('pose evaluator failed' in error['message'] for error in result['errors'])


def test_selected_iper_actor_conflict_is_explicit_and_not_regrouped(experiment):
    module = summary_module()
    run = write_eval(experiment, 'iper', 'final', 'Qwen')
    selected = json.loads((run / 'selected_pairs.jsonl').read_text())
    selected['appearance'] = '999_1'
    jsonl(run / 'selected_pairs.jsonl', [selected])
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']['by_dataset']['iper']
    assert stats['status_counts']['provenance_error'] == 1
    assert stats['metrics']['identity_A']['valid_n'] == 0
    assert set(stats['groups']) == {'007'}


def test_corrupt_score_schema_keeps_cases_and_records_error(experiment):
    module = summary_module()
    run = write_eval(experiment, 'tiktok_v3', 'final', 'Qwen')
    (run / 'scores.json').write_text('{"per_sample": null}')
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']['by_dataset']['tiktok_v3']
    assert stats['expected_n'] == 3
    assert stats['metrics']['identity_A']['valid_n'] == 0
    assert stats['status_counts']['score_missing'] == 3
    assert any(error['kind'] == 'invalid_record_schema' for error in result['errors'])


def test_same_pair_id_with_changed_target_frame_cannot_enter_paired_means(experiment):
    module = summary_module()
    run = write_eval(experiment, 'tiktok_v3', 'final', 'Qwen')
    selected = [json.loads(line) for line in (run / 'selected_pairs.jsonl').read_text().splitlines()]
    selected[0]['target_stem'] = 'wrong_frame'
    jsonl(run / 'selected_pairs.jsonl', selected)
    result = module.summarize(experiment[0])
    stats = result['cohorts']['full36']['stages']['final']['Qwen']['by_dataset']['tiktok_v3']
    assert stats['status_counts']['provenance_error'] == 1
    assert stats['metrics']['identity_A']['valid_n'] == 2


def test_archived_attempt_does_not_duplicate_corrected_fixed_cases(experiment):
    module = summary_module()
    root = experiment[0]
    old = write_eval(experiment, 'iper', 'initial', 'init', failures=(0,))
    archived = root / 'diagnostic_failures/init_initial_iper_missing_cache'
    archived.parent.mkdir(parents=True)
    old.rename(archived)
    (archived / 'rerun_reason.json').write_text(json.dumps({'category': 'infrastructure_condition_cache_path',
        'reason': 'rerun exact same locked pairs with correct feature cache'}))
    write_eval(experiment, 'iper', 'initial', 'init', identities={0: .6})
    result = module.summarize(root)
    stats = result['cohorts']['full36']['stages']['initial']['init']['by_dataset']['iper']
    assert stats['status_counts']['complete'] == 1
    assert stats['metrics']['identity_A']['valid_n'] == 1
    assert not any(error['kind'] == 'duplicate_run' for error in result['errors'])
    attempt = result['archived_attempts'][0]
    assert attempt['failed'] == 1
    assert attempt['reason']['category'] == 'infrastructure_condition_cache_path'
    assert attempt['records'][0]['failure'] == 'generation failed'
    assert attempt['records_link'] == '../diagnostic_failures/init_initial_iper_missing_cache/per_sample.jsonl'
    text = (root / 'report/summary.md').read_text(encoding='utf-8')
    assert 'infrastructure_condition_cache_path' in text
    assert '归档' in text
