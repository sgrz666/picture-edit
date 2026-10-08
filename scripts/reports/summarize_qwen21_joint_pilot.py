#!/usr/bin/env python3
"""Summarize the actual joint-pilot logs and saved-image scores on locked cohorts.

Full36 initial/J/final and fixed12 hotload/mid stay separate. Paired estimates
use case intersections, not differences of unmatched averages. No missing
identity imputation, confidence intervals, significance tests or GPU use.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

METRICS = ('identity_A', 'pose_pck_002', 'pose_pck_005')
ARMS = ('Qwen', 'BFSBody', 'BFSHead')
DATA_FILES = {'tiktok_v3': 'val24_pairs.jsonl', 'iper': 'iper12_pairs.jsonl'}


def error(errors, kind, path, message, **details):
    errors.append({'kind': kind, 'path': str(path), 'message': message, **details})


def read_json(path, errors, default=None):
    if not path.is_file():
        error(errors, 'missing_file', path, f'Missing {path.name}')
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        error(errors, 'malformed_json', path, str(exc))
        return default


def read_jsonl(path, errors):
    if not path.is_file():
        error(errors, 'missing_file', path, f'Missing {path.name}')
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Record is not an object')
            rows.append(row)
        except ValueError as exc:
            error(errors, 'malformed_json', path, str(exc), line=number, raw_record=line)
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def group_for(dataset, row, errors, path):
    if dataset == 'tiktok_v3':
        group = row.get('source_video_group')
    else:
        appearance = row.get('source_appearance', row.get('appearance'))
        group = str(appearance).split('_')[0] if appearance else None
    if not group:
        error(errors, 'missing_group', path, 'Original video group / actor prefix is unavailable', pair_id=row.get('pair_id'))
    return group


def load_protocol(root, errors):
    lock = read_json(root / 'data/lock.json', errors, {})
    manifests, full, monitor = {}, [], []
    for dataset, filename in DATA_FILES.items():
        rows = read_jsonl(root / 'data' / filename, errors)
        manifests[dataset] = rows
        for index, row in enumerate(rows):
            full.append({'dataset': dataset, 'pair_index': index, 'pair_id': row.get('pair_id'),
                         'group': group_for(dataset, row, errors, root / 'data' / filename)})
        name = 'monitor_tiktok_indices' if dataset == 'tiktok_v3' else 'monitor_iper_indices'
        indices = lock.get(name)
        if not isinstance(indices, list):
            error(errors, 'missing_monitor_lock', root / 'data/lock.json', f'No locked {name}; fixed12 comparisons remain empty')
            continue
        if len(indices) != len(set(indices)) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(rows) for i in indices):
            error(errors, 'invalid_monitor_lock', root / 'data/lock.json', f'Invalid {name}', indices=indices)
            continue
        monitor.extend(case for case in full if case['dataset'] == dataset and case['pair_index'] in indices)
    return manifests, full, monitor, lock


def indexed(rows, errors, path):
    result = {}
    if not isinstance(rows, list):
        error(errors, 'invalid_record_schema', path, 'Expected a list of saved case records', payload=rows)
        return result
    for row in rows:
        if not isinstance(row, dict):
            error(errors, 'invalid_record_schema', path, 'Case record is not an object', record=row)
            continue
        index = row.get('pair_index')
        if not isinstance(index, int) or isinstance(index, bool):
            error(errors, 'invalid_pair_index', path, 'Record lacks integer pair_index', record=row)
            continue
        if index in result:
            error(errors, 'duplicate_record', path, 'Duplicate case cannot enter metric denominator', pair_index=index, record=row)
            result[index] = None
        else:
            result[index] = row
    return result


def discover_runs(root, errors):
    result, profiles, excluded = {}, {}, []
    for config_path in sorted(root.rglob('config.json')):
        directory = config_path.parent
        if directory.is_relative_to(root / 'diagnostic_failures'):
            continue
        if not (directory / 'per_sample.jsonl').is_file() and not (directory / 'per_sample.json').is_file():
            continue
        config = read_json(config_path, errors, {})
        if not isinstance(config, dict):
            error(errors, 'invalid_record_schema', config_path, 'Evaluation config must be an object', payload=config)
            continue
        dataset, stage, variant = (config.get(key) for key in ('dataset_type', 'stage', 'variant'))
        if dataset not in DATA_FILES or stage not in ('initial', 'common', 'hotload', 'mid', 'final'):
            continue
        if variant not in ('init', 'J', *ARMS):
            excluded.append({'path': directory.relative_to(root).as_posix(), 'stage': stage, 'variant': variant,
                             'reason': 'Explicit diagnostic column outside the primary comparison'})
            continue
        records_path = directory / 'per_sample.jsonl'
        rows = read_jsonl(records_path, errors) if records_path.is_file() else read_json(directory / 'per_sample.json', errors, [])
        records = indexed(rows, errors, records_path)
        scores_payload = read_json(directory / 'scores.json', errors, {})
        score_rows = scores_payload.get('per_sample', []) if isinstance(scores_payload, dict) else scores_payload
        scores = indexed(score_rows, errors, directory / 'scores.json')
        indices = config.get('indices')
        if not isinstance(indices, list):
            error(errors, 'missing_selection', config_path, 'Configuration has no locked indices; cannot align selected_pairs')
            indices = []
        selected_rows = read_jsonl(directory / 'selected_pairs.jsonl', errors)
        bad = len(selected_rows) != len(indices) or len(set(indices)) != len(indices)
        if bad:
            error(errors, 'selection_mismatch', directory, 'selected_pairs row count / unique indices differs from config', indices=indices, selected_rows=len(selected_rows))
        profile = {name: config.get(name) for name in ('pairs_sha256', 'seed', 'steps', 'resolution', 'align_mode', 'prompt')}
        if dataset in profiles and profiles[dataset] != profile:
            bad = True
            error(errors, 'sampling_mismatch', config_path, 'Fixed dataset or sampling configuration differs', profile=profile)
        profiles.setdefault(dataset, profile)
        run = {'directory': directory, 'config': config, 'records': records, 'scores': scores,
               'selected': dict(zip(indices, selected_rows)), 'indices': indices, 'invalid': bad}
        key = (dataset, stage, variant)
        if key in result:
            error(errors, 'duplicate_run', directory, 'Multiple evaluation runs occupy the same dataset/stage/variant', key=list(key))
            result[key] = None
        else:
            result[key] = run
        for index in set(records) | set(scores):
            if index not in indices:
                error(errors, 'unselected_record', directory, 'Result lies outside locked selection', pair_index=index,
                      generation_record=records.get(index), score_record=scores.get(index))
    return result, excluded


def archived_attempts(root, output, errors):
    attempts = []
    archive_root = root / 'diagnostic_failures'
    if not archive_root.is_dir():
        return attempts

    def relative(path):
        return quote(Path(os.path.relpath(path, output)).as_posix(), safe='/')

    for records_path in sorted(archive_root.rglob('per_sample.jsonl')):
        directory = records_path.parent
        records = read_jsonl(records_path, errors)
        reason_path = directory / 'rerun_reason.json'
        reason = read_json(reason_path, errors, {})
        audit_records = []
        for row in records:
            links = {}
            for name in ('image', 'source', 'target', 'normal', 'pose', 'comparison', 'face_comparison'):
                value = row.get(name)
                if value and not Path(value).is_absolute():
                    artifact = (directory / value).resolve()
                    if artifact.is_relative_to(directory.resolve()) and artifact.is_file():
                        links[name] = relative(artifact)
            audit_records.append({**row, 'artifact_links': links})
        attempts.append({'name': directory.name, 'records_n': len(records),
            'complete': sum(row.get('status') == 'complete' for row in records),
            'failed': sum(row.get('status') != 'complete' for row in records), 'reason': reason,
            'records_link': relative(records_path), 'config_link': relative(directory / 'config.json'),
            'scores_link': relative(directory / 'scores.json') if (directory / 'scores.json').is_file() else None,
            'reason_link': relative(reason_path) if reason_path.is_file() else None, 'records': audit_records})
    return attempts


def manifest_identity(row):
    """Compare actual A/B bindings even when a stale pair_id was retained."""
    return (row.get('source_appearance', row.get('appearance', row.get('sequence_id'))),
            row.get('target_appearance', row.get('appearance', row.get('sequence_id'))),
            row.get('source_stem', row.get('source', {}).get('stem')),
            row.get('target_stem', row.get('target', {}).get('stem')))


def case_result(case, stage, variant, runs, manifests, errors):
    row = {**case, 'stage': stage, 'variant': variant, 'status': 'missing_run', 'generation_status': None,
           'score_status': None, 'source_face_count': None, 'face_count': None, 'pose_detected': None,
           'failure': None, 'metrics': {name: None for name in METRICS}}
    key = (case['dataset'], stage, variant)
    if key not in runs:
        return row
    run = runs[key]
    if run is None:
        row['status'] = 'ambiguous_run'
        return row
    row['run'] = run['directory'].as_posix()
    index = case['pair_index']
    if index not in run['indices']:
        row['status'] = 'not_selected'
        return row
    generation, score = run['records'].get(index), run['scores'].get(index)
    selected = run['selected'].get(index)
    original = manifests[case['dataset']][index]
    if (run['invalid'] or not selected or selected.get('pair_id') != original.get('pair_id')
            or manifest_identity(selected) != manifest_identity(original)
            or (case['dataset'] == 'tiktok_v3' and selected.get('source_video_group') != case['group'])
            or (case['dataset'] == 'iper' and group_for('iper', selected, errors, run['directory']) != case['group'])
            or (generation and generation.get('metadata', {}).get('pair_id') != original.get('pair_id'))):
        row['status'] = 'provenance_error'
        error(errors, 'case_provenance_mismatch', run['directory'], 'Pair/group metadata or run configuration differs from the locked case',
              pair_index=index, selected_record=selected, generation_record=generation)
        return row
    if generation is None:
        row['status'] = 'missing_record'
        error(errors, 'missing_record', run['directory'], 'Selected case has no unique generation record', pair_index=index)
        return row
    row.update(generation_status=generation.get('status'), failure=generation.get('failure'))
    if generation.get('status') != 'complete':
        row['status'] = 'generation_failed'
        error(errors, 'generation_failure', run['directory'], generation.get('failure') or 'Generation did not complete', pair_index=index, record=generation)
        return row
    if score is None:
        row['status'] = 'score_missing'
        error(errors, 'score_missing', run['directory'], 'Completed case has no unique scores.json record', pair_index=index)
        return row
    row.update(score_status=score.get('status'), source_face_count=score.get('source_face_count'),
               face_count=score.get('face_count'), pose_detected=score.get('pose_detected'))
    if score.get('score_failure') or score.get('status') != 'complete':
        row.update(status='score_failed', failure=score.get('score_failure') or score.get('failure') or 'Scoring did not complete')
        error(errors, 'score_failure', run['directory'], row['failure'], pair_index=index, record=score)
        if score.get('status') != 'complete':
            return row
    else:
        row['status'] = 'complete'
    for metric in METRICS:
        value = score.get(metric)
        eligible = metric != 'identity_A' or (row['source_face_count'] == 1 and row['face_count'] == 1)
        if finite(value) and eligible:
            row['metrics'][metric] = float(value)
        elif value is not None and (not finite(value) or not eligible):
            error(errors, 'invalid_metric', run['directory'], f'{metric} is nonfinite or lacks unique-face eligibility', pair_index=index, value=value)
    return row


def stats(cases, include_groups=True):
    result = {'expected_n': len(cases), 'status_counts': dict(Counter(case['status'] for case in cases)), 'cases': cases,
              'detections': {'source_unique_face_n': sum(case['source_face_count'] == 1 for case in cases),
                             'output_unique_face_n': sum(case['face_count'] == 1 for case in cases),
                             'output_no_face_n': sum(case['face_count'] == 0 for case in cases),
                             'output_multiface_n': sum(isinstance(case['face_count'], int) and case['face_count'] > 1 for case in cases),
                             'output_detection_missing_n': sum(case['face_count'] is None for case in cases),
                             'pose_detected_n': sum(case['pose_detected'] is True for case in cases),
                             'pose_detection_missing_n': sum(case['pose_detected'] is None for case in cases)}, 'metrics': {}}
    grouped = {}
    for case in cases:
        if case['group'] is not None:
            grouped.setdefault(case['group'], []).append(case)
    for metric in METRICS:
        values = [case['metrics'][metric] for case in cases if finite(case['metrics'][metric])]
        group_means = {group: mean([case['metrics'][metric] for case in rows if finite(case['metrics'][metric])]) for group, rows in grouped.items()}
        valid_groups = [value for value in group_means.values() if value is not None]
        result['metrics'][metric] = {'mean': mean(values), 'valid_n': len(values), 'expected_n': len(cases),
            'missing_n': len(cases) - len(values), 'group_macro_mean': mean(valid_groups), 'valid_group_n': len(valid_groups),
            'group_n': len(grouped), 'group_means': group_means}
    result['metrics']['identity_A']['source_eligible_n'] = result['detections']['source_unique_face_n']
    if include_groups:
        result['groups'] = {group: stats(rows, False) for group, rows in sorted(grouped.items())}
    return result


def stage_stats(cases):
    # Dataset prefixes prevent similarly named actors and original videos from
    # accidentally becoming one overall cluster.
    overall = stats([{**case, 'group': f'{case["dataset"]}:{case["group"]}' if case['group'] else None} for case in cases])
    overall['by_dataset'] = {dataset: stats([case for case in cases if case['dataset'] == dataset]) for dataset in DATA_FILES}
    return overall


def paired(left, right, label):
    right_by_case = {(case['dataset'], case['pair_index']): case for case in right}
    result = {'direction': label, 'expected_n': len(left), 'metrics': {}}
    for metric in METRICS:
        differences = []
        for case in left:
            other = right_by_case.get((case['dataset'], case['pair_index']))
            a, b = case['metrics'][metric], other['metrics'][metric] if other else None
            if finite(a) and finite(b):
                differences.append({'dataset': case['dataset'], 'pair_index': case['pair_index'], 'pair_id': case['pair_id'],
                                    'group': case['group'], 'left': a, 'right': b, 'difference': b - a})
        groups = {}
        for item in differences:
            if item['group'] is not None:
                groups.setdefault(f'{item["dataset"]}:{item["group"]}', []).append(item['difference'])
        group_means = {group: mean(values) for group, values in sorted(groups.items())}
        result['metrics'][metric] = {'mean_difference': mean([item['difference'] for item in differences]),
            'valid_pair_n': len(differences), 'expected_n': len(left), 'missing_or_invalid_n': len(left) - len(differences),
            'group_macro_mean_difference': mean(list(group_means.values())), 'valid_group_n': len(groups),
            'group_means': group_means, 'per_case_differences': differences,
            'by_dataset': {dataset: {'valid_pair_n': sum(item['dataset'] == dataset for item in differences),
                'expected_n': sum(case['dataset'] == dataset for case in left),
                'mean_difference': mean([item['difference'] for item in differences if item['dataset'] == dataset])} for dataset in DATA_FILES}}
    return result


def training_summary(root, errors):
    result = {'scope': 'Only root/training/{J,Qwen,BFSBody,BFSHead}/metrics.jsonl; private preflight excluded', 'total_formal_steps': 0}
    for arm in ('J', *ARMS):
        path = root / 'training' / arm / 'metrics.jsonl'
        rows = read_jsonl(path, errors)
        valid = {}
        for row in rows:
            step = row.get('step')
            if row.get('error') or row.get('failure') or not isinstance(step, int) or isinstance(step, bool) or step <= 0 or not finite(row.get('loss')):
                error(errors, 'training_error_record', path, str(row.get('error', row.get('failure', 'Invalid formal training row'))), record=row)
                continue
            if step in valid:
                error(errors, 'duplicate_training_step', path, 'Duplicate formal step is excluded from authoritative counts', step=step, record=row)
                valid[step] = None
            else:
                valid[step] = row
        unique_rows = [row for row in valid.values() if row is not None]
        steps = sorted(row['step'] for row in unique_rows)
        last = max(steps, default=0)
        counts = {'formal_steps': len(steps), 'last_logged_step': last,
                  'steps_contiguous_from_one': steps == list(range(1, last + 1)), 'expected_budget': 256,
                  'face_active_steps': sum(row.get('face_active') is True for row in unique_rows),
                  'face_active_flag_missing': sum('face_active' not in row for row in unique_rows),
                  'auxiliary_requested_steps': sum(row.get('auxiliary_requested') is True for row in unique_rows),
                  'auxiliary_used_steps': sum(row.get('auxiliary_used') is True for row in unique_rows),
                  'auxiliary_requested_flag_missing': sum('auxiliary_requested' not in row for row in unique_rows),
                  'auxiliary_used_flag_missing': sum('auxiliary_used' not in row for row in unique_rows),
                  'auxiliary_eligible_faces': sum(row.get('auxiliary_eligible_faces', 0) for row in unique_rows),
                  'logged_step_indices': steps, 'metrics_path': path.relative_to(root).as_posix()}
        result[arm] = counts
        result['total_formal_steps'] += len(steps)
    return result


def format_metric(metric):
    value = '--' if metric['mean'] is None else f'{metric["mean"]:.4f}'
    return f'{value} ({metric["valid_n"]}/{metric["expected_n"]})'


def markdown(summary):
    text = ['# Body + Face 联合 BFS 试验汇总', '',
            '身份分数只统计 A 与输出均检测到唯一脸且评分有效的样本；缺失身份不填补。PCK 列显示各自有效分母，检测失败按既有评分器的实际值保留。',
            '全量初始/J/最终与固定监测子集热加载/中期分别统计。组均值按 TikTok 原视频组、iPER 人物前缀等权；不能把重复帧当成独立人物。原视频组数量有限，不报告置信区间、显著性或优胜结论。', '',
            '| 阶段/底座 | 固定样本 | 有效记录 | ID 对 A（有效/固定） | PCK 2% | PCK 5% |',
            '| --- | ---: | ---: | --- | --- | --- |']
    for cohort in summary['cohorts'].values():
        for stage, variants in cohort['stages'].items():
            for variant, value in variants.items():
                text.append(f'| {stage}/{variant} | {value["expected_n"]} | {value["status_counts"].get("complete", 0)} | '
                            + ' | '.join(format_metric(value['metrics'][metric]) for metric in METRICS) + ' |')
    text.extend(['', '| 配对差值方向 | ID 差值（配对/固定） | PCK 2% 差值 | PCK 5% 差值 |', '| --- | --- | --- | --- |'])
    for comparison in [*summary['paired_final'].values(), *summary['hotload_vs_final'].values()]:
        values = []
        for metric in METRICS:
            item = comparison['metrics'][metric]
            value = '--' if item['mean_difference'] is None else f'{item["mean_difference"]:+.4f}'
            values.append(f'{value} ({item["valid_pair_n"]}/{item["expected_n"]})')
        text.append('| ' + comparison['direction'] + ' | ' + ' | '.join(values) + ' |')
    text.extend(['', '| 正式训练 | 实际记录步数 | Face 激活 | 辅助项请求/使用 | 有效辅助脸数 |', '| --- | ---: | ---: | --- | ---: |'])
    for arm in ('J', *ARMS):
        row = summary['training'][arm]
        text.append(f'| {arm} | {row["formal_steps"]}/256 | {row["face_active_steps"]} | {row["auxiliary_requested_steps"]}/{row["auxiliary_used_steps"]} | {row["auxiliary_eligible_faces"]} |')
    text.extend(['', f'正式训练共记录 {summary["training"]["total_formal_steps"]} 次更新，私有预检更新未计入。',
                 f'保留 {len(summary["errors"])} 条缺失/错误记录；逐案例状态、检测覆盖、组均值及共同有效分母详见 [summary.json](summary.json)。',
                 '差值均为右项减左项，身份和姿态使用各自的共同有效交集。仅热加载/中期子集与最终结果比较时，最终也限制到同一锁定子集。'])
    if summary['archived_attempts']:
        text.extend(['', '归档基础设施尝试保留原样，修正后仍使用相同固定案例；归档尝试不进入主比较或有效分母。'])
        for attempt in summary['archived_attempts']:
            category = attempt['reason'].get('category', 'unspecified')
            text.append(f'- [{attempt["name"]} 原始记录]({attempt["records_link"]})：{attempt["records_n"]} 次尝试，{attempt["complete"]} 完成、{attempt["failed"]} 失败；原因 `{category}`。')
    return '\n'.join(text) + '\n'


def summarize(root, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / 'report'
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    manifests, full, monitor, lock = load_protocol(root, errors)
    runs, excluded = discover_runs(root, errors)
    cache = {}

    def cases(cohort, stage, variant):
        name = (stage, variant)
        if name not in cache:
            cache[name] = [case_result(case, stage, variant, runs, manifests, errors) for case in full]
        selected = {(case['dataset'], case['pair_index']) for case in cohort}
        return [case for case in cache[name] if (case['dataset'], case['pair_index']) in selected]

    full_stages = {stage: {variant: stage_stats(cases(full, stage, variant)) for variant in variants}
                   for stage, variants in (('initial', ('init',)), ('common', ('J',)), ('final', ARMS))}
    monitor_stages = {
        'hotload': {variant: stage_stats(cases(monitor, 'hotload', variant)) for variant in ARMS},
        'mid': {variant: stage_stats(cases(monitor, 'mid', variant)) for variant in ('J', *ARMS)},
    }
    result = {'schema': 'qwen21_joint_pilot_summary_v1', 'protocol': {'full_n': len(full), 'monitor_n': len(monitor),
              'monitor_cases': monitor, 'lock': lock, 'missing_identity_imputed': False,
              'uncertainty': 'No confidence intervals, significance tests or winner claim for this small grouped pilot'},
              'cohorts': {'full36': {'expected_n': len(full), 'stages': full_stages},
                          'fixed12': {'expected_n': len(monitor), 'stages': monitor_stages}},
              'paired_final': {arm: paired(cases(full, 'final', 'Qwen'), cases(full, 'final', arm), f'{arm} final - Qwen final') for arm in ARMS[1:]},
              'hotload_vs_final': {arm: paired(cases(monitor, 'hotload', arm), cases(monitor, 'final', arm), f'{arm} final - {arm} hotload (fixed subset)') for arm in ARMS},
              'training': training_summary(root, errors), 'excluded_diagnostic_columns': excluded,
              'archived_attempts': archived_attempts(root, output, errors), 'errors': errors}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'summary.md').write_text(markdown(result), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: RUN/report')
    args = parser.parse_args(argv)
    result = summarize(args.run, args.output)
    print(json.dumps({'full_cases': result['protocol']['full_n'], 'monitor_cases': result['protocol']['monitor_n'],
                      'formal_steps': result['training']['total_formal_steps'], 'errors': len(result['errors'])}), flush=True)


if __name__ == '__main__':
    main()
