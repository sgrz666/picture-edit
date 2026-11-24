#!/usr/bin/env python3
"""Select among Native RL checkpoints using independent held-out visual scores."""

import argparse
import math
from pathlib import Path
from statistics import fmean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pose_control.qwen21_full.native_v62.training.io import read_json_ref, verify, write_json
from src.pose_control.qwen21_full.native_v62.training.config import digest


REVIEW_SCHEMA = 'native_rl_visual_review_v1'
SCORES_SCHEMA = 'native_rl_visual_scores_v1'


def checked_ref(ref):
    if not ref.get('sha256'):
        raise ValueError('Visual review requires SHA-bound artifacts')
    return verify(ref)


def review_cases(report):
    keys = [(row['source'], row['pair_id'], row['seed']) for row in report['rows']]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Visual review requires a nonempty, unique ODE cohort')
    cases = [{'source': source, 'pair_id': pair_id, 'seed': seed, 'condition': checked_ref(row['condition'])}
             for (source, pair_id, seed), row in zip(keys, report['rows'], strict=True)]
    return cases, [checked_ref(row['image']) for row in report['rows']]


def prepare(run_dir):
    run, run_ref = read_json_ref({'path': str((run_dir / 'run.json').resolve())})
    summary, summary_ref = read_json_ref({'path': str((run_dir / 'summary.json').resolve())})
    if run['evaluation_kind'] != 'standard':
        raise ValueError('Visual selection requires held-out validation')
    identity_sha256 = digest(run['resume_identity'])
    review = summary['visual_review']
    baseline = checked_ref(review['baseline_checkpoint'])
    baseline_ref = checked_ref(review['baseline_report'])
    report, _ = read_json_ref(baseline_ref)
    if report['sft_reference'] != baseline or run['initialization'] != baseline:
        raise ValueError('ODE baseline does not match SFT checkpoint')
    cases, images = review_cases(report)
    candidates = [{'id': baseline['sha256'], 'checkpoint': baseline, 'ode_report': baseline_ref,
                   'optimizer_updates': 0, 'images': images}]
    for eligible in review['eligible']:
        checkpoint = checked_ref(eligible['checkpoint'])
        report_ref = checked_ref(eligible['report'])
        commit = Path(checkpoint['path']).with_suffix('.commit.json')
        marker, _ = read_json_ref({'path': str(commit)})
        report, _ = read_json_ref(report_ref)
        update = marker['counters']['optimizer_updates']
        if (marker.get('schema') != 'native_rl_commit_v3' or marker['checkpoint'] != checkpoint
                or marker['identity_sha256'] != identity_sha256
                or marker.get('ode_gate') != {'passed': True, 'report': report_ref}
                or report['regressed'] is not False or report['optimizer_updates'] != update
                or report['sft_reference'] != baseline):
            raise ValueError('Visual candidate does not match its committed ODE gate')
        candidate_cases, images = review_cases(report)
        if candidate_cases != cases:
            raise ValueError('Visual candidate differs from the fixed SFT ODE cohort')
        candidates.append({'id': checkpoint['sha256'], 'checkpoint': checkpoint, 'ode_report': report_ref,
                           'optimizer_updates': update, 'images': images})
    if len({row['id'] for row in candidates}) != len(candidates):
        raise ValueError('Duplicate checkpoint in visual review')
    return {'schema': REVIEW_SCHEMA, 'run': run_ref, 'summary': summary_ref, 'baseline_id': baseline['sha256'],
            'cases': cases, 'candidates': sorted(candidates, key=lambda row: row['id'])}


def select(review_path, scores_path, min_gain):
    review, review_ref = read_json_ref({'path': str(review_path.resolve())})
    scores, scores_ref = read_json_ref({'path': str(scores_path.resolve())})
    if review.get('schema') != REVIEW_SCHEMA or scores.get('schema') != SCORES_SCHEMA or scores.get('review') != review_ref:
        raise ValueError('Independent scores must bind the exact visual review manifest')
    summary = checked_ref(review['summary'])
    if prepare(Path(summary['path']).parent) != review:
        raise ValueError('Visual review differs from the committed ODE cohort')
    judge = scores['evaluator']
    if judge.get('kind') not in ('human_blind', 'editreward') or not judge.get('protocol'):
        raise ValueError('Expected an independent blind rubric or EditReward protocol')
    if judge['kind'] == 'editreward':
        checked_ref(judge['model_manifest'])
    by_id = {row['id']: row for row in scores['candidates']}
    candidates = {row['id']: row for row in review['candidates']}
    if set(by_id) != set(candidates) or len(by_id) != len(scores['candidates']):
        raise ValueError('Independent scores must cover the full eligible set and SFT baseline')
    means = {}
    for identifier, candidate in candidates.items():
        observations = by_id[identifier]['scores']
        if [item['image_sha256'] for item in observations] != [image['sha256'] for image in candidate['images']]:
            raise ValueError('Independent scores differ from the ordered held-out images')
        values = [item['score'] for item in observations]
        if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
            raise ValueError('Independent visual scores must be finite numbers in [0,1]')
        means[identifier] = fmean(values)
    baseline = review['baseline_id']
    best = max(means, key=lambda identifier: (means[identifier], -candidates[identifier]['optimizer_updates']))
    selected = best if means[best] - means[baseline] > min_gain else baseline
    return {'schema': 'native_rl_visual_selection_v1', 'review': review_ref, 'scores': scores_ref,
            'evaluator': judge, 'mean_scores': means, 'min_gain': min_gain, 'selected_id': selected,
            'selected': candidates[selected]['checkpoint'], 'baseline_id': baseline}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    build = commands.add_parser('prepare', help='Export SFT plus ODE-gate-passing candidates')
    build.add_argument('--run', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    rank = commands.add_parser('select', help='Select with SHA-bound independent scores')
    rank.add_argument('--review', type=Path, required=True)
    rank.add_argument('--scores', type=Path, required=True)
    rank.add_argument('--min-gain', type=float, default=0.)
    rank.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output already exists')
    if args.action == 'select' and not (math.isfinite(args.min_gain) and 0 <= args.min_gain <= 1):
        parser.error('--min-gain must be in [0,1]')
    result = prepare(args.run) if args.action == 'prepare' else select(args.review, args.scores, args.min_gain)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
