#!/usr/bin/env python3
"""Turn blind annotation jsonl into the numbers the contract asks for.

    python tally.py responses/ --truth bundle/truth.json --index bundle/index.json --markdown

Reads one judgement per line (see annotate.html's export). Win rate is signed against the
**reference arm** (the Qwen-Edit baseline, or `skeleton_only` while that arm does not exist),
so a positive number always means the candidate is better. Bootstrap resamples whole
questions, not individual judgements, because three annotators rating one question are not
three independent samples.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

AXES = ('identity', 'action', 'interaction', 'hand')
CHOICES = ('candidate', 'reference', 'tie')
BOOTSTRAP_ROUNDS = 5000
BOOTSTRAP_SEED = 20260927


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_judgements(responses_dir, index, truth):
    expected = {item['id']: item for item in index}
    arms = {item['id']: item for item in truth['items']}
    judgements, excluded = [], []
    for path in sorted(Path(responses_dir).glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = expected.get(row.get('id'))
            if not question:
                excluded.append({'source': path.name, 'id': row.get('id'), 'reason': 'unknown_id'})
                continue
            if row.get('left_sha256') != question['left_sha256'] or \
               row.get('right_sha256') != question['right_sha256']:
                # A stale packet: the bundle was rebuilt and this answer refers to other images.
                excluded.append({'source': path.name, 'id': row['id'], 'reason': 'image_hash_mismatch'})
                continue
            meta = arms[row['id']]
            side_for = {'left': meta['left_arm'], 'right': meta['right_arm']}
            role_of_arm = {truth['candidate_arm']: 'candidate', truth['reference_arm']: 'reference'}
            if row.get('pick') in ('left', 'right'):
                pick = role_of_arm.get(side_for[row['pick']])
            elif row.get('pick') == 'tie':
                pick = 'tie'
            else:
                pick = None
            if pick is None:
                excluded.append({'source': path.name, 'id': row['id'], 'reason': 'no_choice'})
                continue
            signed = {'candidate': 1.0, 'reference': -1.0, 'tie': 0.0}
            arm_of_side = {'left': meta['left_arm'], 'right': meta['right_arm']}
            candidate_scores = {axis: row['ratings'][side].get(axis)
                for side in ('left', 'right') for axis in AXES
                if arm_of_side[side] == truth['candidate_arm']}
            reference_scores = {axis: row['ratings'][side].get(axis)
                for side in ('left', 'right') for axis in AXES
                if arm_of_side[side] == truth['reference_arm']}
            if any(value is None for value in list(candidate_scores.values())
                    + list(reference_scores.values())):
                excluded.append({'source': path.name, 'id': row['id'], 'reason': 'missing_axis'})
                continue
            judgements.append({'id': row['id'], 'pair': question['pair'],
                'annotator': row.get('annotator', path.stem), 'pick': pick,
                'value': signed[pick], 'candidate': candidate_scores,
                'reference': reference_scores})
    return judgements, excluded


def bootstrap_ci(values, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    draws = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        draws.append(mean(sample))
    draws.sort()
    return draws[int(0.025 * rounds)], draws[int(0.975 * rounds)]


def fleiss_kappa(items_per_question):
    """items_per_question: list of lists of category indices, one list per question."""
    rows = [row for row in items_per_question if len(row) >= 2]
    if not rows:
        return None
    categories = len(CHOICES)
    n = len(rows[0])
    if any(len(row) != n for row in rows):
        # General form allows unequal raters; do it properly rather than dropping rows.
        total_pairs = sum(len(row) * (len(row) - 1) for row in rows)
        agree = sum(row.count(c) * (row.count(c) - 1) for row in rows for c in set(row))
        p_within = agree / total_pairs if total_pairs else 0.0
        counts = {c: sum(row.count(c) for row in rows) for c in range(categories)}
        total_judgements = sum(len(row) for row in rows)
        p_between = sum((counts[c] / total_judgements) ** 2 for c in range(categories))
        if p_between >= 1.0:
            return None
        return (p_within - p_between) / (1.0 - p_between)
    table = [[row.count(c) for c in range(categories)] for row in rows]
    p_between = sum((sum(r[c] for r in table) / (n * len(table))) ** 2 for c in range(categories))
    p_within = mean((sum(count ** 2 for count in row) - n) / (n * (n - 1)) for row in table)
    if p_between >= 1.0:
        return None
    return (p_within - p_between) / (1.0 - p_between)


def mean_pairwise_pearson(questions):
    """Approximate ICC: how much do annotators agree on the numbers they wrote."""
    vectors = {}
    for item in questions:
        key = item['id']
        vectors.setdefault(key, []).append([item['candidate'][axis] for axis in AXES])
    scores = []
    for values in vectors.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                x, y = values[i], values[j]
                if x == y:
                    scores.append(1.0)
                    continue
                mx, my = mean(x), mean(y)
                num = sum((a - mx) * (b - my) for a, b in zip(x, y))
                den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
                scores.append(num / den if den else 0.0)
    return mean(scores) if scores else None


def summarize(judgements, truth):
    by_question = {}
    for row in judgements:
        by_question.setdefault(row['id'], []).append(row)
    values = [mean([row['value'] for row in rows]) for rows in by_question.values()]
    wins = sum(1 for row in judgements if row['pick'] == 'candidate')
    losses = sum(1 for row in judgements if row['pick'] == 'reference')
    ties = sum(1 for row in judgements if row['pick'] == 'tie')
    low_ci, high_ci = bootstrap_ci(values) if values else (None, None)

    def plausibility(side):
        rows = [row for row in judgements if all(row[side][axis] >= 3 for axis in AXES)]
        return len(rows) / len(judgements) if judgements else None

    axis_fail = {axis: sum(1 for row in judgements if row['candidate'][axis] <= 2) / len(judgements)
        for axis in AXES} if judgements else {}
    axis_fail_ref = {axis: sum(1 for row in judgements if row['reference'][axis] <= 2) / len(judgements)
        for axis in AXES} if judgements else {}
    votes = [[{'candidate': 0, 'reference': 1, 'tie': 2}[row['pick']] for row in rows]
        for rows in by_question.values()]
    return {
        'questions': len(by_question), 'judgements': len(judgements),
        'annotators': sorted({row['annotator'] for row in judgements}),
        'reference_arm': truth['reference_arm'], 'candidate_arm': truth['candidate_arm'],
        'wins_for_candidate': wins, 'losses': losses, 'ties': ties,
        'tie_rate': ties / len(judgements) if judgements else None,
        'win_rate': mean(values) if values else None,
        'win_rate_ci95': [low_ci, high_ci],
        'contract_win_margin': '+5%' if low_ci is not None and low_ci > 0.05 else 'NOT_MET',
        'plausibility_rate_candidate': plausibility('candidate'),
        'plausibility_rate_reference': plausibility('reference'),
        'plausibility_threshold': {'stage1': 0.70, 'stage2': 0.90},
        'axis_fail_rate_candidate': axis_fail, 'axis_fail_rate_reference': axis_fail_ref,
        'fleiss_kappa_on_choice': fleiss_kappa(votes),
        'mean_pairwise_r_on_axes': mean_pairwise_pearson(judgements),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('responses', type=Path)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    truth = read_json(args.truth)
    index = read_json(args.index)
    judgements, excluded = load_judgements(args.responses, index, truth)
    if not judgements:
        raise SystemExit(f'NO_JUDGEMENTS under {args.responses} (excluded {len(excluded)})')
    summary = summarize(judgements, truth)
    summary['excluded'] = excluded
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    agreement = summary['fleiss_kappa_on_choice']
    if agreement is not None and agreement < 0.4:
        print(f'AGREEMENT_LOW fleiss_kappa={agreement:.3f} < 0.4，这批结论不要进报告')
    rows = [
        ('对照', f"{summary['candidate_arm']} vs {summary['reference_arm']}（正数=前者更好）"),
        ('题数 / 判定数', f"{summary['questions']} / {summary['judgements']}"),
        ('标注者', ', '.join(summary['annotators']) or '-'),
        ('胜/平/负', f"{summary['wins_for_candidate']}/{summary['ties']}/{summary['losses']}"),
        ('win_rate', f"{summary['win_rate']:+.3f} "
            f"CI95 [{summary['win_rate_ci95'][0]:+.3f}, {summary['win_rate_ci95'][1]:+.3f}]"
            f" -> {summary['contract_win_margin']}"),
        ('主观合理比例(候选)', f"{summary['plausibility_rate_candidate']:.3f} "
            f"(阶段一 0.70 / 阶段二 0.90)"),
        ('主观合理比例(参照)', f"{summary['plausibility_rate_reference']:.3f}"),
        ('tie 率', f"{summary['tie_rate']:.3f}"),
        ('不合格率(候选,≤2)', ' '.join(f'{axis}={value:.2f}'
            for axis, value in summary['axis_fail_rate_candidate'].items())),
        ('不合格率(参照,≤2)', ' '.join(f'{axis}={value:.2f}'
            for axis, value in summary['axis_fail_rate_reference'].items())),
        ('成对选择 Fleiss κ', f"{agreement:.3f}" if agreement is not None else 'n/a（每题不足 2 人）'),
        ('轴分两两 Pearson 均值', f"{summary['mean_pairwise_r_on_axes']:.3f}"),
        ('剔除', f"{len(excluded)} " + (str(excluded[:3]) if excluded else '')),
    ]
    for label, value in rows:
        print(f'| {label} | {value} |' if args.markdown else f'{label:24} {value}')
    if args.json:
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
