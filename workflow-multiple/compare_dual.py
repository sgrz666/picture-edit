#!/usr/bin/env python3
"""Compare the two arms, and the two framings, on the pairs where both exist.

`report_dual.py` lists every run and averages per arm. That answers "what does this arm score"
but not the question the batch was run for: **on the same pair, does the other arm do better?**
Averages can mislead here because pairs differ enormously in difficulty - if the arms happened to
be measured on different pair subsets, the means would compare the pairs rather than the arms.

This script only uses pairs where both sides exist, reports the per-pair difference, and counts
wins per metric with an explicit tie tolerance. Directions are declared once: PCK, identity
similarity and person-count improve upwards, error metrics improve downwards.

    add/envs/smplestx_bw/bin/python workflow-multiple/compare_dual.py --by arm
    add/envs/smplestx_bw/bin/python workflow-multiple/compare_dual.py --by framing
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report_dual import lock_held, run_facts

# metric key in run_facts, and whether a larger value is better
METRICS = (('count_ok', True), ('pck', True), ('mpjpe', False), ('torso_ratio', None),
    ('gap_error', False), ('identity', True))
TIE_TOLERANCE = 0.02


def differences(left, right, key, tolerance=TIE_TOLERANCE):
    """Per-pair comparison of two run facts: value, signed delta, and a verdict per metric."""
    row = {'pair': left['pair'], 'left': left.get(key), 'right': right.get(key), 'metrics': {}}
    wins = {'left': 0, 'right': 0, 'tie': 0, 'skipped': 0}
    for name, higher_better in METRICS:
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            row['metrics'][name] = {'left': None, 'right': None, 'verdict': 'not measured'}
            wins['skipped'] += 1
            continue
        delta = a - b
        if higher_better is None:
            # torso ratio is only meaningful as a distance from 1.0, so it never declares a winner
            row['metrics'][name] = {'left': a, 'right': b, 'delta': delta,
                'verdict': 'informative only'}
            continue
        margin = abs(delta)
        if margin <= tolerance:
            verdict = 'tie'
        elif (delta > 0) == higher_better:
            verdict = 'left'
        else:
            verdict = 'right'
        row['metrics'][name] = {'left': a, 'right': b, 'delta': delta, 'verdict': verdict}
        if verdict in wins:
            wins[verdict] += 1
    row['wins'] = wins
    return row


def paired_rows(facts, key):
    """Compare every two-way combination of arms or framings, pair by pair.

    Only pairs with both sides present contribute, and a pair that only one side ever reached is
    reported as missing rather than being dropped silently from the counts.
    """
    values = sorted({fact.get(key) for fact in facts if fact.get(key)})
    rows, missing = [], []
    for left_value, right_value in combinations(values, 2):
        by_pair = {}
        for fact in facts:
            if not fact.get('pair') or fact.get(key) not in (left_value, right_value):
                continue
            side = 'left' if fact[key] == left_value else 'right'
            by_pair.setdefault(fact['pair'], {})[side] = fact
        for pair, sides in sorted(by_pair.items()):
            if 'left' in sides and 'right' in sides:
                rows.append(differences(sides['left'], sides['right'], key))
            else:
                missing.append({'pair': pair, 'left': left_value, 'right': right_value,
                    'only': list(sides)})
    return rows, missing


def summarise(rows):
    summary = {'comparisons': len(rows), 'by_metric': {}}
    for name, _ in METRICS:
        verdicts = [row['metrics'][name]['verdict'] for row in rows]
        values = [row['metrics'][name].get('delta') for row in rows
            if row['metrics'][name].get('delta') is not None]
        summary['by_metric'][name] = {
            'n': len(values), 'left_wins': verdicts.count('left'),
            'right_wins': verdicts.count('right'), 'ties': verdicts.count('tie'),
            'not_measured': verdicts.count('not measured'),
            'median_delta': float(np.median(values)) if values else None,
            'mean_delta': float(np.mean(values)) if values else None}
    return summary


def number(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--by', choices=('arm', 'framing'), required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    live_writer = lock_held()
    facts = [fact for fact in (run_facts(path, live_writer)
        for path in sorted(Path(args.runs).iterdir()) if path.is_dir()) if fact]
    complete = [fact for fact in facts if fact['status'] == 'complete']
    rows, missing = paired_rows(complete, args.by)
    header = f'{"pair":42} {"left":14} {"right":14} ' + ' '.join(f'{name:>12}' for name, _ in METRICS)
    print(header)
    print('-' * len(header))
    for row in rows:
        cells = ' '.join(f'{number((row["metrics"][name].get("delta"))):>7}'
            f'[{row["metrics"][name]["verdict"][:2]}]' for name, _ in METRICS)
        print(f'{row["pair"]:42} {str(row["left"]):14} {str(row["right"]):14} {cells}')
    if not rows:
        print('no pair has both sides complete yet; this table fills in as the batch runs')
    summary = summarise(rows)
    for name, item in summary['by_metric'].items():
        print(f'  {name:10} n={item["n"]:2} left_wins={item["left_wins"]:2} '
            f'right_wins={item["right_wins"]:2} ties={item["ties"]:2} '
            f'not_measured={item["not_measured"]:2} median_delta={number(item["median_delta"])}')
    if missing:
        print('pairs measured on one side only: '
            + ', '.join(f"{item['pair']}({item['left']} vs {item['right']}: only "
                f"{item['only']})" for item in missing))
    print('delta is left minus right; a metric only votes when both sides measured it, and ties '
        f'within {TIE_TOLERANCE} count as ties rather than wins')
    print('torso_ratio never votes: it is a distance from 1.0, not a higher-or-lower-is-better '
        'quantity')
    if args.out:
        from src.pose_edit.common import atomic_json
        atomic_json(args.out, {'by': args.by, 'rows': rows, 'missing': missing,
            'summary': summary})


if __name__ == '__main__':
    main()
