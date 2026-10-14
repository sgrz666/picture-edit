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


def pick(candidates):
    """One run per (pair, treatment): prefer the batch-named directory, then the later name.

    Legacy runs with hand-made names (``hug_skeleton_only``) still count, but when a pair has both
    a legacy run and a batch run for the same treatment the batch one wins, because its name is
    the convention the batch writes and the comparison must not depend on directory order.
    """
    named = [fact for fact in candidates if str(fact.get('run', '')).startswith(str(fact['pair']))]
    pool = named or candidates
    return sorted(pool, key=lambda fact: fact.get('run', ''))[-1]


def paired_rows(facts, key):
    """Compare two treatments of the same pair with every *other* variable held fixed.

    The variables are arm, framing and keypoints source. Comparing any one of them must hold the
    remaining two equal, or the "winner" is whichever treatment happened to run on easier pairs -
    the first version of this function grouped only by (pair, treatment) and paired a with_b
    b_canvas run against a skeleton_only source_fit run, which read as a framing result.
    """
    variables = ('arm', 'framing', 'source')
    held_variables = [name for name in variables if name != key]
    values = sorted({fact.get(key) for fact in facts if fact.get(key)})
    if len(values) < 2:
        return [], []
    left_value, right_value = values[-2], values[-1]
    slots = {}
    for fact in facts:
        pair, treatment = fact.get('pair'), fact.get(key)
        if not pair or not treatment:
            continue
        held = tuple(fact.get(name) for name in held_variables)
        if any(value is None for value in held):
            continue
        slots.setdefault((pair, held), {}).setdefault(treatment, []).append(fact)
    rows, missing = [], []
    for (pair, held), by_treatment in sorted(slots.items()):
        sides = {treatment: pick(items) for treatment, items in by_treatment.items()}
        if left_value in sides and right_value in sides:
            row = differences(sides[left_value], sides[right_value], key)
            row['held'] = dict(zip(held_variables, held))
            row['left_run'] = sides[left_value].get('run')
            row['right_run'] = sides[right_value].get('run')
            rows.append(row)
        elif len(sides) == 1:
            missing.append({'pair': pair,
                'held': dict(zip(held_variables, held)), 'held_variables': held_variables,
                'left': left_value, 'right': right_value, 'only': [next(iter(sides))]})
    return rows, missing


def context_of(row):
    """The held variables as a stable label, e.g. 'framing=b_canvas source=dwpose'."""
    return ' '.join(f'{name}={row["held"][name]}' for name in sorted(row.get('held', {}))) or None


def tally(items):
    out = {'comparisons': len(items), 'by_metric': {}}
    for name, _ in METRICS:
        verdicts = [row['metrics'][name]['verdict'] for row in items]
        values = [row['metrics'][name].get('delta') for row in items
            if row['metrics'][name].get('delta') is not None]
        out['by_metric'][name] = {
            'n': len(values), 'left_wins': verdicts.count('left'),
            'right_wins': verdicts.count('right'), 'ties': verdicts.count('tie'),
            'not_measured': verdicts.count('not measured'),
            'median_delta': float(np.median(values)) if values else None,
            'mean_delta': float(np.mean(values)) if values else None}
    return out


def summarise(rows):
    """Per held context, plus a pooled block that is labelled as such.

    Quoting the pooled block as a single result is what this tool exists to prevent: rows in it can
    differ in the variables that were supposed to stay fixed."""
    summary = tally(rows)
    contexts = {}
    for row in rows:
        contexts.setdefault(context_of(row) or 'no held context', []).append(row)
    summary['by_context'] = {name: tally(items) for name, items in sorted(contexts.items())}
    summary['contexts'] = sorted(contexts)
    summary['quote'] = ('quote the per-context block; the pooled by_metric block mixes rows whose '
        'held variables differ')
    return summary


def number(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


# A cell shows both sides, not only the delta: when one arm cannot be scored at all (with_b with
# the wrong person count has no pose metrics), the other side's value is still the finding.
VERDICT_MARK = {'left': 'L', 'right': 'R', 'tie': '=', 'not measured': '.',
    'informative only': 'i'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--by', choices=('arm', 'framing', 'source'), required=True,
        help='which treatment to compare; the other two are held fixed within a pair')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    live_writer = lock_held()
    facts = [fact for fact in (run_facts(path, live_writer)
        for path in sorted(Path(args.runs).iterdir()) if path.is_dir()) if fact]
    complete = [fact for fact in facts if fact['status'] == 'complete']
    rows, missing = paired_rows(complete, args.by)
    header = (f'{"pair":42} {"left":14} {"right":14} '
        + ' '.join(f'{name:>15}' for name, _ in METRICS))
    print(header)
    print('-' * len(header))
    for row in rows:
        cells = ' '.join(f'{number(row["metrics"][name].get("left"))}|'
            f'{number(row["metrics"][name].get("right"))}'
            f'{VERDICT_MARK[row["metrics"][name]["verdict"]]}' for name, _ in METRICS)
        context = ' '.join(f'{name}={value}' for name, value in sorted(row['held'].items()))
        print(f'{row["pair"]:42} {str(row["left"]):14} {str(row["right"]):14} [{context:34}] {cells}')
    if not rows:
        print('no pair has both sides complete yet; this table fills in as the batch runs')
    summary = summarise(rows)
    for context, block in summary['by_context'].items():
        print(f'--- held [{context}] with {block["comparisons"]} row(s) ---')
        for name, item in block['by_metric'].items():
            print(f'  {name:10} n={item["n"]:2} left_wins={item["left_wins"]:2} '
                f'right_wins={item["right_wins"]:2} ties={item["ties"]:2} '
                f'not_measured={item["not_measured"]:2} median_delta={number(item["median_delta"])}')
    if len(summary['contexts']) > 1:
        print(f'--- POOLED over {len(summary["contexts"])} held contexts: not a single-framing result ---')
    for name, item in summary['by_metric'].items():
        print(f'  {name:10} n={item["n"]:2} left_wins={item["left_wins"]:2} '
            f'right_wins={item["right_wins"]:2} ties={item["ties"]:2} '
            f'not_measured={item["not_measured"]:2} median_delta={number(item["median_delta"])}')
    if missing:
        print('one-sided (the other variables had to match, so these could not vote): '
            + ', '.join(f"{item['pair']} at "
                + ' '.join(f'{name}={value}' for name, value in item['held'].items())
                + f" only ran {item['only'][0]} of {item['left']}/{item['right']}"
                for item in missing))
    print('each cell is left|right then a verdict mark: L=left wins, R=right wins, ==tie within '
        f'{TIE_TOLERANCE}, .=one side did not measure that metric, i=informative only')
    print('a metric only votes when both sides measured it; the summary counts and median delta '
        'below are over the pairs that could vote')
    print('torso_ratio never votes: it is a distance from 1.0, not a higher-or-lower-is-better '
        'quantity')
    if args.out:
        from src.pose_edit.common import atomic_json
        atomic_json(args.out, {'by': args.by, 'rows': rows, 'missing': missing,
            'summary': summary})


if __name__ == '__main__':
    main()
