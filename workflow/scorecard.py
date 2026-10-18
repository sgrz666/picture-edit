#!/usr/bin/env python3
"""Cross-run scorecard for the ground-truth-free metrics written by evaluate2.py.

    python scorecard.py                     # aligned table over every scored run
    python scorecard.py --baseline X --candidate Y
    python scorecard.py --json reports/scorecard.json

It reads only ``runs/*/evaluation/eval2.json`` plus the first-reader ``eval.json`` when
present, and flags records whose image changed or disappeared underneath them — the runs
directory is edited by other sessions while these numbers are produced, so a stale score
must never look like a fresh one.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import read_json, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parent
# (label, extractor, higher_is_better). Headline first, diagnostics after, in reading order:
# did it move, did it land where asked, is it still the same person and scene, is it healthy.
COLUMNS = (
    ('pck3D', lambda summary: (summary or {}).get('body_pck@0.2'), True),
    ('pckDW', lambda r: _summary(r).get('body_pck@0.2'), True),
    ('pckDW*', lambda r: _summary(r).get('body_pck@0.2_strict'), True),
    ('agree', lambda r: (r.get('estimator_agreement') or {}).get('body_agree@0.2'), True),
    ('torsoG', lambda r: (r.get('scale_fidelity') or {}).get(
        'torso_ratio_generated_over_guide'), None),
    ('moved', lambda r: ((r.get('pose_generated_vs_source') or {}).get('summary') or {}).get(
        'body_mpjpe_norm'), None),
    ('idcos', lambda r: (r.get('identity') or {}).get('identity_cosine'), True),
    ('bgMAE', lambda r: (r.get('preservation') or {}).get('background_mae'), False),
    ('bgLPIPS', lambda r: (r.get('preservation') or {}).get('background_lpips'), False),
    ('perMAE', lambda r: (r.get('preservation') or {}).get('person_mae'), False),
    ('shape', lambda r: ((r.get('shape') or {}).get('drift') or {}).get('max_rel_change'), False),
    ('people', lambda r: (r.get('persons_detected') or {}).get('generated'), None),
)


def _summary(record):
    return (record.get('pose_generated_vs_guide') or {}).get('summary') or {}


def load_runs(runs_dir):
    rows = []
    for path in sorted(runs_dir.glob('*/evaluation/eval2.json')):
        run_name = path.parts[-3]
        try:
            record = read_json(path)
        except Exception as error:
            rows.append({'run': run_name, 'state': f'unreadable ({type(error).__name__})',
                'record': None, 'first': None})
            continue
        final = Path(record.get('final_image') or '')
        state = 'ok'
        if not final.is_file():
            state = 'stale: image gone'
        else:
            record_path = path.parent / 'stage2.json'
            if record_path.is_file():
                try:
                    stored = ((read_json(record_path).get('identity') or {}).get(
                        'inputs_sha256') or {}).get('final_image')
                except Exception:
                    stored = None
                if stored and stored != sha256_file(final):
                    state = 'stale: image changed after scoring'
        first = None
        first_path = path.parent / 'eval.json'
        if first_path.is_file():
            try:
                first = (read_json(first_path).get('summary') or {})
            except Exception:
                first = None
        rows.append({'run': run_name, 'state': state, 'record': record, 'first': first})
    return rows


def extract(column, row):
    label, getter, _higher = column
    record, first = row['record'], row['first']
    if record is None:
        return None
    if label == 'pck3D':
        return getter(first)
    try:
        return getter(record)
    except AttributeError:
        return None


def format_value(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def compare(baseline, candidate):
    rows = []
    for column in COLUMNS:
        label, _getter, higher = column
        before, after = extract(column, baseline), extract(column, candidate)
        if before is None or after is None:
            rows.append({'metric': label, 'baseline': before, 'candidate': after,
                'delta': None, 'verdict': 'incomplete'})
            continue
        delta = after - before
        direction = 'better' if (delta > 0) == bool(higher) else 'worse' if delta else 'same'
        rows.append({'metric': label, 'baseline': before, 'candidate': after,
            'delta': delta, 'verdict': direction if higher is not None else 'direction n/a'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    parser.add_argument('--baseline')
    parser.add_argument('--candidate')
    parser.add_argument('--json', help='also write the scorecard to this path')
    args = parser.parse_args()
    rows = load_runs(Path(args.runs_dir))
    if not rows:
        print('no evaluation/eval2.json found; run evaluate2.py --all first')
        return 1

    header = f'{"run":<26} {"state":<26}' + ''.join(f'{column[0]:>10}' for column in COLUMNS)
    print(header)
    for row in rows:
        cells = ''.join(f'{format_value(extract(column, row)):>10}' for column in COLUMNS)
        print(f'{row["run"]:<26} {row["state"]:<26}{cells}')
    print('\nlegend: pck3D evaluate.py/SMPLest-X | pckDW DWPose second reader | pckDW* strict '
        '(dropped joints count as miss) | agreement between the two readers | '
        'torsoG generated torso / requested torso (1.0 = right size) | '
        'moved skeleton displacement vs the source frame | idcos ArcFace vs source | '
        'bg/per preservation over the person-mask union | shape pose-invariant limb ratios')

    if args.baseline or args.candidate:
        if not (args.baseline and args.candidate):
            print('\n--baseline and --candidate must be given together')
            return 1
        index = {row['run']: row for row in rows}
        missing = [name for name in (args.baseline, args.candidate) if name not in index]
        if missing:
            print(f'\nunknown run(s): {", ".join(missing)}')
            return 1
        print(f'\npairwise {args.baseline} -> {args.candidate}')
        for row in compare(index[args.baseline], index[args.candidate]):
            print(f'  {row["metric"]:<9} {format_value(row["baseline"]):>9} -> '
                f'{format_value(row["candidate"]):>9} '
                f'{"-" if row["delta"] is None else "%+.3f" % row["delta"]:>9}  {row["verdict"]}')
        print('  n=1 image per run: a delta is a single observation. Score several seeds per '
            'pair before claiming an improvement, then the deltas can be bootstrapped as in '
            'scripts/eval/eval_iper_four_arm.py.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
