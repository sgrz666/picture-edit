#!/usr/bin/env python3
"""Audit the two-person experiment grid: every pair x arm x framing cell and the evidence behind it.

The paired readouts here are only as trustworthy as the grid under them, and that grid has already
misled twice: one pair was read as an arm result, five pairs as a framing result. Counting run
directories does not answer the question, because a run can exist and be 'complete' yet carry no
pose number (wrong person count) and no identity number (DINOv2 unavailable).

Coverage is read from each run's own artifacts -- run.json status plus evaluation/eval_dual.json --
not from the batch index, so this stays correct for the fit twins, which are driven by their own
script and never enter index.tsv.

Exit code 0 means the whole planned grid is covered, 1 means cells are still owed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('with_b', 'skeleton_only')
FRAMINGS = ('b_canvas', 'source_fit')
SOURCES = ('dwpose', 'fit')


def pair_names(root):
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / 'pair.json').is_file())


def run_status(run_dir):
    """The run's own recorded status; absent or unreadable runs report None."""
    manifest = run_dir / 'run.json'
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text()).get('status')
    except ValueError:
        return None


def evidence(run_dir):
    """What this run can actually be read for."""
    item = {'status': run_status(run_dir), 'has_evaluation': False,
            'pose': False, 'identity': False, 'count_ok': None}
    path = run_dir / 'evaluation' / 'eval_dual.json'
    if not path.is_file():
        return item
    item['has_evaluation'] = True
    summary = json.loads(path.read_text()).get('summary', {})
    item['pose'] = summary.get('body_pck@0.2') is not None
    item['identity'] = summary.get('identity_similarity') is not None
    item['count_ok'] = summary.get('person_count_ok')
    return item


def build_grid(pairs, runs_dir):
    """Every planned cell: dwpose guide for both arms and framings, plus the fit-guide twins."""
    grid = []
    for name, arm, framing, source in cells(pairs):
        item = {'name': name, 'pair': pair_of(name, pairs), 'arm': arm, 'framing': framing,
                'source': source}
        item.update(evidence(runs_dir / name))
        item['covered'] = item['status'] == 'complete' and item['has_evaluation']
        grid.append(item)
    return grid


def cells(pairs):
    """(run name, arm, framing, guide source) for every cell the design plans to fill.

    Fit-guide twins are planned at b_canvas only: they exist to isolate the guide source against
    one ground-truth-guided counterpart, and crossing guide x framing would be a third variable
    the experiment does not claim to measure."""
    planned = []
    for pair in pairs:
        for arm in ARMS:
            for framing in FRAMINGS:
                planned.append((f'{pair}.{arm}.{framing}', arm, framing, 'dwpose'))
        planned.append((f'{pair}.skeleton_only.b_canvas.fit', 'skeleton_only', 'b_canvas', 'fit'))
    return planned


def pair_of(name, pairs):
    for pair in pairs:
        if name.startswith(pair + '.'):
            return pair
    return None


def pairable_rows(grid, key):
    """Pairs that can vote on `key`: both treatments covered, every other variable held equal."""
    held_names = [name for name in ('arm', 'framing', 'source') if name != key]
    groups = {}
    for item in grid:
        if not item['covered'] or item.get(key) is None or item['pair'] is None:
            continue
        groups.setdefault((item['pair'],) + tuple(item[name] for name in held_names), {})[item[key]] = item
    rows = voteable = 0
    for treatments in groups.values():
        if len(treatments) < 2:
            continue
        rows += 1
        if any(one['pose'] or one['identity'] for one in treatments.values()):
            voteable += 1
    return rows, voteable


def main():
    same = pair_names(ROOT / 'pairs')
    cross = pair_names(ROOT / 'pairs_cross')
    pairs = same + cross
    grid = build_grid(pairs, ROOT / 'runs')

    planned = len(grid)
    covered = [item for item in grid if item['covered']]
    batch = [item for item in grid if item['source'] == 'dwpose']
    twins = [item for item in grid if item['source'] == 'fit']
    print(f'pairs: {len(same)} same-identity + {len(cross)} cross-identity = {len(pairs)}')
    print(f'planned cells: {planned} ({len(batch)} dwpose-guided, {len(twins)} fit-guide twins)')
    print(f'covered (run.json complete AND eval_dual.json present): {len(covered)}')
    print(f'  carrying a pose number: {sum(1 for item in covered if item["pose"])}'
          f' | an identity number: {sum(1 for item in covered if item["identity"])}'
          f' | correct person count: {sum(1 for item in covered if item["count_ok"] == 1)}')
    for key in ('arm', 'framing', 'source'):
        rows, voteable = pairable_rows(grid, key)
        print(f'--by {key}: {rows} paired rows, {voteable} carry a metric to vote on')

    missing = [item for item in grid if not item['covered']]
    if missing:
        print(f'\n{len(missing)} cells still owed:')
        for item in missing:
            print(f'  {item["name"]}  status={item["status"] or "never attempted"}')
        return 1
    print('\nGRID_COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
