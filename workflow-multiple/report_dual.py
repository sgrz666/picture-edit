#!/usr/bin/env python3
"""Read-only index over workflow-multiple/runs: which arms exist, what they measured, what is missing."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = Path(__file__).resolve().parent
CURRENT_SCHEMA = 'dual_pose_workflow_v1'


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def lock_held():
    path = ROOT / '.workflow.lock'
    if not path.is_file():
        return False
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def stage_seconds(run_dir):
    facts = {}
    for name in ('geometry', 'prompt', 'generation'):
        record = read_json(run_dir / name / 'stage.json')
        if record:
            facts[name] = (record.get('metadata') or {}).get('seconds')
    return facts


def run_facts(run_dir, live_writer):
    manifest = read_json(run_dir / 'run.json')
    if not manifest:
        return None
    status = manifest.get('status')
    if status == 'running' and not live_writer:
        status = 'abandoned'
    if manifest.get('schema') != CURRENT_SCHEMA:
        status = f'{status}(legacy)'
    settings = manifest.get('settings', {})
    evaluation = read_json(run_dir / 'evaluation' / 'eval_dual.json') or {}
    summary = evaluation.get('summary') or {}
    interaction = (evaluation.get('interaction') or {}).get('error_relative')
    verification = manifest.get('verification') or {}
    return {'run': run_dir.name, 'status': status,
        'verified': verification.get('status'), 'pair': manifest.get('pair', {}).get('name'),
        'kind': manifest.get('pair', {}).get('kind'),
        'framing': settings.get('framing'), 'arm': settings.get('generator_references'),
        'settings': f"{settings.get('resolution')}/{settings.get('steps')}/{settings.get('seed')}",
        'persons': summary.get('persons_detected'), 'count_ok': summary.get('person_count_ok'),
        'pck': summary.get('body_pck@0.2'), 'mpjpe': summary.get('body_mpjpe_norm'),
        'torso_ratio': summary.get('torso_scale_ratio'), 'gap_error': interaction,
        'seconds': stage_seconds(run_dir),
        'error': manifest.get('error')}


def number(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--pairs', type=Path, nargs='*', default=[ROOT / 'pairs',
        ROOT / 'pairs_cross'])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    live_writer = lock_held()
    runs = [fact for fact in (run_facts(path, live_writer)
        for path in sorted(Path(args.runs).iterdir()) if path.is_dir()) if fact]
    available = []
    for directory in args.pairs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        available += [path.name for path in sorted(directory.iterdir())
            if (path / 'pair.json').is_file()]
    built = {fact['pair'] for fact in runs}
    if args.json:
        print(json.dumps({'runs': runs, 'pairs_available': len(available),
            'pairs_without_run': sorted(set(available) - built)}, ensure_ascii=False, indent=2))
        return
    header = (f'{"run":30} {"status":10} {"verify":10} {"pair":26} {"framing":12} '
        f'{"arm":14} res/steps/seed persons count pck mpjpe torso gap  stage seconds')
    print(header)
    print('-' * len(header))
    for fact in sorted(runs, key=lambda item: item['run']):
        timings = ' '.join(f'{name[:4]}:{value:.0f}s' if value else f'{name[:4]}:-'
            for name, value in fact['seconds'].items())
        print(f'{fact["run"]:30} {fact["status"]:10} {str(fact["verified"]):10} '
            f'{str(fact["pair"]):26} {str(fact["framing"]):12} {str(fact["arm"]):14} '
            f'{fact["settings"]:14} {number(fact["persons"], 0):>7} '
            f'{number(fact["count_ok"]):>5} {number(fact["pck"]):>5} {number(fact["mpjpe"]):>5} '
            f'{number(fact["torso_ratio"]):>5} {number(fact["gap_error"]):>4} {timings}')
    missing = sorted(set(available) - built)
    print(f'\npairs: {len(available)} available, {len(available) - len(missing)} with a run, '
        f'{len(missing)} without')
    if missing:
        print('without a run: ' + ', '.join(missing))
    if any(fact['status'] == 'abandoned' for fact in runs):
        print('abandoned runs are ones whose process died before writing a final status')


if __name__ == '__main__':
    if hasattr(sys, 'win_version'):
        raise SystemExit('workflow-multiple/runs lives on the Linux GPU host; run this there')
    main()
