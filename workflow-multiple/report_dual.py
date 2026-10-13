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
        # Runs written before --keypoints-source existed have no such setting; their guides were
        # necessarily the pair's own landmarks, because nothing else could draw them.
        'framing': settings.get('framing'), 'arm': settings.get('generator_references'),
        'source': settings.get('keypoints_source', 'dwpose'),
        'settings': f"{settings.get('resolution')}/{settings.get('steps')}/{settings.get('seed')}",
        'persons': summary.get('persons_detected'), 'count_ok': summary.get('person_count_ok'),
        'regions_ok': summary.get('person_count_regions_ok'),
        'pck': summary.get('body_pck@0.2'), 'mpjpe': summary.get('body_mpjpe_norm'),
        'torso_ratio': summary.get('torso_scale_ratio'), 'gap_error': interaction,
        'identity': summary.get('identity_similarity'),
        'swap': summary.get('identity_swap_suspected'),
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
    header = (f'{"run":54} {"status":9} v {"pair":26} {"framing":11} '
        f'{"arm":13} res/steps/seed persons count reg pck mpjpe torso gap ident swap  stage seconds')
    print(header)
    print('-' * len(header))
    for fact in sorted(runs, key=lambda item: item['run']):
        timings = ' '.join(f'{name[:4]}:{value:.0f}s' if value else f'{name[:4]}:-'
            for name, value in fact['seconds'].items())
        verified = {'validated': 'o', 'mismatch': 'k'}.get(fact['verified'], '-')
        print(f'{fact["run"]:54} {fact["status"]:9} {verified} '
            f'{str(fact["pair"]):26} {str(fact["framing"]):11} {str(fact["arm"]):13} '
            f'{fact["settings"]:14} {number(fact["persons"], 0):>7} '
            f'{number(fact["count_ok"]):>5} {number(fact["regions_ok"]):>3} '
            f'{number(fact["pck"]):>5} {number(fact["mpjpe"]):>5} '
            f'{number(fact["torso_ratio"]):>5} {number(fact["gap_error"]):>4} '
            f'{number(fact["identity"]):>5} {number(fact["swap"]):>4} {timings}')
    missing = sorted(set(available) - built)
    print(f'\npairs: {len(available)} available, {len(available) - len(missing)} with a run, '
        f'{len(missing)} without')
    if missing:
        print('without a run: ' + ', '.join(missing))
    if any(fact['status'] == 'abandoned' for fact in runs):
        print('abandoned runs are ones whose process died before writing a final status')
    print('\nv: o=stages validated, k=mismatch, -=not checked')
    print('count is the whole-image person check, reg the per-region one; YOLOX merges embracing '
          'bodies even on the real Hi4D photographs, so a zero in count on an entangled pair is '
          'often the detector, not the generator - read reg before blaming the picture')
    print('pose columns are only comparable when the detected person count matches; a run with '
          'extra bodies has its landmarks assigned to an arbitrary subset')
    groups = {}
    for fact in runs:
        if fact['status'] != 'complete':
            continue
        groups.setdefault((fact['arm'], fact['framing']), []).append(fact)
    print(f'\n{"arm":14} {"framing":11} runs count_ok pck mpjpe torso gap ident swap_rate')
    for (arm, framing), facts in sorted(groups.items(), key=lambda item: str(item[0])):
        comparable = [fact for fact in facts if fact['count_ok'] == 1.0]

        def mean(key):
            values = [fact[key] for fact in comparable if isinstance(fact[key], (int, float))]
            return sum(values) / len(values) if values else None
        print(f'{str(arm):14} {str(framing):11} {len(facts):4} '
            f'{sum(1 for fact in facts if fact["count_ok"] == 1.0):8.0f} '
            f'{number(mean("pck")):>4} {number(mean("mpjpe")):>5} {number(mean("torso_ratio")):>5} '
            f'{number(mean("gap_error")):>3} {number(mean("identity")):>5} '
            f'{sum(1 for fact in facts if fact["swap"] == 1.0) / len(facts):8.2f}')
    print('arm means are taken over runs that detected the right number of people; swap_rate is '
        'over every run in the group')


if __name__ == '__main__':
    if hasattr(sys, 'win_version'):
        raise SystemExit('workflow-multiple/runs lives on the Linux GPU host; run this there')
    main()
