#!/usr/bin/env python3
"""Read-only index over workflow/runs: what exists, what verifies, what still lacks a run."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import read_json


ROOT = Path(__file__).resolve().parent
STAGE_DIRECTORY = {'geometry': 'geometry', 'prompt': 'prompt', 'generate': 'generation'}
CURRENT_SCHEMA = 'single_person_pose_workflow_v2'
CURRENT_STAGE_SCHEMA = 'single_person_pose_stage_v2'
# experiments/ is flat on the Windows mirror and grouped by family on sg (AGENTS.md 2.1).
PAIRS_CANDIDATES = ('experiments/qwen-2.1-test/data/testset/pairs',
    'experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs')


def default_pairs_dir():
    for relative in PAIRS_CANDIDATES:
        candidate = ROOT.parent / relative
        if candidate.is_dir():
            return candidate
    return ROOT.parent / PAIRS_CANDIDATES[0]


def read_optional(path):
    """common.read_json raises on a missing file; a run lacking an artifact is not an error here.

    Other sessions edit runs/ while this index reads it, so a file can also disappear between
    discovery and read.
    """
    try:
        return read_json(path)
    except (FileNotFoundError, IsADirectoryError):
        return None


def lock_held():
    """Whether some invocation currently owns the workflow, so 'running' can be real."""
    path = ROOT / '.workflow.lock'
    if not path.is_file():
        return False
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def stage_facts(run_dir):
    facts = {}
    for name, directory in STAGE_DIRECTORY.items():
        record = read_optional(run_dir / directory / 'stage.json')
        if record:
            metadata = record.get('metadata') or {}
            facts[name] = {'seconds': metadata.get('seconds'),
                'completed_at': record.get('completed_at_unix'),
                'status': record.get('status')}
    return facts


def prompt_origin(run_dir):
    selection = read_optional(run_dir / 'prompt' / 'generation_prompt.json')
    if selection:
        return {'kind': selection.get('source'), 'sha256': selection.get('sha256'),
            'path': selection.get('source_path')}
    # Runs made before prompt selection existed only recorded the text the generator read.
    generation = read_optional(run_dir / 'generation' / 'generation.json')
    if generation:
        legacy = generation.get('prompt_source') or {}
        hand_written = generation.get('prompt_sha256') != (
            read_optional(run_dir / 'prompt' / 'pe_result.json') or {}).get('compiled_prompt')
        return {'kind': legacy.get('source', 'pe_compiled' if not hand_written else 'manually edited'),
            'sha256': generation.get('prompt_sha256'), 'path': legacy.get('source_path')}
    return {'kind': None, 'sha256': None, 'path': None}


def run_facts(run_dir, live_writer, label=None):
    manifest = read_optional(run_dir / 'run.json')
    if not manifest:
        return None
    status = manifest.get('status')
    if status == 'running' and not live_writer:
        status = 'abandoned'
    stage_schema = (read_optional(run_dir / 'geometry' / 'stage.json') or {}).get('schema')
    if manifest.get('schema') != CURRENT_SCHEMA or stage_schema not in (None, CURRENT_STAGE_SCHEMA):
        status = f'{status}(legacy)'
    settings = manifest.get('settings', {})
    generation = read_optional(run_dir / 'generation' / 'generation.json') or {}
    evaluation = read_optional(run_dir / 'evaluation' / 'eval.json') or {}
    verification = manifest.get('verification')
    if isinstance(verification, str):
        verification = {'status': 'asserted', 'note': verification}
    return {'run': label or run_dir.name, 'status': status,
        'recorded_verification': (verification or {}).get('status'),
        'pair': Path(manifest.get('A_original', '')).parent.name,
        'resolution': settings.get('resolution'), 'steps': settings.get('steps'),
        'seed': settings.get('seed'),
        'final_size': generation.get('final_size'),
        'prompt': prompt_origin(run_dir),
        'stages': stage_facts(run_dir),
        'total_seconds': (manifest.get('finished_at_unix', 0) - manifest.get('started_at_unix', 0))
            if manifest.get('finished_at_unix') else None,
        'evaluation': evaluation or None,
        'error': manifest.get('error')}


def discover_runs(runs_dir):
    """Runs may sit one level under a batch directory (runs/<batch>/<run>/run.json)."""
    for manifest in sorted(runs_dir.rglob('run.json')):
        yield manifest.parent


def collect(runs_dir, pairs_dir):
    live_writer = lock_held()
    runs = [run_facts(path, live_writer, str(path.relative_to(runs_dir)))
        for path in discover_runs(runs_dir)]
    runs = [item for item in runs if item]
    covered = {item['pair'] for item in runs}
    pairs = sorted(path.name for path in pairs_dir.iterdir() if (path / 'A_source.png').is_file()) \
        if pairs_dir.is_dir() else []
    return runs, pairs, covered


def format_row(run):
    stages = run['stages']
    timings = ' '.join(f'{name[:4]}:{stages[name]["seconds"]:.0f}s'
        if stages.get(name, {}).get('seconds') is not None else f'{name[:4]}:-'
        for name in ('geometry', 'prompt', 'generate'))
    evaluation = run['evaluation']
    metrics = ''
    if evaluation:
        summary = evaluation.get('summary') or {}
        metrics = ' '.join(f'{key}={value:.3f}' for key, value in summary.items()
            if isinstance(value, (int, float)))
    prompt = run['prompt']
    size = 'x'.join(str(value) for value in run['final_size']) if run['final_size'] else '-'
    return (f'{run["run"]:44} {run["status"]:16} {str(run["recorded_verification"]):9} '
        f'{run["pair"]:12} {run["resolution"]}/{run["steps"]}/{run["seed"]} '
        f'{str(prompt["kind"] or "no generation"):16} {size:11} {timings:34} {metrics}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--pairs', type=Path, default=default_pairs_dir())
    parser.add_argument('--json', action='store_true', help='Machine-readable output')
    args = parser.parse_args()
    runs, pairs, covered = collect(args.runs, args.pairs)
    if args.json:
        print(json.dumps({'runs': runs, 'pairs': pairs,
            'pairs_without_run': sorted(set(pairs) - covered)}, ensure_ascii=False, indent=2))
        return
    if not pairs:
        print(f'WARNING: no A_source.png under {args.pairs}; "pairs without a run" is meaningless '
            f'here, pass --pairs')
    header = (f'{"run":44} {"status":16} {"verify":9} {"pair":12} res/steps/seed '
        f'{"prompt source":16} {"final":11} {"stage seconds":34} evaluation')
    print(header)
    print('-' * len(header))
    for run in sorted(runs, key=lambda item: item['stages'].get('generate', {})
        .get('completed_at') or 0, reverse=True):
        print(format_row(run))
    missing = sorted(set(pairs) - covered)
    print(f'\npairs: {len(pairs)} known, {len(pairs) - len(missing)} with a run, '
        f'{len(missing)} without: {missing or "none"}')
    newest = max((run['stages'].get('generate', {}).get('completed_at') or 0 for run in runs),
        default=None)
    latest = read_optional(ROOT / 'latest_run.json')
    if latest:
        print(f'latest_run.json -> {Path(latest["run_dir"]).name} ({latest["status"]})')
    if newest:
        print(f'newest generation: {time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))}')


if __name__ == '__main__':
    if os.name != 'posix':
        raise SystemExit('workflow/runs lives on the Linux GPU host; run this there')
    main()
