#!/usr/bin/env python3
"""Run the approved pilot to completion, with an immutable runtime snapshot and stage logs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from core import digest, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.tag):
        raise ValueError('tag must be plain letters/digits/underscore/hyphen')
    config = read_json(args.config)
    root = Path(config['workspace_root'])
    control = root / 'output' / f'campaign_{args.tag}'
    control.mkdir(parents=True, exist_ok=False)
    snapshot = control / 'source_runtime'
    snapshot.mkdir()
    files = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        target = snapshot / path.name
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        files[path.name] = sha256(target)
    write_json(control / 'config.json', config)
    write_json(control / 'provenance.json', {'pid': os.getpid(), 'python': sys.executable,
               'script_sha256': files, 'config_sha256': digest(config),
               'data_index_sha256': sha256(args.data / 'index.json'),
               'cache_index_sha256': sha256(args.cache / 'index.json')})
    run = [sys.executable, '-u', str(snapshot / 'run.py'), '--config', str(control / 'config.json')]
    data_args = ['--data', str(args.data), '--cache', str(args.cache)]
    pilot = f'pilot_{args.tag}'
    probe = f'preflight_{args.tag}'
    inference = f'comparison_{args.tag}'
    stages = [
        ('real_backward_probe', [*run, 'train', '--name', probe, *data_args, '--until-step', '1']),
        ('native_reload_and_inference_probe', [*run, 'infer', '--name', f'probe_inference_{args.tag}',
          *data_args, '--adapter', str(root / 'output' / probe / 'checkpoint-000001'), '--limit', '1']),
        ('train_first_100', [*run, 'train', '--name', pilot, *data_args, '--until-step', '100']),
        ('resume_to_300', [*run, 'train', '--name', pilot, *data_args,
          '--resume', str(root / 'output' / pilot / 'checkpoint-000100')]),
        ('paired_inference', [*run, 'infer', '--name', inference, *data_args,
          '--adapter', str(root / 'output' / pilot / 'checkpoint-000300')]),
        ('cpu_evaluation', [*run, 'evaluate', '--name', f'pilot_report_{args.tag}',
          '--inference', str(root / 'output' / inference)])]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = config['nvml_compat'] + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'], env['MKL_NUM_THREADS'] = '4', '4'
    with (control / 'events.jsonl').open('x', encoding='utf-8', buffering=1) as events:
        def event(kind, **values):
            row = {'time': time.time(), 'event': kind, **values}
            text = json.dumps(row, ensure_ascii=False)
            events.write(text + '\n')
            print(text, flush=True)
        for stage, command in stages:
            event('started', stage=stage, command=command)
            with (control / f'{stage}.log').open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                event('child', stage=stage, pid=child.pid)
                code = child.wait()
            event('completed' if code == 0 else 'failed', stage=stage, returncode=code)
            if code:
                raise SystemExit(code)
        event('campaign_completed', report=str(root / 'report' / f'pilot_report_{args.tag}'))


if __name__ == '__main__':
    main()
