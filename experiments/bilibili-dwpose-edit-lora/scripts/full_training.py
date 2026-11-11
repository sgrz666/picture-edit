#!/usr/bin/env python3
"""Launch full-data frozen caching, a real health check, one full epoch, then comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from core import digest, load_training_rows, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.tag):
        raise ValueError('Invalid run tag')
    config = read_json(args.config)
    rows = load_training_rows(config, args.data)
    if len(rows) != config['steps'] or config.get('initialization') != 'base_and_fresh_lora':
        raise ValueError('This launch is explicitly one complete epoch with fresh LoRA')
    root = Path(config['workspace_root'])
    control = root / 'output' / f'full_campaign_{args.tag}'
    control.mkdir(parents=True, exist_ok=False)
    snapshot = control / 'source_runtime'
    snapshot.mkdir()
    fingerprints = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        target = snapshot / path.name
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        fingerprints[path.name] = sha256(target)
    write_json(control / 'config.json', config)
    write_json(control / 'provenance.json', {'pid': os.getpid(), 'created_at': time.time(),
               'python': sys.executable, 'script_sha256': fingerprints, 'config_sha256': digest(config),
               'data': str(args.data), 'data_index_sha256': sha256(args.data / 'index.json'),
               'train_manifest_sha256': sha256(args.data / config['training_manifest']),
               'train_pairs': len(rows), 'train_videos': len({r['group_id'] for r in rows}),
               'initialization': config['initialization'], 'reuse_cache': str(args.reuse_cache)})
    run = [sys.executable, '-u', str(snapshot / 'run.py'), '--config', str(control / 'config.json')]
    cache_name, train_name = f'cache_full_{args.tag}', f'train_full_{args.tag}'
    cache_path = root / 'cache' / cache_name
    data_args = ['--data', str(args.data), '--cache', str(cache_path)]
    cache_command = [*run, 'cache', '--name', cache_name, '--data', str(args.data)]
    if args.reuse_cache:
        cache_command += ['--reuse-cache', str(args.reuse_cache)]
    inference_name = f'full_comparison_{args.tag}'
    stages = [
        ('full_cache', cache_command),
        ('full_data_backward_and_validation_check', [*run, 'train', '--name', f'preflight_full_{args.tag}',
          *data_args, '--until-step', '1']),
        ('full_train', [*run, 'train', '--name', train_name, *data_args]),
        ('full_comparison', [*run, 'infer', '--name', inference_name, *data_args,
          '--adapter', str(root / 'output' / train_name / f'checkpoint-{config["steps"]:06d}')]),
        ('full_evaluation', [*run, 'evaluate', '--name', f'full_report_{args.tag}',
          '--inference', str(root / 'output' / inference_name)])]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = config['nvml_compat'] + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'], env['MKL_NUM_THREADS'] = '4', '4'
    with (control / 'events.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        def event(kind, **fields):
            row = {'time': time.time(), 'event': kind, **fields}
            message = json.dumps(row, ensure_ascii=False)
            stream.write(message + '\n')
            print(message, flush=True)
        for stage, command in stages:
            event('started', stage=stage, command=command)
            with (control / f'{stage}.log').open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                event('child', stage=stage, pid=child.pid)
                code = child.wait()
            event('completed' if code == 0 else 'failed', stage=stage, returncode=code)
            if code:
                raise SystemExit(code)
        event('full_campaign_completed', report=str(root / 'report' / f'full_report_{args.tag}'))


if __name__ == '__main__':
    main()
