#!/usr/bin/env python3
"""Experiment CLI. Heavy model imports are confined to GPU subcommands."""
import argparse
import json
from pathlib import Path
import re
import traceback

from core import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pilot.json')
    parser.add_argument('command', choices=['prepare-data', 'cache', 'train', 'infer', 'evaluate'])
    parser.add_argument('--name', required=True, help='New immutable data/cache/run/output name')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--inference', type=Path)
    parser.add_argument('--limit', type=int, help='Limit normal inference cases for the real GPU preflight')
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.name):
        parser.error('--name must contain only letters, digits, hyphens or underscores')
    required = {'cache': ['data'], 'train': ['data', 'cache'], 'infer': ['data', 'cache'],
                'evaluate': ['inference']}
    for name in required.get(args.command, []):
        if getattr(args, name) is None:
            parser.error(f'{args.command} requires --{name}')
    config = read_json(args.config)
    if args.command == 'prepare-data':
        from prepare import prepare_data
        result = prepare_data(config, args.name)
    elif args.command == 'cache':
        from runtime import cache_data
        result = cache_data(config, args.data, args.name)
    elif args.command == 'train':
        from runtime import train
        result = train(config, args.data, args.cache, args.name, args.until_step, args.resume)
    elif args.command == 'infer':
        from inference import infer
        result = infer(config, args.data, args.cache, args.name, args.adapter, args.limit)
    else:
        from inference import evaluate
        result = evaluate(config, args.inference, args.name)
    print(json.dumps({'result': str(result)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
