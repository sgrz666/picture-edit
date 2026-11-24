#!/usr/bin/env python3
"""Score saved Native RL candidate groups with a frozen SpatialReward model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    from src.pose_control.qwen21_full.native_rl.spatial_reward import spatial_target
    from src.pose_control.qwen21_full.native_rl.spatial_worker import SpatialWorker
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition, verified
    from src.pose_control.qwen21_full.native_v62.training.io import read_json_ref, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--python', type=Path, help='Interpreter for the isolated model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--references', choices=('rgb', 'guides'), default='rgb')
    parser.add_argument('--mode', choices=('official', 'complementary'), default='official')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-pixels', type=int, default=262144)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output already exists')

    data, ref = read_json_ref({'path': str(args.manifest.resolve())})
    groups = data.get('groups')
    if data.get('schema') != 'native_rl_spatial_candidates_v1' or not isinstance(groups, list) or not groups:
        raise ValueError('Expected a nonempty Native SpatialReward candidate manifest')
    prepared = []
    for group in groups:
        target = spatial_target(read_condition(group['condition']), args.references, slot_sources=group.get('slot_sources'))
        paths = [verified(candidate) for candidate in group['candidates']]
        if not paths:
            raise ValueError('SpatialReward candidate group is empty')
        prepared.append((group, target, paths))

    results = []
    with SpatialWorker(args.model, python=args.python, device=args.device, mode=args.mode,
                       references=args.references, batch_size=args.batch_size, max_pixels=args.max_pixels,
                       timeout=args.timeout) as worker:
        print(json.dumps({'event': 'spatial_ready', 'identity': worker.identity}), flush=True)
        for group, target, paths in prepared:
            rows = worker.score_group(target, paths)
            results.append({'condition': group['condition'], 'spatial_target': target.record(),
                            'candidates': [{**result, 'file': asset} for result, asset in
                                           zip(rows, group['candidates'], strict=True)]})
            print(json.dumps({'event': 'spatial_group_scored', 'group': len(results), 'candidates': len(paths)}), flush=True)
        report = {'schema': 'native_rl_spatial_scores_v1', 'manifest': ref, 'evaluator': worker.identity, 'groups': results}
    write_json(args.output, report)


if __name__ == '__main__':
    main()
