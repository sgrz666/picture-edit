#!/usr/bin/env python3
"""Score saved Native RL candidate groups with a frozen SpatialReward model."""

import argparse
import json
from pathlib import Path
from statistics import pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    from src.pose_control.qwen21_full.native_rl.spatial_reward import DEFAULT_MODE, FUSION_MODES, spatial_target
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
    parser.add_argument('--mode', choices=FUSION_MODES, default=DEFAULT_MODE)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-pixels', type=int, default=262144)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--repeats', type=int, default=1, help='Repeat on identical saved candidates to measure score drift')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output already exists')
    if args.repeats < 1:
        parser.error('--repeats must be positive')

    data, ref = read_json_ref({'path': str(args.manifest.resolve())})
    groups = data.get('groups')
    if data.get('schema') != 'native_rl_spatial_candidates_v1' or not isinstance(groups, list) or not groups:
        raise ValueError('Expected a nonempty Native SpatialReward candidate manifest')
    prepared = []
    for group in groups:
        target = spatial_target(read_condition(group['condition']), args.references, slot_sources=group.get('slot_sources'))
        candidates = group['candidates']
        if not candidates:
            raise ValueError('SpatialReward candidate group is empty')
        for candidate in candidates:
            verified(candidate)
        prepared.append((group, target))

    results = []
    with SpatialWorker(args.model, python=args.python, device=args.device, mode=args.mode,
                       references=args.references, batch_size=args.batch_size, max_pixels=args.max_pixels,
                       timeout=args.timeout) as worker:
        print(json.dumps({'event': 'spatial_ready', 'identity': worker.identity}), flush=True)
        for group, target in prepared:
            candidates = group['candidates']
            rows = worker.score_group(target, candidates)
            if args.repeats > 1:
                repeated = [rows] + [worker.score_group(target, candidates) for _ in range(args.repeats - 1)]
                rows = [{**row, 'repeat_rewards': [other['reward'] for other in copies],
                         'repeat_reward_std': pstdev(other['reward'] for other in copies)}
                        for row, copies in zip(rows, zip(*repeated, strict=True), strict=True)]
            results.append({'condition': group['condition'], 'spatial_target': target.record(),
                            'candidates': [{**result, 'file': asset} for result, asset in
                                           zip(rows, group['candidates'], strict=True)]})
            print(json.dumps({'event': 'spatial_group_scored', 'group': len(results), 'candidates': len(candidates)}), flush=True)
        report = {'schema': 'native_rl_spatial_scores_v1', 'manifest': ref, 'evaluator': worker.identity,
                  'repeats': args.repeats, 'groups': results}
    write_json(args.output, report)


if __name__ == '__main__':
    main()
