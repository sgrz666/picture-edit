#!/usr/bin/env python3
"""Lock a balanced subset of existing qualified pairs; CPU/data only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    old = args.source_run / 'data'
    output = args.output / 'data'
    if output.exists():
        raise ValueError('Data lock directory already exists; verify it instead of overwriting')
    rows = read_rows(old / 'train_pairs.jsonl')
    supervision = json.loads((old / 'face_supervision.json').read_text())['records']
    sequence_status = {}
    excluded, eligible = [], []
    for row in rows:
        seq = row['sequence_id']
        if seq not in sequence_status:
            cache = torch.load(old / 'face_cache' / f'{seq}.pt', map_location='cpu', weights_only=False, mmap=True)
            sequence_status[seq] = {
                'indices': cache['stem_to_idx'], 'identity': cache['valid'].tolist(),
                'geometry': cache['geometry_valid'].tolist(),
            }
            del cache
        state = sequence_status[seq]
        si, ti = state['indices'][row['source_stem']], state['indices'][row['target_stem']]
        key = f"{seq}/{row['target_stem']}"
        if state['identity'][si] and state['geometry'][ti] and supervision[key]['by_resolution']['512']['valid']:
            eligible.append(row)
        else:
            excluded.append(row['pair_id'])
    grouped = defaultdict(list)
    rng = random.Random(1708)
    for row in eligible:
        grouped[row['source_video_group']].append(row)
    for group in grouped.values():
        rng.shuffle(group)
    chosen, used_targets, sequence_uses, sequence_frames = [], set(), Counter(), defaultdict(list)
    group_order = sorted(grouped)
    rng.shuffle(group_order)
    while len(chosen) < 512:
        made_progress = False
        for group_name in group_order:
            available = [row for row in grouped[group_name]
                         if (row['sequence_id'], row['target_stem']) not in used_targets]
            if not available:
                continue
            def priority(row):
                seq = row['sequence_id']
                frame = int(row['target_stem'])
                close = sum(abs(frame - prior) < 8 for prior in sequence_frames[seq])
                return sequence_uses[seq], close
            row = min(available, key=priority)
            grouped[group_name].remove(row)
            chosen.append(row)
            seq = row['sequence_id']
            used_targets.add((seq, row['target_stem']))
            sequence_uses[seq] += 1
            sequence_frames[seq].append(int(row['target_stem']))
            made_progress = True
            if len(chosen) == 512:
                break
        if not made_progress:
            raise ValueError(f'Only {len(chosen)} qualified unique targets available')
    val = read_rows(old / 'val24_pairs.jsonl')
    val_groups = {row['source_video_group'] for row in val}
    assert not val_groups.intersection(row['source_video_group'] for row in chosen)
    assert len(chosen) == len(used_targets) == 512
    output.mkdir(parents=True)
    write_rows(output / 'train512_pairs.jsonl', chosen)
    write_rows(output / 'common256_pairs.jsonl', chosen[:256])
    write_rows(output / 'branch256_pairs.jsonl', chosen[256:])
    shutil.copyfile(old / 'val24_pairs.jsonl', output / 'val24_pairs.jsonl')
    shutil.copyfile(old / 'iper12_pairs.jsonl', output / 'iper12_pairs.jsonl')
    monitor_tiktok = []
    for group in sorted(val_groups):
        indices = sorted([index for index, row in enumerate(val) if row['source_video_group'] == group],
                         key=lambda index: val[index].get('pose_distance', 0))
        monitor_tiktok.extend(indices[position] for position in [0, len(indices) // 2, len(indices) - 1])
    iper = read_rows(output / 'iper12_pairs.jsonl')
    monitor_iper, actors = [], set()
    for index, row in enumerate(iper):
        actor = str(row.get('appearance', row.get('source_appearance', ''))).split('_')[0]
        if actor not in actors:
            monitor_iper.append(index)
            actors.add(actor)
        if len(monitor_iper) == 3:
            break
    if len(monitor_iper) != 3:
        raise ValueError('Cannot select three distinct iPER actors')
    lock = {
        'seed': 1708, 'train_pairs': 512, 'unique_targets': len(used_targets),
        'input_eligible_pairs': len(eligible), 'excluded_pair_ids': excluded,
        'selection': 'round-robin original video groups; prefer unused segments and avoid target gaps <8 frames',
        'group_counts': dict(Counter(row['source_video_group'] for row in chosen)),
        'common_group_counts': dict(Counter(row['source_video_group'] for row in chosen[:256])),
        'branch_group_counts': dict(Counter(row['source_video_group'] for row in chosen[256:])),
        'video_groups': len({row['source_video_group'] for row in chosen}),
        'sequences': len({row['sequence_id'] for row in chosen}),
        'val_tiktok': len(val), 'val_iper': len(iper),
        'monitor_tiktok_indices': monitor_tiktok, 'monitor_iper_indices': monitor_iper,
        'train_val_original_groups_disjoint': True, 'final_sealed_test_used': False,
        'source_run': str(args.source_run),
        'features': str(old / 'face_cache'), 'supervision': str(old / 'face_supervision.json'),
        'cache_tiktok': str(args.source_run / 'native_cache_tiktok'),
        'cache_iper': str(args.source_run / 'native_cache_iper'),
        'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob('*.jsonl')},
    }
    (output / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(lock), flush=True)


if __name__ == '__main__':
    main()
