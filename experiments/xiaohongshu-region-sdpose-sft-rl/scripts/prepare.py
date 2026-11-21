"""Lock a note-disjoint Xiaohongshu pilot, without modifying the dataset."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    return ref(path)


def split_pairs(pairs, frames, seed):
    notes = sorted({p['note_id'] for p in pairs}); parent = {n: n for n in notes}; content = {}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]; n = parent[n]
        return n
    for pair in sorted(pairs, key=lambda x: x['pair_id']):
        if pair['disk_status'] != 'complete':
            raise ValueError('Incomplete pair: ' + pair['pair_id'])
        for key in ('source_frame_id', 'target_frame_id'):
            frame = frames[pair[key]]; note = pair['note_id']
            if frame['clip_id'] != note:
                raise ValueError('Pair RGB belongs to a different note')
            digest = frame['rgb_sha256']
            if digest in content:
                parent[find(note)] = find(content[digest])
            else:
                content[digest] = note
    components = defaultdict(list)
    for note in notes: components[find(note)].append(note)
    groups = sorted(components.values()); random.Random(seed).shuffle(groups)
    if len(groups) < 3: raise ValueError('At least three independent note components required')
    nv = max(1, round(.1 * len(groups))); nt = max(1, round(.1 * len(groups)))
    mapping = {n: ('val' if i < nv else 'test' if i < nv + nt else 'train')
               for i, group in enumerate(groups) for n in group}
    return {**{s: [{**p, 'split': s} for p in sorted(pairs, key=lambda x: x['pair_id'])
                   if mapping[p['note_id']] == s] for s in ('train', 'val', 'test')},
            'note_splits': mapping, 'note_components': groups}


def select_pilot(pairs, poses, count, seed):
    from collections import defaultdict
    groups = defaultdict(list); rng = random.Random(seed)
    for p in sorted(pairs, key=lambda x: x['pair_id']):
        valid = poses[p['target_frame_id']]['valid_mask134']
        if len(valid) != 134: raise ValueError('SDPose geometry-order 134 slots required')
        if sum(valid[2:14]) < 8 or max(sum(valid[92:113]), sum(valid[113:134])) < 8: continue
        groups[p['note_id']].append(p)
    notes = sorted(groups); rng.shuffle(notes)
    if len(notes) < count: raise ValueError(f'Too few hand/body qualified notes: {len(notes)} < {count}')
    chosen = []
    for note in notes[:count]:
        rows = list(groups[note]); rng.shuffle(rows)
        rows.sort(key=lambda p: min(sum(poses[p['target_frame_id']]['valid_mask134'][92:113]),
                                  sum(poses[p['target_frame_id']]['valid_mask134'][113:134])), reverse=True)
        chosen.append(rows[0])
    return sorted(chosen, key=lambda p: p['pair_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data/shangguanrz/datasets/xiaohongshu(single)'))
    parser.add_argument('--train-pairs', type=int, default=78); parser.add_argument('--val-pairs', type=int, default=10)
    args = parser.parse_args(); root = args.root.resolve()
    if not str(root).startswith('/data/shangguanrz/'): raise ValueError('Persistent run root required')
    if (root/'data/selection.json').exists(): raise FileExistsError('Selection already locked')
    frames = {r['frame_id']: r for r in map(json.loads, (args.data_root/'frames/frames.jsonl').read_text().splitlines())}
    pairs = list(map(json.loads, (args.data_root/'pairs/pairs.jsonl').read_text().splitlines()))
    split = split_pairs(pairs, frames, 42)
    poses = {fid: read(args.data_root/'geometry/skeleton/sdpose_2d_kp'/f['clip_id']/(fid+'_keypoints.json'))
             for fid, f in frames.items()}
    chosen = [*select_pilot(split['train'], poses, args.train_pairs, 42),
              *select_pilot(split['val'], poses, args.val_pairs, 43)]
    ids = {p[k] for p in chosen for k in ('source_frame_id', 'target_frame_id')}
    locked = {}
    for fid in sorted(ids):
        f = frames[fid]; kp = args.data_root/'geometry/skeleton/sdpose_2d_kp'/f['clip_id']/(fid+'_keypoints.json')
        if sha(f['rgb_path']) != f['rgb_sha256']: raise ValueError('Original RGB changed: '+fid)
        pose = poses[fid]
        if pose['slot_order'] != 'geometry' or (pose['height'], pose['width']) != tuple(f['original_hw']):
            raise ValueError('SDPose order/original Canvas mismatch')
        guide = args.data_root/'geometry/skeleton/sdpose_2d'/f['clip_id']/(fid+'_skeleton.png')
        locked[fid] = {**f, 'sdpose': ref(kp), 'skeleton': ref(guide), 'split': split['note_splits'][f['clip_id']]}
    write(root/'data/full_split.json', split)
    receipt = {'schema': 'xiaohongshu_region_selection_v1', 'data_root': str(args.data_root), 'seed': 42,
        'pairs': chosen, 'frames': locked, 'dataset_inputs': [ref(args.data_root/p) for p in ('frames/frames.jsonl','pairs/pairs.jsonl')],
        'full_split_counts': {s: {'pairs': len(split[s]), 'notes': len({p['note_id'] for p in split[s]})} for s in ('train','val','test')},
        'pilot_counts': {s: sum(p['split']==s for p in chosen) for s in ('train','val')},
        'policy': 'note-disjoint; duplicate-image SHA components cannot cross split; no val/test training rewards',
        'geometry_status': 'not_prepared', 'canvas_long_side': 512}
    write(root/'data/selection.json', receipt)
    print(json.dumps({'selected': receipt['pilot_counts'], 'unique_frames': len(locked), 'full_split_counts': receipt['full_split_counts']}), flush=True)


if __name__ == '__main__': main()
