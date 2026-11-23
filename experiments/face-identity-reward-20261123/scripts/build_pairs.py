#!/usr/bin/env python3
"""Build same/different identity pairs from the scan, with quality strata."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def size_bucket(side):
    if side < 48:
        return 'tiny'
    if side < 96:
        return 'small'
    return 'large'


def pose_bucket(yaw):
    value = abs(yaw)
    if value < 0.15:
        return 'frontal'
    if value < 0.30:
        return 'half'
    return 'profile'


def sharp_bucket(sharpness):
    return 'blurry' if sharpness < 50.0 else 'sharp'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--tiktok-positives', type=int, default=2)
    parser.add_argument('--iper-positives', type=int, default=3)
    parser.add_argument('--iper-cross', type=int, default=6)
    parser.add_argument('--min-side', type=float, default=40.0)
    args = parser.parse_args()

    scan = json.loads(args.scan.read_text(encoding='utf-8'))
    rng = random.Random(args.seed)

    by_identity = defaultdict(list)
    for record in scan['records']:
        if record['tier'] != 1 or not record['detections']:
            continue
        face = record['detections'][0]
        if face['side'] < args.min_side:
            continue
        by_identity[record['identity']].append({
            'key': f'{record["image_id"]}:0', 'identity': record['identity'],
            'source': record['source'], 'group': record['group'], 'role': record['role'],
            'image': record['image'], 'side': face['side'], 'score': face['score'],
            'quality': face['quality']})

    identities = sorted(by_identity)
    print(f'identities={len(identities)} faces={sum(len(v) for v in by_identity.values())}')

    pairs = []

    def stratum(a, b):
        return {'size': size_bucket(min(a['side'], b['side'])),
                'pose': pose_bucket(max(abs(a['quality']['yaw_proxy']),
                                        abs(b['quality']['yaw_proxy']))),
                'sharp': sharp_bucket(min(a['quality']['sharpness'],
                                          b['quality']['sharpness']))}

    def add(kind, a, b, tier, dataset, note=''):
        pairs.append({'kind': kind, 'tier': tier, 'dataset': dataset,
                      'a': dict(a), 'b': dict(b),
                      'stratum': stratum(a, b), 'note': note})

    # ---- positives: same identity, different frame or clip --------------------
    # Cross-clip pairs (different clothing / background / camera) are the closest
    # real-data analogue of source->generated, so they are sampled first and are
    # allowed to reuse one representative face per clip.
    for identity in identities:
        faces = by_identity[identity]
        source = faces[0]['source']
        quota = args.tiktok_positives if source == 'tiktok' else args.iper_positives
        cross_quota = args.iper_cross if source == 'iper' else 0
        groups = defaultdict(list)
        for face in faces:
            groups[face['group']].append(face)
        keys = sorted(groups)
        cross = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for left in groups[keys[i]]:
                    for right in groups[keys[j]]:
                        cross.append((left, right))
        rng.shuffle(cross)
        chosen, used = [], set()
        for a, b in cross:
            if len(chosen) >= cross_quota:
                break
            if a['key'] in used or b['key'] in used:
                continue
            used.update((a['key'], b['key']))
            chosen.append((a, b))
        cross_count = len(chosen)
        candidates = []
        for group, members in groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.append((members[i], members[j]))
        rng.shuffle(candidates)
        for a, b in candidates:
            if len(chosen) >= cross_count + quota:
                break
            if a['key'] in used or b['key'] in used:
                continue
            used.update((a['key'], b['key']))
            chosen.append((a, b))
        for a, b in chosen:
            note = 'cross_clip' if a['group'] != b['group'] else 'same_clip'
            add('same', a, b, 1, source, note)

    # ---- negatives: different identities, size-matched where possible --------
    pools = defaultdict(list)
    for identity in identities:
        for face in by_identity[identity]:
            pools[size_bucket(face['side'])].append(face)
    positives = [pair for pair in pairs if pair['kind'] == 'same']
    for pair in positives:
        anchor = pair['a']
        bucket = size_bucket(anchor['side'])
        pool = [face for face in pools[bucket] if face['identity'] != anchor['identity']]
        if not pool:
            pool = [face for identity in identities for face in by_identity[identity]
                    if face['identity'] != anchor['identity']]
        partner = rng.choice(pool)
        add('diff', anchor, partner, 1, anchor['source'], 'size_matched')

    # ---- tier 2: beach14 real -> generated ----------------------------------
    # Both the source photo and the generations keep the man on the left and the
    # woman on the right, so left/right face order is a model-independent person
    # assignment. Detection order alone is size-based and would mislabel pairs.
    def two_people(record):
        detections = record['detections']
        if len(detections) < 2:
            return None
        index_of = {id(detection): index for index, detection in enumerate(detections)}
        top = sorted(detections, key=lambda item: item['side'], reverse=True)[:2]
        ordered = sorted(top, key=lambda item: (item['bbox'][0] + item['bbox'][2]) / 2.0)
        left, right = ordered
        # Reject blended generations where the two faces overlap or one is tiny.
        if right['bbox'][0] - left['bbox'][2] <= 0:
            return None
        if min(left['side'], right['side']) < 32.0:
            return None
        return [(index_of[id(detection)], detection) for detection in ordered]

    beach = [record for record in scan['records'] if record['source'] == 'beach14']
    source_record = next((record for record in beach if record['role'] == 'source'), None)
    generated = [record for record in beach if record['role'] == 'generated']
    source_people = two_people(source_record) if source_record else None
    if source_people:
        source_faces = []
        for person, (index, face) in enumerate(source_people):
            source_faces.append({'key': f'{source_record["image_id"]}:{index}',
                                 'identity': f'beach:{person}', 'group': 'source', 'role': 'source',
                                 'image': source_record['image'], 'side': face['side'],
                                 'quality': face['quality'], 'score': face['score']})
        for record in generated:
            people = two_people(record)
            if not people:
                continue
            for person, (index, face) in enumerate(people):
                target = {'key': f'{record["image_id"]}:{index}', 'identity': f'beach:{person}',
                          'group': record['group'], 'role': 'generated',
                          'image': record['image'], 'side': face['side'],
                          'quality': face['quality'], 'score': face['score']}
                for source_face in source_faces:
                    kind = 'same' if source_face['identity'] == target['identity'] else 'diff'
                    add(kind, source_face, target, 2, 'beach14',
                        'left_right_match' if kind == 'same' else 'cross_person')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = defaultdict(int)
    for pair in pairs:
        summary[f'tier{pair["tier"]}_{pair["kind"]}'] += 1
    args.out.write_text(json.dumps({'schema': 1, 'seed': args.seed, 'counts': dict(summary),
                                    'pairs': pairs}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(summary), indent=2))


if __name__ == '__main__':
    main()
