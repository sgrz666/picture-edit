#!/usr/bin/env python3
"""Two-person binding check on the beach14 generations.

For every usable generated frame we know the left/right person assignment from
the source photo. The 2x2 cosine matrix tells us whether the model keeps the
two identities apart, and whether the production "largest face" rule would have
bound the wrong person.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ('glintr100', 'w600k_r50', 'lvface_b', 'adaface_kprpe')


def two_people(record, min_side=32.0):
    detections = record['detections']
    if len(detections) < 2:
        return None
    index_of = {id(detection): index for index, detection in enumerate(detections)}
    top = sorted(detections, key=lambda item: item['side'], reverse=True)[:2]
    ordered = sorted(top, key=lambda item: (item['bbox'][0] + item['bbox'][2]) / 2.0)
    left, right = ordered
    if right['bbox'][0] - left['bbox'][2] <= 0 or min(left['side'], right['side']) < min_side:
        return None
    return [(index_of[id(detection)], detection) for detection in ordered]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--emb', type=Path, required=True)
    parser.add_argument('--condition', default='clean')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    scan = json.loads(args.scan.read_text(encoding='utf-8'))
    bundle = np.load(args.emb / f'{args.condition}.npz', allow_pickle=False)
    ids = {key: index for index, key in enumerate(bundle['ids'].tolist())}

    source = next(record for record in scan['records']
                  if record['source'] == 'beach14' and record['role'] == 'source')
    source_people = two_people(source)
    if not source_people:
        raise SystemExit('source photo needs two separated faces')
    source_keys = [f'{source["image_id"]}:{index}' for index, _ in source_people]

    report = {}
    for model in MODELS:
        if model not in bundle.files:
            continue
        embeddings = bundle[model]
        diagonal, off, largest_rule_wrong, same, diff = [], [], 0, [], []
        frames = 0
        for record in scan['records']:
            if record['source'] != 'beach14' or record['role'] != 'generated':
                continue
            people = two_people(record)
            if not people:
                continue
            frames += 1
            keys = [f'{record["image_id"]}:{index}' for index, _ in people]
            matrix = np.array([[float(embeddings[ids[a]] @ embeddings[ids[b]]) for b in keys]
                               for a in source_keys])
            diagonal.append(matrix[0, 0] + matrix[1, 1])
            off.append(matrix[0, 1] + matrix[1, 0])
            # largest-face rule: the biggest detection is taken as person 0
            if people[0][0] != 0:
                largest_rule_wrong += 1
            same.extend([matrix[0, 0], matrix[1, 1]])
            diff.extend([matrix[0, 1], matrix[1, 0]])
        diagonal, off = np.asarray(diagonal), np.asarray(off)
        report[model] = {
            'frames': frames,
            'binding_accuracy': float((diagonal > off).mean()),
            'mean_diagonal': float(diagonal.mean() / 2),
            'mean_offdiagonal': float(off.mean() / 2),
            'margin': float((diagonal - off).mean() / 2),
            'largest_face_rule_wrong_rate': float(largest_rule_wrong / max(1, frames)),
            'same_mean': float(np.mean(same)), 'diff_mean': float(np.mean(diff)),
        }
        print(model, json.dumps(report[model]), flush=True)

    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
