#!/usr/bin/env python3
"""Extract embeddings for one evaluation condition across all backends."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facekit  # noqa: E402

CONDITIONS = {
    'clean': {'jitter': None, 'degrade': 'clean'},
    'align_j5': {'jitter': {'rotation': 5.0, 'scale': 0.04, 'shift': 3.0}, 'degrade': 'clean'},
    'align_j10': {'jitter': {'rotation': 10.0, 'scale': 0.08, 'shift': 6.0}, 'degrade': 'clean'},
    'blur1.5': {'jitter': None, 'degrade': 'blur1.5'},
    'lowres48': {'jitter': None, 'degrade': 'lowres48'},
    'lowres32': {'jitter': None, 'degrade': 'lowres32'},
    'jpeg20': {'jitter': None, 'degrade': 'jpeg20'},
    'occl': {'jitter': None, 'degrade': 'occl'},
    'genlike': {'jitter': None, 'degrade': 'genlike'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--condition', required=True, choices=sorted(CONDITIONS))
    parser.add_argument('--backends', nargs='+',
                        default=['glintr100', 'w600k_r50', 'lvface_b', 'adaface_kprpe'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--flip-tta', action='store_true',
                        help='average the embedding with its horizontal mirror '
                             '(LVFace reports IJB-C with this augmentation)')
    args = parser.parse_args()

    spec = CONDITIONS[args.condition]
    bundle = np.load(args.crops, allow_pickle=False)
    ids, crops, keypoints = bundle['ids'], bundle['crops'], bundle['keypoints']
    if args.limit:
        ids, crops, keypoints = ids[:args.limit], crops[:args.limit], keypoints[:args.limit]
    print(f'{len(ids)} crops, condition={args.condition}', flush=True)

    if spec['jitter']:
        warped, moved = [], []
        for key, crop, points in zip(ids, crops, keypoints):
            matrix = facekit.jitter_matrix(facekit.stable_seed(str(key)),
                                           spec['jitter']['rotation'], spec['jitter']['scale'],
                                           spec['jitter']['shift'])
            warped.append(cv2.warpAffine(crop, matrix, (112, 112),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            homogeneous = np.concatenate([points.astype(np.float64),
                                          np.ones((points.shape[0], 1))], axis=1)
            moved.append((homogeneous @ matrix.T).astype(np.float32))
        crops, keypoints = np.stack(warped), np.stack(moved)

    crops = np.stack([facekit.degrade(crop, spec['degrade'], facekit.stable_seed(str(key), 7))
                      for key, crop in zip(ids, crops)])

    backends = facekit.build_backends(args.backends, device=args.device)
    payload = {'ids': ids, 'condition': args.condition, 'spec': json.dumps(spec),
               'flip_tta': args.flip_tta}
    if args.flip_tta:
        mirrored = np.stack([crop[:, ::-1].copy() for crop in crops])
        flipped_points = np.stack([np.stack([112.0 - points[:, 0], points[:, 1]], axis=1)
                                   [[1, 0, 2, 4, 3]] for points in keypoints])
    timings = {}
    for name, backend in backends.items():
        start = time.time()
        if name == 'adaface_kprpe':
            embeddings = backend.embed(list(crops), list(keypoints))
            if args.flip_tta:
                mirrored_embeddings = backend.embed(list(mirrored), list(flipped_points))
                embeddings = [facekit._l2(a + b) for a, b in zip(embeddings, mirrored_embeddings)]
        else:
            embeddings = backend.embed(list(crops))
            if args.flip_tta:
                mirrored_embeddings = backend.embed(list(mirrored))
                embeddings = [facekit._l2(a + b) for a, b in zip(embeddings, mirrored_embeddings)]
        elapsed = time.time() - start
        payload[name] = np.stack(embeddings)
        timings[name] = {'seconds': elapsed, 'ms_per_face': 1000 * elapsed / max(1, len(ids)),
                         'dim': int(embeddings[0].size)}
        print(f'{name}: {elapsed:.1f}s ({timings[name]["ms_per_face"]:.1f} ms/face) '
              f'dim={timings[name]["dim"]}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    (args.out.with_suffix('.timing.json')).write_text(json.dumps(timings, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
