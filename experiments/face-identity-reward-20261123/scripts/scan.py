#!/usr/bin/env python3
"""Scan candidate images: detect faces, cache aligned crops and quality stats.

Sources
  tiktok  340 sequences (one identity each), frames spread over the clip
  iper    103 appearance clips over 32 subjects, source + target frames
  beach14 56 native-v7 generations of one dual-person source (tier-2 only)

Outputs (under --out)
  scan.json      one record per image: source, identity, detections, quality
  crops.npz      aligned 112x112 crops + in-crop keypoints for the base condition
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facekit  # noqa: E402

TIKTOK_ROOT = Path('/data/shangguanrz/pic-edit-datasets-test/TikTokDataset/TikTok_dataset/TikTok_dataset')
IPER_ROOT = Path('/data/shangguanrz/pic-edit-datasets-test/iPER/iper_sampled_6src64tgt')
BEACH_ROOT = Path('/data/shangguanrz/pic-edit-experiments/two/workflow-v2/experiments/2.beach14')
BEACH_SOURCE = BEACH_ROOT / 'results/_shared_inputs/A_reference.png'


def spread(items, count):
    if len(items) <= count:
        return list(items)
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in sorted(set(picks.tolist()))]


def tiktok_images(per_sequence):
    records = []
    for sequence in sorted(path.name for path in TIKTOK_ROOT.iterdir() if path.is_dir()):
        frames = sorted((TIKTOK_ROOT / sequence / 'images').glob('*.png'))
        for frame in spread(frames, per_sequence):
            records.append({'image': str(frame), 'source': 'tiktok',
                            'identity': f'tt:{sequence}', 'group': sequence,
                            'role': 'frame', 'tier': 1})
    return records


def iper_images(sources, targets):
    records = []
    for appearance in sorted(path.name for path in IPER_ROOT.iterdir()
                             if path.is_dir() and '_' in path.name):
        subject = appearance.split('_')[0]
        root = IPER_ROOT / appearance / 'rgb_1024'
        for frame in spread(sorted((root / 'source_standard').glob('*.png')), sources):
            records.append({'image': str(frame), 'source': 'iper',
                            'identity': f'iper:{subject}', 'group': appearance,
                            'role': 'source', 'tier': 1})
        for frame in spread(sorted((root / 'target').glob('*.png')), targets):
            records.append({'image': str(frame), 'source': 'iper',
                            'identity': f'iper:{subject}', 'group': appearance,
                            'role': 'target', 'tier': 1})
    return records


def beach_images():
    records = [{'image': str(BEACH_SOURCE), 'source': 'beach14', 'identity': 'beach',
                'group': 'source', 'role': 'source', 'tier': 2}]
    for target in sorted(BEACH_ROOT.joinpath('results_native_v7').glob('t*')):
        for layout in ('native_2ref', 'native_4ref'):
            for seed in (42, 43):
                image = target / layout / 'generation' / f'final_image_seed{seed}.png'
                if image.is_file():
                    records.append({'image': str(image), 'source': 'beach14',
                                    'identity': 'beach', 'group': f'{target.name}/{layout}/{seed}',
                                    'role': 'generated', 'tier': 2})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tiktok-frames', type=int, default=5)
    parser.add_argument('--iper-sources', type=int, default=3)
    parser.add_argument('--iper-targets', type=int, default=3)
    parser.add_argument('--max-faces', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = tiktok_images(args.tiktok_frames) + iper_images(args.iper_sources,
                                                              args.iper_targets) + beach_images()
    print(f'{len(records)} candidate images', flush=True)

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda'
                 else ['CPUExecutionProvider'])
    detector = facekit.Detector(providers=providers, det_size=640)

    scanned, crops, points, ids = [], {}, {}, []
    failures = 0
    for number, record in enumerate(records, 1):
        bgr = facekit.read_bgr(record['image'])
        image_id = hashlib.sha1(record['image'].encode('utf-8')).hexdigest()[:12]
        detections = detector.detect(bgr)[:args.max_faces]
        entry = {**record, 'image_id': image_id, 'image_hw': [int(bgr.shape[0]), int(bgr.shape[1])],
                 'detections': []}
        kept = 0
        for index, detection in enumerate(detections):
            crop, in_crop, _ = facekit.align_face(bgr, detection['keypoints'])
            quality = facekit.face_quality(crop, in_crop)
            entry['detections'].append({
                'bbox': detection['bbox'], 'score': detection['score'], 'side': detection['side'],
                'keypoints': detection['keypoints'].tolist(), 'quality': quality})
            if record['source'] == 'beach14' or index == 0:
                key = f'{image_id}:{index}'
                crops[key] = crop
                points[key] = in_crop
                ids.append(key)
            kept += 1
        if not kept:
            failures += 1
        scanned.append(entry)
        if number % 200 == 0:
            print(f'{number}/{len(records)} detected={sum(1 for s in scanned if s["detections"])} '
                  f'failed={failures}', flush=True)

    (args.out / 'scan.json').write_text(json.dumps({
        'schema': 1,
        'detector': {'model': 'scrfd_10g_bnkps.onnx', 'det_size': 640,
                     'providers': detector.providers},
        'counts': {'images': len(scanned),
                   'with_face': sum(1 for item in scanned if item['detections']),
                   'no_face': failures},
        'records': scanned}, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.out / 'crops.npz',
                        ids=np.array(ids),
                        crops=np.stack([crops[key] for key in ids]),
                        keypoints=np.stack([points[key] for key in ids]))
    print(json.dumps({'images': len(scanned), 'no_face': failures, 'crops': len(ids)}), flush=True)


if __name__ == '__main__':
    main()
