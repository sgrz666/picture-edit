#!/usr/bin/env python3
"""Compare SCRFD detection recall at det_size 640 (production) vs 1024."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facekit  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=250)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    scan = json.loads(args.scan.read_text(encoding='utf-8'))
    picks = defaultdict(list)
    for record in scan['records']:
        if len(picks[record['source']]) < args.per_source:
            picks[record['source']].append(record)

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda'
                 else ['CPUExecutionProvider'])
    small = facekit.Detector(providers=providers, det_size=640)
    large = facekit.Detector(providers=providers, det_size=1024)

    report = {}
    for source, records in picks.items():
        stats = {'images': 0, 'hit640': 0, 'hit1024': 0, 'sides640': [], 'sides1024': []}
        for record in records:
            bgr = facekit.read_bgr(record['image'])
            stats['images'] += 1
            found_small = small.detect(bgr)
            found_large = large.detect(bgr)
            stats['hit640'] += int(bool(found_small))
            stats['hit1024'] += int(bool(found_large))
            if found_small:
                stats['sides640'].append(found_small[0]['side'])
            if found_large:
                stats['sides1024'].append(found_large[0]['side'])
        import numpy as np
        report[source] = {
            'images': stats['images'],
            'recall640': stats['hit640'] / max(1, stats['images']),
            'recall1024': stats['hit1024'] / max(1, stats['images']),
            'median_side640': float(np.median(stats['sides640'])) if stats['sides640'] else None,
            'median_side1024': float(np.median(stats['sides1024'])) if stats['sides1024'] else None}
        print(source, json.dumps(report[source]), flush=True)

    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
