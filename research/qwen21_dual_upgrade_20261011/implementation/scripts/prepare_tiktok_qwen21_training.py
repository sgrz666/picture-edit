#!/usr/bin/env python3
"""Audit strict TikTok splits and write separate eligible training manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def check_split_disjoint(splits):
    groups, sequences = {}, {}
    for split, rows in splits.items():
        if any(not row.get('source_video_group') for row in rows):
            raise ValueError(f'{split}: missing source_video_group provenance')
        groups[split] = {row['source_video_group'] for row in rows}
        sequences[split] = {row['sequence_id'] for row in rows}
    names = sorted(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if groups[first] & groups[second] or sequences[first] & sequences[second]:
                raise ValueError(f'Split overlap: {first}/{second}')
    return {name: {'sequences': len(sequences[name]), 'video_groups': len(groups[name])} for name in names}


def inspect_body_pair(row, assets_root, bundles):
    seq, stem = row['sequence_id'], row['target_stem']
    reasons = []
    if row['source_stem'] == stem:
        reasons.append('source_equals_target')
    for key in ('source_rgb', 'target_rgb', 'target_person_mask', 'target_normal',
                'target_dwpose', 'target_dwpose_json'):
        if not row.get(key) or not Path(row[key]).is_file():
            reasons.append(f'missing_{key}')
    part = Path(assets_root) / seq / 'part_512' / f'{stem}.png'
    if not part.is_file():
        reasons.append('missing_part_map')
    if seq not in bundles:
        path = Path(assets_root) / seq / 'v6_conditions.pt'
        bundles[seq] = torch.load(path, map_location='cpu', weights_only=False) if path.is_file() else {}
    bundle = bundles[seq]
    index = bundle.get('stem_to_idx', {}).get(stem)
    visible = 0
    if index is None:
        reasons.append('missing_v6_target_stem')
    else:
        global_params, points = bundle['smplx_global'][index], bundle['kps_25_coords'][index]
        if global_params.shape != (26,) or not torch.isfinite(global_params).all():
            reasons.append('nonfinite_or_wrong_shape_smplx')
        if points.shape != (25, 3) or not torch.isfinite(points).all():
            reasons.append('nonfinite_or_wrong_shape_keypoints')
        else:
            valid = (points[:, 2] > .05) & ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(-1)
            visible = int(valid.sum())
            if visible < 8:
                reasons.append('fewer_than_8_body_keypoints')
    return {'pair_id': row.get('pair_id'), 'sequence_id': seq, 'target_stem': stem,
            'visible_body_keypoints': visible}, reasons


def diverse_subset(rows, count):
    if count <= 0:
        return rows
    selected, used = [], set()
    for row in rows:
        if row['sequence_id'] not in used:
            selected.append(row)
            used.add(row['sequence_id'])
            if len(selected) == count:
                return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1')
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-train-count', type=int, default=4)
    parser.add_argument('--smoke-val-count', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.output / 'readiness.json').exists():
        raise FileExistsError('Use a new preparation output directory; frozen manifests must not be overwritten')
    splits = {split: read_rows(args.pairs_root / f'{split}_pairs.jsonl') for split in ('train', 'val', 'test')}
    provenance = check_split_disjoint(splits)
    if any(not splits[split] for split in splits):
        raise ValueError('All three source manifests must be nonempty')
    args.output.mkdir(parents=True, exist_ok=True)
    bundles, selected, rejected, counts = {}, {}, [], {}
    # Test rows are used only to check group provenance; no test RGB/features are opened.
    for split in ('train', 'val'):
        kept, reasons_count = [], Counter()
        for row in splits[split]:
            record, reasons = inspect_body_pair(row, args.assets_root, bundles)
            if reasons:
                rejected.append({'split': split, **record, 'reasons': reasons})
                reasons_count.update(reasons)
            else:
                kept.append(row)
        if not kept:
            raise ValueError(f'No eligible {split} pairs')
        selected[split] = kept
        write_rows(args.output / f'{split}_pairs.jsonl', kept)
        counts[split] = {'input_pairs': len(splits[split]), 'eligible_pairs': len(kept),
                         'excluded_pairs': len(splits[split]) - len(kept), 'exclusion_reasons': dict(reasons_count),
                         **provenance[split]}
    write_rows(args.output / 'exclusions.jsonl', rejected)
    smoke = args.output / 'smoke'
    smoke.mkdir()
    for split, count in (('train', args.smoke_train_count), ('val', args.smoke_val_count)):
        subset = diverse_subset(selected[split], count)
        write_rows(smoke / f'{split}_pairs.jsonl', subset)
        # Decode representative images before the expensive model load.
        for row in subset:
            for key in ('source_rgb', 'target_rgb', 'target_normal', 'target_person_mask'):
                with Image.open(row[key]) as image:
                    image.load()
    report = {
        'dataset': 'tiktok_v3', 'architecture': 'qwen21_full_v1', 'splits': counts,
        'test': {'pairs': len(splits['test']), **provenance['test'], 'rgb_or_features_used': False},
        'split_overlap': False, 'body_status': 'eligible_for_bounded_training_after_runtime_smoke',
        'face_status': 'requires_full_ArcFace_DINO_cache_and_supervision',
        'alignment': 'resize; same normalized coordinates for RGB, geometry and face',
        'input_sha256': {split: hashlib.sha256((args.pairs_root / f'{split}_pairs.jsonl').read_bytes()).hexdigest() for split in splits},
        'output_sha256': {split: hashlib.sha256((args.output / f'{split}_pairs.jsonl').read_bytes()).hexdigest() for split in ('train', 'val')},
        'original_assets_modified': False, 'minimum_visible_body_keypoints': 8,
        'limits': ['Raw-video group disjointness does not certify identity disjointness',
                   'A smoke run does not establish generative quality or convergence',
                   'Only representative images are decoded; every selected runtime sample still validates conditions'],
    }
    (args.output / 'readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
