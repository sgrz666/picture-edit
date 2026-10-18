#!/usr/bin/env python3
"""Paired Face variant comparisons on common visible faces and pose groups."""

from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
from pathlib import Path

import numpy as np


FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-sample', required=True, type=Path)
    parser.add_argument('--baseline', default='body')
    parser.add_argument('--manual-views', type=Path)
    parser.add_argument('--legacy-audit-csv', type=Path)
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'face_group_summary.json')
    return parser.parse_args()


def mean(values):
    return float(np.mean(values)) if len(values) else None


def paired(rows, baseline, candidate, metric, eligible):
    matches = []
    for row in rows:
        if not eligible(row):
            continue
        a, b = row['variants'][baseline], row['variants'][candidate]
        if metric.startswith('identity_') and not (a['generated_face_reliable'] and b['generated_face_reliable']):
            continue
        x, y = a.get(metric), b.get(metric)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            matches.append((float(x), float(y)))
    if not matches:
        return {'n': 0, 'baseline': None, 'candidate': None, 'delta': None, 'bootstrap_95_ci': None}
    array = np.asarray(matches)
    differences = array[:, 1] - array[:, 0]
    rng = np.random.default_rng(42)
    draws = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    return {'n': len(matches), 'baseline': mean(array[:, 0]), 'candidate': mean(array[:, 1]),
            'delta': mean(differences), 'bootstrap_95_ci': np.quantile(draws, [.025, .975]).tolist()}


def main():
    args = arguments()
    rows = json.loads(args.per_sample.read_text())
    if args.manual_views:
        audit = json.loads(args.manual_views.read_text())
        if len(audit['views']) != len(rows):
            raise ValueError('manual view count differs from the evaluation count')
        for row, code in zip(rows, audit['views']):
            row['source_view'], row['target_view'] = code[:2]
    legacy_visible, legacy_all_front = set(), set()
    if args.legacy_audit_csv:
        audit_rows = {}
        with args.legacy_audit_csv.open(newline='', encoding='utf-8') as stream:
            for entry in csv.DictReader(stream):
                audit_rows.setdefault(int(entry['pair_idx']), []).append(entry)
        for index, entries in audit_rows.items():
            if len(entries) != 6:
                continue
            if all(entry['face_visible'] == 'True' and entry['view'] in ('F', 'P') and
                   float(entry['min_side_px'] or 0) >= 32 for entry in entries):
                legacy_visible.add(index)
                if all(entry['view'] == 'F' for entry in entries):
                    legacy_all_front.add(index)
    modes = list(rows[0]['variants'])
    if args.baseline not in modes:
        raise ValueError('baseline absent from evaluation')
    groups = {'all': lambda row: True,
              'reliable_visible_face': lambda row: row['strict_reference_usable'],
              'source_face_usable': lambda row: row['source_face_usable']}
    if 'source_face_side_original' in rows[0]:
        groups.update({
            'source_face_below_32px_at_512': lambda row: row['source_face_side_original'] is not None and
                row['source_face_side_original'] < 64,
            'source_face_at_least_32px_at_512': lambda row: row['source_face_side_original'] is not None and
                row['source_face_side_original'] >= 64,
        })
    if 'person_seen_in_face_training' in rows[0]:
        groups.update({
            'person_seen_in_face_training': lambda row: row['person_seen_in_face_training'],
            'person_unseen_in_face_training': lambda row: not row['person_seen_in_face_training'],
            'reliable_face_person_seen': lambda row: row['strict_reference_usable'] and
                row['person_seen_in_face_training'],
            'reliable_face_person_unseen': lambda row: row['strict_reference_usable'] and
                not row['person_seen_in_face_training'],
        })
    if args.manual_views:
        groups.update({
            'same_visible_orientation': lambda row: row['source_view'] == row['target_view'] and row['target_view'] != 'B',
            'changed_visible_orientation': lambda row: row['source_view'] != row['target_view'] and
                row['source_view'] != 'B' and row['target_view'] != 'B',
            'source_or_target_back': lambda row: 'B' in (row['source_view'], row['target_view']),
            'reliable_face_same_orientation': lambda row: row['strict_reference_usable'] and
                row['source_view'] == row['target_view'] and row['target_view'] != 'B',
            'reliable_face_changed_orientation': lambda row: row['strict_reference_usable'] and
                row['source_view'] != row['target_view'] and
                row['source_view'] != 'B' and row['target_view'] != 'B',
        })
    if args.legacy_audit_csv:
        current_faces_at_least_32 = lambda row: all(
            variant['generated_face_reliable'] and variant['generated_face_side'] >= 32
            for variant in row['variants'].values())
        groups.update({
            'prior_audit_visible_32': lambda row: row['pair_idx'] in legacy_visible and
                current_faces_at_least_32(row),
            'prior_audit_all_front_32': lambda row: row['pair_idx'] in legacy_all_front and
                current_faces_at_least_32(row),
        })
    output = {'rows': len(rows), 'baseline': args.baseline, 'groups': {}}
    for group, eligible in groups.items():
        subset = [row for row in rows if eligible(row)]
        output['groups'][group] = {'pairs': len(subset), 'comparisons': {}}
        for candidate in modes:
            if candidate == args.baseline:
                continue
            output['groups'][group]['comparisons'][candidate] = {
                metric: paired(rows, args.baseline, candidate, metric, eligible)
                for metric in ('identity_cosine', 'identity_target_cosine', 'pose_pck_002',
                               'background_mae_source', 'psnr', 'ssim')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
