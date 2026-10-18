#!/usr/bin/env python3
"""Score the 12 RGB-reference outputs with the unchanged cross-person metrics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from scripts.eval.eval_qwen21_bfs_rgb import (
    DEFAULT, PAIR_IDS, PROTOCOL, VARIANTS, PREVIOUS, validate_saved, sha256,
)
from scripts.eval import score_qwen21_native_value as shared
from scripts.reports import build_qwen21_native_value_report as figures

LABELS = {'native': '原生 A＋B', 'bfs_body': 'BFS Body', 'bfs_head': 'BFS Head'}


def validate_index(root, partial=False):
    lock_path = root / 'experiment_lock.json'
    pairs = json.loads((root / 'data/pairs.json').read_text())
    prompts = json.loads((root / 'data/prompts.json').read_text())
    rows = shared.read_jsonl(root / 'generations.jsonl')
    expected = {(pid, variant) for pid in PAIR_IDS for variant in VARIANTS}
    keys = [(r['pair_id'], r['method']) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected or (not partial and set(keys) != expected):
        raise ValueError('RGB audit must cover exactly four pairs x three arms')
    if [p['pair_id'] for p in pairs] != list(PAIR_IDS):
        raise ValueError('Locked pair order differs')
    for row in rows:
        validate_saved(row, prompts[row['pair_id']], row['references'], row['method'], sha256(lock_path))
    return pairs, rows


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[shared.group_key(row)][row['method']].append(row)
    return {'groups': {g: {m: shared.aggregate(rs) for m, rs in methods.items()} for g, methods in groups.items()},
            'paired_differences': {g: {m + '_minus_native': shared.paired_delta(methods['native'], methods[m])
                                      for m in VARIANTS[1:] if m in methods and 'native' in methods}
                                   for g, methods in groups.items()},
            'formal_identity_review_complete': all(r['face_selection'][role]['reviewed']
                                                   for r in rows for role in ('source', 'target', 'generated'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(4)
    pairs, indexed = validate_index(root, args.allow_partial)
    by_id = {p['pair_id']: p for p in pairs}
    directory = root / 'evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    prior = PREVIOUS / 'supplement_combinations/evaluation'
    for name in ('face_detection_cache.json', 'face_review.json'):
        path = directory / name
        if not path.exists():
            path.write_bytes((prior / name).read_bytes())
    old = {shared.row_key(r): r for r in json.loads((directory / 'evaluated_samples.json').read_text())} \
          if (directory / 'evaluated_samples.json').exists() else {}
    baselines = {(r['pair_id'], r['resolution']): r for r in json.loads((prior / 'evaluated_samples.json').read_text())
                 if r['method'] == 'native_ab' and r['steps'] == 30}
    scorer = None
    evaluated, reused = [], 0
    for row in sorted(indexed, key=shared.row_key):
        cached = old.get(shared.row_key(row))
        if cached and cached['image_sha256'] == row['image_sha256']:
            value = {**cached, **row}
            reused += 1
        elif args.review_only:
            raise ValueError('Review-only requires prior scores for every image')
        elif row['method'] == 'native' and row.get('cost_measurement') == 'historical':
            cached = baselines[row['pair_id'], row['resolution']]
            if cached['image_sha256'] != row['image_sha256']:
                raise ValueError('Reused baseline metric image differs')
            value = {**cached, **row}
            reused += 1
        else:
            if scorer is None:
                scorer = shared.Scorer(root)
            value = scorer.score(row, by_id[row['pair_id']])
        evaluated.append(value)
        print(json.dumps({'evaluated': len(evaluated), 'pair': row['pair_id'], 'method': row['method'],
                          'pck': value['metrics']['pose_pck_002']}), flush=True)
    manual = json.loads((directory / 'face_review.json').read_text())
    shared.apply_face_review(evaluated, manual)
    statistics = summarize(evaluated)
    shared.write_tables(root, evaluated, statistics)
    # Registries are process-local: older experiment entrypoints are unchanged.
    shared.ALL_METHODS = VARIANTS
    shared.LABELS.update(LABELS)
    shared.review_material(root, evaluated)
    figures.CN.update(LABELS)
    panels = [figures.figure(root, evaluated, p, 1024, VARIANTS, 'bfs_rgb') for p in pairs]
    panels = [p for p in panels if p is not None]
    shared.save(root / 'report_figures/manifest.json', panels)
    definitions = json.loads((directory / 'metric_definitions.json').read_text())
    definitions.update(multi_stage_cost='Single edit only; no second stage.',
                       generation_B_rgb='B RGB is intentionally the second native image reference in all arms.',
                       cost_measurement='Native reused measurement is historical; new LoRA results measured this run.')
    shared.save(directory / 'metric_definitions.json', definitions)
    auxiliary = [r for r in json.loads((prior / 'evaluated_samples.json').read_text())
                 if r['pair_id'] in PAIR_IDS and r['method'] == 'native_skeleton'
                 and r['resolution'] == 1024 and r['steps'] == 30]
    if len(auxiliary) != 4:
        raise ValueError('Missing historical skeleton auxiliary comparison')
    shared.save(directory / 'historical_skeleton_reference.json', [
        {'pair_id': r['pair_id'], 'dataset': r['dataset'], 'image': r['image'],
         'image_sha256': r['image_sha256'], 'input_protocol': 'A RGB + skeleton + detailed action text',
         'metrics': r['metrics'], 'face_selection': r['face_selection']} for r in auxiliary])
    shared.save(directory / 'evaluation_protocol.json', {
        'formal_rows': len(evaluated), 'reused_metric_rows': reused, 'panels': len(panels),
        'reference_roles': ['A', 'B'], 'target_rgb_in_native_references': True,
        'cross_person_reconstruction_metrics': False, 'pose_background_resolution': 512,
        'face_resolution': 'native 1024 generated / original A,B',
        'identity_primary': 'Three-arm manually confirmed common reliable faces, separately by dataset; n<3 descriptive only',
        'formal_identity_review_complete': statistics['formal_identity_review_complete'],
        'scoring_source': str(Path(shared.__file__)), 'scoring_source_sha256': sha256(shared.__file__)})
    print(json.dumps({'complete': len(evaluated) == 12, 'panels': len(panels),
                      'manual_complete': statistics['formal_identity_review_complete']}), flush=True)


if __name__ == '__main__':
    main()
