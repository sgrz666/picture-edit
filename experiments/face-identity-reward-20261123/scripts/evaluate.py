#!/usr/bin/env python3
"""Score the identity-pair benchmark: AUC / EER / TAR / gaps / fusion / paired tests."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

MODELS = ('glintr100', 'w600k_r50', 'lvface_b', 'adaface_kprpe')
BASELINE = 'glintr100'


def eer_and_tar(same, diff):
    scores = np.concatenate([same, diff])
    labels = np.concatenate([np.ones_like(same), np.zeros_like(diff)])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[index] + fnr[index]) / 2)
    tar = {}
    for target in (1e-2, 1e-3):
        valid = fpr <= target
        tar[f'tar@{target:g}'] = float(tpr[valid].max()) if valid.any() else float('nan')
    return eer, tar


def describe(same, diff, with_ci=True, repeats=600, seed=0):
    same, diff = np.asarray(same, np.float64), np.asarray(diff, np.float64)
    if same.size < 5 or diff.size < 5:
        return {'pairs_same': int(same.size), 'pairs_diff': int(diff.size), 'status': 'insufficient'}
    labels = np.concatenate([np.ones_like(same), np.zeros_like(diff)])
    scores = np.concatenate([same, diff])
    auc = float(roc_auc_score(labels, scores))
    eer, tar = eer_and_tar(same, diff)
    pooled = np.sqrt((same.var(ddof=1) + diff.var(ddof=1)) / 2)
    output = {'pairs_same': int(same.size), 'pairs_diff': int(diff.size), 'auc': auc,
              'eer': eer, **tar,
              'same_mean': float(same.mean()), 'same_std': float(same.std(ddof=1)),
              'diff_mean': float(diff.mean()), 'diff_std': float(diff.std(ddof=1)),
              'gap': float(same.mean() - diff.mean()),
              'dprime': float((same.mean() - diff.mean()) / max(pooled, 1e-9)),
              'rank_snr': float((same.mean() - diff.mean()) / max(same.std(ddof=1), 1e-9))}
    if with_ci:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(repeats):
            index = rng.integers(0, scores.size, scores.size)
            if len(np.unique(labels[index])) < 2:
                continue
            values.append(roc_auc_score(labels[index], scores[index]))
        output['auc_ci95'] = ([float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                              if values else [float('nan'), float('nan')])
    return output


def paired_delta(same_a, diff_a, same_b, diff_b, repeats=800, seed=0):
    """Bootstrap the AUC difference (b - a) over the same pair resample."""
    labels = np.concatenate([np.ones_like(same_a), np.zeros_like(diff_a)])
    score_a = np.concatenate([same_a, diff_a])
    score_b = np.concatenate([same_b, diff_b])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        index = rng.integers(0, labels.size, labels.size)
        if len(np.unique(labels[index])) < 2:
            continue
        deltas.append(roc_auc_score(labels[index], score_b[index]) -
                      roc_auc_score(labels[index], score_a[index]))
    deltas = np.asarray(deltas)
    return {'delta': float(roc_auc_score(labels, score_b) - roc_auc_score(labels, score_a)),
            'ci95': [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            'p_better': float((deltas > 0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--emb', type=Path, required=True)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', default=['clean'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text(encoding='utf-8'))['pairs']
    scan = json.loads(args.scan.read_text(encoding='utf-8'))

    detection = defaultdict(lambda: {'images': 0, 'with_face': 0})
    for record in scan['records']:
        stats = detection[record['source']]
        stats['images'] += 1
        stats['with_face'] += int(bool(record['detections']))
    beach = [record for record in scan['records']
             if record['source'] == 'beach14' and record['role'] == 'generated']
    detection['beach14_generated_dual'] = {
        'images': len(beach), 'with_face': sum(1 for r in beach if len(r['detections']) >= 2)}

    identities = sorted({pair['a']['identity'] for pair in pairs if pair['tier'] == 1})
    rng = np.random.default_rng(args.seed)
    shuffled = list(identities)
    rng.shuffle(shuffled)
    calibration = set(shuffled[:max(1, len(shuffled) // 2)])

    report = {'detection': {key: {**value,
                                  'fail_rate': 1 - value['with_face'] / max(1, value['images'])}
                            for key, value in detection.items()},
              'split': {'identities': len(identities), 'calibration': len(calibration)},
              'conditions': {}}

    for condition in args.conditions:
        bundle = np.load(args.emb / f'{condition}.npz', allow_pickle=False)
        ids = {key: index for index, key in enumerate(bundle['ids'].tolist())}
        usable = [model for model in MODELS if model in bundle.files]

        detail = {}
        for model in usable:
            embeddings = bundle[model]
            rows = []
            for pair in pairs:
                a, b = ids.get(pair['a']['key']), ids.get(pair['b']['key'])
                if a is None or b is None:
                    continue
                rows.append({'kind': pair['kind'], 'tier': pair['tier'], 'dataset': pair['dataset'],
                             'value': float(embeddings[a] @ embeddings[b]),
                             'stratum': pair['stratum'], 'identity': pair['a']['identity'],
                             'note': pair.get('note', '')})
            detail[model] = rows

        def subset(model, **filters):
            output = []
            for row in detail[model]:
                if all(row[key] == value for key, value in filters.items()):
                    output.append(row)
            return output

        condition_report = {'models': {}, 'fusion': {}, 'strata': {}, 'delta_vs_baseline': {}}
        for model in usable:
            entry = {'all': describe([r['value'] for r in subset(model, kind='same')],
                                     [r['value'] for r in subset(model, kind='diff')])}
            for tier in (1, 2):
                entry[f'tier{tier}'] = describe(
                    [r['value'] for r in subset(model, tier=tier, kind='same')],
                    [r['value'] for r in subset(model, tier=tier, kind='diff')])
            for dataset in ('tiktok', 'iper'):
                entry[dataset] = describe(
                    [r['value'] for r in subset(model, dataset=dataset, kind='same')],
                    [r['value'] for r in subset(model, dataset=dataset, kind='diff')],
                    with_ci=False)
            # Hard positives only: same person across clips (different clothing,
            # lighting and pose) is the closest real-data analogue to source->generated.
            entry['cross_clip'] = describe(
                [r['value'] for r in subset(model, tier=1, kind='same', note='cross_clip')],
                [r['value'] for r in subset(model, tier=1, kind='diff')])
            condition_report['models'][model] = entry

        if BASELINE in usable:
            base = {kind: [r['value'] for r in subset(BASELINE, tier=1, kind=kind)]
                    for kind in ('same', 'diff')}
            for model in usable:
                if model == BASELINE:
                    continue
                other = {kind: [r['value'] for r in subset(model, tier=1, kind=kind)]
                         for kind in ('same', 'diff')}
                condition_report['delta_vs_baseline'][model] = paired_delta(
                    base['same'], base['diff'], other['same'], other['diff'])

        for key in ('size', 'pose', 'sharp'):
            buckets = {}
            for value in sorted({row['stratum'][key] for row in detail[usable[0]] if row['tier'] == 1}):
                buckets[value] = {
                    model: describe([r['value'] for r in subset(model, tier=1, kind='same')
                                     if r['stratum'][key] == value],
                                    [r['value'] for r in subset(model, tier=1, kind='diff')
                                     if r['stratum'][key] == value], with_ci=False)
                    for model in usable}
            condition_report['strata'][key] = buckets

        # ---- fusion on held-out identities ----------------------------------
        if len(usable) >= 2:
            calib_rows = [row for row in detail[usable[0]] if row['identity'] in calibration]
            test_index = [index for index, row in enumerate(detail[usable[0]])
                          if row['identity'] not in calibration]
            if len(test_index) < 40:
                test_index = list(range(len(detail[usable[0]])))

            def standardise(values):
                array = np.asarray(values, np.float64)
                return float(array.mean()), float(array.std(ddof=1) + 1e-9)

            stats = {}
            for model in usable:
                stats[model] = standardise([row['value'] for row in calib_rows])

            def z_fuse(members, index):
                return float(np.mean([(detail[model][index]['value'] - stats[model][0]) /
                                      stats[model][1] for model in members]))

            for label, members in (('z:glintr100+lvface_b', ['glintr100', 'lvface_b']),
                                   ('z:lvface_b+adaface_kprpe', ['lvface_b', 'adaface_kprpe']),
                                   ('z:glintr100+lvface_b+adaface_kprpe',
                                    ['glintr100', 'lvface_b', 'adaface_kprpe'])):
                if not all(member in usable for member in members):
                    continue
                condition_report['fusion'][label] = describe(
                    [z_fuse(members, i) for i in test_index if detail[usable[0]][i]['kind'] == 'same'],
                    [z_fuse(members, i) for i in test_index if detail[usable[0]][i]['kind'] == 'diff'])

            for label, members in (('lr:glintr100+lvface_b', ['glintr100', 'lvface_b']),
                                   ('lr:lvface_b+adaface_kprpe', ['lvface_b', 'adaface_kprpe']),
                                   ('lr:glintr100+lvface_b+adaface_kprpe',
                                    ['glintr100', 'lvface_b', 'adaface_kprpe'])):
                if not all(member in usable for member in members):
                    continue
                calib_index = [index for index, row in enumerate(detail[usable[0]])
                               if row['identity'] in calibration]
                features = np.array([[detail[model][index]['value'] for model in members]
                                     for index in calib_index])
                labels = np.array([1 if detail[usable[0]][index]['kind'] == 'same' else 0
                                   for index in calib_index])
                if labels.min() == labels.max():
                    continue
                model = LogisticRegression(max_iter=2000, C=1.0)
                model.fit(features, labels)
                test_features = np.array([[detail[member][index]['value'] for member in members]
                                          for index in test_index])
                probability = model.predict_proba(test_features)[:, 1]
                condition_report['fusion'][label] = describe(
                    [probability[i] for i, index in enumerate(test_index)
                     if detail[usable[0]][index]['kind'] == 'same'],
                    [probability[i] for i, index in enumerate(test_index)
                     if detail[usable[0]][index]['kind'] == 'diff'])
                condition_report['fusion'][f'{label}_weights'] = {
                    name: float(weight) for name, weight in zip(members, model.coef_[0])}

        report['conditions'][condition] = condition_report
        summary = ' '.join(f'{model}={condition_report["models"][model]["tier1"].get("auc", float("nan")):.4f}'
                           for model in usable)
        print(f'[{condition}] {summary}', flush=True)

    (args.out / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding='utf-8')
    print('wrote', args.out / 'results.json')


if __name__ == '__main__':
    main()
