#!/usr/bin/env python3
"""Actor-clustered, paired BFS statistics; preserves runner results and missing cases."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VARIANTS = ('body', 'nolora', 'bfs', 'alternative')
ACTOR_SOURCE = 'https://svip-lab.github.io/dataset/iPER_dataset.html'


def actor_id(appearance):
    return appearance.split('_', 1)[0]


def paired_stat(records, variant, baseline, metric):
    import numpy as np
    groups = collections.defaultdict(list)
    stems = []
    for record in records:
        a = record['metrics'][variant].get(metric)
        b = record['metrics'][baseline].get(metric)
        if not record['source_face_valid'] or a is None or b is None:
            continue
        groups[actor_id(record['appearance'])].append(float(a-b))
        stems.append(record['stem'])
    values = [x for group in groups.values() for x in group]
    means = np.array([np.mean(group) for group in groups.values()])
    bootstrap = np.random.default_rng(42).choice(means, size=(10000, len(means)), replace=True).mean(axis=1) if len(means) else None
    return {'n_pairs': len(values), 'n_actors': len(groups), 'mean_delta_per_pair': float(np.mean(values)) if values else None,
            'mean_delta_per_actor': float(means.mean()) if len(means) else None,
            'actor_cluster_bootstrap_95_ci': [float(v) for v in np.quantile(bootstrap, [.025, .975])] if len(means) > 1 else None,
            'wins': sum(v > 0 for v in values), 'losses': sum(v < 0 for v in values),
            'ties': sum(v == 0 for v in values), 'stems': stems,
            'cohort': 'valid source face and paired metric availability; identity requires both outputs detected'}


def analyze(records):
    import numpy as np
    summary = {}
    common_identity = [r for r in records if r['source_face_valid'] and all(r['metrics'][v].get('identity_A') is not None for v in VARIANTS)]
    for variant in VARIANTS:
        all_metrics = [r['metrics'][variant] for r in records]
        attempted = [r for r in records if r['source_face_valid']]
        summary[variant] = {'n_total': len(records), 'n_complete': sum(m['status'] == 'complete' for m in all_metrics),
                            'n_face_detected': sum(m['face_detected'] for m in all_metrics),
                            'n_unique_face': sum(m.get('face_unique', m['face_detected']) for m in all_metrics),
                            'n_unique_face_on_eligible': sum(r['metrics'][variant].get('face_unique', r['metrics'][variant]['face_detected']) for r in attempted),
                            'n_source_face_eligible': len(attempted),
                            'n_source_face_fallback': sum(not r['source_face_valid'] for r in records),
                            'n_generated_face_failure_on_eligible': sum(r['metrics'][variant]['status'] == 'complete' and not r['metrics'][variant]['face_detected'] for r in attempted),
                            'n_generated_multiface_on_eligible': sum((r['metrics'][variant].get('face_count') or 0) > 1 for r in attempted),
                            'n_unexpected_missing_on_eligible': sum(r['metrics'][variant]['status'] != 'complete' for r in attempted)}
        for metric in ('identity_A', 'pose_pck_002', 'outside_head_mae_255'):
            values = [m[metric] for m in all_metrics if m.get(metric) is not None]
            summary[variant][metric + '_mean'] = float(np.mean(values)) if values else None
            summary[variant][metric + '_n'] = len(values)
        values = [r['metrics'][variant]['identity_A'] for r in common_identity]
        summary[variant]['common_identity_mean'] = float(np.mean(values)) if values else None
        summary[variant]['common_identity_n'] = len(values)
        values = [r['metrics'][variant]['pose_pck_002'] for r in attempted if r['metrics'][variant].get('pose_pck_002') is not None]
        summary[variant]['eligible_pose_pck_002_mean'] = float(np.mean(values)) if values else None
        summary[variant]['eligible_pose_pck_002_n'] = len(values)
        for metric in ('pose_pck_002', 'outside_head_mae_255'):
            # Explicit policy: source-face-unavailable cases retain C. This is posthoc evaluation,
            # not a claim that BFS generated a successful image for those six cases.
            metrics = [(r['metrics'][variant] if r['source_face_valid'] else r['metrics']['body']) for r in records]
            values = [m[metric] for m in metrics if m.get(metric) is not None]
            summary[variant]['with_source_fallback_' + metric + '_mean'] = float(np.mean(values)) if values else None
            summary[variant]['with_source_fallback_' + metric + '_n'] = len(values)
    transitions, paired = {}, {}
    for variant in ('bfs', 'alternative'):
        for baseline in ('nolora', 'body'):
            key = variant + '_minus_' + baseline
            paired[key] = {metric: paired_stat(records, variant, baseline, metric) for metric in ('identity_A', 'pose_pck_002')}
            table = collections.Counter()
            for r in records:
                if not r['source_face_valid']:
                    continue
                a, b = r['metrics'][variant], r['metrics'][baseline]
                if a['status'] != 'complete' or b['status'] != 'complete':
                    table['unexpected_missing'] += 1
                else:
                    table[f'baseline_face_{bool(b.get("face_unique", b["face_detected"]))}_variant_face_{bool(a.get("face_unique", a["face_detected"]))}'] += 1
            transitions[key] = dict(table)
    actors = {}
    for actor in sorted({actor_id(r['appearance']) for r in records}):
        selected = [r for r in records if actor_id(r['appearance']) == actor]
        actors[actor] = {'n_pairs': len(selected), 'appearances': sorted({r['appearance'] for r in selected}),
                         'n_source_face_eligible': sum(r['source_face_valid'] for r in selected), 'variants': {}}
        for variant in VARIANTS:
            values = [r['metrics'][variant].get('identity_A') for r in selected if r['metrics'][variant].get('identity_A') is not None]
            actors[actor]['variants'][variant] = {'identity_mean': float(np.mean(values)) if values else None, 'identity_n': len(values)}
    return {'summary': summary, 'paired': paired, 'face_detection_transitions': transitions, 'by_actor': actors,
            'cohort': {'n_pairs': len(records), 'n_actors': len(actors), 'n_appearances': len({r['appearance'] for r in records}),
                       'source_face_eligible': sum(r['source_face_valid'] for r in records),
                       'common_identity_n': len(common_identity), 'actor_mapping_source': ACTOR_SOURCE},
            'output_face_audit_complete': all('face_count' in r['metrics'][v] for r in records for v in VARIANTS if r['metrics'][v]['status'] == 'complete'),
            'limitations': ['eight actors, ten clothing appearances; validation sample, not an independent unseen-actor test',
                            'identity means / confidence intervals are conditional on detection; consult failure transitions',
                            'paired PCK contrasts use common eligible samples; raw aggregate PCK denominators differ',
                            'fallback summaries copy Body metrics for source-face-unavailable cases; no BFS generation occurred there',
                            'one seed and one strength; simplified Body, no current Full Face comparison',
                            'body PCK does not measure head angle/expression', 'whole-image second stage may alter clothing/background']}


def self_test():
    assert actor_id('001_2') == actor_id('001_5') == '001'
    rows = []
    for appearance, delta in [('001_2', .1), ('001_5', .1), ('006_1', -.1)]:
        metrics = {v: {'status': 'complete', 'face_detected': True, 'identity_A': .5, 'pose_pck_002': 1., 'outside_head_mae_255': 0.} for v in VARIANTS}
        metrics['bfs']['identity_A'] += delta
        rows.append({'stem': appearance, 'appearance': appearance, 'source_face_valid': True, 'metrics': metrics})
    result = analyze(rows)
    stat = result['paired']['bfs_minus_nolora']['identity_A']
    assert stat['n_actors'] == 2 and abs(stat['mean_delta_per_actor']) < 1e-12
    assert abs(stat['mean_delta_per_pair'] - .1/3) < 1e-12
    rows[0]['metrics']['bfs']['face_detected'] = False
    rows[0]['metrics']['bfs']['identity_A'] = None
    result = analyze(rows)
    assert result['summary']['bfs']['n_generated_face_failure_on_eligible'] == 1
    assert result['cohort']['common_identity_n'] == 2
    assert result['face_detection_transitions']['bfs_minus_nolora']['baseline_face_True_variant_face_False'] == 1
    assert result['paired']['bfs_minus_body']['pose_pck_002']['n_pairs'] == 3
    print('PASS: actual-actor clustering; unequal group weights; output detection failure retained; paired pose cohort')


def audit_outputs(run, records):
    from PIL import Image
    from eval_iper_four_arm import IdentityMetric
    from scripts.eval.bfs_compare import face_detection
    identity = IdentityMetric()
    for i, record in enumerate(records):
        for variant in VARIANTS:
            metrics = record['metrics'][variant]
            path = run / 'images' / f'{record["stem"]}_{variant}.png'
            metrics.setdefault('identity_A_before_unique_face_audit', metrics.get('identity_A'))
            if path.exists():
                bbox, count = face_detection(identity, Image.open(path).convert('RGB'))
                metrics.update({'face_count': count, 'face_unique': count == 1, 'output_face_bbox': bbox})
                if count != 1 or not record['source_face_valid']:
                    metrics['identity_A'] = None
            else:
                metrics.update({'face_count': None, 'face_unique': False, 'output_face_bbox': None})
        print(f'FACE AUDIT {i+1}/{len(records)}', flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(records):
    sampled = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'
    assets = ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast'
    paths = {Path(r[key]) for r in records for key in ('source_path', 'target_path')}
    paths.update(assets / r['appearance'] / 'v6_conditions.pt' for r in records)
    paths.update(ROOT / name for name in ('eval_iper_four_arm.py', 'src/data/iper_dataset.py',
                 'src/data/tiktok_dataset.py', 'src/pose_control/qwen21/body_adapter.py', 'src/pose_control/qwen21/runtime.py',
                 'models/antelopev2/scrfd_10g_bnkps.onnx', 'models/antelopev2/glintr100.onnx',
                 'models/DWPose/yolox_l.onnx', 'models/DWPose/dw-ll_ucoco_384.onnx',
                 'scripts/eval/summarize_bfs_comparison.py'))
    return {'sampled_root': str(sampled), 'assets_root': str(assets),
            'hash_capture_scope': 'postrun read-only hashes of input RGB, target condition bundles and metric/control code; not every backbone shard',
            'sha256': {str(p): file_hash(p) for p in sorted(paths) if p.is_file()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_compare_30_20261004'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    records = json.loads((args.run / 'records.json').read_text())
    original_records = args.run / 'records_before_face_audit.json'
    if not original_records.exists():
        original_records.write_bytes((args.run / 'records.json').read_bytes())
    audit_outputs(args.run, records)
    (args.run / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    result = analyze(records)
    result_path = args.run / 'results.json'
    raw_path = args.run / 'raw_runner_results.json'
    if not raw_path.exists():
        raw_path.write_bytes(result_path.read_bytes())
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (args.run / 'provenance.json').write_text(json.dumps(provenance(records), ensure_ascii=False, indent=2), encoding='utf-8')
    flat = [{'stem': r['stem'], 'appearance': r['appearance'], 'actor': actor_id(r['appearance']), 'variant': v,
             'source_face_valid': r['source_face_valid'], **r['metrics'][v]} for r in records for v in VARIANTS]
    with (args.run / 'per_pair.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({'cohort': result['cohort'], 'summary': result['summary'], 'paired': result['paired']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
