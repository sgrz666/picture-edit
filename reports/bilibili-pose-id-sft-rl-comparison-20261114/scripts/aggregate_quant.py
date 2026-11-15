#!/usr/bin/env python3
"""Aggregate the verified SFT/RL tables used by the Word report.

Inputs are the already-reviewed report assets. This script only reads them and
writes a single JSON for the docx builder; it does not re-run any metric.
"""
import json
from pathlib import Path

REPORT = Path('D:/temp/codex/worktrees/native-spatial-weights/pictureedit/experiments/'
              'bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1')
OUT = Path('D:/codeplus/pictureedit/reports/bilibili-pose-id-sft-rl-comparison-20261114/assets/quant.json')

analysis = json.loads((REPORT / 'analysis.json').read_text(encoding='utf-8'))
steps = [0, 50, 100, 150, 200, 250, 300]

# Absolute metric means: use the per-image records when present, because they
# keep the fixed ID-applicable denominator explicit.
checkpoints = []
for step in steps:
    summary = next(row for row in analysis['checkpoints'] if row['step'] == step)
    rows = json.loads((REPORT / f'source/update_{step:06d}/per_image.json').read_text(encoding='utf-8'))
    id_rows = [row for row in rows if row['id_applicable']]
    ids = [row['identity_cosine'] for row in id_rows if row['identity_cosine'] is not None]
    buf = [row['buffalo_cosine'] for row in id_rows if row['buffalo_cosine'] is not None]
    nme = [row['nme_diagonal'] for row in rows if row['nme_diagonal'] is not None]
    checkpoints.append({
        'step': step,
        'images': summary['images'],
        'pose_failures': summary['pose_failures'],
        'id_applicable': summary['id_applicable'],
        'id_failures': summary['id_failures'],
        'pose_reward': summary['pose_reward'],
        'pck_002_diagonal': summary['pck_002_diagonal'],
        'pck_005_diagonal': summary['pck_005_diagonal'],
        'nme_diagonal': sum(nme) / len(nme),
        'identity_reward': summary['identity_reward'],
        'buffalo_reward': summary['buffalo_reward'],
        'identity_cosine': sum(ids) / len(ids),
        'buffalo_cosine': sum(buf) / len(buf),
        'identity_n': len(ids),
        'buffalo_n': len(buf),
    })

# Pair rows: SFT (base) + RL200 + RL300 on the same pair_id.
pair200 = {row['pair_id']: row for row in analysis['paired']['200']}
pair300 = {row['pair_id']: row for row in analysis['paired']['300']}
pairs = []
for row in analysis['paired']['200']:
    pid = row['pair_id']
    r300 = pair300[pid]
    category = 'id_na' if not row['id_applicable'] or row['delta_identity_cosine'] is None else (
        'both' if row['delta_pose_reward'] > 0 and row['delta_identity_cosine'] > 0 else
        'regression' if row['delta_pose_reward'] < 0 and row['delta_identity_cosine'] < 0 else
        'tradeoff')
    pairs.append({
        'pair_id': pid,
        'clip_id': row['clip_id'],
        'id_applicable': row['id_applicable'],
        'category': category,
        'pose_base': row['base_pose_reward'],
        'pose_200': row['rl_pose_reward'],
        'pose_delta_200': row['delta_pose_reward'],
        'pose_300': r300['rl_pose_reward'],
        'pose_delta_300': r300['delta_pose_reward'],
        'pck002_base': row['base_pck_002_diagonal'],
        'pck002_200': row['rl_pck_002_diagonal'],
        'pck002_delta_200': row['delta_pck_002_diagonal'],
        'pck002_300': r300['rl_pck_002_diagonal'],
        'pck002_delta_300': r300['delta_pck_002_diagonal'],
        'pck005_base': row['base_pck_005_diagonal'],
        'pck005_200': row['rl_pck_005_diagonal'],
        'pck005_delta_200': row['delta_pck_005_diagonal'],
        'pck005_300': r300['rl_pck_005_diagonal'],
        'pck005_delta_300': r300['delta_pck_005_diagonal'],
        'id_base': row['base_identity_cosine'],
        'id_200': row['rl_identity_cosine'],
        'id_delta_200': row['delta_identity_cosine'],
        'id_300': r300['rl_identity_cosine'],
        'id_delta_300': r300['delta_identity_cosine'],
        'buffalo_base': row['base_buffalo_cosine'],
        'buffalo_200': row['rl_buffalo_cosine'],
        'buffalo_delta_200': row['delta_buffalo_cosine'],
        'buffalo_300': r300['rl_buffalo_cosine'],
        'buffalo_delta_300': r300['delta_buffalo_cosine'],
        'nme_base': row['base_nme_diagonal'],
        'nme_200': row['rl_nme_diagonal'],
        'nme_delta_200': row['delta_nme_diagonal'],
        'nme_300': r300['rl_nme_diagonal'],
        'nme_delta_300': r300['delta_nme_diagonal'],
    })

# Per-clip absolute and delta values for both checkpoints of interest.
by_clip = {}
for step in (200, 300):
    rows = []
    for clip in sorted({row['clip_id'] for row in pairs}):
        selected = [row for row in pairs if row['clip_id'] == clip]
        id_rows = [row for row in selected if row['id_applicable'] and row[f'id_delta_{step}'] is not None]
        rows.append({
            'clip_id': clip,
            'pairs': len(selected),
            'id_n': len(id_rows),
            'pose_base': sum(row['pose_base'] for row in selected) / len(selected),
            'pose_rl': sum(row[f'pose_{step}'] for row in selected) / len(selected),
            'pose_delta': sum(row[f'pose_delta_{step}'] for row in selected) / len(selected),
            'pck002_base': sum(row['pck002_base'] for row in selected) / len(selected),
            'pck002_rl': sum(row[f'pck002_{step}'] for row in selected) / len(selected),
            'pck002_delta': sum(row[f'pck002_delta_{step}'] for row in selected) / len(selected),
            'id_base': (sum(row['id_base'] for row in id_rows) / len(id_rows)) if id_rows else None,
            'id_rl': (sum(row[f'id_{step}'] for row in id_rows) / len(id_rows)) if id_rows else None,
            'id_delta': (sum(row[f'id_delta_{step}'] for row in id_rows) / len(id_rows)) if id_rows else None,
            'buffalo_delta': (sum(row[f'buffalo_delta_{step}'] for row in id_rows) / len(id_rows)) if id_rows else None,
        })
    by_clip[str(step)] = rows

counts = {key: sum(row['category'] == key for row in pairs) for key in ('both', 'tradeoff', 'regression', 'id_na')}
id_valid = [row for row in pairs if row['id_applicable']]
joint = {
    'id_valid': len(id_valid),
    'both_up': sum(row['pose_delta_200'] > 0 and row['id_delta_200'] > 0 for row in id_valid),
    'tradeoff': sum((row['pose_delta_200'] > 0) != (row['id_delta_200'] > 0) for row in id_valid),
    'both_down': sum(row['pose_delta_200'] < 0 and row['id_delta_200'] < 0 for row in id_valid),
    'pose_wins': sum(row['pose_delta_200'] > 0 for row in pairs),
    'pose_losses': sum(row['pose_delta_200'] < 0 for row in pairs),
    'pck002_wins': sum(row['pck002_delta_200'] > 0 for row in pairs),
    'pck002_losses': sum(row['pck002_delta_200'] < 0 for row in pairs),
    'pck002_ties': sum(row['pck002_delta_200'] == 0 for row in pairs),
    'id_wins': sum(row['id_delta_200'] > 0 for row in id_valid),
    'id_losses': sum(row['id_delta_200'] < 0 for row in id_valid),
    'buffalo_wins': sum(row['buffalo_delta_200'] > 0 for row in id_valid),
    'buffalo_losses': sum(row['buffalo_delta_200'] < 0 for row in id_valid),
}

stats = {str(step): analysis['statistics'][str(step)] for step in (200, 300)}
result = {
    'source_report': str(REPORT),
    'scope': analysis['scope'],
    'checkpoints': checkpoints,
    'stats': stats,
    'by_clip': by_clip,
    'pairs': pairs,
    'category_counts': counts,
    'joint': joint,
    'clip_pose_delta_range_200': [
        min(row['pose_delta'] for row in by_clip['200']),
        max(row['pose_delta'] for row in by_clip['200'])],
    'clips_positive_pose_200': sum(row['pose_delta'] > 0 for row in by_clip['200']),
    'clips_positive_pose_300': sum(row['pose_delta'] > 0 for row in by_clip['300']),
    'clips_positive_id_200': sum((row['id_delta'] or 0) > 0 for row in by_clip['200']),
    'clips_positive_id_300': sum((row['id_delta'] or 0) > 0 for row in by_clip['300']),
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('category_counts', 'joint', 'clip_pose_delta_range_200',
                                               'clips_positive_pose_200', 'clips_positive_pose_300',
                                               'clips_positive_id_200', 'clips_positive_id_300')},
                 ensure_ascii=False, indent=2))
for row in checkpoints:
    print(row['step'], f"pose={row['pose_reward']:.6f}", f"pck002={row['pck_002_diagonal']:.6f}",
          f"pck005={row['pck_005_diagonal']:.6f}", f"nme={row['nme_diagonal']:.6f}",
          f"id={row['identity_cosine']:.6f}", f"buf={row['buffalo_cosine']:.6f}",
          f"fail={row['pose_failures']}/{row['id_failures']}")
