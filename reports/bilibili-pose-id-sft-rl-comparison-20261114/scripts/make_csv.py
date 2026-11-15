#!/usr/bin/env python3
"""Write the 50-row per-pair CSV used as the report's machine-readable appendix."""
import csv
import json
from pathlib import Path

ROOT = Path('D:/codeplus/pictureedit/reports/bilibili-pose-id-sft-rl-comparison-20261114')
REPORT = Path('D:/temp/codex/worktrees/native-spatial-weights/pictureedit/experiments/'
              'bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1')
quant = json.loads((ROOT / 'assets/quant.json').read_text(encoding='utf-8'))
base_per = json.loads((ROOT / 'assets/base/per_image.json').read_text(encoding='utf-8'))
base = {row['pair_id']: row for row in base_per}
order = []
for row in quant['pairs']:
    if row['clip_id'] not in order:
        order.append(row['clip_id'])
pairs = []
for clip in order:
    pairs.extend([row for row in quant['pairs'] if row['clip_id'] == clip])

columns = [
    'case_no', 'clip_id', 'pair_id', 'id_applicable', 'category',
    'pose_sft', 'pose_rl200', 'delta_pose_200', 'pose_rl300', 'delta_pose_300',
    'pck002_sft', 'pck002_rl200', 'delta_pck002_200', 'pck002_rl300', 'delta_pck002_300',
    'pck005_sft', 'pck005_rl200', 'delta_pck005_200', 'pck005_rl300', 'delta_pck005_300',
    'nme_sft', 'nme_rl200', 'delta_nme_200', 'nme_rl300', 'delta_nme_300',
    'glintr_sft', 'glintr_rl200', 'delta_glintr_200', 'glintr_rl300', 'delta_glintr_300',
    'buffalo_sft', 'buffalo_rl200', 'delta_buffalo_200', 'buffalo_rl300', 'delta_buffalo_300',
    'base_pose', 'base_pck002', 'base_pck005', 'base_nme', 'base_glintr', 'base_buffalo',
    'image_A', 'image_pose', 'image_GT', 'image_base', 'image_sft', 'image_rl200', 'image_rl300',
]
out = ROOT / 'assets/per_pair_metrics.csv'
with out.open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for number, row in enumerate(pairs, 1):
        b = base[row['pair_id']]
        writer.writerow({
            'case_no': number,
            'clip_id': row['clip_id'],
            'pair_id': row['pair_id'],
            'id_applicable': 'yes' if row['id_applicable'] else 'no',
            'category': row['category'],
            'pose_sft': row['pose_base'], 'pose_rl200': row['pose_200'],
            'delta_pose_200': row['pose_delta_200'], 'pose_rl300': row['pose_300'],
            'delta_pose_300': row['pose_delta_300'],
            'pck002_sft': row['pck002_base'], 'pck002_rl200': row['pck002_200'],
            'delta_pck002_200': row['pck002_delta_200'], 'pck002_rl300': row['pck002_300'],
            'delta_pck002_300': row['pck002_delta_300'],
            'pck005_sft': row['pck005_base'], 'pck005_rl200': row['pck005_200'],
            'delta_pck005_200': row['pck005_delta_200'], 'pck005_rl300': row['pck005_300'],
            'delta_pck005_300': row['pck005_delta_300'],
            'nme_sft': row['nme_base'], 'nme_rl200': row['nme_200'],
            'delta_nme_200': row['nme_delta_200'], 'nme_rl300': row['nme_300'],
            'delta_nme_300': row['nme_delta_300'],
            'glintr_sft': row['id_base'], 'glintr_rl200': row['id_200'],
            'delta_glintr_200': row['id_delta_200'], 'glintr_rl300': row['id_300'],
            'delta_glintr_300': row['id_delta_300'],
            'buffalo_sft': row['buffalo_base'], 'buffalo_rl200': row['buffalo_200'],
            'delta_buffalo_200': row['buffalo_delta_200'], 'buffalo_rl300': row['buffalo_300'],
            'delta_buffalo_300': row['buffalo_delta_300'],
            'base_pose': b['pose_reward'], 'base_pck002': b['pck_002_diagonal'],
            'base_pck005': b['pck_005_diagonal'], 'base_nme': b['nme_diagonal'],
            'base_glintr': b['identity_cosine'], 'base_buffalo': b['buffalo_cosine'],
            'image_A': str(REPORT / f'context/{row["pair_id"]}/A.png'),
            'image_pose': str(REPORT / f'context/{row["pair_id"]}/pose.png'),
            'image_GT': str(REPORT / f'context/{row["pair_id"]}/GT.png'),
            'image_base': str(ROOT / f'assets/base/{row["pair_id"]}.png'),
            'image_sft': str(REPORT / f'images/0/{row["pair_id"]}.png'),
            'image_rl200': str(REPORT / f'images/200/{row["pair_id"]}.png'),
            'image_rl300': str(REPORT / f'images/300/{row["pair_id"]}.png'),
        })
print(out, 'rows', len(pairs))
