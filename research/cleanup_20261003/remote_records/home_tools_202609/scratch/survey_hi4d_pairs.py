#!/usr/bin/env python3
"""Survey Hi4D native_interaction_v4_2 pairs to pick diverse two-person test samples."""
import json
import os
from collections import Counter

BASE = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1/native_interaction_v4_2'

for split in ('test', 'val', 'train'):
    path = os.path.join(BASE, f'main_pairs_{split}.jsonl')
    if not os.path.isfile(path):
        continue
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f'=== {split}: {len(rows)} 条 ===')
    if not rows:
        continue
    for field in ('pair', 'source_action', 'target_action', 'camera_id',
                  'pair_type', 'pair_relation', 'target_sampling_case'):
        c = Counter(r.get(field) for r in rows)
        print(f'  {field:22s} {dict(c)}')
    print(f'  same_action True/False : '
          f'{sum(1 for r in rows if r.get("same_action"))}/'
          f'{sum(1 for r in rows if not r.get("same_action"))}')
    print(f'  contact vertices total : {Counter(r.get("target_contact_vertex_count_total") for r in rows)}')
    # 源帧是否确实非接触
    print(f'  source_contact 全 False: {all(not r.get("source_contact") for r in rows)}')
    print(f'  target_contact 全 True : {all(r.get("target_contact") for r in rows)}')
    print()

# 挑样本：跨动作优先 + 目标接触顶点数 >= 10（真的抱上）
rows = [json.loads(l) for l in open(os.path.join(BASE, 'main_pairs_test.jsonl')) if l.strip()]
strong = [r for r in rows if (r.get('target_contact_vertex_count_total') or 0) >= 10]
print(f'=== 目标接触顶点 >=10 的测试样本: {len(strong)} ===')
by_pair = {}
for r in strong:
    by_pair.setdefault(r['pair'], []).append(r)
for p, rs in sorted(by_pair.items()):
    print(f'  {p}: {len(rs)} 条  跨动作={sum(1 for x in rs if not x.get("same_action"))}')
print()
print('=== 候选样本（每条附接触强度与动作对）===')
for r in sorted(strong, key=lambda x: -(x.get('target_contact_vertex_count_total') or 0))[:12]:
    print(f'  {r["pair"]:8s} {r["source_action"]:10s} -> {r["target_action"]:8s} '
          f'cam{r["camera_id"]:>3}  src#{r["source_frame_id"]:<4} tgt#{r["target_frame_id"]:<4} '
          f'contact={r["target_contact_vertex_count_total"]:<4} cross={not r["same_action"]}')
