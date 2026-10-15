#!/usr/bin/env python3
"""Build a Hi4D two-person interaction test set from the native_interaction_v4_2 N2C pairs.

Task (project's own MAIN_N2C definition):
  source = same identity pair + same camera, two people NOT in contact  -> appearance/identity
  target = same identity pair + same camera, two people IN contact      -> interaction pose
"""
import json
import os
import shutil
from collections import defaultdict

from PIL import Image, ImageDraw

H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
PAIRS = os.path.join(H, 'native_interaction_v4_2/main_pairs_test.jsonl')
OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/interaction'
SET = os.path.join(OUT, 'testset')

rows = [json.loads(l) for l in open(PAIRS) if l.strip()]

# 选择策略：
#  1) 目标必须是无歧义的「拥抱」(hug)，且接触顶点数高（确实抱住）
#  2) 源与目标跨动作（taichi/yoga/dance -> hug），构成真正的 N2C
#  3) 同一 pair 下源动作去重，保证多样
# 选择策略：覆盖多种交互类型；每种目标动作取最多 2 个样本；
# 优先跨动作（源动作 != 目标动作）且目标接触强度高。
PRIORITY = ['hug00', 'taichi00', 'yoga00', 'backhug02', 'dance02', 'pose02']
TAKE_PER_ACTION = 2
MEANING = {'hug00': 'embrace', 'taichi00': 'taichi partner push-hands',
           'yoga00': 'partner yoga', 'backhug02': 'embrace from behind',
           'dance02': 'partner dance', 'pose02': 'partner pose'}

cand_all = [r for r in rows if (r.get('target_contact_vertex_count_total') or 0) >= 20]
print(f'接触>=20 的候选: {len(cand_all)}')
from collections import Counter as _C
print('  目标动作分布:', dict(_C(r['target_action'] for r in cand_all)))

picked, used_tgt = [], set()
for act in PRIORITY:
    group = [r for r in cand_all if r['target_action'] == act]
    group.sort(key=lambda r: (r['same_action'], -(r['target_contact_vertex_count_total'] or 0)))
    taken = 0
    for r in group:
        tkey = (r['pair'], r['target_frame_id'])
        if tkey in used_tgt or r['same_action']:
            continue
        used_tgt.add(tkey)
        picked.append(r)
        taken += 1
        if taken >= TAKE_PER_ACTION:
            break
    # 若无跨动作可用，退化为同动作
    if taken == 0:
        for r in group:
            tkey = (r['pair'], r['target_frame_id'])
            if tkey in used_tgt:
                continue
            used_tgt.add(tkey)
            picked.append(r)
            taken += 1
            if taken >= TAKE_PER_ACTION:
                break

print(f'选定 {len(picked)} 个样本:')
for r in picked:
    print(f"  {r['pair']}  {r['source_action']}#{r['source_frame_id']} -> "
          f"{r['target_action']}#{r['target_frame_id']}  contact={r['target_contact_vertex_count_total']}  "
          f"cross={not r['same_action']}")

os.makedirs(SET, exist_ok=True)
manifest = []
for i, r in enumerate(picked, start=1):
    sid = f'inter_{i:02d}_{r["pair"]}_{r["source_action"].rstrip("0123456789")}'
    d = os.path.join(SET, sid)
    os.makedirs(d, exist_ok=True)
    rec = {'id': sid, 'pair': r['pair'], 'camera_id': r['camera_id'],
           'interaction': MEANING.get(r['target_action'], r['target_action']),
           'source_action': r['source_action'], 'target_action': r['target_action'],
           'cross_action': not r['same_action'],
           'source_frame_id': r['source_frame_id'], 'target_frame_id': r['target_frame_id'],
           'target_contact_vertex_count_total': r['target_contact_vertex_count_total'],
           'source_contact': r['source_contact'], 'target_contact': r['target_contact']}
    for role, key in (('source', 'source_rgb'), ('target', 'target_rgb')):
        src = r[key]
        dst = os.path.join(d, f'{role}.png')
        im = Image.open(src).convert('RGB')
        im.save(dst, quality=95)
        rec[f'{role}_path'] = dst
        rec[f'{role}_size'] = list(im.size)
    # 逐人掩码，供后续 mask 引导变体使用
    for role in ('source', 'target'):
        for who in ('A', 'B'):
            k = f'{role}_person_mask_{who}'
            if r.get(k) and os.path.isfile(r[k]):
                dst = os.path.join(d, f'{role}_mask_{who}.png')
                shutil.copyfile(r[k], dst)
                rec[k] = dst
    manifest.append(rec)

with open(os.path.join(SET, 'manifest.json'), 'w') as fh:
    json.dump({'task': 'Hi4D MAIN_N2C  double-person non-contact -> contact',
               'source': 'Hi4D_pilot_v1/native_interaction_v4_2 main_pairs_test.jsonl',
               'note': 'source = appearance/identity, target = interaction pose',
               'samples': manifest}, fh, indent=2, ensure_ascii=False)

# 预览图：source | target 并排
TH = 400
PAD = 8
rows_n = len(manifest)
W = PAD + 2 * (TH + PAD)
Hh = PAD + rows_n * (TH + 30 + PAD)
sheet = Image.new('RGB', (W, Hh), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for i, rec in enumerate(manifest):
    y = PAD + i * (TH + 30 + PAD)
    for c, role in enumerate(('source', 'target')):
        im = Image.open(rec[f'{role}_path']).convert('RGB')
        im.thumbnail((TH, TH))
        x = PAD + c * (TH + PAD)
        sheet.paste(im, (x + (TH - im.width) // 2, y))
        draw.rectangle([x, y, x + TH, y + TH], outline=(170, 170, 170))
    draw.text((PAD + 4, y + TH + 8),
              f"{rec['id']}   SOURCE  {rec['source_action']} (non-contact)", fill=(10, 10, 10))
    draw.text((PAD + TH + PAD + 4, y + TH + 8),
              f"TARGET  {rec['target_action']} (contact={rec['target_contact_vertex_count_total']})",
              fill=(10, 10, 10))
sheet.save(os.path.join(SET, 'sample_preview.png'), optimize=True)
print('wrote', os.path.join(SET, 'sample_preview.png'), sheet.size)
print('DONE')
