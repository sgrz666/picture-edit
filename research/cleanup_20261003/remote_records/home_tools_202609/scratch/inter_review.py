#!/usr/bin/env python3
"""Review sheet: per interaction type, show SOURCE | TARGET | detailed | minimal | w_ref."""
import json
import os

from PIL import Image, ImageDraw

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test'
SET = os.path.join(BASE, 'data/interaction/testset')
GEN = os.path.join(BASE, 'experiment/results/interaction_v1')
OUT = GEN

man = json.load(open(os.path.join(SET, 'manifest.json')))['samples']
by_id = {s['id']: s for s in man}
res = json.load(open(os.path.join(GEN, 'results.json')))

order = ['embrace', 'taichi partner push-hands', 'partner yoga',
         'embrace from behind', 'partner dance', 'partner pose']
types = sorted({r['interaction'] for r in res},
               key=lambda t: order.index(t) if t in order else 99)

TH_W, TH_H = 210, 300
PAD = 8
LABEL = 26
COLS = ['SOURCE', 'TARGET ref', 'detailed', 'minimal', 'w_ref']

for pi, chunk in enumerate([types[:2], types[2:4], types[4:]], start=1):
    if not chunk:
        continue
    W = PAD + len(COLS) * (TH_W + PAD)
    H = PAD + len(chunk) * (TH_H + LABEL + PAD)
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for ri, t in enumerate(chunk):
        rows = [r for r in res if r['interaction'] == t]
        s = by_id[rows[0]['sample']]
        y = PAD + ri * (TH_H + LABEL + PAD)
        paths = [s['source_path'], s['target_path']]
        for arm in ('detailed', 'minimal', 'w_ref'):
            m = [r for r in rows if r['arm'] == arm]
            paths.append(os.path.join(GEN, 'images', m[0]['id'] + '.png') if m else None)
        for ci, (label, p) in enumerate(zip(COLS, paths)):
            x = PAD + ci * (TH_W + PAD)
            if p and os.path.isfile(p):
                im = Image.open(p).convert('RGB')
                im.thumbnail((TH_W, TH_H))
                sheet.paste(im, (x + (TH_W - im.width) // 2, y))
            draw.rectangle([x, y, x + TH_W, y + TH_H], outline=(170, 170, 170))
            draw.text((x + 3, y + TH_H + 7), label, fill=(10, 10, 10))
        draw.text((PAD + 3, y - 1), t[:44], fill=(150, 30, 30))
    out = os.path.join(OUT, f'inter_review_p{pi}.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)
