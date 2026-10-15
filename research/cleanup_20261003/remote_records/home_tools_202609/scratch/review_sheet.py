#!/usr/bin/env python3
"""Compact review sheet: A | B | generated, 3 samples per page."""
import json
import os

from PIL import Image, ImageDraw

SET = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/testset'
GEN = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/gen_v1'
OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/gen_v1'

TH_W, TH_H = 280, 420
PAD = 8
LABEL = 24
PER_PAGE = 3

man = {p['pair']: p for p in json.load(open(os.path.join(SET, 'manifest.json')))['pairs']}
res = json.load(open(os.path.join(GEN, 'results.json')))

order = ['hard_01', 'hard_02', 'hard_03', 'hard_04', 'hard_05', 'hard_05__minimal',
         'medium_01', 'medium_01__minimal', 'medium_02']
items = [r for k in order for r in res if r['id'] == k]

pages = [items[i:i + PER_PAGE] for i in range(0, len(items), PER_PAGE)]
for pi, chunk in enumerate(pages, start=1):
    rows = len(chunk)
    W = PAD + 3 * (TH_W + PAD)
    H = PAD + rows * (TH_H + LABEL + PAD)
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(chunk):
        p = man[r['base_pair']]
        y = PAD + i * (TH_H + LABEL + PAD)
        ims = [Image.open(p['A_path']).convert('RGB'),
               Image.open(p['B_path']).convert('RGB'),
               Image.open(os.path.join(GEN, 'images', r['id'] + '.png')).convert('RGB')]
        for c, im in enumerate(ims):
            im.thumbnail((TH_W, TH_H))
            x = PAD + c * (TH_W + PAD)
            sheet.paste(im, (x + (TH_W - im.width) // 2, y))
            draw.rectangle([x, y, x + TH_W, y + TH_H], outline=(170, 170, 170))
        draw.text((PAD + 4, y + TH_H + 6), 'A  ' + r['base_pair'], fill=(10, 10, 10))
        draw.text((PAD + TH_W + PAD + 4, y + TH_H + 6), 'B  (pose ref)', fill=(10, 10, 10))
        draw.text((PAD + 2 * (TH_W + PAD) + 4, y + TH_H + 6), 'OUTPUT  ' + r['id'], fill=(10, 10, 10))
    out = os.path.join(OUT, f'review_p{pi}.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)
