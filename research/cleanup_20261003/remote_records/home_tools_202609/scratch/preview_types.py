#!/usr/bin/env python3
"""Preview one sample per interaction type, source|target, 3 rows per page."""
import json
import os

from PIL import Image, ImageDraw

SET = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/interaction/testset'
man = json.load(open(os.path.join(SET, 'manifest.json')))['samples']
by_int = {}
for s in man:
    by_int.setdefault(s['interaction'], []).append(s)
one = [v[0] for v in by_int.values()]
order = ['embrace', 'taichi partner push-hands', 'partner yoga',
         'embrace from behind', 'partner dance', 'partner pose']
one.sort(key=lambda s: order.index(s['interaction']) if s['interaction'] in order else 99)

TH = 400
PAD = 8
for pi, chunk in enumerate([one[:3], one[3:]], start=1):
    if not chunk:
        continue
    W = PAD + 2 * (TH + PAD)
    Hh = PAD + len(chunk) * (TH + 30 + PAD)
    sheet = Image.new('RGB', (W, Hh), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(chunk):
        y = PAD + i * (TH + 30 + PAD)
        for c, role in enumerate(('source', 'target')):
            im = Image.open(rec[f'{role}_path']).convert('RGB')
            im.thumbnail((TH, TH))
            x = PAD + c * (TH + PAD)
            sheet.paste(im, (x + (TH - im.width) // 2, y))
            draw.rectangle([x, y, x + TH, y + TH], outline=(170, 170, 170))
        draw.text((PAD + 4, y + TH + 8),
                  f"{rec['id']}  SOURCE {rec['source_action']} (non-contact)", fill=(10, 10, 10))
        draw.text((PAD + TH + PAD + 4, y + TH + 8),
                  f"TARGET {rec['target_action']} = {rec['interaction']}", fill=(10, 10, 10))
    out = os.path.join(SET, f'preview_types_p{pi}.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)
