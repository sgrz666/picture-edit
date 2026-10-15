#!/usr/bin/env python3
"""Render B (pose reference) images large enough to describe poses precisely."""
import json
import os

from PIL import Image, ImageDraw

SET = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/testset'
TH = 460
PAD = 10
LABEL = 30
COLS = 2
PER_FILE = 4

man = json.load(open(os.path.join(SET, 'manifest.json')))
pairs = man['pairs']

for part, chunk in enumerate([pairs[:PER_FILE], pairs[PER_FILE:]], start=1):
    if not chunk:
        continue
    rows = (len(chunk) + COLS - 1) // COLS
    W = PAD + COLS * (TH + PAD)
    H = PAD + rows * (TH + LABEL + PAD)
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(chunk):
        col, row = i % COLS, i // COLS
        x = PAD + col * (TH + PAD)
        y = PAD + row * (TH + LABEL + PAD)
        im = Image.open(p['B_path']).convert('RGB')
        im.thumbnail((TH, TH))
        sheet.paste(im, (x + (TH - im.width) // 2, y))
        draw.rectangle([x, y, x + TH, y + TH], outline=(160, 160, 160))
        draw.text((x + 4, y + TH + 8), f"{p['pair']}  B  {im.width}x{im.height}", fill=(10, 10, 10))
    out = os.path.join(SET, f'B_sheet_p{part}.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)
