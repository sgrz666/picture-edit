#!/usr/bin/env python3
"""Render a side-by-side preview of each test pair for visual validation."""
import json
import os

from PIL import Image, ImageDraw

SET = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/testset'
OUT = os.path.join(SET, 'pair_preview.png')

TH = 400
PAD = 10
LABEL = 28

man = json.load(open(os.path.join(SET, 'manifest.json')))
pairs = man['pairs']

for part, chunk in enumerate([pairs[:4], pairs[4:]], start=1):
    if not chunk:
        continue
    rows = len(chunk)
    width = PAD + (TH + PAD) * 2 + 230
    height = PAD + rows * (TH + LABEL + PAD)
    sheet = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(chunk):
        y = PAD + i * (TH + LABEL + PAD)
        a = Image.open(p['A_path']).convert('RGB')
        b = Image.open(p['B_path']).convert('RGB')
        a.thumbnail((TH, TH))
        b.thumbnail((TH, TH))
        sheet.paste(a, (PAD + (TH - a.width) // 2, y))
        sheet.paste(b, (PAD + TH + PAD + (TH - b.width) // 2, y))
        draw.rectangle([PAD, y, PAD + TH, y + TH], outline=(170, 170, 170))
        draw.rectangle([PAD + TH + PAD, y, PAD + TH + PAD + TH, y + TH], outline=(170, 170, 170))
        draw.text((PAD, y + TH + 6), f"A (appearance, heavy)  {p['A_size']}", fill=(10, 10, 10))
        draw.text((PAD + TH + PAD, y + TH + 6), f"B (pose, slim)  {p['B_size']}", fill=(10, 10, 10))
        draw.text((PAD + (TH + PAD) * 2 + 12, y + 16), p['pair'], fill=(10, 10, 10))
        draw.text((PAD + (TH + PAD) * 2 + 12, y + 38), 'left = A', fill=(90, 90, 90))
        draw.text((PAD + (TH + PAD) * 2 + 12, y + 56), 'right = B', fill=(90, 90, 90))
    out = os.path.join(SET, f'pair_preview_p{part}.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)
