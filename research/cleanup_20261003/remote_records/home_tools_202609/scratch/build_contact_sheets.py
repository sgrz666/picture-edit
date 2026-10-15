#!/usr/bin/env python3
"""Build labelled contact sheets from candidate buckets for fast visual review."""
import json
import os
import sys

from PIL import Image, ImageDraw

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/candidates'
SHEET_DIR = os.path.join(BASE, 'contact_sheets')

COLS = 4
ROWS_PER_PAGE = 3
THUMB_W = 260
THUMB_H = 360
PAD = 8
LABEL_H = 26
FONT = None


def build(bucket):
    src = os.path.join(BASE, bucket)
    if not os.path.isdir(src):
        print(f'skip {bucket}: no dir')
        return
    names = sorted(n for n in os.listdir(src)
                   if n.lower().endswith(('.jpg', '.jpeg', '.png')))
    if not names:
        print(f'skip {bucket}: no images')
        return

    per_page = COLS * ROWS_PER_PAGE
    pages = [names[i:i + per_page] for i in range(0, len(names), per_page)]
    os.makedirs(SHEET_DIR, exist_ok=True)
    for page_index, chunk in enumerate(pages, start=1):
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new('RGB', (COLS * (THUMB_W + PAD) + PAD,
                                 rows * (THUMB_H + LABEL_H + PAD) + PAD), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(chunk):
            idx = names.index(name) + 1
            col, row = i % COLS, i // COLS
            x = PAD + col * (THUMB_W + PAD)
            y = PAD + row * (THUMB_H + LABEL_H + PAD)
            try:
                im = Image.open(os.path.join(src, name)).convert('RGB')
            except Exception:
                continue
            im.thumbnail((THUMB_W, THUMB_H))
            ox = x + (THUMB_W - im.width) // 2
            oy = y + (THUMB_H - im.height) // 2
            sheet.paste(im, (ox, oy))
            draw.rectangle([x, y, x + THUMB_W, y + THUMB_H], outline=(170, 170, 170))
            label = f'#{idx} {im.width}x{im.height} {name[:30]}'
            draw.text((x + 3, y + THUMB_H + 6), label, fill=(15, 15, 15))
        out = os.path.join(SHEET_DIR, f'{bucket}_p{page_index}.png')
        sheet.save(out, optimize=True)
        print(f'wrote {out}  ({len(chunk)} items, {sheet.width}x{sheet.height})')

    with open(os.path.join(SHEET_DIR, f'{bucket}_index.json'), 'w') as fh:
        json.dump({str(i + 1): n for i, n in enumerate(names)}, fh, indent=2)
    print(f'{bucket}: {len(names)} images, {len(pages)} sheet page(s)')


if __name__ == '__main__':
    buckets = sys.argv[1:] or ['heavy', 'slim']
    for b in buckets:
        build(b)
