import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path('two_rl_sheet')
META = json.loads((SRC / 'meta.json').read_text(encoding='utf8'))
OUT = Path('out')
OUT.mkdir(exist_ok=True)

FONT = 'C:/Windows/Fonts/msyh.ttc'
f_title = ImageFont.truetype(FONT, 26)
f_sub = ImageFont.truetype(FONT, 14)
f_label = ImageFont.truetype(FONT, 15)
f_tag = ImageFont.truetype(FONT, 13)
f_foot = ImageFont.truetype(FONT, 13)

IMG_H = 420
PAD = 28
GAP_X = 28
GAP_Y = 24
CELL_W = 730
LABEL_H = 46
CELL_H = LABEL_H + IMG_H
COLS, ROWS = 2, 5
PER_SHEET = COLS * ROWS

W = PAD * 2 + CELL_W * COLS + GAP_X * (COLS - 1)
HEADER_H = 62
FOOTER_H = 40
H = PAD + HEADER_H + ROWS * (CELL_H + GAP_Y) - GAP_Y + FOOTER_H + PAD

BG = (255, 255, 255)
INK = (27, 36, 48)
MUTED = (110, 122, 138)
LINE = (223, 229, 236)
ACCENT = (15, 118, 110)


def fit(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    nw = max(1, round(w * IMG_H / h))
    return im.resize((nw, IMG_H), Image.LANCZOS), (w, h)


def build(sheet_idx, items):
    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)

    d.text((PAD, PAD), f'小红书 two-rl · 随机配对（seed 20261124）· 第 {sheet_idx}/2 张', font=f_title, fill=INK)
    d.text((PAD, PAD + 34), f'每格左 A（source）｜右 B（target）· 共 {len(items)} 对 · 数据集 xiaohognshu(two-rl)',
           font=f_sub, fill=MUTED)

    for i, item in enumerate(items):
        col, row = i % COLS, i // COLS
        x = PAD + col * (CELL_W + GAP_X)
        y = PAD + HEADER_H + row * (CELL_H + GAP_Y)

        a, a_size = fit(SRC / item['a'])
        b, b_size = fit(SRC / item['b'])
        label = f"{item['idx']:02d}  {item['pair_id']}"
        d.text((x + 2, y + 1), label, font=f_label, fill=INK)
        lw = d.textlength(label, font=f_label)
        size_txt = f"{a_size[0]}×{a_size[1]}  ·  {item['source_tag']}"
        d.text((x + 16 + lw, y + 3), size_txt, font=f_sub, fill=MUTED)
        files = f"{item['a_name']}  |  {item['b_name']}"
        while d.textlength(files, font=f_sub) > CELL_W - 8 and len(files) > 12:
            files = files[:-2]
        d.text((x + 2, y + 24), files, font=f_sub, fill=MUTED)

        total = a.width + 12 + b.width
        sx = x + (CELL_W - total) // 2
        iy = y + LABEL_H
        canvas.paste(a, (sx, iy))
        canvas.paste(b, (sx + a.width + 12, iy))
        d.rectangle([sx, iy, sx + a.width - 1, iy + IMG_H - 1], outline=LINE, width=1)
        d.rectangle([sx + a.width + 12, iy, sx + a.width + 12 + b.width - 1, iy + IMG_H - 1], outline=LINE, width=1)

        for tx, tw, name in ((sx, a.width, 'A'), (sx + a.width + 12, b.width, 'B')):
            d.rectangle([tx + 6, iy + 6, tx + 6 + 20, iy + 6 + 20], fill=ACCENT)
            d.text((tx + 11, iy + 8), name, font=f_tag, fill=(255, 255, 255))

    d.text((PAD, H - PAD - 20), '抽样口径：pairs.jsonl 中 disk_status=complete 且两图存在（814 对）· Python random.seed(20261124) · 不做人工筛选',
           font=f_foot, fill=MUTED)
    path = OUT / f'two_rl_random20_sheet{sheet_idx}.png'
    canvas.save(path)
    print('saved', path, canvas.size)


build(1, META[:PER_SHEET])
build(2, META[PER_SHEET:])
