import glob, os
import numpy as np
from PIL import Image, ImageDraw

TEST = '/home/shangguanrz/project/pic-edit/experiments/16-test'
B1 = f'{TEST}/stage1_baselines/images'
B2 = f'{TEST}/stage2_adapters/images'
S = 512
W = int(open('/dev/null').read() or 0) if False else None

def firstof(*paths):
    for q in paths:
        if os.path.isfile(q):
            return q
    raise FileNotFoundError(' / '.join(paths))


def load(p, H=440):
    im = Image.open(p).convert('RGB')
    w = max(1, round(im.width * H / im.height))
    return im.resize((w, H), Image.LANCZOS)

def lb(im, H=440):
    w = max(1, round(im.width * H / im.height))
    return im.resize((w, H), Image.LANCZOS)

A = lb(Image.open(glob.glob(f'{TEST}/source/p-A/*.jpg')[0]).convert('RGB'))
B = lb(Image.open(glob.glob(f'{TEST}/source/p-B/*.jpg')[0]).convert('RGB'))
H = 440
KEYS = ['zh_simple', 'zh_complex', 'en_simple', 'en_complex']

rows = []
rows.append(('REFERENCE', [A, B], ['A (外貌来源)', 'B (姿态参考)']))
rows.append(('DeepGen 基干', [load(firstof(f'{B1}/deepgen_base_{k}_cropped.png',
                                           f'{B1}/deepgen_base_{k}_raw.png'), H)
                                for k in KEYS], KEYS))
rows.append(('DeepGen + adapter', [load(firstof(f'{B2}/deepgen_adapter_{k}_raw.png',
                                                    f'{B2}/deepgen_adapter_source_{k}_raw.png'), H)
                                     for k in KEYS], KEYS))
rows.append(('Qwen-2.1 基干', [load(f'{B1}/qwen_base_{k}_raw.png', H) for k in KEYS], KEYS))
rows.append(('Qwen-2.1 + adapter', [load(firstof(f'{B2}/qwen_adapter_{k}_raw.png',
                                                  f'{B2}/qwen_adapter_source_{k}_raw.png'), H)
                                       for k in KEYS], KEYS))

cellw = max(max(sum(im.width + 8 for im in ims) for _, ims, _ in rows) + 20, 1100)
totalh = sum(H + 46 for _ in rows) + 16
c = Image.new('RGB', (cellw, totalh), (255, 255, 255))
d = ImageDraw.Draw(c)
try:
    from PIL import ImageFont
    f1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    f2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
except Exception:
    f1 = f2 = None
y = 8
for title, ims, labels in rows:
    d.text((10, y), title, fill=(150, 30, 30), font=f1)
    y += 24
    x = 10
    for im, lb2 in zip(ims, labels):
        c.paste(im, (x, y))
        d.rectangle([x, y, x + im.width, y + H], outline=(160, 160, 160))
        d.text((x + 3, y + H + 3), lb2, fill=(30, 30, 30), font=f2)
        x += im.width + 8
    y += H + 46
c.save(f'{TEST}/stage2_adapters/grid_16.png', optimize=True)
print('wrote grid_16.png', c.size)
print('rows:', [r[0] for r in rows])
print('每行图数:', [len(r[1]) for r in rows])
