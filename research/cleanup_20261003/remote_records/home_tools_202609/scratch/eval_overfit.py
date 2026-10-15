import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment'
D = f'{BASE}/results/overfit_inter01/images'
HI4D = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
S = 512

src = Image.open(f'{HI4D}/raw/pair00/taichi00/images/4/000041.jpg').convert('RGB').resize((S, S), Image.LANCZOS)
tgt = Image.open(f'{HI4D}/raw/pair00/hug00/images/4/000108.jpg').convert('RGB').resize((S, S), Image.LANCZOS)
# 用目标帧的两人掩码联合外接框作为评估区域
G = np.load(f'{HI4D}/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz', allow_pickle=True)
ma = np.asarray(Image.fromarray((G['semantic_A'] > 0).astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) > 127
mb = np.asarray(Image.fromarray((G['semantic_B'] > 0).astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) > 127
u = ma | mb
ys, xs = np.where(u)
mx = int((xs.max() - xs.min()) * 0.06); my = int((ys.max() - ys.min()) * 0.06)
box = (max(0, xs.min()-mx), max(0, ys.min()-my), min(S, xs.max()+mx), min(S, ys.max()+my))
print('评估区域 box =', box)
ta = np.asarray(tgt.crop(box), np.float32)

def m(img):
    a = np.asarray(img.crop(box), np.float32)
    mse = float(np.mean((a - ta) ** 2))
    return float(-10 * np.log10(mse / 255**2)), float(ssim(a, ta, channel_axis=2, data_range=255))

steps = sorted({int(os.path.basename(p).split('_')[1]) for p in glob.glob(f'{D}/step_*_on.png')})
print()
print(f'{"step":>6} | {"OFF psnr":>9} {"ssim":>7} | {"ON psnr":>9} {"ssim":>7} | {"Δssim":>8}')
print('-' * 62)
rows = []
for st in steps:
    off = Image.open(f'{D}/step_{st:04d}_off.png').convert('RGB')
    on = Image.open(f'{D}/step_{st:04d}_on.png').convert('RGB')
    po, so = m(off); pn, sn = m(on)
    rows.append({'step': st, 'off_psnr': po, 'off_ssim': so, 'on_psnr': pn, 'on_ssim': sn})
    print(f'{st:6d} | {po:9.2f} {so:7.4f} | {pn:9.2f} {sn:7.4f} | {sn-so:+8.4f}')
json.dump(rows, open(f'{BASE}/results/overfit_inter01/metrics.json', 'w'), indent=2)

# 预览表：SOURCE | TARGET | OFF | ON，按 step 分行
T = 230
rows_img = []
for st in steps:
    rows_img.append((st, [src, tgt,
                          Image.open(f'{D}/step_{st:04d}_off.png').convert('RGB'),
                          Image.open(f'{D}/step_{st:04d}_on.png').convert('RGB')]))
W = 4 * (T + 6) + 6
Hh = len(rows_img) * (T + 24 + 6) + 6
sheet = Image.new('RGB', (W, Hh), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for ri, (st, ims) in enumerate(rows_img):
    y = 6 + ri * (T + 24 + 6)
    for ci, im in enumerate(ims):
        im = im.resize((T, T))
        x = 6 + ci * (T + 6)
        sheet.paste(im, (x, y))
        dr.rectangle([x, y, x + T, y + T], outline=(160, 160, 160))
    dr.text((6, y + T + 5), f'step {st}', fill=(10, 10, 10))
    dr.text((6 + (T+6), y + T + 5), 'TARGET GT', fill=(10, 10, 10))
    dr.text((6 + 2*(T+6), y + T + 5), 'OFF', fill=(10, 10, 10))
    dr.text((6 + 3*(T+6), y + T + 5), 'ON (adapter)', fill=(150, 30, 30))
sheet.save(f'{BASE}/results/overfit_inter01/overfit_review.png', optimize=True)
print()
print('wrote overfit_review.png', sheet.size)
