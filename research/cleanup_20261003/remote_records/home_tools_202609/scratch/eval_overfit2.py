import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment'
D = f'{BASE}/results/overfit_inter01/images'
HI4D = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'

# 三联条布局：3 个 384x384 贴片，间距 8，顶部 26px 标签
X = [8 + i * (384 + 8) for i in range(3)]
BOX = lambda i: (X[i], 26, X[i] + 384, 410)

def load_panel(path, i):
    return Image.open(path).convert('RGB').crop(BOX(i))

# 目标 GT 面板（第 2 格）
tgt_panel = load_panel(f'{D}/step_0000_on.png', 1)
G = np.load(f'{HI4D}/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz', allow_pickle=True)
def mask_panel(m):
    return np.asarray(Image.fromarray((m > 0).astype(np.uint8) * 255).resize((384, 384), Image.NEAREST)) > 127
u = mask_panel(G['semantic_A']) | mask_panel(G['semantic_B'])
ys, xs = np.where(u)
mx = int((xs.max()-xs.min())*0.06); my = int((ys.max()-ys.min())*0.06)
box = (max(0, xs.min()-mx), max(0, ys.min()-my), min(384, xs.max()+mx), min(384, ys.max()+my))
print('人物区域 box =', box)
ta = np.asarray(tgt_panel.crop(box), np.float32)

def m(img):
    a = np.asarray(img.crop(box), np.float32)
    mse = float(np.mean((a - ta)**2))
    return float(-10*np.log10(mse/255**2)), float(ssim(a, ta, channel_axis=2, data_range=255))

steps = sorted({int(os.path.basename(p).split('_')[1]) for p in glob.glob(f'{D}/step_*_on.png')})
print()
print(f'{"step":>6} | {"OFF psnr":>8} {"ssim":>7} | {"ON psnr":>8} {"ssim":>7} | {"Δssim":>8} {"Δpsnr":>7}')
print('-'*70)
rows = []
for st in steps:
    off = load_panel(f'{D}/step_{st:04d}_off.png', 2)
    on = load_panel(f'{D}/step_{st:04d}_on.png', 2)
    po, so = m(off); pn, sn = m(on)
    rows.append({'step': st, 'off_psnr': round(po,3), 'off_ssim': round(so,5),
                 'on_psnr': round(pn,3), 'on_ssim': round(sn,5)})
    print(f'{st:6d} | {po:8.3f} {so:7.4f} | {pn:8.3f} {sn:7.4f} | {sn-so:+8.4f} {pn-po:+7.3f}')
json.dump(rows, open(f'{BASE}/results/overfit_inter01/metrics.json','w'), indent=2)

# 预览表
T = 240
sheet = Image.new('RGB', (4*(T+6)+6, len(steps)*(T+24+6)+6), (255,255,255))
dr = ImageDraw.Draw(sheet)
for ri, st in enumerate(steps):
    y = 6 + ri*(T+24+6)
    ims = [load_panel(f'{D}/step_{st:04d}_on.png', 0), load_panel(f'{D}/step_{st:04d}_on.png', 1),
           load_panel(f'{D}/step_{st:04d}_off.png', 2), load_panel(f'{D}/step_{st:04d}_on.png', 2)]
    for ci, im in enumerate(ims):
        x = 6 + ci*(T+6)
        sheet.paste(im.resize((T, T)), (x, y))
        dr.rectangle([x, y, x+T, y+T], outline=(160,160,160))
    dr.text((6, y+T+5), f'step {st}  |  SOURCE', fill=(10,10,10))
    dr.text((6+(T+6), y+T+5), 'TARGET GT', fill=(10,10,10))
    dr.text((6+2*(T+6), y+T+5), 'OFF', fill=(10,10,10))
    dr.text((6+3*(T+6), y+T+5), 'ON (adapter)', fill=(150,30,30))
sheet.save(f'{BASE}/results/overfit_inter01/overfit_review.png', optimize=True)
print(); print('wrote overfit_review.png', sheet.size)
