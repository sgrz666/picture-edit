import glob
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment'
D = f'{BASE}/results/overfit_inter01/images'
HI4D = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
S = 512
box = (186, 118, 415, 427)

tgt = Image.open(f'{HI4D}/raw/pair00/hug00/images/4/000108.jpg').convert('RGB').resize((S, S), Image.LANCZOS)
ta = np.asarray(tgt.crop(box), dtype=np.float32)
print('tgt crop shape', ta.shape, ' mean', ta.mean(), ' std', ta.std())

for name in ('step_0000_off', 'step_0000_on', 'step_0300_on', 'step_0500_on'):
    im = Image.open(f'{D}/{name}.png').convert('RGB')
    a = np.asarray(im.crop(box), dtype=np.float32)
    mse = float(np.mean((a - ta) ** 2))
    psnr = -10 * np.log10(mse / 255 ** 2)
    sv = ssim(a, ta, channel_axis=2, data_range=255)
    print(f'{name:18s} crop{a.shape} mean={a.mean():8.3f} std={a.std():7.3f}  '
          f'mse={mse:9.2f} psnr={psnr:7.3f} ssim={sv:.6f}')

# 再做一次两图之间的直接差异
x = np.asarray(Image.open(f'{D}/step_0000_on.png').convert('RGB').crop(box), np.float32)
y = np.asarray(Image.open(f'{D}/step_0500_on.png').convert('RGB').crop(box), np.float32)
print()
print('ON(0) vs ON(500) 在评估区内: max|d|=', float(np.abs(x - y).max()), ' 均值=', float(np.abs(x - y).mean()))
