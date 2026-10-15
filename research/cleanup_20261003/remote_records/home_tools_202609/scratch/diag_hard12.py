import json
import os

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test'
SET = f'{BASE}/data/testset'
GEN = f'{BASE}/experiment/results/gen_v1/images'
man = {p['pair']: p for p in json.load(open(f'{SET}/manifest.json'))['pairs']}

def load(p, size=None):
    im = Image.open(p).convert('RGB')
    if size:
        im = im.resize(size, Image.LANCZOS)
    return im

S = (300, 400)
p1, p2 = man['hard_01'], man['hard_02']
A1, B1, O1 = load(p1['A_path'], S), load(p1['B_path'], S), load(f'{GEN}/hard_01.png', S)
A2, B2, O2 = load(p2['A_path'], S), load(p2['B_path'], S), load(f'{GEN}/hard_02.png', S)

def cmp(x, y, label):
    a, b = np.asarray(x, np.float32), np.asarray(y, np.float32)
    mse = float(np.mean((a-b)**2))
    print(f'  {label:34s} mse={mse:9.2f}  psnr={-10*np.log10(mse/255**2):7.2f}  '
          f'ssim={ssim(a, b, channel_axis=2, data_range=255):.4f}')

print('=== 输入与输出的成对相似度（统一缩放到 300x400） ===')
cmp(A1, A2, 'A(hard_01) vs A(hard_02)')
cmp(B1, B2, 'B(hard_01) vs B(hard_02)')
cmp(O1, O2, 'OUTPUT(hard_01) vs OUTPUT(hard_02)')
print()
cmp(A1, O1, 'hard_01: A  vs OUTPUT')
cmp(B1, O1, 'hard_01: B  vs OUTPUT')
cmp(A2, O2, 'hard_02: A  vs OUTPUT')
cmp(B2, O2, 'hard_02: B  vs OUTPUT')

# 拼图
T = (300, 400)
lab = Image.new('RGB', (6*(T[0]+6)+6, T[1]+30), (255, 255, 255))
dr = ImageDraw.Draw(lab)
for i, (im, name) in enumerate([(A1, 'A1 (hard_01 src)'), (B1, 'B1 (hard_01 pose)'),
                                (O1, 'OUT hard_01'), (A2, 'A2 (hard_02 src)'),
                                (B2, 'B2 (hard_02 pose)'), (O2, 'OUT hard_02')]):
    x = 6 + i*(T[0]+6)
    lab.paste(im, (x, 24))
    dr.rectangle([x, 24, x+T[0], 24+T[1]], outline=(150,150,150))
    dr.text((x+4, 6), name, fill=(10,10,10))
lab.save(f'{BASE}/experiment/results/hard12_compare.png', optimize=True)
print()
print('wrote hard12_compare.png', lab.size)

# 各自原尺寸
for nm, p in (('A1', p1['A_path']), ('B1', p1['B_path']), ('A2', p2['A_path']), ('B2', p2['B_path'])):
    im = Image.open(p)
    print(f'  {nm}: {im.size}  {os.path.basename(p)}')
for nm in ('hard_01', 'hard_02'):
    im = Image.open(f'{GEN}/{nm}.png')
    print(f'  {nm} output: {im.size}')
