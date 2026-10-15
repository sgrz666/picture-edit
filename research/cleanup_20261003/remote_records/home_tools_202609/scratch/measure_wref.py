#!/usr/bin/env python3
"""Quantify: how close is the w_ref output to the GT target? Is the model copying or regenerating?"""
import json
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test'
SET = os.path.join(BASE, 'data/interaction/testset')
GEN = os.path.join(BASE, 'experiment/results/interaction_v1')

man = {s['id']: s for s in json.load(open(os.path.join(SET, 'manifest.json')))['samples']}
res = json.load(open(os.path.join(GEN, 'results.json')))

SIZE = (470, 640)   # 统一到参考帧的宽高比


def load(p):
    return np.asarray(Image.open(p).convert('RGB').resize(SIZE, Image.LANCZOS), dtype=np.float32)


def metrics(a, b):
    mse = float(np.mean((a - b) ** 2))
    psnr = float('inf') if mse == 0 else float(-10 * np.log10(mse / 255 ** 2))
    s = float(ssim(a, b, channel_axis=2, data_range=255))
    return psnr, s


print(f"{'sample':34s} {'arm':9s} {'vs TARGET psnr/ssim':>22s} {'vs SOURCE psnr/ssim':>22s}")
print('-' * 92)
agg = {}
for r in res:
    s = man[r['sample']]
    out = load(os.path.join(GEN, 'images', r['id'] + '.png'))
    tgt = load(s['target_path'])
    src = load(s['source_path'])
    pt, st = metrics(out, tgt)
    ps, ss = metrics(out, src)
    print(f"{r['id']:34s} {'':9s} {pt:10.2f} dB / {st:.3f}   {ps:10.2f} dB / {ss:.3f}")
    agg.setdefault(r['arm'], []).append((pt, st, ps, ss))

print('-' * 92)
print(f"{'arm':10s} {'vs TARGET psnr':>15s} {'ssim':>8s} {'vs SOURCE psnr':>16s} {'ssim':>8s}   n")
for arm, v in agg.items():
    a = np.array(v)
    print(f"{arm:10s} {a[:,0].mean():15.2f} {a[:,1].mean():8.3f} {a[:,2].mean():16.2f} {a[:,3].mean():8.3f}   {len(v)}")

# 参照：target 自己 vs source（两个不同帧的差异基线）
print()
print('=== 参照基线（两帧本身差异）===')
ts, ss_ = [], []
for sid, s in man.items():
    tgt = load(s['target_path']); src = load(s['source_path'])
    p, q = metrics(tgt, src)
    ts.append(q)
print(f"  TARGET vs SOURCE (两帧原始差异) SSIM 均值 = {np.mean(ts):.3f}  ← 这是「完全不同」的水平")

# 参照：把 target 自身做一次 VAE 往返无法在此计算，改用 JPEG 质量损失模拟下界不适用；
# 改给出「target 与原尺寸缩放回自身」= 1.0 的理论上界说明
print(f"  TARGET vs TARGET (完全相同的上界) SSIM = 1.000")
print()
print('=== 每个交互类型 w_ref 与 target 的 SSIM ===')
for t in sorted({r['interaction'] for r in res}):
    v = [x for x in agg['w_ref'] if True]
    ids = [r for r in res if r['interaction'] == t and r['arm'] == 'w_ref']
    if not ids:
        continue
    r = ids[0]; s = man[r['sample']]
    out = load(os.path.join(GEN, 'images', r['id'] + '.png'))
    _, sv = metrics(out, load(s['target_path']))
    print(f'  {t:32s} w_ref SSIM={sv:.3f}')
