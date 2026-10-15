#!/usr/bin/env python3
"""Re-measure w_ref fidelity inside the PERSON region only (green screen dominates full-frame SSIM)."""
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


def person_bbox(sample, size, margin=0.06):
    """目标帧两人掩码的联合外接框（归一化坐标）。"""
    w, h = size
    union = None
    for who in ('A', 'B'):
        p = sample.get(f'target_person_mask_{who}')
        if not p or not os.path.isfile(p):
            continue
        m = np.asarray(Image.open(p).convert('L').resize(size, Image.NEAREST)) > 127
        union = m if union is None else (union | m)
    if union is None or not union.any():
        return (0, 0, w, h)
    ys, xs = np.where(union)
    mx, my = int((xs.max() - xs.min()) * margin), int((ys.max() - ys.min()) * margin)
    return (max(0, xs.min() - mx), max(0, ys.min() - my),
            min(w, xs.max() + mx), min(h, ys.max() + my))


def load_crop(p, size, box):
    im = Image.open(p).convert('RGB').resize(size, Image.LANCZOS)
    return np.asarray(im.crop(box), dtype=np.float32)


def metrics(a, b):
    mse = float(np.mean((a - b) ** 2))
    psnr = float('inf') if mse == 0 else float(-10 * np.log10(mse / 255 ** 2))
    return psnr, float(ssim(a, b, channel_axis=2, data_range=255))


SIZE = (470, 640)
print(f"{'id':36s} {'out vs TARGET':>18s} {'out vs SOURCE':>18s}")
print('-' * 76)
agg = {}
tgt_src = []
for r in res:
    s = man[r['sample']]
    box = person_bbox(s, SIZE)
    out = load_crop(os.path.join(GEN, 'images', r['id'] + '.png'), SIZE, box)
    tgt = load_crop(s['target_path'], SIZE, box)
    src = load_crop(s['source_path'], SIZE, box)
    pt, st = metrics(out, tgt)
    ps, ss = metrics(out, src)
    print(f"{r['id']:36s} {pt:8.2f} dB {st:.3f}   {ps:8.2f} dB {ss:.3f}")
    agg.setdefault(r['arm'], []).append((pt, st, ps, ss))
    if r['arm'] == 'w_ref':
        p2, s2 = metrics(tgt, src)
        tgt_src.append((p2, s2))

print('-' * 76)
print(f"{'arm':10s} | {'vs TARGET psnr':>14s} {'ssim':>7s} | {'vs SOURCE psnr':>14s} {'ssim':>7s} | n")
for arm, v in agg.items():
    a = np.array(v)
    print(f"{arm:10s} | {a[:,0].mean():14.2f} {a[:,1].mean():7.3f} | {a[:,2].mean():14.2f} {a[:,3].mean():7.3f} | {len(v)}")

a = np.array(tgt_src)
print()
print(f"参照：TARGET vs SOURCE 仅人物区域  = {a[:,0].mean():.2f} dB / SSIM {a[:,1].mean():.3f}   (「完全不同」的基线)")
print(f"参照：TARGET vs TARGET             = inf       / SSIM 1.000   (完全复制的上界)")
