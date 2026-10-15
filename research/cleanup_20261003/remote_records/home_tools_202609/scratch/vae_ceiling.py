#!/usr/bin/env python3
"""Estimate the copy ceiling: run the GT target through Qwen's VAE and measure round-trip SSIM.

If a hypothetical perfect copy is decoded through the same VAE the pipeline uses, the best
achievable SSIM vs the original is this round-trip number. Anything far below it is NOT a copy.
"""
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image
from skimage.metrics import structural_similarity as ssim

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test'
SET = os.path.join(BASE, 'data/interaction/testset')
MODEL = '/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1'

man = {s['id']: s for s in json.load(open(os.path.join(SET, 'manifest.json')))['samples']}
vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(MODEL, 'vae'), dtype=torch.bfloat16).to('cuda')
vae.eval()

SIZE = (768, 1024)     # 生成尺寸


def person_bbox(sample, size, margin=0.06):
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


print(f"{'sample':34s} {'VAE round-trip (person region)':>32s}")
print('-' * 70)
vals = []
for sid, s in man.items():
    im = Image.open(s['target_path']).convert('RGB').resize(SIZE, Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    rgb = arr / 127.5 - 1.0
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    x = np.concatenate([rgb, alpha], axis=2)          # H,W,4  <- VAE 原生 RGBA
    x = torch.from_numpy(x).permute(2, 0, 1)[None].unsqueeze(2).to('cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode()
        rec = vae.decode(lat).sample
    dec = rec[0]
    if dec.dim() == 4:
        dec = dec[:, 0]
    dec = dec[:3]
    out = ((dec.float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)
    box = person_bbox(s, SIZE)
    a = np.asarray(im.crop(box), dtype=np.float32)
    b = out[box[1]:box[3], box[0]:box[2]]
    mse = float(np.mean((a - b) ** 2))
    sv = float(ssim(a, b, channel_axis=2, data_range=255))
    psnr = float('inf') if mse == 0 else float(-10 * np.log10(mse / 255 ** 2))
    vals.append((psnr, sv))
    print(f"{sid:34s} {psnr:14.2f} dB / SSIM {sv:.3f}")
    for who in ('A', 'B'):
        pass

a = np.array(vals)
print('-' * 70)
print(f"VAE 往返上限（完美复制能达到的最好水平）= {a[:,0].mean():.2f} dB / SSIM {a[:,1].mean():.3f}")
print()
print("对照（人物区域）:")
print("  完全复制上界（VAE 往返） : SSIM ~%.3f" % a[:,1].mean())
print("  TARGET vs SOURCE（不同姿态）: SSIM 0.470")
print("  w_ref 输出 vs TARGET        : SSIM 0.469")
