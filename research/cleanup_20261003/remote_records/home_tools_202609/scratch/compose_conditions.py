#!/usr/bin/env python3
"""Compose every adapter input condition for Hi4D inter_01_pair00_taichi into one figure."""
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
sys.path.insert(0, '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/code')
import hi4d_conditions as hc  # noqa: E402

HI4D = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/overfit_inter01/input_conditions.png'
S = 512
PAIR, SRC_ACT, TGT_ACT, SRC_F, TGT_F, CAM = 'pair00', 'taichi00', 'hug00', 41, 108, 4

FONT_PATHS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']


def font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = font(19)
F_LABEL = font(15)
F_SMALL = font(12)

PART_COLORS = [
    (232, 92, 92), (240, 166, 72), (245, 222, 96), (176, 220, 90),
    (98, 200, 140), (86, 196, 202), (96, 152, 228), (140, 118, 220),
    (204, 104, 200), (232, 110, 156), (150, 112, 84), (120, 160, 110),
    (110, 130, 190), (190, 160, 120),
]

batch = hc.build_sample(PAIR, TGT_ACT, TGT_F, CAM, resolution=S)
dbg = batch['_debug']
problems = hc.validate(batch, resolution=S)
print('契约校验:', '通过' if not problems else problems)


def t2img(t, mode='rgb'):
    a = t[0].detach().float().numpy()
    if mode == 'rgb':
        v = ((a.transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(v)
    if a.shape[0] == 1:
        v = (a[0] * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(v).convert('RGB')
    if mode == 'max':                       # 25 通道 heatmap 取最大投影
        v = (np.clip(a.max(axis=0), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(v).convert('RGB')
    if mode == 'parts':
        lab = a.argmax(axis=0)
        has = a.sum(axis=0) > 0
        out = np.full((S, S, 3), 236, dtype=np.uint8)
        for i, c in enumerate(PART_COLORS):
            out[(lab == i) & has] = c
        return Image.fromarray(out)
    raise ValueError(mode)


def load_rgb(act, frame):
    return Image.open(f'{HI4D}/raw/{PAIR}/{act}/images/{CAM}/{frame:06d}.jpg').convert('RGB').resize((S, S), Image.LANCZOS)


src_rgb = load_rgb(SRC_ACT, SRC_F)
tgt_rgb = load_rgb(TGT_ACT, TGT_F)

T = 232                      # 单格边长
GAP = 8
COLW = T + GAP
ROWS = 6
W = GAP + 4 * COLW
H = 74 + ROWS * (T + 46) + 250
sheet = Image.new('RGB', (W, H), (252, 252, 252))
dr = ImageDraw.Draw(sheet)

dr.rectangle([0, 0, W, 56], fill=(28, 32, 38))
dr.text((14, 8), 'Qwen21 body adapter — input conditions', font=F_TITLE, fill=(245, 245, 245))
dr.text((14, 32), f'Hi4D  {PAIR}  source={SRC_ACT}#{SRC_F} (non-contact)  ->  '
                  f'target={TGT_ACT}#{TGT_F} (contact, cam{CAM})', font=F_SMALL, fill=(180, 190, 200))

y = 74


def tiles(items, row_label):
    """items: list of (PIL image, caption)."""
    global y
    dr.text((GAP, y - 17), row_label, font=F_LABEL, fill=(40, 40, 40))
    for i, (im, cap) in enumerate(items):
        x = GAP + i * COLW
        sheet.paste(im.resize((T, T)), (x, y))
        dr.rectangle([x, y, x + T, y + T], outline=(170, 170, 170))
        dr.text((x + 3, y + T + 4), cap, font=F_SMALL, fill=(30, 30, 30))
    y += T + 46


# ---- 0 上下文 ----
tiles([(src_rgb, 'SOURCE RGB'),
       (tgt_rgb, 'TARGET RGB (GT)'),
       (t2img(batch['normal_a']), 'normal_a   [3,512,512]'),
       (t2img(batch['normal_b']), 'normal_b   [3,512,512]')],
      '0.  reference images + surface normals   ([-1,1])')

# ---- 1 掩码 / 深度 ----
tiles([(t2img(batch['human_mask_a'], 'g'), 'human_mask_a   [1,512,512]'),
       (t2img(batch['human_mask_b'], 'g'), 'human_mask_b   [1,512,512]'),
       (t2img(batch['depth_a'], 'g'), 'depth_a   [1,512,512]'),
       (t2img(batch['depth_b'], 'g'), 'depth_b   [1,512,512]')],
      '1.  person masks + normalised depth')

# ---- 2 接触 ----
contact = t2img(batch['contact_raster'])
ca = Image.fromarray((batch['contact_raster'][0, 0].numpy() * 255).clip(0, 255).astype(np.uint8)).convert('RGB')
cb = Image.fromarray((batch['contact_raster'][0, 1].numpy() * 255).clip(0, 255).astype(np.uint8)).convert('RGB')
tiles([(ca, 'contact_raster  ch0 = A   [2,512,512]'),
       (cb, 'contact_raster  ch1 = B'),
       (t2img(batch['pose_heatmap_a'], 'max'), 'pose_heatmap_a   [25,512,512]'),
       (t2img(batch['pose_heatmap_b'], 'max'), 'pose_heatmap_b   [25,512,512]')],
      '2.  inter-person contact raster + BODY_25 heatmaps   (max projection)')

# ---- 3 部位 ----
legend_w = 2 * COLW - GAP
legend = Image.new('RGB', (legend_w, T), (255, 255, 255))
ld = ImageDraw.Draw(legend)
ld.text((6, 6), 'part_onehot 14 classes  (all-zero = background)',
        font=F_SMALL, fill=(30, 30, 30))
for i, (name, col) in enumerate(zip(hc.PART_NAMES, PART_COLORS)):
    cx = 8 + (i % 2) * (legend_w // 2)
    cy = 28 + (i // 2) * 27
    ld.rectangle([cx, cy + 2, cx + 14, cy + 16], fill=col, outline=(120, 120, 120))
    ld.text((cx + 20, cy), f'{i:2d}  {name}', font=F_LABEL, fill=(30, 30, 30))

tiles([(t2img(batch['part_onehot_a'], 'parts'), 'part_onehot_a   [14,512,512]'),
       (t2img(batch['part_onehot_b'], 'parts'), 'part_onehot_b   [14,512,512]'),
       (legend, 'legend'),
       (Image.new('RGB', (T, T), (255, 255, 255)), '')],
      '3.  14-class one-hot body parts')


# ---- 4 数值向量 ----
def bar_panel(vec_a, vec_b, title, labels=None, vmin=None, vmax=None):
    """把一个或多个向量画成条形图（长度按最大绝对值归一）。"""
    Wp, Hp = 2 * COLW - GAP, T
    im = Image.new('RGB', (Wp, Hp), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((4, 4), title, font=F_LABEL, fill=(25, 25, 25))
    series = [v for v in (vec_a, vec_b) if v is not None]
    peak = max(float(np.abs(v).max()) for v in series) or 1.0
    top = 30
    row_h = (Hp - top - 12) // max(1, len(series))
    pal = [(60, 110, 200), (210, 110, 60)]
    for si, v in enumerate(series):
        v = np.asarray(v, dtype=np.float32)
        n = len(v)
        bw = (Wp - 90) / n
        base = top + si * row_h + row_h * 0.5
        d.line([(60, base), (Wp - 12, base)], fill=(200, 200, 200))
        for i, val in enumerate(v):
            bx = 60 + i * bw
            hgt = (val / peak) * (row_h * 0.40)
            col = pal[si % 2] if val >= 0 else (200, 70, 70)
            if hgt >= 0:
                d.rectangle([bx + 1, base - hgt, bx + bw - 2, base], fill=col)
            else:
                d.rectangle([bx + 1, base, bx + bw - 2, base - hgt], fill=col)
        lab = labels[si] if labels else f'series {si}'
        d.text((4, top + si * row_h + 2), lab, font=F_SMALL, fill=(40, 40, 40))
        d.text((Wp - 74, top + si * row_h + 2), f'max|v|={peak:.2f}', font=F_SMALL, fill=(120, 120, 120))
    return im


y += 0
dr.text((GAP, y - 17), '4.  numeric condition vectors   (batch dim omitted)', font=F_LABEL, fill=(40, 40, 40))
p1 = bar_panel(batch['smplx_global_a'][0].numpy(), batch['smplx_global_b'][0].numpy(),
               'smplx_global   [B, 26]', ['person A  [26]', 'person B  [26]'])
sheet.paste(p1, (GAP, y))
p2 = bar_panel(batch['relative_geometry'][0].numpy(), None,
               'relative_geometry   [B, 11]', ['inter-person  [11]'])
sheet.paste(p2, (GAP + 2 * COLW, y))
y += T + 46

# ---- 5 说明 ----
dr.rectangle([GAP, y, W - GAP, y + 200], fill=(244, 246, 248), outline=(200, 205, 212))
lines = [
    'Shapes / ranges (contract from SMPLXConditionInjector._validate_person):',
    '  normal [B,3,512,512] in [-1,1]   pose_heatmap [B,25,512,512] in [0,1] (OpenPose BODY_25)',
    '  part_onehot [B,14,512,512] binary, per-pixel sum <= 1, all-zero = background',
    '  smplx_global [B,26]   human_mask [B,1,512,512] in [0,1]   depth [B,1,512,512]',
    '  relative_geometry [B,11]   contact_raster [B,2,512,512]   person_b_valid [B] bool   task_id [B] int64 = HUG(2)',
    '',
    'Documented deviations from the iPER convention:',
    '  (1) pose_heatmap is projected from SMPL-24 joints (K_runtime + cam extrinsics), NOT from the',
    '      bundled DWPose npz: that file is 134-pt but its first 17 are NOT in COCO order.',
    '  (2) part_onehot is approximated from the BODY_25 skeleton by rasterising limb capsules;',
    '      Hi4D ships no body-part segmentation (semantic_A/B are binary person masks only).',
    '  (3) smplx_global / relative_geometry layouts are defined in hi4d_conditions.py docstring.',
]
for i, ln in enumerate(lines):
    dr.text((GAP + 10, y + 8 + i * 14), ln, font=F_SMALL,
            fill=(35, 35, 35) if not ln.startswith('  (') else (150, 60, 30))

sheet.save(OUT, optimize=True)
print('wrote', OUT, sheet.size)

# 同时导出条件清单 json
manifest = {
    'sample': 'inter_01_pair00_taichi',
    'hi4d': {'pair': PAIR, 'source': f'{SRC_ACT}#{SRC_F}', 'target': f'{TGT_ACT}#{TGT_F}', 'camera': CAM},
    'geometry_npz': dbg['geom_path'],
    'task_id': 2, 'task_name': 'HUG',
    'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                    'min': float(v.min()), 'max': float(v.max())}
                for k, v in batch.items() if torch.is_tensor(v)},
    'part_names': hc.PART_NAMES,
    'body25_names': hc.BODY25_NAMES,
    'contract_checks': problems or 'all passed',
}
open(OUT.replace('.png', '.json'), 'w').write(json.dumps(manifest, indent=2, ensure_ascii=False))
print('wrote json manifest')
