#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_contact_panels.py

Stitches the 6 strong-contact interaction groups into 5-column horizontal
comparison panel strips (2572x560):
[GT RGB | Old Pose | DWPose Combined | Person A DWPose | Person B DWPose]
with clean headers, labels, and borders.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = Path(r"D:\codeplus\pictureedit\reports\assets\contact_samples")
OUT_DIR = Path(r"D:\codeplus\pictureedit\reports\assets\contact_panels_5col")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\sg\.gemini\antigravity\brain\dbbc3185-48d6-4986-ac87-0bef1bc51ead")

GROUPS = [
    {
        "id": "group1_backhug",
        "title": "案例 1 · 背后环抱 (Back-Hug) | pair02/backhug02/000044",
        "rgb": "backhug_rgb.png",
        "old": "backhug_oldpose.png",
        "dw": "backhug_dwpose.png",
        "dw_a": "backhug_dw_A.png",
        "dw_b": "backhug_dw_B.png",
    },
    {
        "id": "group2_highfive",
        "title": "案例 2 · 掌心相触击掌 (High-Five) | pair12/highfive12/000035",
        "rgb": "highfive_contact_rgb.png",
        "old": "highfive_contact_oldpose.png",
        "dw": "highfive_contact_dwpose.png",
        "dw_a": "highfive_dw_A.png",
        "dw_b": "highfive_dw_B.png",
    },
    {
        "id": "group3_dance",
        "title": "案例 3 · 舞蹈搭肩牵手 (Dance) | pair14/dance14/000114",
        "rgb": "dance14_rgb.png",
        "old": "dance14_oldpose.png",
        "dw": "dance14_dwpose.png",
        "dw_a": "dance14_dw_A.png",
        "dw_b": "dance14_dw_B.png",
    },
    {
        "id": "group4_kiss",
        "title": "案例 4 · 亲密贴面相迎 (Kiss) | pair22/kiss22/000022",
        "rgb": "kiss22_rgb.png",
        "old": "kiss22_oldpose.png",
        "dw": "kiss22_dwpose.png",
        "dw_a": "kiss22_dw_A.png",
        "dw_b": "kiss22_dw_B.png",
    },
    {
        "id": "group5_fight",
        "title": "案例 5 · 格斗推搡抓肩 (Fight) | pair12/fight12/000030",
        "rgb": "fight12_rgb.png",
        "old": "fight12_oldpose.png",
        "dw": "fight12_dwpose.png",
        "dw_a": "fight12_dw_A.png",
        "dw_b": "fight12_dw_B.png",
    },
    {
        "id": "group6_deephug",
        "title": "案例 6 · 正面双臂深拥 (Deep-Hug) | pair00/hug00/000089",
        "rgb": "deephug_rgb.png",
        "old": "deephug_oldpose.png",
        "dw": "deephug_dwpose.png",
        "dw_a": "deephug_dw_A.png",
        "dw_b": "deephug_dw_B.png",
    },
]

HEADER_H = 48
IMG_W, IMG_H = 512, 512
BORDER = 3
COLS = 5
TOTAL_W = IMG_W * COLS + BORDER * (COLS - 1)
TOTAL_H = HEADER_H + IMG_H

font = ImageFont.truetype("msyh.ttc", 20)

LABELS = [
    "① 真实实拍图 (GT RGB)",
    "② 旧版 24 关节点简笔",
    "③ 3D DWPose 复合双人",
    "④ Person A 单人 DWPose",
    "⑤ Person B 单人 DWPose",
]

COLORS = [
    (226, 232, 240), # Light Slate
    (248, 113, 113), # Light Red
    (96, 165, 250),  # Vibrant Blue
    (52, 211, 153),  # Emerald Green
    (251, 191, 36),  # Amber Gold
]

for g in GROUPS:
    panel = Image.new("RGB", (TOTAL_W, TOTAL_H), (15, 23, 42)) # Slate Dark header
    draw = ImageDraw.Draw(panel)

    # Load 5 images
    im_rgb = Image.open(SRC_DIR / g["rgb"]).convert("RGB").resize((IMG_W, IMG_H))
    im_old = Image.open(SRC_DIR / g["old"]).convert("RGB").resize((IMG_W, IMG_H))
    im_dw = Image.open(SRC_DIR / g["dw"]).convert("RGB").resize((IMG_W, IMG_H))
    im_dwa = Image.open(SRC_DIR / g["dw_a"]).convert("RGB").resize((IMG_W, IMG_H))
    im_dwb = Image.open(SRC_DIR / g["dw_b"]).convert("RGB").resize((IMG_W, IMG_H))

    images = [im_rgb, im_old, im_dw, im_dwa, im_dwb]

    # Paste images & draw headers
    for i in range(COLS):
        col_x0 = i * (IMG_W + BORDER)
        panel.paste(images[i], (col_x0, HEADER_H))

        label = LABELS[i]
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = col_x0 + (IMG_W - text_w) // 2
        ty = (HEADER_H - text_h) // 2 - 2
        draw.text((tx, ty), label, font=font, fill=COLORS[i])

        # Vertical divider line
        if i > 0:
            div_x = col_x0 - BORDER
            draw.rectangle([div_x, 0, div_x + BORDER - 1, TOTAL_H], fill=(51, 65, 85))

    out_file = OUT_DIR / f"{g['id']}_5col_panel.png"
    panel.save(out_file)
    panel.save(ARTIFACT_DIR / f"{g['id']}_5col_panel.png")
    print(f"Created: {out_file.name}")

print("All 6 5-column panels generated successfully!")
