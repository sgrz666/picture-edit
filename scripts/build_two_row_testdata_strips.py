# -*- coding: utf-8 -*-
"""
测试集评测结果双排可视化合成脚本 (2-Row Test Evaluation Comparison Strips)

结构设计：
- 第 1 排：输入控制条件 (Input Conditioning Signals)
  * DWPose (姿态骨架) | Normal (几何法线) | Depth (空间深度) | Part Map (身体部位解析)
- 第 2 排：编辑生成对比 (Visual Results & Comparison)
  * 原图 (Source) | 目标真值 (Ground Truth) | 无控制基线 (Adapter OFF) | 所提方法 (Adapter ON V6.3)
- 附带量化指标：每个样本标注 PSNR / SSIM，直观呈现所提适配器的控制精度提升。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_font(size: int, bold: bool = False):
    font_path = r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"
    if Path(font_path).exists():
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()

def draw_text_center(draw: ImageDraw.ImageDraw, box, text: str, font, fill):
    x0, y0, x1, y1 = box
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((x0 + x1 - tw) // 2, (y0 + y1 - th) // 2), text, fill=fill, font=font)

# Precompute 24-color palette for part map
cmap = plt.get_cmap("tab20")
palette_lut = (cmap(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8)
extra_colors = np.array([
    [255, 128, 0], [0, 255, 128], [128, 0, 255], [255, 0, 128]
], dtype=np.uint8)
all_palette = np.vstack([palette_lut, extra_colors])

def colorize_part_map(part_pil: Image.Image) -> Image.Image:
    arr = np.array(part_pil.convert("L"))
    if arr.max() == 0:
        return part_pil.convert("RGB")
    if arr.max() <= 24:
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        for val in range(1, 25):
            mask = (arr == val)
            if mask.any():
                rgb[mask] = all_palette[(val - 1) % len(all_palette)]
        return Image.fromarray(rgb)
    return part_pil.convert("RGB")

def create_two_row_strip(
    pair_idx: int,
    app: str,
    src_stem: str,
    tgt_stem: str,
    img_dir: Path,
    cond_dir: Path,
    backup_dir: Path,
    m_on: dict,
    m_off: dict,
    cell: int = 384,
) -> Image.Image:
    # 1. Load conditions (Row 1)
    c_dwpose = Image.open(cond_dir / f"cond_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}_dwpose.png").convert("RGB")
    c_normal = Image.open(cond_dir / f"cond_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}_normal.png").convert("RGB")
    c_depth = Image.open(cond_dir / f"cond_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}_depth.png").convert("RGB")
    raw_part = Image.open(cond_dir / f"cond_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}_part.png")
    c_part = colorize_part_map(raw_part)

    # 2. Load visual results (Row 2)
    src_img = Image.open(img_dir / f"src_{pair_idx:03d}_{app}_{src_stem}.png").convert("RGB")
    gt_img = Image.open(img_dir / f"gt_{pair_idx:03d}_{app}_{tgt_stem}.png").convert("RGB")
    pred_on = Image.open(img_dir / f"pred_{pair_idx:03d}_{app}_{tgt_stem}.png").convert("RGB")
    
    # Extract gen_off from 1-row backup eval strip (crop: [768:1152, 30:414])
    backup_eval = backup_dir / f"eval_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}.png"
    if not backup_eval.exists():
        backup_eval = img_dir / f"eval_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}.png"
    eval_strip = Image.open(backup_eval).convert("RGB")
    gen_off = eval_strip.crop((768, 30, 1152, 414))

    # Resize all to cell size
    r1_imgs = [
        c_dwpose.resize((cell, cell), Image.LANCZOS),
        c_normal.resize((cell, cell), Image.LANCZOS),
        c_depth.resize((cell, cell), Image.LANCZOS),
        c_part.resize((cell, cell), Image.LANCZOS),
    ]
    r2_imgs = [
        src_img.resize((cell, cell), Image.LANCZOS),
        gt_img.resize((cell, cell), Image.LANCZOS),
        gen_off.resize((cell, cell), Image.LANCZOS),
        pred_on.resize((cell, cell), Image.LANCZOS),
    ]

    # Quantitative metrics
    psnr_on = m_on.get("psnr", 0.0)
    ssim_on = m_on.get("ssim_skimage", m_on.get("ssim", 0.0))
    psnr_off = m_off.get("psnr", 0.0)
    ssim_off = m_off.get("ssim_skimage", m_off.get("ssim", 0.0))
    psnr_diff = psnr_on - psnr_off

    # Layout dimensions
    LABEL_W = 168
    col_w = cell
    grid_w = col_w * 4
    total_w = LABEL_W + grid_w

    header_h = 44         # Top title bar
    col_header_h = 36     # Per-column title bar
    r1_h = col_header_h + cell
    row_sep_h = 10        # Separator between row 1 and row 2
    r2_caption_h = 42     # Per-column bottom metrics / caption
    r2_h = col_header_h + cell + r2_caption_h
    total_h = header_h + r1_h + row_sep_h + r2_h

    # Canvas
    bg_color = (248, 249, 250)
    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Fonts
    f_title = load_font(19, bold=True)
    f_sub = load_font(14, bold=False)
    f_row_label = load_font(17, bold=True)
    f_row_sub = load_font(13, bold=False)
    f_col_title = load_font(15, bold=True)
    f_metrics = load_font(13, bold=True)
    f_cap = load_font(13, bold=False)

    # --- Top Title Bar ---
    draw.rectangle([0, 0, total_w, header_h], fill=(31, 35, 40))
    title_left = f"测试集评测样本 #{pair_idx:03d}   |   人物外观: {app}   |   动作迁移"
    title_right = "模型: Unified SMPL-X Adapter V6.3 (Step 12,000 Checkpoint)"
    draw_text_center(draw, (20, 0, 500, header_h), title_left, f_title, (255, 255, 255))
    tb_r = draw.textbbox((0, 0), title_right, font=f_sub)
    draw.text((total_w - (tb_r[2] - tb_r[0]) - 20, (header_h - (tb_r[3] - tb_r[1])) // 2), title_right, fill=(180, 195, 215), font=f_sub)

    # --- Row 1: Conditions ---
    y1 = header_h
    # Left sidebar label for Row 1
    draw.rectangle([0, y1, LABEL_W, y1 + r1_h], fill=(36, 41, 46))
    mid_y1 = y1 + r1_h // 2
    draw_text_center(draw, (0, mid_y1 - 36, LABEL_W, mid_y1 - 10), "① 输入控制条件", f_row_label, (255, 255, 255))
    draw_text_center(draw, (0, mid_y1 - 6, LABEL_W, mid_y1 + 16), "Input Conditions", f_row_sub, (200, 205, 215))
    draw_text_center(draw, (0, mid_y1 + 20, LABEL_W, mid_y1 + 42), "(4通道几何控制信号)", f_row_sub, (130, 200, 255))

    # Row 1 Columns
    r1_cols = [
        ("DWPose 姿态骨架", "#7030A0", r1_imgs[0]),
        ("Normal 几何法线", "#C00000", r1_imgs[1]),
        ("Depth 空间深度", "#ED7D31", r1_imgs[2]),
        ("Part Map 部位解析", "#70AD47", r1_imgs[3]),
    ]
    for j, (col_title, col_color, c_img) in enumerate(r1_cols):
        cx = LABEL_W + j * col_w
        # Header
        draw.rectangle([cx, y1, cx + col_w - 1, y1 + col_header_h - 1], fill=col_color)
        draw_text_center(draw, (cx, y1, cx + col_w - 1, y1 + col_header_h - 1), col_title, f_col_title, (255, 255, 255))
        # Image
        canvas.paste(c_img, (cx, y1 + col_header_h))
        # Subtle border
        draw.rectangle([cx, y1 + col_header_h, cx + col_w - 1, y1 + col_header_h + cell - 1], outline=(220, 224, 230), width=1)

    # Separator
    y_sep = y1 + r1_h
    draw.rectangle([0, y_sep, total_w, y_sep + row_sep_h], fill=(225, 230, 238))

    # --- Row 2: Visual Results ---
    y2 = y_sep + row_sep_h
    # Left sidebar label for Row 2
    draw.rectangle([0, y2, LABEL_W, y2 + r2_h], fill=(24, 34, 48))
    mid_y2 = y2 + (col_header_h + cell) // 2
    draw_text_center(draw, (0, mid_y2 - 36, LABEL_W, mid_y2 - 10), "② 编辑效果对比", f_row_label, (255, 255, 255))
    draw_text_center(draw, (0, mid_y2 - 6, LABEL_W, mid_y2 + 16), "Visual Comparison", f_row_sub, (200, 205, 215))
    draw_text_center(draw, (0, mid_y2 + 20, LABEL_W, mid_y2 + 42), "(消融基线 vs 所提模型)", f_row_sub, (140, 255, 180))

    # Row 2 Columns
    diff_sign = f"+{psnr_diff:.2f}" if psnr_diff >= 0 else f"{psnr_diff:.2f}"
    r2_cols = [
        ("原图 (Source)", "#1F3864", r2_imgs[0], "输入参考图\n身份 / 服装 / 背景", None, False),
        ("目标真值 (Ground Truth)", "#385723", r2_imgs[1], "编辑目标真值\n上界参照 (GT)", None, False),
        ("无控制基线 (Adapter OFF)", "#595959", r2_imgs[2], f"PSNR: {psnr_off:.2f} dB   |   SSIM: {ssim_off:.4f}", (242, 244, 247), False),
        ("所提方法 (Adapter ON V6.3)", "#0F7038", r2_imgs[3], f"PSNR: {psnr_on:.2f} dB   |   SSIM: {ssim_on:.4f} ({diff_sign} dB)", (232, 248, 236), True),
    ]

    for j, (col_title, col_color, r_img, cap_text, pill_bg, is_best) in enumerate(r2_cols):
        cx = LABEL_W + j * col_w
        # Header
        draw.rectangle([cx, y2, cx + col_w - 1, y2 + col_header_h - 1], fill=col_color)
        title_color = (255, 235, 110) if is_best else (255, 255, 255)
        draw_text_center(draw, (cx, y2, cx + col_w - 1, y2 + col_header_h - 1), col_title, f_col_title, title_color)
        # Image
        canvas.paste(r_img, (cx, y2 + col_header_h))
        # Image border (highlighted for best)
        border_color = (16, 140, 60) if is_best else (220, 224, 230)
        border_w = 2 if is_best else 1
        draw.rectangle([cx, y2 + col_header_h, cx + col_w - 1, y2 + col_header_h + cell - 1], outline=border_color, width=border_w)

        # Bottom Caption / Metrics
        cap_y0 = y2 + col_header_h + cell
        cap_y1 = cap_y0 + r2_caption_h
        if pill_bg:
            draw.rectangle([cx + 4, cap_y0 + 4, cx + col_w - 5, cap_y1 - 4], fill=pill_bg, outline=border_color, width=1)
            tcolor = (12, 115, 45) if is_best else (170, 45, 45)
            draw_text_center(draw, (cx, cap_y0, cx + col_w, cap_y1), cap_text, f_metrics, tcolor)
        else:
            lines = cap_text.split("\n")
            if len(lines) == 2:
                draw_text_center(draw, (cx, cap_y0 + 2, cx + col_w, cap_y0 + 20), lines[0], f_cap, (60, 64, 67))
                draw_text_center(draw, (cx, cap_y0 + 20, cx + col_w, cap_y1 - 2), lines[1], f_cap, (120, 124, 128))
            else:
                draw_text_center(draw, (cx, cap_y0, cx + col_w, cap_y1), cap_text, f_cap, (60, 64, 67))

    return canvas

def main():
    parser = argparse.ArgumentParser(description="Build 2-row evaluation comparison strips")
    parser.add_argument("--testdata-dir", default=r"D:\codeplus\pictureedit\可视化\testdata", help="Root testdata dir")
    parser.add_argument("--cell", type=int, default=384, help="Cell size in pixels")
    args = parser.parse_args()

    root_dir = Path(args.testdata_dir)
    img_dir = root_dir / "images"
    cond_dir = root_dir / "conditions"
    backup_dir = root_dir / "images_1row_backup"
    metrics_file = root_dir / "eval_metrics.json"

    assert img_dir.exists(), f"Images dir not found: {img_dir}"
    assert cond_dir.exists(), f"Conditions dir not found: {cond_dir}"
    assert metrics_file.exists(), f"Metrics file not found: {metrics_file}"

    # 1. Ensure backup directory exists and backup original 1-row strips
    backup_dir.mkdir(parents=True, exist_ok=True)
    existing_eval_strips = sorted(list(img_dir.glob("eval_*.png")))
    print(f"[1/4] Found {len(existing_eval_strips)} eval strip images.")
    
    backed_up = 0
    for p in existing_eval_strips:
        dst = backup_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
            backed_up += 1
    print(f"[2/4] Backed up {backed_up} original 1-row strips to {backup_dir}")

    # 2. Read evaluation metrics
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    per_sample_on = data["per_sample"]
    per_sample_off = data["per_sample_adapter_off"]
    num_samples = len(per_sample_on)
    print(f"[3/4] Processing {num_samples} evaluation pairs...")

    # 3. Assemble each sample into 2-row layout
    for idx in range(num_samples):
        m_on = per_sample_on[idx]
        m_off = per_sample_off[idx]
        app = m_on["appearance"]
        src_stem = m_on["source_stem"]
        tgt_stem = m_on["target_stem"]
        pair_idx = m_on["pair_idx"]

        strip = create_two_row_strip(
            pair_idx=pair_idx,
            app=app,
            src_stem=src_stem,
            tgt_stem=tgt_stem,
            img_dir=img_dir,
            cond_dir=cond_dir,
            backup_dir=backup_dir,
            m_on=m_on,
            m_off=m_off,
            cell=args.cell,
        )

        out_path = img_dir / f"eval_{pair_idx:03d}_{app}_{src_stem}_{tgt_stem}.png"
        strip.save(out_path, quality=95)

        if (idx + 1) % 10 == 0 or idx == num_samples - 1:
            print(f"  Processed [{idx + 1:02d}/{num_samples:02d}]: {out_path.name}")

    print(f"[4/4] Successfully rebuilt all {num_samples} evaluation images into 2-row layout in {img_dir}!")

if __name__ == "__main__":
    main()
