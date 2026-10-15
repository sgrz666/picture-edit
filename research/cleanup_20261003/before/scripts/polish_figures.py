import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEST_DIR = Path(r"D:\codeplus\pictureedit\expeiment\tiktok\one-piece test")

def get_font(size, bold=False):
    font_names = ["msyhbd.ttc", "simhei.ttf", "arialbd.ttf"] if bold else ["msyh.ttc", "simsun.ttc", "arial.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            pass
    return ImageFont.load_default()

def add_label_box(img, title, subtitle=None, size=512, font_size=16):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    f_title = get_font(font_size, bold=True)
    f_sub = get_font(max(10, font_size - 4), bold=False)
    
    tb = draw.textbbox((0, 0), title, font=f_title)
    th = tb[3] - tb[1]
    banner_h = th + 14
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=f_sub)
        banner_h += (sb[3] - sb[1]) + 6

    banner = Image.new("RGBA", (img.width, banner_h), (15, 23, 42, 230))
    img.paste(banner, (0, 0), banner)
    d = ImageDraw.Draw(img)
    tw = tb[2] - tb[0]
    d.text(((img.width - tw) // 2, 6), title, font=f_title, fill=(255, 255, 255))
    if subtitle:
        sw = sb[2] - sb[0]
        d.text(((img.width - sw) // 2, th + 10), subtitle, font=f_sub, fill=(203, 213, 225))
    return img.convert("RGB")

def polish_scorecard():
    metrics_p = TEST_DIR / "quantitative_metrics.json"
    with open(metrics_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    b = data["full_body"]
    face = data["face_roi"]
    lh = data["left_hand_roi"]
    rh = data["right_hand_roi"]

    def fmt_delta(val, unit="", invert=False):
        sign = "+" if val > 0 else ""
        text = f"{sign}{val:.2f}{unit}" if abs(val) >= 0.01 else f"{sign}{val:.4f}{unit}"
        is_pos = (val > 0) if not invert else (val < 0)
        color = (5, 150, 105) if is_pos else ((220, 38, 38) if abs(val) > 0.05 else (71, 85, 105))
        return text, color

    rows = [
        ("Full Body (512x512)", "PSNR (dB)", f"{b['baseline']['psnr']:.2f}", f"{b['full_adapter']['psnr']:.2f}", fmt_delta(b['full_adapter']['psnr'] - b['baseline']['psnr'], " dB")),
        ("Full Body (512x512)", "SSIM", f"{b['baseline']['ssim']:.4f}", f"{b['full_adapter']['ssim']:.4f}", fmt_delta(b['full_adapter']['ssim'] - b['baseline']['ssim'])),
        ("Face ROI (280x280)", "PSNR (dB)", f"{face['baseline']['psnr']:.2f}", f"{face['full_adapter']['psnr']:.2f}", fmt_delta(face['full_adapter']['psnr'] - face['baseline']['psnr'], " dB")),
        ("Face ROI (280x280)", "L1 Error", f"{face['baseline']['l1']:.4f}", f"{face['full_adapter']['l1']:.4f}", fmt_delta(face['full_adapter']['l1'] - face['baseline']['l1'], invert=True)),
        ("Left Hand ROI", "PSNR (dB)", f"{lh['baseline']['psnr']:.2f}", f"{lh['full_adapter']['psnr']:.2f}", fmt_delta(lh['full_adapter']['psnr'] - lh['baseline']['psnr'], " dB")),
        ("Left Hand ROI", "L1 Error", f"{lh['baseline']['l1']:.4f}", f"{lh['full_adapter']['l1']:.4f}", fmt_delta(lh['full_adapter']['l1'] - lh['baseline']['l1'], invert=True)),
        ("Right Hand ROI", "PSNR (dB)", f"{rh['baseline']['psnr']:.2f}", f"{rh['full_adapter']['psnr']:.2f}", fmt_delta(rh['full_adapter']['psnr'] - rh['baseline']['psnr'], " dB")),
        ("Right Hand ROI", "L1 Error", f"{rh['baseline']['l1']:.4f}", f"{rh['full_adapter']['l1']:.4f}", fmt_delta(rh['full_adapter']['l1'] - rh['baseline']['l1'], invert=True)),
    ]

    # Canvas dimensions
    width, height = 1300, 720
    canvas = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    f_title = get_font(24, bold=True)
    f_sub = get_font(13, bold=False)
    f_th = get_font(15, bold=True)
    f_td = get_font(14, bold=False)
    f_td_bold = get_font(14, bold=True)

    title_text = "Quantitative Evaluation Scorecard: TikTok Sample 00001"
    sub_text = "Pure 28-Step Generative Diffusion (Zero GT Leakage) - Baseline vs Full Adapter Trained"
    
    tb = draw.textbbox((0, 0), title_text, font=f_title)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((width - (tb[2] - tb[0])) // 2, 22), title_text, font=f_title, fill=(15, 23, 42))
    draw.text(((width - (sb[2] - sb[0])) // 2, 58), sub_text, font=f_sub, fill=(71, 85, 105))

    # Table layout
    table_x = 50
    table_y = 100
    table_w = width - 100
    col_widths = [320, 180, 240, 240, 220]
    headers = ["Evaluation Region", "Metric", "Baseline (Face OFF)", "Full Adapter (Face ON)", "Delta / Improvement"]

    # Header Row
    header_h = 55
    draw.rectangle([table_x, table_y, table_x + table_w, table_y + header_h], fill=(15, 23, 42))
    cur_x = table_x
    for i, h in enumerate(headers):
        w = col_widths[i]
        hb = draw.textbbox((0, 0), h, font=f_th)
        hw = hb[2] - hb[0]
        hh = hb[3] - hb[1]
        draw.text((cur_x + (w - hw) // 2, table_y + (header_h - hh) // 2), h, font=f_th, fill=(255, 255, 255))
        cur_x += w

    # Body Rows
    row_y = table_y + header_h
    row_h = 56
    for idx, r in enumerate(rows):
        bg = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
        draw.rectangle([table_x, row_y, table_x + table_w, row_y + row_h], fill=bg)
        # horizontal separator line
        draw.line([table_x, row_y + row_h, table_x + table_w, row_y + row_h], fill=(226, 232, 240), width=1)
        
        cur_x = table_x
        # Col 0: Region
        w0 = col_widths[0]
        b0 = draw.textbbox((0, 0), r[0], font=f_td_bold)
        draw.text((cur_x + 25, row_y + (row_h - (b0[3] - b0[1])) // 2), r[0], font=f_td_bold, fill=(30, 41, 59))
        cur_x += w0

        # Col 1: Metric
        w1 = col_widths[1]
        b1 = draw.textbbox((0, 0), r[1], font=f_td)
        draw.text((cur_x + (w1 - (b1[2] - b1[0])) // 2, row_y + (row_h - (b1[3] - b1[1])) // 2), r[1], font=f_td, fill=(51, 65, 85))
        cur_x += w1

        # Col 2: Baseline
        w2 = col_widths[2]
        b2 = draw.textbbox((0, 0), r[2], font=f_td)
        draw.text((cur_x + (w2 - (b2[2] - b2[0])) // 2, row_y + (row_h - (b2[3] - b2[1])) // 2), r[2], font=f_td, fill=(15, 23, 42))
        cur_x += w2

        # Col 3: Full Adapter
        w3 = col_widths[3]
        b3 = draw.textbbox((0, 0), r[3], font=f_td_bold)
        draw.text((cur_x + (w3 - (b3[2] - b3[0])) // 2, row_y + (row_h - (b3[3] - b3[1])) // 2), r[3], font=f_td_bold, fill=(15, 23, 42))
        cur_x += w3

        # Col 4: Delta
        w4 = col_widths[4]
        d_text, d_color = r[4]
        b4 = draw.textbbox((0, 0), d_text, font=f_td_bold)
        draw.text((cur_x + (w4 - (b4[2] - b4[0])) // 2, row_y + (row_h - (b4[3] - b4[1])) // 2), d_text, font=f_td_bold, fill=d_color)

        row_y += row_h

    # Outer border
    draw.rectangle([table_x, table_y, table_x + table_w, row_y], outline=(203, 213, 225), width=2)

    out_p = TEST_DIR / "06_quantitative_evaluation_summary.png"
    canvas.save(out_p, quality=95)
    print("Saved:", out_p)

def polish_fullbody_panel():
    src = Image.open(TEST_DIR / "input_source_0014.png").resize((512, 512))
    norm = Image.open(TEST_DIR / "input_target_normal_0074.png").resize((512, 512))
    gt = Image.open(TEST_DIR / "target_gt_reference_0074.png").resize((512, 512))
    off = Image.open(TEST_DIR / "pure_inference_face_off.png").resize((512, 512))
    on = Image.open(TEST_DIR / "pure_inference_face_on.png").resize((512, 512))

    panels = [
        (src, "Source Frame (0014)", "Input Identity"),
        (norm, "Target 3D Normal (0074)", "Pose & Surface Guidance"),
        (gt, "Target GT Reference", "Ground Truth Target"),
        (off, "Baseline: Face OFF", "PSNR: 11.24 dB | Base Adapter"),
        (on, "Full Adapter: Face ON", "PSNR: 11.24 dB | 200 Steps Trained"),
    ]
    labeled = [add_label_box(img, t, s, size=512, font_size=18) for img, t, s in panels]
    total_w = sum(img.width for img in labeled) + (len(labeled) - 1) * 12 + 40
    total_h = 512 + 100
    canvas = Image.new("RGB", (total_w, total_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    f_main = get_font(24, bold=True)
    f_sub = get_font(13, bold=False)
    
    t_text = "TikTok 00001 Single-Sample Overfit: Pure 28-Step Generative Diffusion (Zero GT Leakage)"
    s_text = "Comparing Input Source, Target Normal, Ground Truth, Baseline (Face OFF), and Full Adapter Trained (Face ON)"
    tb = draw.textbbox((0, 0), t_text, font=f_main)
    sb = draw.textbbox((0, 0), s_text, font=f_sub)
    draw.text(((total_w - (tb[2] - tb[0])) // 2, 16), t_text, font=f_main, fill=(15, 23, 42))
    draw.text(((total_w - (sb[2] - sb[0])) // 2, 50), s_text, font=f_sub, fill=(71, 85, 105))

    x_off = 20
    for img in labeled:
        canvas.paste(img, (x_off, 80))
        x_off += img.width + 12
    out_p = TEST_DIR / "03_genuine_inference_fullbody_comparison.png"
    canvas.save(out_p, quality=95)
    print("Saved:", out_p)

def polish_face_zoom():
    src = Image.open(TEST_DIR / "input_source_0014.png").resize((512, 512))
    gt = Image.open(TEST_DIR / "target_gt_reference_0074.png").resize((512, 512))
    off = Image.open(TEST_DIR / "pure_inference_face_off.png").resize((512, 512))
    on = Image.open(TEST_DIR / "pure_inference_face_on.png").resize((512, 512))

    crop_tgt = (115, 95, 295, 275)
    crop_src = (120, 100, 300, 280)
    z = 320

    f_src = src.crop(crop_src).resize((z, z), Image.Resampling.LANCZOS)
    f_gt = gt.crop(crop_tgt).resize((z, z), Image.Resampling.LANCZOS)
    f_off = off.crop(crop_tgt).resize((z, z), Image.Resampling.LANCZOS)
    f_on = on.crop(crop_tgt).resize((z, z), Image.Resampling.LANCZOS)

    items = [
        (f_src, "Source Frame (0014)", "Input Identity"),
        (f_gt, "Target GT Reference", "Ground Truth Target"),
        (f_off, "Baseline: Face OFF", "PSNR: 11.49 dB | SSIM: 0.5711"),
        (f_on, "Full Adapter: Face ON", "PSNR: 11.41 dB | SSIM: 0.5650"),
    ]
    labeled = [add_label_box(img, t, s, size=z, font_size=16) for img, t, s in items]
    total_w = sum(img.width for img in labeled) + (len(labeled) - 1) * 12 + 40
    total_h = z + 95
    canvas = Image.new("RGB", (total_w, total_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    f_main = get_font(22, bold=True)
    f_sub = get_font(13, bold=False)

    t_text = "Facial ROI Zoom-In Comparison (Pure 28-Step Diffusion from Random Noise)"
    s_text = "Evaluated on cropped facial bounding box [115, 95, 295, 275]"
    tb = draw.textbbox((0, 0), t_text, font=f_main)
    sb = draw.textbbox((0, 0), s_text, font=f_sub)
    draw.text(((total_w - (tb[2] - tb[0])) // 2, 14), t_text, font=f_main, fill=(15, 23, 42))
    draw.text(((total_w - (sb[2] - sb[0])) // 2, 45), s_text, font=f_sub, fill=(71, 85, 105))

    x_off = 20
    for img in labeled:
        canvas.paste(img, (x_off, 72))
        x_off += img.width + 12
    out_p = TEST_DIR / "04_genuine_inference_face_roi_zoom.png"
    canvas.save(out_p, quality=95)
    print("Saved:", out_p)

def polish_hand_zoom():
    src = Image.open(TEST_DIR / "input_source_0014.png").resize((512, 512))
    gt = Image.open(TEST_DIR / "target_gt_reference_0074.png").resize((512, 512))
    off = Image.open(TEST_DIR / "pure_inference_face_off.png").resize((512, 512))
    on = Image.open(TEST_DIR / "pure_inference_face_on.png").resize((512, 512))

    lh_box = (127, 414, 257, 512)
    rh_box = (18, 70, 151, 202)
    z = 280

    rows = [
        ("Left Hand (Target Box: [127, 414, 257, 512])", [
            (src.crop(lh_box).resize((z, z)), "Source LH", "Input Frame 0014"),
            (gt.crop(lh_box).resize((z, z)), "Target GT LH", "Ground Truth Reference"),
            (off.crop(lh_box).resize((z, z)), "Baseline LH", "PSNR: 9.96 dB | L1: 0.2386"),
            (on.crop(lh_box).resize((z, z)), "Full Adapter LH", "PSNR: 10.35 dB (+0.38) | L1: 0.2245"),
        ]),
        ("Right Hand (Target Box: [18, 70, 151, 202])", [
            (src.crop(rh_box).resize((z, z)), "Source RH", "Input Frame 0014"),
            (gt.crop(rh_box).resize((z, z)), "Target GT RH", "Ground Truth Reference"),
            (off.crop(rh_box).resize((z, z)), "Baseline RH", "PSNR: 7.31 dB | L1: 0.3795"),
            (on.crop(rh_box).resize((z, z)), "Full Adapter RH", "PSNR: 7.16 dB | L1: 0.3858"),
        ]),
    ]

    total_w = 4 * z + 3 * 12 + 40
    total_h = 2 * z + 2 * 45 + 100
    canvas = Image.new("RGB", (total_w, total_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    f_main = get_font(22, bold=True)
    f_sub = get_font(13, bold=False)
    f_section = get_font(15, bold=True)

    t_text = "Hand ROI Zoom-In Comparison: Left & Right Hands (SMPL-X 21-Points Detailed Pose Control)"
    s_text = "SMPL-X Keypoints extracted from joint_proj_model[25:65] + Left/Right Hand Pose parameters"
    tb = draw.textbbox((0, 0), t_text, font=f_main)
    sb = draw.textbbox((0, 0), s_text, font=f_sub)
    draw.text(((total_w - (tb[2] - tb[0])) // 2, 14), t_text, font=f_main, fill=(15, 23, 42))
    draw.text(((total_w - (sb[2] - sb[0])) // 2, 45), s_text, font=f_sub, fill=(71, 85, 105))

    y_off = 75
    for row_title, items in rows:
        draw.text((20, y_off), row_title, font=f_section, fill=(30, 41, 59))
        y_off += 28
        x_off = 20
        for img, t, s in items:
            lbl = add_label_box(img, t, s, size=z, font_size=15)
            canvas.paste(lbl, (x_off, y_off))
            x_off += z + 12
        y_off += z + 22

    out_p = TEST_DIR / "05_genuine_inference_hand_roi_zoom.png"
    canvas.save(out_p, quality=95)
    print("Saved:", out_p)

if __name__ == "__main__":
    polish_scorecard()
    polish_fullbody_panel()
    polish_face_zoom()
    polish_hand_zoom()
    print("ALL POLISHED FIGURES GENERATED!")
