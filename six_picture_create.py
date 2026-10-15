#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the six-panel overview figure for one single-person pose-transfer workflow run.

The figure stitches the six artifacts that matter for inspecting a run, in pipeline order:

    1. inputs/A_reference.png             source photo  (identity / outfit / scene)
    2. inputs/B_rgb.png                   target photo  (action reference only)
    3. geometry/B_poseguide_overlay.png   estimator fit check on B
    4. geometry/B_3d_dwpose.png           aligned pose guide  -> the ONLY one the generator sees
    5. geometry/A_poseguide_alignment.png aligned guide overlaid on A (diagnostic only)
    6. generation/final_image.png         final result

Panels 4 and 5 are the same aligned guide, differing only in the background image
(black canvas vs. the source photo). That is the fastest way to tell them apart.

Usage
-----
    # one run
    python3 six_picture_create.py <run_dir> [-o OUT.png]

    # every finished run under a directory
    python3 six_picture_create.py --all <runs_dir>

    # only check whether a run has all six artifacts
    python3 six_picture_create.py <run_dir> --check

Options
-------
    -o / --out        output path (default: <this_script_dir>/../output/<run>_six_panel.png)
    --title           override the figure title
    --note            override the footer note
    --cell            panel edge length in px (default 620)
    --gap             gap between panels (default 22)
    --scale-threshold X   below this the footer warns that the guide is undersized (default 0.65)
    --lang zh|en     force label language (default: zh when a CJK font is available)
    --check           validate artifacts and print a report, do not render
    --all             treat the positional argument as a directory of runs and process each

Exit status
-----------
    0  figure(s) written
    1  invalid arguments or missing artifacts in single-run mode
    2  no CJK font available and --lang zh requested
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- panel table
# (relative path, panel name, role line, fill, stroke, title colour)
PANELS = [
    ("inputs/A_reference.png",            "A_reference",
     "源照片 · 身份 / 服装 / 场景",        (230, 241, 251), (24, 95, 165),  (4, 44, 83)),
    ("inputs/B_rgb.png",                  "B_rgb",
     "目标照片 · 仅提供动作 / 表情",       (241, 239, 232), (136, 135, 128), (44, 44, 41)),
    ("geometry/B_poseguide_overlay.png",  "B_poseguide_overlay",
     "估计器在 B 上的拟合校验",            (250, 238, 218), (133, 79, 11),  (65, 36, 2)),
    ("geometry/B_3d_dwpose.png",          "B_3d_dwpose",
     "对齐后姿态引导 · 唯一进入生成器",    (234, 243, 222), (59, 109, 17),  (23, 52, 4)),
    ("geometry/A_poseguide_alignment.png", "A_poseguide_alignment",
     "对齐引导叠加到 A · 仅诊断",          (250, 238, 218), (133, 79, 11),  (65, 36, 2)),
    ("generation/final_image.png",        "final_image",
     "最终生成结果",                       (225, 245, 238), (15, 110, 86),  (4, 52, 44)),
]
PANELS_EN = [
    ("inputs/A_reference.png",            "A_reference",
     "source photo - identity / outfit / scene", (230, 241, 251), (24, 95, 165),  (4, 44, 83)),
    ("inputs/B_rgb.png",                  "B_rgb",
     "target photo - action reference only",      (241, 239, 232), (136, 135, 128), (44, 44, 41)),
    ("geometry/B_poseguide_overlay.png",  "B_poseguide_overlay",
     "estimator fit check on B",                  (250, 238, 218), (133, 79, 11),  (65, 36, 2)),
    ("geometry/B_3d_dwpose.png",          "B_3d_dwpose",
     "aligned pose guide - the only one the generator sees", (234, 243, 222), (59, 109, 17), (23, 52, 4)),
    ("geometry/A_poseguide_alignment.png", "A_poseguide_alignment",
     "aligned guide overlaid on A - diagnostic only",        (250, 238, 218), (133, 79, 11),  (65, 36, 2)),
    ("generation/final_image.png",        "final_image",
     "final generated result",                    (225, 245, 238), (15, 110, 86),  (4, 52, 44)),
]

# CJK-capable fonts, most preferred first. The workflow machines ship Noto Sans CJK.
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]
MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
]

HEADER, CAPTION, FOOTER = 132, 64, 76
PAD, COLS, ROWS = 26, 3, 2


def pick_font(candidates, size):
    for path in candidates:
        if os.path.isfile(path):
            try:
                # .ttc bundles several faces; face 0 always covers the CJK block.
                return ImageFont.truetype(path, size, index=0)
            except Exception:
                continue
    return None


def detect_lang(requested):
    """Return 'zh' when a CJK font is usable, else 'en'."""
    if requested == "en":
        return "en"
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                ImageFont.truetype(path, 12, index=0)
                return "zh"
            except Exception:
                continue
    return "en"


def read_metadata(run_dir: Path):
    """Pull the numbers that explain the figure: scale, canvas, valid joints."""
    meta = {}
    stage = run_dir / "geometry" / "stage.json"
    if not stage.is_file():
        return meta
    try:
        data = json.loads(stage.read_text(encoding="utf-8"))
    except Exception:
        return meta
    checks = data.get("metadata", {}).get("camera_checks", {})
    transform = checks.get("B_to_generation", {}) or {}
    meta["scale"] = transform.get("scale")
    meta["canvas"] = checks.get("generation_canvas")
    meta["valid_b"] = checks.get("valid_B_joints")
    return meta


def auto_note(meta, threshold):
    """A footer note that says what the figure shows, driven by the alignment scale."""
    scale = meta.get("scale")
    if scale is None:
        return ("注：未读到 geometry/stage.json，无法给出缩放比；请人工核对骨架与 A 的比例是否匹配。")
    if scale < 0.5:
        return (f"注：本 run 的 scale 仅 {scale:.2f} —— B 的姿态被等比缩小后，骨架只占 A 人物的一小部分，"
                f"生成器可能把过小的骨架当成独立的「小人」物体渲染，姿态迁移容易失败。")
    if scale < threshold:
        return (f"注：本 run 的 scale 为 {scale:.2f}，骨架偏小；若结果里出现多余人形伪影，"
                f"根源就在这里。")
    return f"注：本 run 的 scale 为 {scale:.2f}，骨架尺寸与 A 较匹配，对齐正常。"


def auto_note_en(meta, threshold):
    scale = meta.get("scale")
    if scale is None:
        return ("Note: geometry/stage.json unreadable, so no scale is reported; "
                "check the guide/A proportion by eye.")
    if scale < 0.5:
        return (f"Note: scale is only {scale:.2f} - after the uniform shrink the guide covers a "
                f"small fraction of the source person, so the generator may render the guide "
                f"itself as a separate small figure and the transfer tends to fail.")
    if scale < threshold:
        return (f"Note: scale is {scale:.2f}, the guide is undersized; extra human-shaped "
                f"artefacts in the result come from this.")
    return f"Note: scale is {scale:.2f}, the guide matches the source person well."


def subtitle(meta, lang):
    canvas = meta.get("canvas")
    valid_b = meta.get("valid_b")
    scale = meta.get("scale")
    parts = []
    if lang == "zh":
        parts.append("同一 demo 流程的另一个输入对" if False else "动作迁移 workflow 单次运行")
    else:
        parts.append("single-person pose-transfer run")
    if canvas:
        parts.append(f"{'生成画布' if lang == 'zh' else 'canvas'} {canvas[0]}×{canvas[1]}")
    if scale is not None:
        parts.append(f"B→画布缩放比 scale = {scale:.4f}")
    if valid_b:
        parts.append(f"{'有效 B 关节' if lang == 'zh' else 'valid B joints'} {valid_b}")
    return "　".join(parts) if lang == "zh" else "  |  ".join(parts)


def check_run(run_dir: Path):
    """Report which of the six artifacts exist; return the list of missing ones."""
    missing = []
    print(f"=== {run_dir}")
    for rel, name, _role, *_ in PANELS:
        path = run_dir / rel
        if path.is_file():
            size = path.stat().st_size
            print(f"  OK      {rel:<42} {size/1024:9.0f} KB")
        else:
            print(f"  MISSING {rel}")
            missing.append(rel)
    meta = read_metadata(run_dir)
    if meta:
        print(f"  meta    scale={meta.get('scale')} canvas={meta.get('canvas')} "
              f"valid_B={meta.get('valid_b')}")
    else:
        print("  meta    geometry/stage.json 不可用")
    return missing


def build(run_dir: Path, out_path: Path, lang, title, note, cell, gap, threshold):
    f_body = pick_font(FONT_CANDIDATES, 18)
    f_head = pick_font(FONT_CANDIDATES, 32)
    f_name = pick_font(FONT_CANDIDATES, 20)
    f_role = pick_font(FONT_CANDIDATES, 15)
    f_note = pick_font(FONT_CANDIDATES, 16)
    if lang == "zh" and f_body is None:
        print("错误：找不到可用的中文字体，请用 --lang en。", file=sys.stderr)
        return 2
    # English fallback keeps working even when no font is found at all.
    if f_head is None:
        f_head = ImageFont.load_default()

    table = PANELS if lang == "zh" else PANELS_EN
    meta = read_metadata(run_dir)
    if title is None:
        title = f"动作迁移 workflow · {run_dir.name}" if lang == "zh" else f"Pose transfer workflow - {run_dir.name}"
    if note is None:
        note = auto_note(meta, threshold) if lang == "zh" else auto_note_en(meta, threshold)
    sub = subtitle(meta, lang)
    flow = ("流程：A_reference + B_rgb  →  geometry  →  prompt  →  generate  →  final_image"
            if lang == "zh" else
            "flow: A_reference + B_rgb  ->  geometry  ->  prompt  ->  generate  ->  final_image")

    width = PAD * 2 + cell * COLS + gap * (COLS - 1)
    height = HEADER + ROWS * (cell + CAPTION) + gap + FOOTER
    canvas = Image.new("RGB", (width, height), (252, 252, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 26), title, font=f_head, fill=(15, 15, 20))
    draw.text((PAD, 70), sub, font=f_body, fill=(95, 100, 110))
    draw.text((PAD, 96), flow, font=f_role, fill=(140, 145, 155))

    missing = []
    for index, (rel, name, role, fill, stroke, text_colour) in enumerate(table):
        row, col = divmod(index, COLS)
        x = PAD + col * (cell + gap)
        y = HEADER + row * (cell + CAPTION + gap)
        draw.rectangle([x, y, x + cell, y + cell], fill=(244, 244, 246), outline=(205, 205, 210))
        path = run_dir / rel
        if path.is_file():
            try:
                image = Image.open(path).convert("RGB")
                ratio = min((cell - 16) / image.width, (cell - 16) / image.height)
                new_w, new_h = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
                canvas.paste(image.resize((new_w, new_h), Image.LANCZOS),
                             (x + (cell - new_w) // 2, y + (cell - new_h) // 2))
            except Exception as error:
                draw.text((x + 14, y + 14), f"读取失败: {error}", font=f_role, fill=(160, 40, 40))
                missing.append(rel)
        else:
            draw.text((x + 14, y + 14), "缺失", font=f_name, fill=(160, 40, 40))
            missing.append(rel)
        draw.rounded_rectangle([x, y + cell + 8, x + cell, y + cell + CAPTION - 8],
                               radius=8, fill=fill, outline=stroke, width=2)
        draw.text((x + 14, y + cell + 16), f"{index + 1}.  {name}", font=f_name, fill=text_colour)
        draw.text((x + 14, y + cell + 40), role, font=f_role, fill=(85, 90, 100))

    draw.text((PAD, height - FOOTER + 18), note, font=f_note, fill=(170, 60, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)

    flag = "" if not missing else f"（缺失 {len(missing)} 项：{', '.join(missing)}）"
    print(f"OK  {out_path}{flag}")
    return 0


def main(argv=None):
    here = Path(__file__).resolve().parent
    default_out_dir = here.parent / "output"

    parser = argparse.ArgumentParser(
        description="把一次 workflow run 的六件关键产物拼成一张带标注的图。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 six_picture_create.py ../../runs/demo\n"
               "  python3 six_picture_create.py --all ../../runs\n"
               "  python3 six_picture_create.py ../../runs/demo --check\n")
    parser.add_argument("run_dir", help="run 目录，例如 workflow/runs/demo")
    parser.add_argument("-o", "--out", help="输出图片路径")
    parser.add_argument("--title", help="覆盖图题")
    parser.add_argument("--note", help="覆盖页脚注释")
    parser.add_argument("--cell", type=int, default=620, help="单格边长 px（默认 620）")
    parser.add_argument("--gap", type=int, default=22, help="格间距 px（默认 22）")
    parser.add_argument("--scale-threshold", type=float, default=0.65,
                        help="低于此缩放比时页脚提示骨架偏小（默认 0.65）")
    parser.add_argument("--lang", choices=("zh", "en", "auto"), default="auto",
                        help="标签语言，默认 auto（有中文字体就用中文）")
    parser.add_argument("--check", action="store_true", help="只校验六件产物是否齐全，不出图")
    parser.add_argument("--all", action="store_true", help="把 run_dir 当作 runs 目录，逐个处理")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"错误：目录不存在 {run_dir}", file=sys.stderr)
        return 1

    lang = detect_lang(args.lang)

    if args.check:
        missing = check_run(run_dir)
        return 1 if missing else 0

    if args.all:
        status = 0
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "geometry").is_dir():
                continue
            out = args.out or (default_out_dir / f"{child.name}_six_panel.png")
            result = build(child, Path(out), lang, args.title, args.note,
                           args.cell, args.gap, args.scale_threshold)
            if result == 2:
                return 2
            if not (child / "generation" / "final_image.png").is_file():
                print(f"  提示：{child.name} 没有 generation/final_image.png，最后一格为空。")
                status = 1
        return status

    out = Path(args.out) if args.out else (default_out_dir / f"{run_dir.name}_six_panel.png")
    return build(run_dir, out, lang, args.title, args.note,
                 args.cell, args.gap, args.scale_threshold)


if __name__ == "__main__":
    sys.exit(main())
