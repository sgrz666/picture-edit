# -*- coding: utf-8 -*-
"""把 03 的四联评测图扩充为 04 模板样式（补入条件控制列）。

对 eval_results 中的每一个评测样本，合成两排模板图：
  ① 控制条件：原图 | DWPose 运动 | Normal 法线 | Depth 深度 | Part Map 部位
  ② 编辑结果：目标图像 GT | Adapter OFF | Adapter ON
底部深色说明栏给出 OFF / ON 各自的 PSNR 与 SSIM（取自 eval_metrics.json）。

条件图按需从远端批量拉取（单次 tar 流式传输，避免逐文件 scp）。

用法：
  python scripts/build_eval_template_panels.py --src .codex_tmp/vis_pull --out 可视化 [--limit 10]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

CELL = 360
LABEL_W = 170
GAP = 10
BG = (255, 255, 255)

HOST = "sg"
REMOTE_ASSETS = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"

ROW1 = [
    ("原图 (Source)", "#1F3864", "quad:0", "输入参考图\n身份 / 服装 / 背景"),
    ("DWPose (运动)", "#7030A0", "cond:dwpose_512", "姿态骨架\n控制信号"),
    ("Normal (法线)", "#C00000", "cond:normal_512", "表面法线\n几何控制"),
    ("Depth (深度)", "#ED7D31", "cond:depth_vis_512", "空间深度\n层级控制"),
    ("Part Map (部位)", "#70AD47", "cond:part_512", "身体部位\n划分控制"),
]
ROW2 = [
    ("目标图像 (GT)", "#538135", "quad:1", "编辑目标真值\n上界参照"),
    ("Adapter OFF", "#44546A", "quad:2", "无控制基线\n仅文本与参考图"),
    ("Adapter ON (最终)", "#1F1F1F", "quad:3", None),
]
GROUPS = [("① 控制条件", ROW1), ("② 真值与编辑结果", ROW2)]

FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]
RE_EVAL = re.compile(r"^eval_\d+_([\d_]+?)_(source_.+?)_(target_.+)\.png$")


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_text_center(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    tb = draw.textbbox((0, 0), text, font=font)
    draw.text(((x0 + x1 - (tb[2] - tb[0])) // 2, (y0 + y1 - (tb[3] - tb[1])) // 2), text, fill=fill, font=font)


def fetch_conditions(pairs: list[dict], staging: Path) -> None:
    """用一次 tar 流把所需条件图全部拉回。"""
    dest = staging / "eval_conditions"
    dest.mkdir(parents=True, exist_ok=True)
    names = set()
    for p in pairs:
        for cond in ("normal_512", "part_512", "depth_vis_512", "dwpose_512"):
            names.add(f"{p['appearance']}/{cond}/target/{p['target_stem']}.png")
    names = sorted(n for n in names if not (dest / n).exists())
    if not names:
        print("[info] 条件图已全部在本地缓存")
        return
    print(f"[info] 拉取 {len(names)} 个条件图文件（tar 流）")
    r = subprocess.run(
        ["ssh", HOST, f"cd {REMOTE_ASSETS} && tar czf - -T -"],
        input="\n".join(names).encode("utf-8"), capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        print(f"[warn] 条件图拉取失败：{r.stderr.decode('utf-8', 'ignore')[:200]}")
        return
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)
    got = sum(1 for n in names if (dest / n).exists())
    print(f"[info] 已获取 {got}/{len(names)} 个条件图")


def crop_quad(path: Path) -> list[Image.Image]:
    im = Image.open(path).convert("RGB")
    w = im.width // 4
    top = 33 if im.height > 414 - 1 else 0
    return [im.crop((i * w, top, (i + 1) * w, im.height)) for i in range(4)]


def build_panel(pair: dict, quad: Path, cond_dir: Path, out_path: Path, cell, fonts) -> bool:
    quads = crop_quad(quad)
    max_cols = max(len(g[1]) for g in GROUPS)
    body_w = max_cols * cell
    W = LABEL_W + body_w
    rows, total_h = [], 0
    for _, cols in GROUPS:
        cw = body_w // len(cols)
        header_h, caption_h = max(26, cw // 9), max(34, cw // 7)
        rows.append((cw, header_h, caption_h))
        total_h += cw + header_h + caption_h
    H = total_h + GAP * (len(GROUPS) - 1)

    board = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(board)
    f_head, f_cap, f_label, f_metric = fonts
    y = 0
    for (label, cols), (cw, header_h, caption_h) in zip(GROUPS, rows):
        draw.rectangle([0, y, LABEL_W - 1, y + header_h + cw + caption_h - 1], fill="#2C2C2A")
        draw_text_center(draw, (0, y + header_h, LABEL_W - 1, y + header_h + cw), label, f_label, (255, 255, 255))
        for j, (title, color, src, cap) in enumerate(cols):
            x0 = LABEL_W + j * cw
            draw.rectangle([x0, y, x0 + cw - 1, y + header_h - 1], fill=color)
            fcolor = (255, 210, 90) if title.startswith("Adapter ON") else (255, 255, 255)
            draw_text_center(draw, (x0, y, x0 + cw - 1, y + header_h - 1), title, f_head, fcolor)

            cy = y + header_h
            if src.startswith("quad:"):
                im = quads[int(src.split(":")[1])]
            else:
                p = cond_dir / f"{pair['appearance']}/{src.split(':')[1]}/target/{pair['target_stem']}.png"
                im = Image.open(p) if p.exists() else None
            if im is None:
                draw.rectangle([x0, cy, x0 + cw - 1, cy + cw - 1], fill=(235, 235, 235))
                draw_text_center(draw, (x0, cy, x0 + cw - 1, cy + cw - 1), "缺图", f_label, (120, 120, 120))
            else:
                im = im.convert("RGB")
                if src.startswith("cond:") and ImageStat.Stat(im.convert("L")).mean[0] < 20:
                    im = ImageOps.autocontrast(im, cutoff=1)
                s = min(cw / im.width, cw / im.height)
                rs = im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                board.paste(rs, (x0 + (cw - rs.width) // 2, cy + (cw - rs.height) // 2))

            cy2 = cy + cw
            draw.rectangle([x0, cy2, x0 + cw - 1, cy2 + caption_h - 1], fill=(26, 26, 26))
            m = pair.get("metrics", {}).get(title)
            if m:
                draw_text_center(draw, (x0, cy2, x0 + cw - 1, cy2 + caption_h - 1), m, f_metric, (120, 220, 160))
            elif cap:
                for k, line in enumerate(cap.split("\n")):
                    draw_text_center(draw, (x0, cy2 + k * (caption_h // 3), x0 + cw - 1, cy2 + (k + 1) * (caption_h // 3) + 2), line, f_cap, (190, 190, 190))
        y += header_h + cw + caption_h + GAP
    board.save(out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".codex_tmp/vis_pull")
    ap.add_argument("--out", default="可视化")
    ap.add_argument("--subdir", default="03_最终评测四联图/模板版_带条件控制")
    ap.add_argument("--cell", type=int, default=CELL)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本（0 = 全部）")
    args = ap.parse_args()

    staging = Path(args.src)
    quad_dir = staging / "eval_images"
    metrics_path = staging / "eval_metrics.json"
    out = Path(args.out) / args.subdir
    out.mkdir(parents=True, exist_ok=True)

    metrics_on, metrics_off = {}, {}
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in data.get("per_sample", []):
            metrics_on[(row["appearance"], row["source_stem"], row["target_stem"])] = row
        for row in data.get("per_sample_adapter_off", []):
            metrics_off[(row["appearance"], row["source_stem"], row["target_stem"])] = row

    pairs = []
    for p in sorted(quad_dir.glob("eval_*.png")):
        m = RE_EVAL.match(p.name)
        if not m:
            continue
        appearance, src_stem, tgt_stem = m.groups()
        key = (appearance, src_stem, tgt_stem)
        met = {}
        if key in metrics_on:
            met["Adapter OFF"] = f"PSNR {metrics_off[key]['psnr']:.1f} | SSIM {metrics_off[key]['ssim']:.3f}" if key in metrics_off else ""
            met["Adapter ON (最终)"] = f"PSNR {metrics_on[key]['psnr']:.1f} | SSIM {metrics_on[key]['ssim']:.3f}"
        pairs.append({"appearance": appearance, "source_stem": src_stem, "target_stem": tgt_stem,
                      "quad": p, "metrics": met, "idx": int(re.match(r"eval_(\d+)", p.name).group(1))})
    if args.limit:
        pairs = pairs[: args.limit]

    fetch_conditions(pairs, staging)
    cond_dir = staging / "eval_conditions"
    fonts = (load_font(max(14, args.cell // 24)), load_font(max(11, args.cell // 34)),
             load_font(max(14, args.cell // 20)), load_font(max(12, args.cell // 28)))

    n = 0
    for pair in pairs:
        p = out / f"评测{pair['idx'] + 1:03d}_{pair['appearance']}_{pair['source_stem'].replace('source_', '')}_to_{pair['target_stem'].replace('target_', '')}_模板图.png"
        if build_panel(pair, pair["quad"], cond_dir, p, args.cell, fonts):
            n += 1
    print(f"完成：{n} 张模板图 → {out.resolve()}")


if __name__ == "__main__":
    main()
