# -*- coding: utf-8 -*-
"""训练进程可视化整理模块（iPER V6.3）。

把远端拉回的实验产物（multistage_evolution_showcase / eval_results / eval_metrics.json）
整理为结构化的本地「可视化」文件夹，不生成网页，仅输出图片与一份 Markdown 指标汇总。

目录约定：
  可视化/
  ├── 00_总览与说明/     总览大图 + selected_showcase_pairs.jsonl + eval_metrics.json
  ├── 01_条件控制信号/   5 样本 × 6 栏条件图（原图/法线/深度/部位/骨架/目标GT，中文表头）
  ├── 02_训练阶段演化/   每样本 Step 0→3000 演化条带（已带 PSNR/SSIM 标注）
  └── 03_最终评测四联图/ 50 样本 Source|GT|OFF|ON 四联图，文件名含 PSNR/SSIM

用法：
  python scripts/build_visualization_folder.py \
      --src  .codex_tmp/vis_pull \
      --out  可视化
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HEADER_H = 36
CELL = 256
BG = (255, 255, 255)
FG = (30, 30, 30)
LINE = (210, 210, 210)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

COND_COLS = [
    ("原图 Source", "source"),
    ("法线 Normal", "cond_normal"),
    ("深度 Depth", "cond_depth"),
    ("部位划分 Part", "cond_part"),
    ("姿态骨架 DWPose", "cond_dwpose"),
    ("目标真值 GT", "target_gt"),
]


def load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit_cell(img: Image.Image, cell: int = CELL) -> Image.Image:
    img = img.convert("RGB")
    s = min(cell / img.width, cell / img.height)
    w, h = round(img.width * s), round(img.height * s)
    img = img.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (cell, cell), BG)
    canvas.paste(img, ((cell - w) // 2, (cell - h) // 2))
    return canvas


def build_condition_grid(sample_id: str, show_dir: Path, out_path: Path, font) -> None:
    """单样本 1×6 条件控制条带（参考图 1 样式，带中文表头）。"""
    imgs = []
    for _, key in COND_COLS:
        p = show_dir / f"{sample_id}_{key}.png"
        imgs.append(fit_cell(Image.open(p)) if p.exists() else Image.new("RGB", (CELL, CELL), (240, 240, 240)))
    W, H = CELL * len(imgs), CELL + HEADER_H
    board = Image.new("RGB", (W, H), BG)
    board.paste(imgs[0], (0, HEADER_H))
    for i, im in enumerate(imgs[1:], 1):
        board.paste(im, (i * CELL, HEADER_H))
    draw = ImageDraw.Draw(board)
    for i, (label, _) in enumerate(COND_COLS):
        x0 = i * CELL
        draw.rectangle([x0, 0, x0 + CELL - 1, HEADER_H - 1], outline=LINE)
        tb = draw.textbbox((0, 0), label, font=font)
        draw.text((x0 + (CELL - tb[2] + tb[0]) // 2, (HEADER_H - tb[3] + tb[1]) // 2), label, fill=FG, font=font)
    board.save(out_path)


def parse_eval_name(stem: str) -> tuple[str, str, str] | None:
    m = re.match(r"eval_\d+_([\d_]+?)_(source_.+?)_(target_.+)$", stem)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".codex_tmp/vis_pull", help="远端拉回资产所在目录")
    ap.add_argument("--out", default="可视化", help="输出目录")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    show_dir = src / "multistage_evolution_showcase"
    eval_dir = src / "eval_images"
    metrics_path = src / "eval_metrics.json"
    if not show_dir.exists():
        raise SystemExit(f"缺少 showcase 目录: {show_dir}")

    font = load_font(18)
    for sub in ("00_总览与说明", "01_条件控制信号", "02_训练阶段演化", "03_最终评测四联图"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sample_ids = sorted(
        {re.sub(r"_(source|target_gt|step_\d+|evolution_strip|cond_\w+)\.png$", "", p.name)
         for p in show_dir.glob("sample_*.png")}
    )

    for i, sid in enumerate(sample_ids, 1):
        appearance = re.sub(r"^sample_\d+_", "", sid)
        tag = f"样本{i}_{appearance}"
        build_condition_grid(sid, show_dir, out / "01_条件控制信号" / f"{tag}_条件控制六栏.png", font)
        strip = show_dir / f"{sid}_evolution_strip.png"
        if strip.exists():
            shutil.copy2(strip, out / "02_训练阶段演化" / f"{tag}_Step0-3000演化条带.png")

    master = show_dir / "v6_multistage_evolution_master_grid.png"
    if master.exists():
        shutil.copy2(master, out / "00_总览与说明" / "多阶段演化总览_5样本x9栏.png")
    for extra in ("selected_showcase_pairs.jsonl",):
        if (show_dir / extra).exists():
            shutil.copy2(show_dir / extra, out / "00_总览与说明" / extra)
    if metrics_path.exists():
        shutil.copy2(metrics_path, out / "00_总览与说明" / "eval_metrics.json")

    n_eval = 0
    metrics: dict[tuple[str, str, str], dict] = {}
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in data.get("per_sample", []):
            metrics[(row["appearance"], row["source_stem"], row["target_stem"])] = row
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("eval_*.png")):
            key = parse_eval_name(p.stem)
            row = metrics.get(key) if key else None
            if row:
                num = int(re.match(r"eval_(\d+)", p.stem).group(1))
                name = f"评测{num + 1:03d}_{key[0]}_{key[1].replace('source_', '')}_to_{key[2].replace('target_', '')}_ON-PSNR{row['psnr']:.1f}_SSIM{row['ssim']:.3f}.png"
            else:
                name = p.name
            shutil.copy2(p, out / "03_最终评测四联图" / name)
            n_eval += 1

    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        s_on, s_off = data.get("summary", {}), data.get("summary_adapter_off", {})
        rows = data.get("per_sample", [])
        best = sorted(rows, key=lambda r: -r["psnr"])[:5]
        worst = sorted(rows, key=lambda r: r["psnr"])[:5]
        lines = [
            "# iPER V6.3 实验可视化汇总",
            "",
            f"- 评测样本数：{s_on.get('num_samples')}（checkpoint：`{Path(data.get('adapter_checkpoint', '')).name}`）",
            "",
            "| 指标 | Adapter ON | Adapter OFF | 提升 |",
            "|---|---:|---:|---:|",
            f"| PSNR (dB) | {s_on.get('avg_psnr', 0):.2f} | {s_off.get('avg_psnr', 0):.2f} | +{s_on.get('avg_psnr', 0) - s_off.get('avg_psnr', 0):.2f} |",
            f"| SSIM | {s_on.get('avg_ssim', 0):.4f} | {s_off.get('avg_ssim', 0):.4f} | +{s_on.get('avg_ssim', 0) - s_off.get('avg_ssim', 0):.4f} |",
            f"| MAE | {s_on.get('avg_mae', 0):.2f} | {s_off.get('avg_mae', 0):.2f} | {s_on.get('avg_mae', 0) - s_off.get('avg_mae', 0):.2f} |",
            "",
            "## PSNR 最高 5 个样本",
            "",
            "| 样本 | PSNR | SSIM |",
            "|---|---:|---:|",
        ]
        for r in best:
            lines.append(f"| {r['appearance']} ({r['source_stem']}→{r['target_stem']}) | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        lines += ["", "## PSNR 最低 5 个样本", "", "| 样本 | PSNR | SSIM |", "|---|---:|---:|"]
        for r in worst:
            lines.append(f"| {r['appearance']} ({r['source_stem']}→{r['target_stem']}) | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        lines += [
            "",
            "## 文件夹说明",
            "",
            "- `01_条件控制信号/`：每样本 [原图|法线|深度|部位划分|姿态骨架|目标真值] 六栏控制图。",
            "- `02_训练阶段演化/`：每样本 Step 0/500/1000/1500/2000/2500/3000 + 真值的演化条带（图内已标注逐步 PSNR/SSIM）。",
            "- `03_最终评测四联图/`：50 个测试样本 [Source|Ground Truth|Adapter OFF|Adapter ON] 四联图，文件名含 ON 的 PSNR/SSIM。",
            "- 总览大图：`00_总览与说明/多阶段演化总览_5样本x9栏.png`。",
            "",
        ]
        (out / "00_总览与说明" / "指标汇总.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"完成：{len(sample_ids)} 个样本条件图/演化条带，{n_eval} 张评测四联图 → {out.resolve()}")


if __name__ == "__main__":
    main()
