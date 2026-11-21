# -*- coding: utf-8 -*-
"""编辑效果对比图合成模块（三排：控制条件 + 真值/训练早期 + 训练中后期）。

三排结构中的 step 列由样本目录里实际存在的 step_* 图自动枚举，不再写死：
  ① 控制条件：原图 | DWPose | Normal | Depth | Part Map
  ② 目标真值 + 前半段训练阶段（GT | Step 0 | … ）
  ③ 后半段训练阶段（… | Step 3000 最终）

用法：
  python scripts/build_edit_effect_strips.py --src .codex_tmp/vis_pull --out 可视化 [--cell 360]
  python scripts/build_edit_effect_strips.py --showcase-dir .codex_tmp/vis_pull/iper_v6_showcase_expanded --out 可视化
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

HOST = "sg"
REMOTE_SAMPLED = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt"
REMOTE_ASSETS = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"
COND_MAP = {
    "cond_dwpose": "dwpose_512",
    "cond_normal": "normal_512",
    "cond_depth": "depth_vis_512",
    "cond_part": "part_512",
}


def tar_fetch(root: str, names: list[str], dest: Path) -> int:
    names = sorted({n for n in names if n})
    if not names:
        return 0
    r = subprocess.run(["ssh", HOST, f"cd {root} && tar czf - -T -"],
                       input="\n".join(names).encode("utf-8"), capture_output=True)
    if r.returncode != 0 or not r.stdout:
        print(f"[warn] 拉取失败（{root.split('/')[-1]}）：{r.stderr.decode('utf-8', 'ignore')[:160]}")
        return 0
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)
    return sum(1 for n in names if (dest / n).exists())


def ensure_assets(show_dir: Path) -> None:
    """若缺少源图 / 真值 / 条件图，按 selected_showcase_pairs.jsonl 从远端补齐。"""
    pf = show_dir / "selected_showcase_pairs.jsonl"
    if not pf.exists():
        return
    pairs = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
    raw = show_dir.parent / "showcase_raw"
    (raw / "sampled").mkdir(parents=True, exist_ok=True)
    (raw / "assets").mkdir(parents=True, exist_ok=True)

    want_sampled, want_assets, jobs = [], [], []
    for i, p in enumerate(pairs, 1):
        app = p["appearance"]
        sid = f"sample_{i}_{app}"
        src = p["source"].get("rgb_1024")
        tgt = p["target"].get("rgb_1024")
        tstem = Path(tgt).stem if tgt else ""
        for key, fname in [("source", "source"), ("target_gt", "target_gt")]:
            if not (show_dir / f"{sid}_{fname}.png").exists():
                rel = src if key == "source" else tgt
                if rel:
                    rel_full = f"{app}/{rel}"
                    want_sampled.append(rel_full)
                    jobs.append((raw / "sampled" / rel_full, show_dir / f"{sid}_{fname}.png"))
        for ckey, rdir in COND_MAP.items():
            if not (show_dir / f"{sid}_{ckey}.png").exists() and tstem:
                rel = f"{app}/{rdir}/target/{tstem}.png"
                want_assets.append(rel)
                jobs.append((raw / "assets" / rel, show_dir / f"{sid}_{ckey}.png"))
    if not jobs:
        return
    print(f"[info] 补齐源图/真值 {len(want_sampled)} 个，条件图 {len(want_assets)} 个")
    tar_fetch(REMOTE_SAMPLED, want_sampled, raw / "sampled")
    tar_fetch(REMOTE_ASSETS, want_assets, raw / "assets")
    copied = 0
    for srcp, dstp in jobs:
        if srcp.exists():
            Image.open(srcp).convert("RGB").save(dstp)
            copied += 1
    print(f"[info] 已写入 {copied}/{len(jobs)} 个缺失文件")

CELL = 360
LABEL_W = 170
GAP = 10
BG = (255, 255, 255)

COND_COLS = [
    ("原图 (Source)", "#1F3864", "source", "输入参考图\n身份 / 服装 / 背景"),
    ("DWPose (运动)", "#7030A0", "cond_dwpose", "姿态骨架\n控制信号"),
    ("Normal (法线)", "#C00000", "cond_normal", "表面法线\n几何控制"),
    ("Depth (深度)", "#ED7D31", "cond_depth", "空间深度\n层级控制"),
    ("Part Map (部位)", "#70AD47", "cond_part", "身体部位\n划分控制"),
]
GT_COL = ("目标图像 (GT)", "#538135", "target_gt", "编辑目标真值\n上界参照")

FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def psnr_ssim(img: Image.Image, gt: Image.Image) -> tuple[float, float]:
    a = np.asarray(img.convert("RGB").resize((512, 512), Image.LANCZOS), dtype=np.float64)
    b = np.asarray(gt.convert("RGB").resize((512, 512), Image.LANCZOS), dtype=np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = 10.0 * np.log10(255.0**2 / mse) if mse > 1e-12 else 99.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x, mu_y = a.mean(), b.mean()
    s_x, s_y = a.std(), b.std()
    s_xy = ((a - mu_x) * (b - mu_y)).mean()
    ssim = ((2 * mu_x * mu_y + c1) * (2 * s_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (s_x**2 + s_y**2 + c2))
    return psnr, float(ssim)


def discover_steps(show_dir: Path, sample_id: str) -> list[tuple[str, str, str, str | None]]:
    """按编号自动枚举该样本的 step 列。"""
    found = []
    for p in show_dir.glob(f"{sample_id}_step_*.png"):
        m = re.match(rf"^{re.escape(sample_id)}_step_(\d+)\.png$", p.name)
        if m:
            found.append(int(m.group(1)))
    cols = []
    for i, n in enumerate(sorted(found)):
        last = i == len(found) - 1
        title = f"Step {n} (初始)" if n == 0 else (f"Step {n} (最终)" if last else f"Step {n}")
        color = "#1F1F1F" if last else "#44546A"
        cap = "无控制基线\nAdapter OFF" if n == 0 else None
        cols.append((title, color, f"step_{n}", cap))
    return cols


def build_groups(steps: list) -> list[tuple[str, list]]:
    half = len(steps) // 2 if len(steps) > 1 else len(steps)
    row2 = [GT_COL] + steps[:half]
    row3 = steps[half:]
    groups = [("① 控制条件", list(COND_COLS)), ("② 真值与训练早期", row2)]
    if row3:
        groups.append(("③ 训练中后期", row3))
    return groups


def draw_text_center(draw: ImageDraw.ImageDraw, box, text: str, font, fill):
    x0, y0, x1, y1 = box
    tb = draw.textbbox((0, 0), text, font=font)
    draw.text(((x0 + x1 - (tb[2] - tb[0])) // 2, (y0 + y1 - (tb[3] - tb[1])) // 2), text, fill=fill, font=font)


def enhance(im: Image.Image, key: str) -> Image.Image:
    im = im.convert("RGB")
    if key.startswith("cond_") and ImageStat.Stat(im.convert("L")).mean[0] < 20:
        im = ImageOps.autocontrast(im, cutoff=1)
    return im


def build_panel(show_dir: Path, sample_id: str, out_path: Path, cell: int, fonts) -> bool:
    steps = discover_steps(show_dir, sample_id)
    if not steps:
        return False
    groups = build_groups(steps)
    all_cols = [c for _, cols in groups for c in cols]

    cache, metrics = {}, {}
    gt_path = show_dir / f"{sample_id}_target_gt.png"
    gt = Image.open(gt_path) if gt_path.exists() else None
    for _, _, key, _ in all_cols:
        p = show_dir / f"{sample_id}_{key}.png"
        if p.exists():
            cache[key] = Image.open(p)
            if gt is not None and key.startswith("step_"):
                metrics[key] = psnr_ssim(cache[key], gt)
    if len(cache) < 7:
        return False

    max_cols = max(len(cols) for _, cols in groups)
    body_w = max_cols * cell
    W = LABEL_W + body_w
    rows, total_h = [], 0
    for _, cols in groups:
        cw = body_w // len(cols)
        header_h, caption_h = max(26, cw // 9), max(34, cw // 7)
        rows.append((cw, header_h, caption_h))
        total_h += cw + header_h + caption_h
    H = total_h + GAP * (len(groups) - 1)

    f_head, f_cap, f_label, f_metric = fonts
    board = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(board)
    y = 0
    for (label, cols), (cw, header_h, caption_h) in zip(groups, rows):
        draw.rectangle([0, y, LABEL_W - 1, y + header_h + cw + caption_h - 1], fill="#2C2C2A")
        draw_text_center(draw, (0, y + header_h, LABEL_W - 1, y + header_h + cw), label, f_label, (255, 255, 255))
        for j, (title, color, key, cap) in enumerate(cols):
            x0 = LABEL_W + j * cw
            draw.rectangle([x0, y, x0 + cw - 1, y + header_h - 1], fill=color)
            fcolor = (255, 210, 90) if title.endswith("(最终)") else (255, 255, 255)
            draw_text_center(draw, (x0, y, x0 + cw - 1, y + header_h - 1), title, f_head, fcolor)

            cy = y + header_h
            im = cache.get(key)
            if im is None:
                draw.rectangle([x0, cy, x0 + cw - 1, cy + cw - 1], fill=(235, 235, 235))
                draw_text_center(draw, (x0, cy, x0 + cw - 1, cy + cw - 1), "缺图", f_label, (120, 120, 120))
            else:
                im = enhance(im, key)
                s = min(cw / im.width, cw / im.height)
                rs = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                board.paste(rs, (x0 + (cw - rs.width) // 2, cy + (cw - rs.height) // 2))

            cy2 = cy + cw
            draw.rectangle([x0, cy2, x0 + cw - 1, cy2 + caption_h - 1], fill=(26, 26, 26))
            if key in metrics:
                psnr, ssim = metrics[key]
                draw_text_center(draw, (x0, cy2, x0 + cw - 1, cy2 + caption_h - 1), f"PSNR {psnr:.1f} | SSIM {ssim:.3f}", f_metric, (120, 220, 160))
            elif cap:
                for k, line in enumerate(cap.split("\n")):
                    draw_text_center(draw, (x0, cy2 + k * (caption_h // 3), x0 + cw - 1, cy2 + (k + 1) * (caption_h // 3) + 2), line, f_cap, (190, 190, 190))
        y += header_h + cw + caption_h + GAP

    board.save(out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".codex_tmp/vis_pull")
    ap.add_argument("--showcase-dir", default="", help="直接指定 showcase 目录（优先于 --src）")
    ap.add_argument("--out", default="可视化")
    ap.add_argument("--subdir", default="04_编辑效果条带")
    ap.add_argument("--cell", type=int, default=CELL)
    ap.add_argument("--stack", action="store_true")
    args = ap.parse_args()

    show_dir = Path(args.showcase_dir) if args.showcase_dir else Path(args.src) / "multistage_evolution_showcase"
    out = Path(args.out) / args.subdir
    out.mkdir(parents=True, exist_ok=True)

    fonts = (load_font(max(14, args.cell // 24)), load_font(max(11, args.cell // 34)),
             load_font(max(14, args.cell // 20)), load_font(max(12, args.cell // 28)))

    ensure_assets(show_dir)
    sample_ids = sorted(
        {re.sub(r"_(source|target_gt|step_\d+|evolution_strip|cond_\w+)\.png$", "", p.name) for p in show_dir.glob("sample_*.png")}
    )
    panels = []
    for i, sid in enumerate(sample_ids, 1):
        appearance = re.sub(r"^sample_\d+_", "", sid)
        p = out / f"样本{i}_{appearance}_编辑效果图.png"
        if build_panel(show_dir, sid, p, args.cell, fonts):
            panels.append(p)
            print(f"[ok] {p.name}（step 列 {len(discover_steps(show_dir, sid))} 个）")

    if args.stack and len(panels) > 1:
        imgs = [Image.open(p) for p in panels]
        W = max(im.width for im in imgs)
        H = sum(im.height for im in imgs) + 12 * (len(imgs) - 1)
        stack = Image.new("RGB", (W, H), BG)
        y = 0
        for im in imgs:
            stack.paste(im, (0, y))
            y += im.height + 12
        stack.save(out / "样本1-N_编辑效果总览.png")
        print("[ok] 样本1-N_编辑效果总览.png")

    print(f"完成：{len(panels)} 张三排编辑效果图 → {out.resolve()}")


if __name__ == "__main__":
    main()
