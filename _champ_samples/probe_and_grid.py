# -*- coding: utf-8 -*-
"""CHAMP(复旦) 样本探针 + 预览网格生成。

用途：判断 HF champ_trainning_sample 的 832 个网络视频，能否用于
      pose-transfer A/B/C 三元组（同人 + 同服装 + 多姿态）。
输出：
  - meta.json        每个视频的分辨率/帧数/fps/时长/码率
  - preview_grid.jpg 每个视频抽 8 帧，纵向一行的网格预览
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(HERE, "videos")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
N_SAMPLES = 8          # 每个视频抽帧数
THUMB_H = 220          # 每帧缩略图高度


def ffprobe(path):
    """用 ffmpeg -i 的 stderr 解析基础参数（不依赖 ffprobe）。"""
    p = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True, errors="ignore")
    return p.stderr


def read_meta(path):
    """解析 ffmpeg stderr 拿分辨率/fps/时长（imageio 的 pyav 后端返回 None，不可靠）。"""
    import re
    info = {"file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1048576, 2)}
    txt = ffprobe(path)
    m = re.search(r"Video:\s*(\w+).*?,\s*(\d+)x(\d+)", txt)
    if m:
        info["codec"], info["w"], info["h"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"(\d+\.?\d*)\s*fps", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration_s"] = round(h * 3600 + mi * 60 + s, 2)
    m = re.search(r"(\d+)\s*kb/s", txt)
    if m:
        info["bitrate_kbps"] = int(m.group(1))
    return info


def load_font(size=18):
    for cand in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                 "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    files = sorted(f for f in os.listdir(VDIR) if f.endswith(".mp4"))
    metas = []
    for f in files:
        p = os.path.join(VDIR, f)
        m = read_meta(p)
        metas.append(m)
        print(f"{m['file'][:16]}  {m.get('w')}x{m.get('h')}  "
              f"{m.get('fps')}fps  {m.get('duration_s')}s  {m['size_mb']}MB  "
              f"{m.get('codec')}  {m.get('bitrate_kbps')}kbps")

    json.dump(metas, open(os.path.join(HERE, "meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 生成预览网格 ----
    font = load_font(18)
    rows = []
    for f in files:
        p = os.path.join(VDIR, f)
        try:
            frames = iio.imiter(p, plugin="pyav")
            buf = [fr for fr in frames]
        except Exception as e:
            print("  !! 读帧失败", f, e)
            continue
        if not buf:
            continue
        idxs = np.linspace(0, len(buf) - 1, N_SAMPLES).astype(int)
        thumbs = []
        for i in idxs:
            im = Image.fromarray(buf[i]).convert("RGB")
            w = int(im.width * THUMB_H / im.height)
            thumbs.append(im.resize((w, THUMB_H), Image.LANCZOS))
        row_w = sum(t.width for t in thumbs) + 4 * (len(thumbs) - 1)
        row = Image.new("RGB", (row_w, THUMB_H + 30), (255, 255, 255))
        x = 0
        for t in thumbs:
            row.paste(t, (x, 30))
            x += t.width + 4
        d = ImageDraw.Draw(row)
        label = f"{f[:12]}...  {len(buf)}帧  {metas[files.index(f)].get('w')}x{metas[files.index(f)].get('h')}"
        d.text((4, 5), label, fill=(0, 0, 0), font=font)
        rows.append(row)
        print(f"  {f[:16]}  抽帧 {len(buf)} 帧 -> 网格行 ok")

    if rows:
        W = max(r.width for r in rows)
        H = sum(r.height for r in rows) + 6 * (len(rows) - 1)
        grid = Image.new("RGB", (W, H), (230, 230, 230))
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height + 6
        out = os.path.join(HERE, "preview_grid.jpg")
        grid.save(out, quality=88)
        print("saved:", out, grid.size)


if __name__ == "__main__":
    main()
