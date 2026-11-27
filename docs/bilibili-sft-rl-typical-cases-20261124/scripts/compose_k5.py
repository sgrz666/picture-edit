"""Compose SFT->RL200 comparison sheets for the Bilibili pose/ID GRPO experiment."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(ROOT, "k5_cases")
META = {m["case_no"]: m for m in json.load(open(os.path.join(ROOT, "k5_cases", "meta.json"), encoding="utf8"))}

FONT_DIR = "C:/Windows/Fonts"
F_REG = os.path.join(FONT_DIR, "msyh.ttc")
F_BOLD = os.path.join(FONT_DIR, "msyhbd.ttc")


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def load(pid, kind):
    if kind in ("0", "200", "300"):
        p = os.path.join(CASES, "images", kind, f"{pid}.png")
    else:
        p = os.path.join(CASES, "context", pid, f"{kind}.png")
    return Image.open(p).convert("RGB")


def fit(im, w, h, bg=(255, 255, 255)):
    r = min(w / im.width, h / im.height)
    nw, nh = int(im.width * r), int(im.height * r)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(im2, ((w - nw) // 2, (h - nh) // 2))
    return canvas


INK = (23, 32, 46)
MUTED = (108, 120, 136)
LINE = (223, 229, 237)
SOFT = (246, 248, 251)
UP = (192, 38, 38)
DOWN = (26, 118, 60)
ACCENT = (37, 99, 235)
WARN = (176, 106, 18)

TAGS = {
    "both": ("姿态 + ID 双升", UP),
    "id_na": ("仅姿态可评", MUTED),
    "tradeoff": ("取舍：一升一降", WARN),
    "regression": ("反例：两项均降", DOWN),
}

CAPTIONS = {
    37: "抬腿：SFT 抬腿偏低、脚尖下垂；RL200 膝盖抬高、小腿回收，更接近真值，ID 也同步提升。",
    31: "交叉站姿：SFT 双脚落地、双臂外张；RL200 收成真值的单腿交叉站立，PCK@0.02 从 70.6% 到 94.1%。",
    5: "抱臂跪姿：SFT 手没到脸、躯干直立；RL200 双手收到面前、上身对齐，但下半身仍未跪下去。",
    19: "举手：SFT 把双手都抬到脑后；RL200 只抬右手、左手放下，与真值一致。",
    20: "取舍：RL200 手臂略微外张、Pose 下降，但脸部与源图 A 更一致，ID cos +0.084。",
    36: "反例：SFT 左手贴脸更接近真值，RL200 手落到胸前，Pose 与 ID 同时下降。",
}


def draw_sheet(case_nos, out_path, title, subtitle, note=None, img_w=248):
    img_h = int(img_w * 672 / 384)
    margin = 54
    label_w = 212
    gap = 14
    head_h = 0
    metric_h = 58
    cap_h = 30
    row_gap = 40
    top = 142

    W = margin * 2 + label_w + gap + 5 * img_w + 4 * gap
    row_h = head_h + img_h + metric_h + cap_h
    H = top + len(case_nos) * row_h + (len(case_nos) - 1) * row_gap + margin + (40 if note else 0)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    d.text((margin, 26), title, font=font(30, True), fill=INK)
    d.text((margin, 74), subtitle, font=font(14.5), fill=MUTED)
    d.line([(margin, top - 38), (W - margin, top - 38)], fill=LINE, width=1)

    cols = ["A 源图", "B 目标骨架", "B 真值", "SFT 9520（RL 前）", "RL 200（RL 后）"]
    x0 = margin + label_w + gap
    for i, c in enumerate(cols):
        x = x0 + i * (img_w + gap)
        color = ACCENT if i == 4 else (INK if i == 3 else MUTED)
        d.text((x, top - 30), c, font=font(14.5, i >= 3), fill=color)

    for ri, no in enumerate(case_nos):
        m = META[no]
        pid = m["pair_id"]
        y = top + ri * (row_h + row_gap)

        if ri:
            d.line([(margin, y - row_gap // 2), (W - margin, y - row_gap // 2)], fill=(240, 243, 247), width=1)

        # left label block
        d.text((margin, y - 2), f"#{no:02d}", font=font(23, True), fill=INK)
        d.text((margin, y + 30), m["clip_id"], font=font(12), fill=MUTED)
        tag, tcol = TAGS.get(m["category"], (m["category"], MUTED))
        d.text((margin, y + 50), tag, font=font(12.5, True), fill=tcol)

        dp = m["delta_pose_200"]
        dpck = m["delta_pck002_200"]
        dy = y + img_h + 10
        def col(v):
            return UP if v > 1e-9 else (DOWN if v < -1e-9 else MUTED)
        d.text((margin, dy), f"ΔPose {dp:+.4f}", font=font(13, True), fill=col(dp))
        d.text((margin, dy + 20), f"ΔPCK  {dpck*100:+.1f}pp", font=font(13, True), fill=col(dpck))
        if m["delta_glintr_200"] is not None:
            dg = m["delta_glintr_200"]
            d.text((margin, dy + 40), f"ΔID   {dg:+.4f}", font=font(13, True), fill=col(dg))
        else:
            d.text((margin, dy + 40), "ΔID   不适用", font=font(13), fill=MUTED)
        d.line([(margin + label_w + gap // 2, y + 2), (margin + label_w + gap // 2, y + img_h + metric_h - 8)], fill=(238, 242, 247), width=1)

        imgs = [
            fit(load(pid, "A"), img_w, img_h),
            fit(load(pid, "pose"), img_w, img_h, (0, 0, 0)),
            fit(load(pid, "GT"), img_w, img_h),
            fit(load(pid, "0"), img_w, img_h),
            fit(load(pid, "200"), img_w, img_h),
        ]
        for i, im in enumerate(imgs):
            x = x0 + i * (img_w + gap)
            canvas.paste(im, (x, y + head_h))
            d.rectangle([x, y + head_h, x + img_w - 1, y + head_h + img_h - 1], outline=LINE, width=1)

        xr = x0 + 4 * (img_w + gap)
        d.rectangle([xr - 2, y + head_h - 2, xr + img_w + 1, y + head_h + img_h + 1], outline=ACCENT, width=2)

        ys = y + img_h + 10
        xs = x0 + 3 * (img_w + gap)
        d.text((xs, ys), f"Pose {m['pose_sft']:.3f}   PCK {m['pck002_sft']*100:.1f}%", font=font(12.5), fill=MUTED)
        id_s = f"ID cos {m['glintr_sft']:.3f}" if m["glintr_sft"] is not None else "ID 不适用"
        d.text((xs, ys + 19), id_s, font=font(12.5), fill=MUTED)

        d.text((xr, ys), f"Pose {m['pose_rl200']:.3f}   PCK {m['pck002_rl200']*100:.1f}%", font=font(12.5, True), fill=INK)
        id_r = f"ID cos {m['glintr_rl200']:.3f}" if m["glintr_rl200"] is not None else "ID 不适用"
        d.text((xr, ys + 19), id_r, font=font(12.5, True), fill=INK)

        cap = CAPTIONS.get(no, "")
        if cap:
            d.text((x0, y + img_h + metric_h + 2), cap, font=font(12.5), fill=(70, 82, 98))

    if note:
        d.text((margin, H - 46), note, font=font(11.5), fill=MUTED)
        d.text((margin, H - 28), "数据来源：sg:/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1（per_pair_metrics.csv + images/）",
               font=font(11.5), fill=(150, 160, 172))

    canvas.save(out_path)
    print("saved", out_path, canvas.size)


if __name__ == "__main__":
    draw_sheet(
        [37, 31, 19, 5],
        os.path.join(ROOT, "fig1_typical.png"),
        "Bilibili 单人姿态迁移：SFT 9520 → RL 200 典型样本前后对比",
        "同一固定 50 对验证集 · 384×672 · seed42 · 30 步 ODE · CFG1 · 五列均为原始输出，未做任何修饰或挑选式修图",
        note="Pose = SDPose 连续姿态奖励（0–1）；PCK@0.02 = COCO17 关节命中率；ID cos = glintr100 余弦（17 个 ID 不适用样本不计入）。红=上升、绿=下降。",
    )
    draw_sheet(
        [20, 36],
        os.path.join(ROOT, "fig2_boundary.png"),
        "边界与反例：平均收益之外的两类样本",
        "同一 50 对验证集内挑出的取舍样本与唯一反例，用于界定上面那组典型图的适用边界",
        note="50 对中 Pose 38 升 12 降；33 个可评 ID 样本 30 升 3 降。均值提升不排除个别退化，选点偏差与 5 个视频的小簇数也在报告里写明了。",
    )
