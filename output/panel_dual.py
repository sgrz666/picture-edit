#!/usr/bin/env python
"""panel_dual.py — 把双人 run 的八张图拼成带中文标注的面板。

每个 run 固定八格（复刻人工面板的 4+4 版式）：

    上排: ① 源场景 A   ② 目标姿态 B   ③ B＋骨架叠加   ④ A＋骨架叠加
    下排: ⑤ slot1 骨架  ⑥ slot2 骨架   ⑦ 双人合并骨架  ⑧ 生成结果

八格的来历（runs/<name>/ 下）：
    ① inputs/A_reference.png      A 源帧：身份、服装、场景都从它来
    ② inputs/B_reference.png      B 目标帧：只提供交互姿态
    ③ geometry/B_skeleton_overlay.png  目标骨架叠回 B 图，核对检测与定标
    ④ geometry/A_skeleton_overlay.png  同一套骨架叠到 A 图，即"想要的动作"
    ⑤ geometry/skeleton_1.png     B 帧从左起第 1 人的单人骨架（黑底）
    ⑥ geometry/skeleton_2.png     B 帧从左起第 2 人的单人骨架（黑底）
    ⑦ geometry/skeleton_both.png  两张单人骨架合并，不跨人连线（核对用）
    ⑧ generation/final_image.png  Qwen-2.1 基座的生成结果（无 adapter/LoRA）

注意 ⑦ 不直接进生成器：skeleton_only 臂给生成器的是 ①⑤⑥ 三张，
with_b 臂给 ①②⑤⑥ 四张；PE 两种臂都看全四张（A、B、骨架1、骨架2）。
脚本按 generation.json 的 reference_images 角色给格子打"生成器输入"标，
skeleton_only 臂下 ② 会打"仅 PE 可见"。

用法：
    python panel_dual.py --run hug_skeleton_only
    python panel_dual.py --all [--out panels] [--runs-dir runs]
    python panel_dual.py --run a,b --run c        # 逗号分隔或重复传参
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 字体

def _load_cjk():
    """返回 (regular, bold, mono) 三个 ImageFont 工厂；找不到 CJK 时降级并告警。"""
    reg_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    bold_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    mono_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def open_font(path, size):
        if path.endswith(".ttc"):
            # ttc 内含多个地区变体，挑 Noto Sans CJK SC；挑不到用 0 号
            for idx in range(10):
                try:
                    f = ImageFont.truetype(path, size, index=idx)
                    name = " ".join(str(x) for x in f.getname())
                    if "SC" in name:
                        return f
                except Exception:
                    break
            return ImageFont.truetype(path, size, index=0)
        return ImageFont.truetype(path, size)

    def first_existing(cands, what):
        for p in cands:
            if Path(p).exists():
                return p
        print(f"[panel] 警告：找不到{what}，中文可能显示为方块", file=sys.stderr)
        return None

    reg_path = first_existing(reg_candidates, "CJK 常规字体")
    bold_path = first_existing(bold_candidates, "CJK 粗体")
    mono_path = first_existing(mono_candidates, "等宽字体")

    def make(kind, size):
        path = {"reg": reg_path, "bold": bold_path, "mono": mono_path}[kind]
        if path is None:
            return ImageFont.load_default()
        try:
            return open_font(path, size)
        except Exception:
            return ImageFont.load_default()

    return make


# ---------------------------------------------------------------- 版式常量

W = 1600
MARGIN = 28
GAP = 16
COLS = 4
CELL_PAD = 6
IMG_H = 470
CAPTION_H = 96
HEADER_H = 138
ROW_GAP = 18
FOOTER_H = 112

INK = "#1F2328"
SUB = "#57606A"
FAINT = "#8C959F"
BORDER = "#D0D7DE"
CAP_BG = "#F6F8FA"
PLACE_BG = "#FFF5F5"
PLACE_INK = "#CF222E"
BADGE_GEN_BG = "#DAF5E4"
BADGE_GEN_INK = "#1A7F37"
BADGE_GEN_BD = "#4AC26B"
BADGE_PE_BG = "#EEF1F4"
BADGE_PE_INK = "#57606A"
BADGE_PE_BD = "#AFB8C1"

CELL_W = (W - 2 * MARGIN - (COLS - 1) * GAP) // COLS
CELL_H = IMG_H + CAPTION_H

# 八格定义：顺序即版式（上排 4 + 下排 4）
CELL_DEFS = [
    dict(key="A_reference", rel="inputs/A_reference.png",
         name="① 源场景 A", desc="身份 / 服装 / 场景都从它来"),
    dict(key="B_reference", rel="inputs/B_reference.png",
         name="② 目标姿态 B", desc="只提供交互姿态"),
    dict(key="B_skeleton_overlay", rel="geometry/B_skeleton_overlay.png",
         name="③ B＋骨架叠加", desc="骨架叠回目标图，核对检测与定标"),
    dict(key="A_skeleton_overlay", rel="geometry/A_skeleton_overlay.png",
         name="④ A＋骨架叠加", desc="把目标动作叠到源场景上"),
    dict(key="skeleton_1", rel="geometry/skeleton_1.png",
         name="⑤ slot 1 单人骨架", desc=""),
    dict(key="skeleton_2", rel="geometry/skeleton_2.png",
         name="⑥ slot 2 单人骨架", desc=""),
    dict(key="skeleton_both", rel="geometry/skeleton_both.png",
         name="⑦ 双人合并骨架", desc="两副骨架叠在一起，不跨人连线"),
    dict(key="final_image", rel="generation/final_image.png",
         name="⑧ 生成结果", desc="Qwen-2.1 基座出图，无 adapter / LoRA"),
]


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def truncate_to(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def draw_pill(draw, xy, text, font, bg, ink, bd):
    x, y = xy
    tw = draw.textlength(text, font=font)
    pad_x, pad_y = 8, 4
    w = tw + 2 * pad_x
    h = font.size + 2 * pad_y
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h / 2,
                           fill=bg, outline=bd, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=ink)
    return w


def fit_image(path, box_w, box_h):
    """读图并按比例缩放进 (box_w, box_h)，返回 PIL.Image；失败返回 None。"""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    iw, ih = im.size
    if iw <= 0 or ih <= 0:
        return None
    scale = min(box_w / iw, box_h / ih)
    if scale < 1:
        im = im.resize((max(1, int(iw * scale)), max(1, int(ih * scale))),
                       Image.LANCZOS)
    return im


def slot_desc(run_dir, slot):
    """从 dual_geometry.json 取该 slot 的说明：从左起第几人、Hi4D 侧、有效点数。"""
    geo = read_json(run_dir / "geometry" / "dual_geometry.json")
    if not geo:
        return "B 帧从左起第 %d 人" % slot
    for p in geo.get("persons", []):
        if p.get("slot") == slot:
            side = p.get("hi4d_side", "?")
            valid = p.get("valid_canvas", p.get("valid_native", "?"))
            return "B 帧从左起第 %d 人 · Hi4D side %s · %s 点" % (slot, side, valid)
    return "B 帧从左起第 %d 人" % slot


def header_lines(run_dir, run_name):
    """头部三行：标题 / 配对信息 / 臂与设置。"""
    rj = read_json(run_dir / "run.json") or {}
    pair = rj.get("pair", {}) or {}
    man = pair.get("manifest", {}) or {}
    st = rj.get("settings", {}) or {}

    title = "双人姿态迁移面板 — %s" % run_name

    bits = []
    if pair.get("name"):
        bits.append("配对 %s" % pair["name"])
    if pair.get("kind"):
        bits.append({"same_identity": "同一对演员",
                     "cross_pair": "不同演员"}.get(pair["kind"], pair["kind"]))
    if man:
        bits.append("%s #%s → %s #%s（cam%s）" % (
            man.get("source_action", "?"), man.get("source_frame_id", "?"),
            man.get("target_action", "?"), man.get("target_frame_id", "?"),
            man.get("camera_id", "?")))
    line2 = " · ".join(bits) if bits else "（run.json 缺失，仅按目录名生成）"

    sbits = []
    if st.get("generator_references"):
        sbits.append("臂 %s" % st["generator_references"])
    if st.get("framing"):
        sbits.append("定标 %s" % st["framing"])
    if st.get("resolution"):
        sbits.append("%s px" % st["resolution"])
    if st.get("steps"):
        sbits.append("%s 步" % st["steps"])
    if st.get("seed") is not None:
        sbits.append("seed %s" % st["seed"])
    if rj.get("status"):
        sbits.append("状态 %s" % rj["status"])
    line3 = " · ".join(sbits)

    gen = read_json(run_dir / "generation" / "generation.json")
    if gen:
        g = []
        if gen.get("seconds"):
            g.append("生成 %.0f s" % gen["seconds"])
        if gen.get("peak_gpu_memory_gib"):
            g.append("峰值显存 %.1f GiB" % gen["peak_gpu_memory_gib"])
        if g:
            line3 = (line3 + " · " if line3 else "") + " · ".join(g)

    return title, line2, line3


def metric_lines(run_dir):
    """尾部指标两行；无评测返回空列表。"""
    ev = read_json(run_dir / "evaluation" / "eval_dual.json")
    if not ev:
        return []
    s = ev.get("summary", {}) or {}
    inter = ev.get("interaction", {}) or {}
    rc = ev.get("region_count", {}) or {}
    regions = ",".join(str(r.get("found")) for r in rc.get("regions", [])) or "—"

    def ok(v):
        return "✓" if v else "✗"

    def num(v, fmt="{:.3f}", dash="—"):
        return dash if v is None else fmt.format(v)

    line1 = ("评测（DWPose 回估，basis=%s）：整图检出 %s 人 %s · 各人物区域检出 [%s] %s"
             % (ev.get("pose_basis", "?"),
                num(s.get("persons_detected"), "{:.0f}"),
                ok(s.get("person_count_ok")), regions,
                ok(s.get("person_count_regions_ok"))))
    line2 = ("PCK@0.2 %s · MPJPE(归一) %s · 躯干比 %s · 髋距误差 %s · 身份相似 %s · 疑似换人 %s"
             % (num(s.get("body_pck@0.2")), num(s.get("body_mpjpe_norm")),
                num(s.get("torso_scale_ratio")),
                num(inter.get("error_relative"), "{:.1%}"),
                num(s.get("identity_similarity")),
                "是" if s.get("identity_swap_suspected") else "否"))
    return [line1, line2]


def build_panel(run_dir, run_name, fonts):
    """拼一个 run 的面板，返回 (PIL.Image, 缺失文件列表)。"""
    make = fonts
    f_title = make("bold", 30)
    f_info = make("reg", 19)
    f_name = make("bold", 23)
    f_desc = make("reg", 18)
    f_file = make("mono", 15)
    f_badge = make("bold", 15)
    f_metric = make("reg", 18)

    gen = read_json(run_dir / "generation" / "generation.json") or {}
    arm = gen.get("generator_references") or (
        read_json(run_dir / "run.json") or {}).get("settings", {}).get(
        "generator_references")
    gen_roles = {r.get("role") for r in gen.get("reference_images", [])}

    mlines = metric_lines(run_dir)
    footer_h = FOOTER_H if mlines else 10
    H = MARGIN + HEADER_H + 14 + 2 * CELL_H + ROW_GAP + 12 + footer_h + MARGIN

    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)

    # ---- 头部 ----
    title, line2, line3 = header_lines(run_dir, run_name)
    y = MARGIN
    d.text((MARGIN, y), title, font=f_title, fill=INK)
    y += 42
    d.text((MARGIN, y), line2, font=f_info, fill=SUB)
    y += 27
    d.text((MARGIN, y), line3, font=f_info, fill=SUB)
    d.line([(MARGIN, MARGIN + HEADER_H - 8), (W - MARGIN, MARGIN + HEADER_H - 8)],
           fill=BORDER, width=1)

    # ---- 八格 ----
    missing = []
    for i, cd in enumerate(CELL_DEFS):
        row, col = divmod(i, COLS)
        x0 = MARGIN + col * (CELL_W + GAP)
        y0 = MARGIN + HEADER_H + 14 + row * (CELL_H + ROW_GAP)
        path = run_dir / cd["rel"]

        # 图片区
        box_w, box_h = CELL_W - 2 * CELL_PAD, IMG_H - 2 * CELL_PAD
        im = fit_image(path, box_w, box_h) if path.exists() else None
        if im is None:
            missing.append(cd["rel"])
            d.rectangle([x0, y0, x0 + CELL_W, y0 + IMG_H],
                        fill=PLACE_BG, outline=BORDER, width=1)
            msg = "缺失 %s" % cd["rel"]
            tw = d.textlength(msg, font=f_desc)
            d.text((x0 + (CELL_W - tw) / 2, y0 + IMG_H / 2 - 10), msg,
                   font=f_desc, fill=PLACE_INK)
        else:
            ix = x0 + (CELL_W - im.size[0]) // 2
            iy = y0 + CELL_PAD + (box_h - im.size[1]) // 2
            d.rectangle([x0, y0, x0 + CELL_W, y0 + IMG_H],
                        fill="#F0F0F0", outline=BORDER, width=1)
            img.paste(im, (ix, iy))

        # 说明区
        cy = y0 + IMG_H
        d.rectangle([x0, cy, x0 + CELL_W, cy + CAPTION_H],
                    fill=CAP_BG, outline=BORDER, width=1)
        tx = x0 + 10
        ty = cy + 8
        name = cd["name"]
        badge = None
        if cd["key"] in gen_roles:
            badge = ("生成器输入", BADGE_GEN_BG, BADGE_GEN_INK, BADGE_GEN_BD)
        elif cd["key"] == "B_reference" and arm == "skeleton_only":
            badge = ("仅 PE 可见", BADGE_PE_BG, BADGE_PE_INK, BADGE_PE_BD)
        name_max = CELL_W - 20 - (150 if badge else 0)
        d.text((tx, ty), truncate_to(d, name, f_name, name_max),
               font=f_name, fill=INK)
        if badge:
            bw = draw_pill(d, (x0 + CELL_W - 10 - 140, ty + 3),
                           badge[0], f_badge, badge[1], badge[2], badge[3])
            del bw
        ty += 30
        desc = cd["desc"]
        if cd["key"] in ("skeleton_1", "skeleton_2"):
            desc = slot_desc(run_dir, 1 if cd["key"] == "skeleton_1" else 2)
        for ln in (desc,):
            d.text((tx, ty), truncate_to(d, ln, f_desc, CELL_W - 20),
                   font=f_desc, fill=SUB)
            ty += 24
        d.text((tx, ty), truncate_to(d, cd["rel"], f_file, CELL_W - 20),
               font=f_file, fill=FAINT)

    # ---- 尾部指标 ----
    if mlines:
        fy = MARGIN + HEADER_H + 14 + 2 * CELL_H + ROW_GAP + 12
        d.rectangle([MARGIN, fy, W - MARGIN, fy + footer_h],
                    fill=CAP_BG, outline=BORDER, width=1)
        d.text((MARGIN + 12, fy + 12), mlines[0], font=f_metric, fill=INK)
        d.text((MARGIN + 12, fy + 44), mlines[1], font=f_metric, fill=SUB)
        d.text((MARGIN + 12, fy + 76),
               "读法：整图人数只在两人分开时可信，相拥/纠缠姿态看各人物区域；"
               "身份相似是 DINOv2 外观代理，不是生物身份。",
               font=make("reg", 15), fill=FAINT)

    return img, missing


def main():
    ap = argparse.ArgumentParser(description="双人 run 八图标注面板生成")
    ap.add_argument("--run", action="append", default=[],
                    help="run 名（runs/ 下的目录名），可逗号分隔或重复传参")
    ap.add_argument("--all", action="store_true", help="对 runs/ 下全部 run 生成")
    ap.add_argument("--runs-dir", default=str(SCRIPT_DIR / "runs"))
    ap.add_argument("--out", default=str(SCRIPT_DIR / "panels"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for chunk in args.run:
        names.extend([n.strip() for n in chunk.split(",") if n.strip()])
    if args.all:
        names = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not names:
        ap.error("至少给一个 --run 或加 --all")

    fonts = _load_cjk()
    index = {}
    n_ok = 0
    for name in names:
        run_dir = runs_dir / name
        if not run_dir.is_dir():
            print("[panel] 跳过（目录不存在）: %s" % name, file=sys.stderr)
            continue
        have = [cd["rel"] for cd in CELL_DEFS if (run_dir / cd["rel"]).exists()]
        if not have:
            print("[panel] 跳过（八张图全缺）: %s" % name, file=sys.stderr)
            continue
        panel, missing = build_panel(run_dir, name, fonts)
        out_path = out_dir / ("%s.png" % name)
        panel.save(out_path)
        index[name] = dict(panel=str(out_path), missing=missing,
                           eval=bool(metric_lines(run_dir)))
        n_ok += 1
        if not args.quiet:
            print("%-58s -> %s  (%d/8 图%s)"
                  % (name, out_path, 8 - len(missing),
                     "，缺: " + ", ".join(missing) if missing else ""))

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print("[panel] 完成 %d 张面板 -> %s（索引 index.json）" % (n_ok, out_dir))


if __name__ == "__main__":
    main()
