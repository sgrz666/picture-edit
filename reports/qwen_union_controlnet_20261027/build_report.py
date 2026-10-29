"""Build the Word report for the Qwen-Image-2.1-Fun-ControlNet-Union experiment family.

Reads staged images under assets/ (copied read-only from sg) and emits a .docx.
"""
import hashlib
import json
import os
import shutil
from collections import defaultdict

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DOCIMG = os.path.join(HERE, "_docimg")
OUT = os.path.join(HERE, "Qwen-Image-2.1-Fun-ControlNet-Union_实验汇报_20261027.docx")

USABLE_CM = 24.0  # landscape A4 minus 2cm margins each side

# ---------------------------------------------------------------- image prep
shutil.rmtree(DOCIMG, ignore_errors=True)
os.makedirs(DOCIMG, exist_ok=True)
_cache = {}


def prep(rel_path, max_edge=900, quality=88):
    """Downscale+re-encode so the deck stays portable. Originals are never touched."""
    src = rel_path if os.path.isabs(rel_path) else os.path.join(ASSETS, rel_path)
    key = (src, max_edge)
    if key in _cache:
        return _cache[key]
    dst = os.path.join(DOCIMG, "%s_%s.jpg" % (hashlib.md5(src.encode()).hexdigest()[:12], max_edge))
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_edge:
            s = max_edge / float(max(w, h))
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        im.save(dst, "JPEG", quality=quality, optimize=True)
    out = (dst, w, h)
    _cache[key] = out
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- doc helpers
doc = Document()

# landscape + tight margins
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(1.6)

style = doc.styles["Normal"]
style.font.name = "等线"
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
for lvl, sz in ((1, 20), (2, 15), (3, 12.5), (4, 11)):
    st = doc.styles["Heading %d" % lvl]
    st.font.name = "等线"
    st.font.size = Pt(sz)
    st.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")


def H(level, text):
    doc.add_heading(text, level=level)


def P(text, bold=False, size=None, italic=False, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    return p


def BULLETS(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def NOTE(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    return p


def TABLE(header, rows, widths=None, font=9.0):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(font)
        r.font.name = "等线"
        r.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
        shade(hdr[i], "DCE6F1")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            r.font.size = Pt(font)
            r.font.name = "等线"
            r.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = tcPr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): hexcolor})
    tcPr.append(sh)


def caption_cell(cell, text, size=8.0):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.name = "等线"
    r.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


IMG_COUNT = 0
USABLE_H_CM = 17.8  # landscape A4 height minus top/bottom margins


def _fit(w, h, cell_cm, fit_h_cm):
    """Pick a display width so the picture fits both the column and the page height."""
    if fit_h_cm:
        return min(cell_cm, fit_h_cm * w / float(h))
    return cell_cm


def GRID(items, cols=4, cell_cm=None, header=None, max_edge=900, fit_h_cm=None):
    """items = [(rel_path, caption)] laid out in a borderless grid with captions."""
    global IMG_COUNT
    IMG_COUNT += len(items)
    n = len(items)
    rows = (n + cols - 1) // cols
    ncols = cols
    t = doc.add_table(rows=(1 if header else 0) + rows, cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    base = 0
    if header:
        base = 1
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            parts = str(h).split("\n")
            for j, part in enumerate(parts):
                r = p.add_run(part)
                r.bold = True
                r.font.size = Pt(9)
                r.font.name = "等线"
                r.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
                if j < len(parts) - 1:
                    r.add_break()
            shade(c, "EFEFEF")
    # leave room for the header row and one caption line per image row
    avail_h = USABLE_H_CM - 1.2 - (0.9 if header else 0.0)
    per_row = avail_h / max(1, rows) - 0.9
    fh = fit_h_cm or per_row
    for idx, (path, cap) in enumerate(items):
        cell = t.rows[base + idx // ncols].cells[idx % ncols]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        img, w, h = prep(path, max_edge)
        cw = cell_cm or (USABLE_CM / cols - 0.25)
        run.add_picture(img, width=Cm(_fit(w, h, cw, fh)))
        caption_cell(cell, cap)
    return t


def FIG(rel_path, cap, width_cm=16.0, max_edge=1400, fit_h_cm=None):
    global IMG_COUNT
    IMG_COUNT += 1
    img, w, h = prep(rel_path, max_edge)
    if fit_h_cm:
        width_cm = min(width_cm, fit_h_cm * w / float(h))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img, width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(cap)
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    r.font.name = "等线"
    r.element.rPr.rFonts.set(qn("w:eastAsia"), "等线")


def PAGEBREAK():
    doc.add_page_break()


# ---------------------------------------------------------------- content data
A1 = "exp1_migration"
A2 = "exp2_hard05_6arm"
A3 = "exp3_schemeB_lora"
A4 = "exp4_proadapter"

ARMS = ["native", "union_s0", "union_s05", "union_s10"]
ARM_CN = {
    "native": "native\n(现工作流基线)",
    "union_s0": "Union scale=0\n(对照组)",
    "union_s05": "Union scale=0.5",
    "union_s10": "Union scale=1.0",
}
SAMPLES = [
    ("single", "hard_02"), ("single", "hard_05_torso"), ("single", "medium_02"),
    ("dual", "hug_pair00_084"), ("dual", "dance_pair02_120"), ("dual", "cross_backhug_pair02_037"),
]

matrix = defaultdict(dict)
for fn in sorted(os.listdir(os.path.join(ASSETS, A1, "images"))):
    kind, sample, arm, seed = fn[:-4].split("__")
    matrix[(kind, sample)][(arm, seed[:6])] = os.path.join(A1, "images", fn)

METRICS = [
    ("hard_02", "单人", "native", "0.750", "0.786", "0.066", "0.031", "0.170", "0.935", "1.00", "0.916", "157.9", "3.75"),
    ("hard_02", "单人", "union_s0", "0.750", "0.786", "0.066", "0.031", "0.170", "0.935", "1.00", "0.916", "241.0", "3.76"),
    ("hard_02", "单人", "union_s05", "0.857", "0.929", "0.071", "0.026", "0.247", "0.980", "1.00", "0.908", "236.1", "3.76"),
    ("hard_02", "单人", "union_s10", "0.964", "0.964", "0.054", "0.012", "0.028", "0.992", "1.00", "0.585", "236.1", "3.76"),
    ("hard_05_torso", "单人", "native", "0.389", "0.667", "0.378", "0.085", "0.320", "1.000", "1.00", "0.920", "156.8", "3.75"),
    ("hard_05_torso", "单人", "union_s0", "0.389", "0.667", "0.378", "0.085", "0.320", "1.000", "1.00", "0.920", "236.1", "3.76"),
    ("hard_05_torso", "单人", "union_s05", "0.389", "0.806", "0.326", "0.038", "0.319", "0.985", "1.00", "0.894", "236.2", "3.76"),
    ("hard_05_torso", "单人", "union_s10", "0.389", "0.944", "0.286", "0.028", "0.313", "0.989", "1.00", "0.577", "236.4", "3.76"),
    ("medium_02", "单人", "native", "0.844", "0.875", "0.101", "0.071", "0.096", "0.818", "1.00", "0.907", "158.6", "3.75"),
    ("medium_02", "单人", "union_s0", "0.844", "0.875", "0.101", "0.071", "0.096", "0.818", "1.00", "0.907", "239.6", "3.76"),
    ("medium_02", "单人", "union_s05", "0.906", "0.906", "0.091", "0.050", "0.130", "0.746", "1.00", "0.925", "239.7", "3.76"),
    ("medium_02", "单人", "union_s10", "0.938", "0.969", "0.070", "0.021", "0.034", "0.909", "1.00", "0.865", "239.3", "3.76"),
    ("hug_pair00_084", "双人", "native", "0.068", "0.131", "0.788", "—", "—", "—", "1.50", "0.700", "195.9", "3.89"),
    ("hug_pair00_084", "双人", "union_s0", "0.068", "0.131", "0.788", "—", "—", "—", "1.50", "0.700", "300.1", "4.70"),
    ("hug_pair00_084", "双人", "union_s05", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.711", "299.1", "4.70"),
    ("hug_pair00_084", "双人", "union_s10", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.685", "299.0", "4.70"),
    ("dance_pair02_120", "双人", "native", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.630", "196.6", "3.90"),
    ("dance_pair02_120", "双人", "union_s0", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.630", "299.2", "4.71"),
    ("dance_pair02_120", "双人", "union_s05", "0.269", "0.327", "0.307", "—", "—", "—", "1.50", "0.721", "298.5", "4.71"),
    ("dance_pair02_120", "双人", "union_s10", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.715", "299.3", "4.71"),
    ("cross_backhug_pair02_037", "双人", "native", "0.456", "0.548", "0.405", "—", "—", "—", "3.50", "0.839", "195.2", "3.82"),
    ("cross_backhug_pair02_037", "双人", "union_s0", "0.456", "0.548", "0.405", "—", "—", "—", "3.50", "0.839", "296.1", "4.68"),
    ("cross_backhug_pair02_037", "双人", "union_s05", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.664", "296.2", "4.68"),
    ("cross_backhug_pair02_037", "双人", "union_s10", "0.000", "0.000", "—", "—", "—", "—", "1.00", "0.708", "294.6", "4.68"),
]

INPUT_DETAIL = {
    "hard_02": ("single", "640×928", "workflow/runs/scale_anchor_20261013/hard_02",
                "inputs/A_reference.png", "geometry/B_3d_dwpose.png", "2021 字符"),
    "hard_05_torso": ("single", "640×928", "workflow/runs/test/hard_05_torso",
                      "inputs/A_reference.png", "geometry/B_3d_dwpose.png", "2041 字符"),
    "medium_02": ("single", "640×928", "workflow/runs/scale_anchor_20261013/medium_02",
                  "inputs/A_reference.png", "geometry/B_3d_dwpose.png", "3028 字符"),
    "hug_pair00_084": ("dual", "672×896", "workflow-multiple/runs/hug_pair00_084_c4_r000",
                       "inputs/A_reference.png + B_reference.png", "geometry/skeleton_both.png", "3065 字符"),
    "dance_pair02_120": ("dual", "672×896", "workflow-multiple/runs/dance_pair02_120_c4_r459.skeleton_only.b_canvas",
                         "inputs/A_reference.png + B_reference.png", "geometry/skeleton_both.png", "3105 字符"),
    "cross_backhug_pair02_037": ("dual", "672×896", "workflow-multiple/runs/cross_backhug_pair02_037_r545_from_pair00.skeleton_only.b_canvas",
                                 "inputs/A_reference.png + B_reference.png", "geometry/skeleton_both.png", "2678 字符"),
}

H05 = [
    ("T1", "Control Only", "Native（B 原生布局）", "0.6667", "0.6667", "0.1935", "0.0238", "0.1791", "7.98", "7.91", "114.09"),
    ("T2a", "Control Only", "source_person（全身框对齐）", "0.8889", "0.8889", "0.1127", "0.5000", "0.0127", "8.03", "8.02", "110.38"),
    ("T2b", "Control Only", "source_torso（躯干尺度对齐）", "0.6667", "0.6667", "0.1847", "0.0000", "0.1863", "15.28", "15.27", "110.62"),
    ("T3", "Inpaint + Control", "Native（B 原生布局）", "0.3889", "0.3889", "0.3058", "0.2857", "0.2443", "30.79", "31.16", "110.90"),
    ("T4a", "Inpaint + Control", "source_person（全身框对齐）", "0.0556", "0.0556", "1.2671", "0.0000", "0.2066", "30.93", "31.21", "111.22"),
    ("T4b", "Inpaint + Control", "source_torso（躯干尺度对齐）", "0.6111", "0.6111", "0.1941", "0.0000", "0.0746", "30.76", "30.76", "110.86"),
]

# =============================================================== 封面
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("\n\nQwen-Image-2.1-Fun-ControlNet-Union")
r.bold = True
r.font.size = Pt(30)
r.font.name = "等线"
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("多条件控制网络迁移 · 适配 · 训练 全量测试实验汇报")
r.bold = True
r.font.size = Pt(18)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("\n实验族：experiments/6.qwen-controlnet union\n数据截至 2026-10-27 · 全部指标取自服务器实测产物")
r.font.size = Pt(12)
r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
PAGEBREAK()

# =============================================================== 0 一页纸
H(1, "〇、汇报要点（一页纸）")
P("本轮围绕阿里 PAI / VideoX-Fun 开源的 Qwen-Image-2.1-Fun-Controlnet-Union（16 个控制块、129 通道、7.1 GB 单权重）"
  "共完成 4 条实验线。核心问题是：官方 Union 控制分支能否在「人物动作迁移」这个任务上，同时做到姿态跟得住、"
  "身份保得住、双人分得开。", bold=False)
TABLE(
    ["#", "实验线", "落点（服务器）", "规模", "一句话结论"],
    [
        ("一", "官方 Union 原生迁移 + 三关验证（方案 A）",
         "experiments/6.qwen-controlnet union/qwen21_union_migration_v1",
         "48 张出图矩阵 + 20 张官方复现/探针图",
         "控制通道在单人上确实有效（PCK@0.2 最高 0.750→0.964），代价是满强度时身份分崩、骨架线漏进画面、双人两人并成一人。判定：需要进一步输入适配。"),
        ("二", "hard_05 单人「3 布局 × 2 模式」六臂对比",
         "workflow/experiment/qwen-union",
         "6 臂 × 1 张 = 6 张出图 + 13 张条件图",
         "Union 把骨架跟随率从 Native 的 0.3333 拉到 0.6111~0.8889；但身份保留（ArcFace 0.07~0.24）远逊于 Native 多图注入（0.8259）。"),
        ("三", "多数据集 LoRA 训练适配（方案 B）",
         "experiments/6.qwen-controlnet union/qwen21_union_dataset_adaptation_v1",
         "1536/72/128 数据锁定 + 4 张 smoke 出图 + 8 张数据 QA 面板",
         "数据、契约、确定性三套门禁全绿（117 项测试、恢复训练逐位差 0），但 overfit200 只降 17.79%（门槛 20%）→ 门禁未过，正式 512 训练与 1024 张 test 未启动。"),
        ("四", "Pro Adapter 人物任务分支",
         "experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1",
         "22 张出图（step0/step200 配对、val 对照、seed 消融、CFG 消融）",
         "200 步训练全局 MSE 仅 -5.67%（门槛 20%）；25 步分支消融 -1.61%~-1.85%，全部远低于门槛。区域 loss 有降（contact -64.37%）但换不成全局收益。"),
    ],
    widths=[0.9, 4.6, 6.2, 4.2, 8.1],
)
P("")
P("总结论：官方 Union 的几何控制能力可复现、可接入、单人有效；但「基座带参考图 + 我们的密集骨架渲染」这一组合下，"
  "满强度会出现骨架漏线，双人会出现人物合并。当前判定停在「需要进一步输入适配」，训练线（方案 B / Pro Adapter）"
  "两轮均未过效果门禁。", bold=True)
PAGEBREAK()

# =============================================================== 1 公共条件
H(1, "一、公共资产与运行环境（四条实验线共用）")
P("四条实验线跑在同一套固定版本的权重、源码与环境上。下表所有值均取自服务器 versions.json / _checksums.json / run.json，"
  "不随时间漂移。")
TABLE(
    ["项", "值", "证据"],
    [
        ("基座模型", "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1（Qwen-Image-2.1 7B DiT，复用现有，不重复下载）", "各实验 README"),
        ("Union 控制权重", "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union\nrevision 8a4702014d4dabb5f896fcba917e2ee0a961465f\n单文件 7,550,979,904 字节（7.1 GB）", "models/*/_checksums.json"),
        ("权重 SHA256", "65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd", "同上"),
        ("控制结构", "16 个官方控制块 control_blocks.0..15 + control_img_in；180 个控制键全 BF16；129 通道（64 骨架/深度等 8 类结构条件 + 65 keep-mask/masked-latent）；注入层 [0,2,…,30]", "research/qwen21_controlnet_transfer_20261015"),
        ("上游源码", "VideoX-Fun commit 13bb3c625e8b1e457d7124d92207b49e00e4a6fe（浅取固定 commit，只读副本随实验目录留存）", "source/VideoX-Fun"),
        ("Python 环境", "qwen21 解释器 + --system-site-packages venv（未向 qwen21 环境塞任何包，遵守红线 7）；仅补 omegaconf、ftfy", "versions.json"),
        ("torch / diffusers / transformers", "2.8.0+cu128 / 0.41.0.dev0 / 5.17.0（装包前后逐字一致）", "versions.json"),
        ("GPU", "RTX 6000D 85GB（总容量 89,171,099,648 B），单卡串行 + 官方 group offload", "run.json / execution_summary.json"),
        ("nvidia-smi 状态", "返回码 18（Driver/library version mismatch，NVML 580.178）不可用；可行性与占卡一律以 torch.cuda.mem_get_info + /proc 句柄实测为准", "16_gpu_probe.py"),
        ("大产物落盘", "/data/shangguanrz/pic-edit-experiments/<实验名>/（/ 盘常年紧张）", "AGENTS.md 第 6 节"),
    ],
    widths=[4.2, 13.6, 6.2],
)
P("")
H(2, "1.1 为什么必须补一层「参考图输入适配」")
BULLETS([
    "官方 control pipeline 把参考图路径写死为 None（pipeline_qwenimage21_control.py:630：input_image_sizes, input_images, vae_images = [], None, None），"
    "官方示例无法证明源人物身份能被保持——这正是要补的那一层。",
    "QwenImage21ControlTransformer2DModel.from_pretrained 加载后会主动把 before_proj/after_proj 归零并 control_img_in.reset_parameters()"
    "（qwenimage21_transformer2d_control.py:191-198）。因此 Union 权重必须在 from_pretrained 之后再加载，且加载完要复查非零。",
    "控制条件在投影前 scatter 到图像 token 位（:288-290），长度必须等于 N_ref + N_target。官方只覆盖目标段；一旦引入参考图，"
    "就要在 129 维输入前面补 N_ref 行零。补零不等于该位置输出为零——control_img_in 带已训练 bias。",
    "KV cache 在控制路径上本来就不生效（:763 的条件含 control_context is None），全部实验显式关闭，不声称拿到缓存加速。",
])
PAGEBREAK()

# =============================================================== 2 实验一
H(1, "二、实验一：官方 Union 原生迁移与三关验证（方案 A）")
H(2, "2.1 实验目的")
P("把官方 Union 预训练控制分支原样搬到 sg 上跑通，然后只补一层「参考图输入」适配，验证不训练能否直接用于本项目的人物动作迁移。"
  "第一轮不引入 target-only 注入、不训 LoRA、不加新 Adapter，也不与现有控制模块混合。三关依次判定，前一关不过就先定位原因："
  "① 官方复现；② 接入契约；③ 效果比较。结论只允许三种：可直接用官方权重 / 需要进一步输入适配 / 需要另立训练实验。")

H(2, "2.2 实验条件")
TABLE(
    ["维度", "取值"],
    [
        ("运行名", "matrix_20261015T134829（48/48 完成，complete=true，生成失败 0 次）"),
        ("实验设计", "6 组样本 × 4 个臂 × 2 个 seed = 48 张"),
        ("四个臂", "native = 现工作流基线（不挂 Union）；union_s0 = 挂 Union 但 control_context_scale=0（对照组）；\n"
                  "union_s05 = scale 0.5；union_s10 = scale 1.0"),
        ("唯一变量", "只有 control_context_scale 不同。模型、提示词、seed、步数、画布、采样器、KV cache 设置全部一致。"),
        ("采样", "40 步；true_cfg_scale = 1.0；seed = {42, 43}；bfloat16；use_kv_cache = False"),
        ("画布", "单人 640×928；双人 672×896；全部 32 对齐"),
        ("精度/显存", "峰值 3.75–3.90 GiB（native）、4.68–4.71 GiB（双人接入版）；显存闸门最终定为 8 GiB（约 1.7 倍余量）"),
        ("单张耗时", "native 157–197 s；接入版单人 236–241 s；双人 295–300 s"),
        ("模型侧唯一改动", "scripts/reference_union_pipeline.py 的 ReferenceUnionPipeline（补参考图输入 + 控制 token 前缀补零）"),
        ("评测口径", "严格 PCK（漏检/未对齐记 0）+ 检测覆盖率；双人先按髋部做人物分配，近并列标 ambiguous；\n"
                    "身份分 = DINOv2 余弦；身份互换用 src.pose_edit.appearance.judge_appearance（差 > 0.02/人 判疑似互换）"),
        ("判定规则", "「最佳 Union 档相对 native 的严格 PCK@0.2」提高 > 0.02 记一个改善样本；身份分下降 > 0.05 记一个身份退化样本；\n"
                    "改善 ≥ 75% 且无身份退化 → 直接使用；有改善但未全覆盖 → 需要输入适配；全不改善 → 需要另立训练实验"),
    ],
    widths=[3.6, 20.4],
)

H(2, "2.3 第 1 关：官方复现（已通过）")
P("权重核验：checkpoints 里 180 个控制键、模型里 180 个控制参数全部 shape 相符；from_pretrained 先置零 34 个投影张量，"
  "Union 权重加载后 restored_after_union_load = 180、still_zero_after_load = []、shape/存在性不匹配 = []、逐位不同 = []；"
  "loader_missing_keys = 297 全是基座自有键，unexpected = 0。")
TABLE(
    ["控制类型", "条件", "实测", "判定"],
    [
        ("Pose", "同 seed 43、40 步、CFG 1，scale 0 → 1", "平均绝对差 33.81 / 255；变化像素占比 87.08%", "响应有效"),
        ("Depth", "同上", "平均绝对差 69.23 / 255；变化像素占比 90.85%", "响应有效"),
        ("Inpaint", "mask 白色占比 54.61%", "白区内平均变化 79.71 vs 黑区 2.25", "「白色 = 重绘」语义成立"),
    ],
    widths=[3.0, 7.0, 8.6, 5.4],
)
NOTE("提示词为按控制图内容自拟（prompts_are_official: false）——模型卡未随 demo 发布提示词，故本关只验「控制通道有效」，不声称复现卡面像素。")
P("")
FIG(os.path.join(A1, "runs_official_formal_20261015T133449", "panel_control_response_pose.png"),
    "图 2-1  Pose 控制响应面板：控制图 | 官方卡面参考 | scale 0 | scale 1（runs/official_formal_20261015T133449）", 22.0, 1500)
FIG(os.path.join(A1, "runs_official_formal_20261015T133449", "panel_control_response_depth.png"),
    "图 2-2  Depth 控制响应面板：控制图 | 官方卡面参考 | scale 0 | scale 1", 22.0, 1500)
PAGEBREAK()

P("下面 8 张为第 1 关的逐条件出图（40 步正式跑，seed 43、CFG 1.0）：")
GRID([
    (os.path.join(A1, "runs_official_formal_20261015T133449", "pose_scale0.0_seed43_40step.png"), "Pose · scale 0.0"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "pose_scale1.0_seed43_40step.png"), "Pose · scale 1.0"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "depth_scale0.0_seed43_40step.png"), "Depth · scale 0.0"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "depth_scale1.0_seed43_40step.png"), "Depth · scale 1.0"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "inpaint_scale1.0_seed43_40step.png"), "Inpaint · scale 1.0"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "official_reference_pose.png"), "官方卡面参考 · Pose"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "official_reference_depth.png"), "官方卡面参考 · Depth"),
    (os.path.join(A1, "runs_official_formal_20261015T133449", "official_reference_inpaint.png"), "官方卡面参考 · Inpaint"),
], cols=4, max_edge=860, fit_h_cm=7.0)
NOTE("同一次运行的 8 步低分辨率预检产物另存于 runs/official_smoke_20261015T131036，不作为效果证据。")
PAGEBREAK()

H(2, "2.4 第 2 关：接入契约 A/B/C/D（已通过）")
P("产物：runs/contract_gate_adapter_20261015T133049/contract_gate.json。对照量选「单步 noise_pred」而不是最终图——"
  "1 步采样会把 latent 打成 NaN（早期踩到的坑，steps 默认 2 并已复核有限），单步预测才是「接入层有没有改变数值路径」的直接证据。")
TABLE(
    ["检查", "做法", "实测结果"],
    [
        ("A", "不传参考图 vs 原版官方 wrapper，比单步 transformer 预测 noise_pred", "逐位相等（bitwise_equal: true），max/mean 绝对差 0.0，形状 (1,1056,64)，无 NaN/Inf"),
        ("B", "control_context_scale = 0 vs 完全不传控制图", "同样逐位相等"),
        ("C", "抓取真实入参", "控制条件 (1, 9148, 129) = 参考 8124 + 目标 1024 token、通道 129；img_shapes 三段覆盖 2 参考 + 1 目标；"
                                "参考段 Leading 行全零；输出画布 512×512 与请求一致"),
        ("D", "交换参考图顺序", "token 数不变，单步预测最大差 5.0625（输出确实随顺序改变）"),
    ],
    widths=[1.6, 8.4, 14.0],
)
NOTE("早一个时间戳的 contract_gate_adapter_20261015T132235 那次 gate_pass=false，是 1 步采样打出的 NaN，已改用 2 步 + 单步预测对照，不是接入层的问题。")

H(2, "2.5 实验输入：六组冻结配对（逐文件 SHA256 记在 input_manifest.json）")
TABLE(
    ["样本", "类型", "画布", "来源 run", "源图输入", "控制图输入", "冻结提示词长度"],
    [(k, v[0], v[1], v[2], v[3], v[4], v[5]) for k, v in INPUT_DETAIL.items()],
    widths=[3.4, 1.4, 1.9, 8.6, 3.4, 3.4, 3.9], font=8.0,
)
P("")
P("单人三组输入面板（source = 源人物照片；pose_ref = 姿态参考骨架；control = 送入 Union 的 129 通道骨架控制图）：", bold=True)
for s in ("hard_02", "hard_05_torso", "medium_02"):
    GRID([(os.path.join(A1, "panels", "%s_seed42.png" % s), "%s 输入面板（source | pose_ref | control）" % s)], cols=1, cell_cm=17.0, max_edge=1000)
P("")
P("双人三组输入（source_A / source_B 两位源人物；skeleton_both 合并骨架控制图；pose_reference_0 单人骨架）：", bold=True)
for s in ("hug_pair00_084", "dance_pair02_120", "cross_backhug_pair02_037"):
    items = []
    for fn in ("source_A.png", "source_B.png", "skeleton_both.png", "pose_reference_0.png"):
        items.append((os.path.join(A1, "input_dual_" + s, fn), "%s · %s" % (s, fn[:-4])))
    GRID(items, cols=4, header=["源人物 A", "源人物 B", "合并骨架（控制图）", "单人骨架参考"], max_edge=760, fit_h_cm=8.2)
P("")
P("双人三组的冻结输入面板（source_A | source_B | control 全景）：", bold=True)
for s in ("hug_pair00_084", "dance_pair02_120", "cross_backhug_pair02_037"):
    GRID([(os.path.join(A1, "panels", "%s_seed42.png" % s), "%s 输入面板" % s)], cols=1, cell_cm=19.0, max_edge=1000)
P("")
P("输入冻结后的画布预览（runs/inputs_preview，用于核对 32 对齐与 letterbox 补边）：", bold=True)
GRID([(os.path.join(A1, "runs_inputs_preview", f), "inputs_preview · %s" % f[:-4])
      for f in sorted(os.listdir(os.path.join(ASSETS, A1, "runs_inputs_preview")))], cols=4, max_edge=700)
PAGEBREAK()

H(2, "2.6 第 3 关：48 张出图矩阵（全部收录）")
P("下面每个样本一张 4 列 × 2 行的对照表：列 = 四个臂，行 = 两个 seed。"
  "union_s0 与 native 在全部 24 个 (样本, seed) 配对上逐字节相同（15_report.py 直接比 SHA256），"
  "所以 scale 0.5 / 1.0 上的任何差异都只来自 Union 控制通道，不可能混入 pipeline 切换的影响。")
for kind, s in SAMPLES:
    H(3, "2.6.%d  %s（%s）" % (SAMPLES.index((kind, s)) + 1, s, "单人" if kind == "single" else "双人"))
    items = []
    for seed in ("seed42", "seed43"):
        for arm in ARMS:
            items.append((matrix[(kind, s)][(arm, seed)], "%s · %s · %s" % (s, arm, seed)))
    GRID(items, cols=4, header=[ARM_CN[a] for a in ARMS], max_edge=820, fit_h_cm=6.0)
PAGEBREAK()

H(2, "2.7 量化指标总表（24 行，两 seed 均值）")
TABLE(
    ["样本", "类型", "臂", "PCK@0.2\nstrict", "PCK@0.5\nstrict", "MPJPE\n(归一)", "脸\nMPJPE", "手\nMPJPE", "检测\n覆盖", "检出\n人数", "身份分", "耗时\n(s)", "峰值\n(GiB)"],
    METRICS, widths=[3.5, 1.3, 1.9, 1.6, 1.6, 1.5, 1.4, 1.4, 1.5, 1.5, 1.5, 1.5, 1.6], font=8.0,
)
P("")
P("相对 native 的变化（判据：最佳 Union 档的严格 PCK@0.2）：", bold=True)
TABLE(
    ["样本", "native", "最佳 Union 档", "Δ", "身份分变化", "记为"],
    [
        ("hard_02", "0.750", "union_s10 0.964", "+0.214", "0.916 → 0.585（掉 0.332）", "改善 + 身份退化"),
        ("hard_05_torso", "0.389", "union_s0 0.389", "+0.000", "0.920 → 0.920", "持平（但 PCK@0.5 0.667→0.944、MPJPE 0.378→0.286、脸 0.085→0.028）"),
        ("medium_02", "0.844", "union_s10 0.938", "+0.094", "0.907 → 0.865", "改善"),
        ("hug_pair00_084", "0.068", "union_s0 0.068", "+0.000", "0.700 → 0.700", "持平（s05/s10 归零）"),
        ("dance_pair02_120", "0.000", "union_s05 0.269", "+0.269", "0.630 → 0.721", "改善"),
        ("cross_backhug_pair02_037", "0.456", "union_s0 0.456", "+0.000", "0.839 → 0.839", "持平（s05/s10 归零）"),
    ],
    widths=[4.4, 1.9, 3.4, 1.5, 4.6, 8.2], font=8.5,
)
NOTE("双人退化的直接原因是检出人数从 native 的 1.5 / 1.0 / 3.5 掉到 1.0——加了控制通道后两个人被合并成一个人，严格口径按定义记 0。退化发生在「人物可分性」这一层，不是关节角度层。")

H(2, "2.8 消融探针：骨架线漏进画面是不是「重复条件」造成的？")
P("原本怀疑是「同一张骨架既在 reference_images 里、又作为控制图」造成重复条件。21_probe_ref_ablation.py 只拿掉参考图里的骨架，"
  "其余变量（提示词、控制图、seed、步数、画布）全部不变。")
GRID([
    (os.path.join(A1, "probe_ablation", "single__hard_02__norefs_ctrl00__seed42.png"), "去掉骨架参考图 · control scale 0.0"),
    (os.path.join(A1, "probe_ablation", "single__hard_02__norefs_ctrl10__seed42.png"), "去掉骨架参考图 · control scale 1.0"),
], cols=2, cell_cm=11.0, max_edge=900)
P("结果：scale 1.0 下火柴人照样出现 → 排除「重复条件」这一解释。漏线是控制通道在「基座带参考图 + 我们这种密集脸/手点云骨架渲染」"
  "下满强度时的行为，与官方 demo 的稀疏火柴人不同，很可能被当成可绘制的纹理。")

H(2, "2.9 实验一结论")
P("判定：需要进一步输入适配——不是「可直接用官方权重」，也还不到「必须另立训练实验」。", bold=True)
BULLETS([
    "接入层本身是对的：24/24 个配对上 native 与 union_s0 逐字节相同，差异全部归因于控制通道。",
    "控制通道在单人上确实有效：严格 PCK@0.2 随强度单调上升（hard_02 0.750→0.857→0.964；medium_02 0.844→0.906→0.938）。",
    "代价出在外观与人物可分性上，不是出在姿态上：scale 1.0 时单人身份分集体下滑（0.916→0.585、0.920→0.577、0.907→0.865），"
    "面板里能直接看到白色骨架线被画进画面（scale 0.5 干净）。",
    "下一步优先级：① 强度上限取 0.5（单人 0.5 已是「姿态改善 + 身份不掉 + 无火柴人」）；② 换控制图渲染形式为官方那种稀疏骨架；"
    "③ 双人按人分别注入或保留身份线索。④ 以上做完仍拿不到收益，才转「另立训练实验」。",
])
NOTE("一处操作教训：首轮显存闸门设 24 GiB 过保守，而本机同卡另有 21 小时的 PE 重写任务占着 ~62 GiB，矩阵在 21.3 GiB 空闲时白等约 3 小时。"
     "实测本矩阵峰值 4.7 GiB，最终闸门定 8 GiB 并加了续跑（跳过 records.jsonl 里已完成组合）。闸门只影响调度，不影响结果。")
NOTE("一处指标口径的坑（已修）：单人基准必须用 npz 里的 keypoints_generation_px（画布坐标）。keypoints_B_px 是裁剪前坐标，"
     "画布 928 高时 y 能到 1184，用它比画布会把工作流自己的 final_canvas.png 也算成 PCK@0.2 = 0；换对之后同一张图是 0.923。")
PAGEBREAK()

# =============================================================== 3 实验二
H(1, "三、实验二：hard_05 单人「3 种骨架布局 × 2 种生成模式」六臂对比")
H(2, "3.1 实验目的")
P("在单人姿态编辑 workflow 上，用官方 Union 在标准测试样本 hard_05 做可复现的姿态控制对比，固定提示词、步数、seed、控制强度与画布，"
  "跑 3 × 2 = 6 组（T1, T2a, T2b, T3, T4a, T4b）。回答四个问题：官方 Union 能否准确跟随我们的骨架控制图？保留 B 原始空间布局 vs "
  "把 B 骨架对齐到 A 的位置哪种更好？引入 A 源图 + Inpainting Mask 能否保住身份/衣服/背景？source_person 与 source_torso 哪种对齐更适合 Union？")

H(2, "3.2 实验条件")
TABLE(
    ["维度", "取值"],
    [
        ("落点", "workflow/experiment/qwen-union（在「6.qwen-controlnet union」族内以 hard_05_single_experiment 符号链接暴露）"),
        ("样本", "hard_05，画布 640×928；来源 workflow/runs/test/hard_05 与 hard_05_torso"),
        ("模型", "基座 Qwen-Image-2.1 7B DiT + Qwen-Image-2.1-Fun-Controlnet-Union.safetensors（7.1 GB 官方单权重）"),
        ("固定超参", "Steps = 40；Seed = 42；true_cfg_scale = 1.0；control_context_scale = 1.0"),
        ("统一提示词", "PE-I2I 确定性语义编译并冻结，SHA256 = 47a60b3f9e620c8cdde6cd89f00ea6b315228fd13072ff97bad110dd24db9da7"),
        ("两个因子", "生成模式：control_only（不传 A 源图、不传 mask）/ inpaint_control（传 A 源图 + Inpainting Mask）；\n"
                    "骨架布局：native（B 原生等比 letterbox，scale 0.661）/ source_person（全身框对齐 A，scale 0.410）/ source_torso（躯干尺度对齐 A，scale 0.977）"),
        ("控制条件制作", "prepare_union_controls.py：SMPLest-X 关键点映射 + YOLO11x-seg 人体分割 + 骨段膨胀生成 mask"),
        ("Mask 覆盖率", "mask_native 30.1% / mask_source_person 27.0% / mask_source_torso 34.8%；底模 mask_A_person 21.7%"),
        ("评测", "官方 DWPose（yolox_l + dw-ll_ucoco_384）取 134 关节点；ArcFace（scrfd_10g + glintr100）取人脸特征；背景指标针对未涂抹区域计算"),
        ("单臂耗时", "110.38–114.09 s（Native Qwen 对照组约 45 s）"),
    ],
    widths=[3.6, 20.4],
)

H(2, "3.3 实验输入")
GRID([
    (os.path.join(A2, "workflow_source", "hard_05__A_reference.png"), "A · 源人物参考图（身份/衣服/场景来源）"),
    (os.path.join(A2, "workflow_source", "hard_05__B_rgb.png"), "B · 目标姿态照片"),
    (os.path.join(A2, "workflow_source", "hard_05__geometry_B_3d_dwpose.png"), "B · DWPose 骨架（几何中间产物）"),
    (os.path.join(A2, "workflow_source", "hard_05__generation__final_canvas.png"), "Native Qwen 基线产物 final_canvas.png"),
], cols=4, max_edge=820, fit_h_cm=8.2)
P("")
P("三种骨架布局的控制图与对应 Inpainting Mask（白 = 重绘，黑 = 保留）：", bold=True)
GRID([
    (os.path.join(A2, "controls", "pose_native.png"), "pose_native（scale 0.661）"),
    (os.path.join(A2, "controls", "pose_source_person.png"), "pose_source_person（scale 0.410）"),
    (os.path.join(A2, "controls", "pose_source_torso.png"), "pose_source_torso（scale 0.977）"),
    (os.path.join(A2, "controls", "mask_native.png"), "mask_native（30.1%）"),
    (os.path.join(A2, "controls", "mask_source_person.png"), "mask_source_person（27.0%）"),
    (os.path.join(A2, "controls", "mask_source_torso.png"), "mask_source_torso（34.8%）"),
], cols=3, max_edge=760, fit_h_cm=7.2)
P("")
P("Mask 叠加与目标区域检查图：", bold=True)
GRID([
    (os.path.join(A2, "controls", "overlay_mask_native.png"), "overlay · native"),
    (os.path.join(A2, "controls", "overlay_mask_source_person.png"), "overlay · source_person"),
    (os.path.join(A2, "controls", "overlay_mask_source_torso.png"), "overlay · source_torso"),
    (os.path.join(A2, "controls", "mask_A_person.png"), "mask_A_person 底模（21.7%）"),
    (os.path.join(A2, "controls", "target_region_native.png"), "target_region · native"),
    (os.path.join(A2, "controls", "target_region_source_person.png"), "target_region · source_person"),
    (os.path.join(A2, "controls", "target_region_source_torso.png"), "target_region · source_torso"),
], cols=3, max_edge=760, fit_h_cm=7.2)

H(2, "3.4 六臂出图结果")
FIG(os.path.join(A2, "comparison.png"),
    "图 3-1  九宫格总览：第一行 = 源 A / 目标 B / Native Qwen 基线；二、三行 = 六个 Union 臂（T1,T2a,T2b / T3,T4a,T4b）",
    17.5, 1500, fit_h_cm=13.4)
PAGEBREAK()
P("六臂单图（逐臂，640×928 原图）：", bold=True)
GRID([
    (os.path.join(A2, "output", "T1__final_image.png"), "T1 · Control Only · Native 布局"),
    (os.path.join(A2, "output", "T2a__final_image.png"), "T2a · Control Only · source_person"),
    (os.path.join(A2, "output", "T2b__final_image.png"), "T2b · Control Only · source_torso"),
    (os.path.join(A2, "output", "T3__final_image.png"), "T3 · Inpaint+Ctrl · Native 布局"),
    (os.path.join(A2, "output", "T4a__final_image.png"), "T4a · Inpaint+Ctrl · source_person"),
    (os.path.join(A2, "output", "T4b__final_image.png"), "T4b · Inpaint+Ctrl · source_torso"),
], cols=3, max_edge=860, fit_h_cm=7.2)
FIG(os.path.join(A2, "workflow_source", "hard_05_torso__generation__final_canvas.png"),
    "图 3-2  对照参考：hard_05_torso 的 Native Qwen 工作流产物（Body PCK@0.2 = 0.3333，ID Cosine = 0.8259）", 12.0, 1200)

H(2, "3.5 定量评测矩阵")
TABLE(
    ["组别", "模式", "骨架布局", "Body\nPCK@0.2", "Strict\nPCK", "Norm\nMPJPE", "Hand\nPCK", "ID\nCosine", "背景 PSNR\n(自身)", "背景 PSNR\n(公共)", "耗时(s)"],
    H05, widths=[1.5, 3.0, 4.6, 1.9, 1.7, 1.7, 1.6, 1.7, 2.0, 2.0, 1.7], font=8.5,
)
NOTE("Native Qwen 对照组（final_canvas.png）：Body PCK@0.2 = 0.3333，Strict PCK = 0.3333，Hand PCK = 0.3810，ID Cosine = 0.8259，耗时约 45 s。"
     "六组 DWPose 检出率均为 100%，人脸 Face_Detected 全部为 True。")

H(2, "3.6 实验二结论（逐问回答）")
BULLETS([
    "① 官方 Union 能否准确跟随骨架控制图？能。纯控制模式下 T2a Body PCK 达 0.8889，Inpainting 约束下 T4b 仍有 0.6111，"
    "显著高于 Native Qwen 的 0.3333。129 通道控制分支的零样本姿态几何塑造能力很强。",
    "② 保留 B 原生布局 vs 对齐到 A？对齐到 A 综合可用性更优。B 原生布局在 B 站位偏左或景别差异大时，Inpainting 必须跨大半个画布"
    "同时「消除原人 + 凭空造人」，重绘面积从 27% 涨到 30.1%~34.8%，背景破坏与重绘伪影严重。",
    "③ 引入 A 源图 + Mask 的效果？必须引入。不传 A 的 T1/T2 背景 PSNR 只有 7.98~15.28 dB（背景完全崩解为随机圣诞树）、"
    "身份相似度 0.01~0.19；引入 Inpainting 后（T3/T4）Mask 外部背景锁到 30.76~30.93 dB、SSIM 达 0.70，服装主色调与材质被继承。",
    "④ source_person vs source_torso？source_torso 更适合 Union。source_person 把全身 bbox 强压到 A 的画框，本例缩放比 0.410，"
    "在 Inpainting 下模型认不出躯干骨架，T4a 的 Body PCK 断崖跌到 0.0556、Norm MPJPE 飙到 1.2671；"
    "source_torso 锁躯干骨骼长度（scale 0.977），T4b 拿到 0.6111 与 0.1659 的低肢体比例漂移。",
    "⑤ 关键短板：官方 Union 用 VAE Inpainting 拼接而非深层人脸特征交叉注意力，面部细微身份保留（0.07~0.24）"
    "明显逊色于 Native Qwen 多图注入（0.8259）。这直接催生了「Native 文本/图像注意力 + Union 姿态控制分支」双轨融合架构的设想。",
])
PAGEBREAK()

# =============================================================== 4 实验三
H(1, "四、实验三：多数据集 LoRA 训练适配（方案 B）")
H(2, "4.1 实验目的")
P("检验官方预训练 Union 接入人物参考图后，控制分支 LoRA 能否改善单人、双人姿态控制；并在相同数据、公共初始化、采样与更新预算下，"
  "比较 joint（全序列更新）与 target-only（仅目标残差注入）两种策略。正式四组为 native、union_frozen、union_lora_joint、union_lora_target。")

H(2, "4.2 实验条件")
TABLE(
    ["维度", "取值"],
    [
        ("数据规模（真实）", "train 1536 / val 72 / test 128；overfit6 是 train 内每数据集两条（不额外增加训练样本）"),
        ("数据集与防泄漏分组", "iPER 512/24/32（appearance）；TikTok 512/24/32（strict_v1 source_video_group 再按 sequence 轮转）；\n"
                              "Hi4D 512/24/32（人物 pair，按动作族 / same-cross action / 实际接触数分层）；3DPW 0/0/32（sequence/pose_bin，仅 test 参与）"),
        ("抽样与去重", "抽样 seed 20261015；排序键 SHA256(\"union_b_v1:\" + sample_id)；按 canonical path 与解码 RGB 像素去重；\n"
                       "iPER/TikTok 同序列间隔 ≥ 8 帧，Hi4D 同 action/camera ≥ 2 帧，3DPW source/target ≥ 8 帧"),
        ("排除量", "38464 条重复目标对应的 source 行、2675 条时间近邻、247 条 3DPW 近帧配对（未降低间隔、未借用其他 split）"),
        ("数据 lock", "lock.json SHA256 = e61eef1e5bc4b2fb24593c28639e425280aeb85263329699e6d2d829d41fe503；\n"
                      "prepared registry SHA256 = 328bbb029a62946f97aa6a2386e379ef5610e6b4075bb27002a68603cff7d7b8（6880 个 prepared 资产：4608 张 RGB 条件图 + 2272 份关键点 JSON）"),
        ("条件预处理", "source、人物 Pose 参考图、combined control 全部准备为 512×512；RGB 补白、Pose 补黑、mask 用 nearest；\n"
                       "关键点用实际 resize 后 x/y 比例；已有 134 点不再次插 neck；Hi4D 用源 mask 中心决定参考图左右顺序"),
        ("原图尺寸", "iPER 1024×1024（Pose 512×512）；TikTok 604×1080；Hi4D 940×1280；本轮 3DPW 样本 1080×1920"),
        ("可训练部分", "基座与原始 Union 全冻结；只训精确 130 个 Linear 的 LoRA A/B，共 23,267,344 个 FP32 参数；rank = alpha = 16、dropout = 0"),
        ("优化器", "AdamW lr = 1e-4，32 updates warmup 后 constant；global gradient norm = 1；BF16 计算"),
        ("批次构造", "每次更新依次取 iPER/TikTok/Hi4D 各一条，micro-batch = 1、accumulation = 3；512 updates 正好各经过锁定的 512 条一次"),
        ("两臂唯一差异", "只改残差注入范围（joint vs target-only），共享初始化（initialization/lora_rank16_seed3407.pt）、sample/noise/timestep 轨迹"),
        ("VAE", "固定 posterior mode；缓存的目标 latent 只在训练监督路径读取"),
        ("确定性", "首次 CUDA 初始化前固定 CUBLAS_WORKSPACE_CONFIG=:4096:8、torch.use_deterministic_algorithms(True, warn_only=False)、\n"
                   "cudnn.benchmark=False、cudnn.deterministic=True；该策略计入缓存/训练/门禁签名"),
        ("正式推理", "512×512、40 steps、CFG = 1、Union control scale = 1、BF16、KV cache 关闭、seeds 42/43"),
        ("显存", "GPU smoke 峰值 allocated ≈ 27.88 GiB、reserved ≈ 29.13 GiB"),
    ],
    widths=[4.6, 19.4], font=8.5,
)

H(2, "4.3 门禁与工程结果")
TABLE(
    ["门禁", "要求", "实测", "结论"],
    [
        ("CPU 契约测试", "全部通过", "本地 116 passed / 1 skipped；服务器 117 passed", "通过"),
        ("GPU 确定性 smoke", "连续 4 updates 与保存恢复后的 2+2 updates，最终 LoRA 与 optimizer 最大绝对差 = 0", "smoke_deterministic_20261016_v2 差值均为 0", "通过"),
        ("首轮 smoke_20261016T1755", "同上", "最终 LoRA 最大差约 1.37e-5，未达 1e-6 门槛", "失败（证据保留，随后靠固定 CUBLAS_WORKSPACE_CONFIG 修复）"),
        ("overfit6 可学习性", "200 updates 后固定平均 loss 至少下降 20%", "0.1598302 → 0.1314015，仅下降 17.7868%", "gate_pass = false，未通过"),
        ("后续阶段", "完整缓存 / 两组 512 训练 / val12 五组诊断 60 张 / test128 四组双 seed 1024 张", "因上一门禁失败而未启动", "未执行"),
    ],
    widths=[3.6, 6.4, 6.6, 7.4], font=8.5,
)
P("")
P("真实 GPU smoke 的两张两步出图（iPER / Hi4D 各一张，只用于验证链路跑通，不构成效果结论）：", bold=True)
GRID([
    (os.path.join(A3, "runs_smoke_deterministic_20261016_v2", "iper_two_step.png"), "smoke_deterministic_20261016_v2 · iPER 两步"),
    (os.path.join(A3, "runs_smoke_deterministic_20261016_v2", "hi4d_two_step.png"), "smoke_deterministic_20261016_v2 · Hi4D 两步"),
    (os.path.join(A3, "runs_smoke_pilot_b_20261016T1841", "iper_two_step.png"), "smoke_pilot_b_20261016T1841 · iPER 两步"),
    (os.path.join(A3, "runs_smoke_pilot_b_20261016T1841", "hi4d_two_step.png"), "smoke_pilot_b_20261016T1841 · Hi4D 两步"),
], cols=4, max_edge=760, fit_h_cm=7.0)
NOTE("这两步图是链路验证产物，分辨率与步数都不等于正式推理（正式为 40 步、512×512）。方案 B 没有产出任何 40 步正式评测图。")

H(2, "4.4 数据 QA 面板（8 张，四个数据集各两张）")
P("面板右侧的目标 RGB 只用于 QA，不进入推理条件或提示词。summary.json 记录尺寸、偏移、人物编号与有效点数。")
for fn in sorted(os.listdir(os.path.join(ASSETS, A3, "data_qa"))):
    FIG(os.path.join(A3, "data_qa", fn), fn[:-4], 22.5, 1600)
H(2, "4.5 实验三结论")
BULLETS([
    "工程侧全部立住：数据四层零串味、lock/registry 双 SHA256 钉死、117 项契约测试、恢复训练逐位差 0。",
    "效果侧未立住：overfit200 只降 17.79%（门槛 20%），门禁按原样保留失败，没有放宽门槛也没有追加预算。",
    "因此方案 B 的正式四组对比（native / union_frozen / union_lora_joint / union_lora_target）从未产生可比出图，"
    "不能据此判断训练收益或注入范围优劣。这条线的结论是「可学习性门禁未过」，不是「训练更差」。",
])
PAGEBREAK()

# =============================================================== 5 实验四
H(1, "五、实验四：Pro Adapter 人物任务分支")
H(2, "5.1 实验目的")
P("在保留官方 Union 16 个原生控制块与 129 通道语义（reference / VLM / VAE 路径不变）的前提下，把人物几何、owner、visibility、"
  "接触、source 外观-FaceID bank 作为零输出任务分支接到 control stream，验证能否改善单/双人控制；并做分支消融与工程门禁。")

H(2, "5.2 实验条件")
TABLE(
    ["维度", "取值"],
    [
        ("架构", "hidden 4096 / width 512 / heads 8 / source_dim 1600 / stage_indices [0,4,8,12] / official_control_blocks 16 / control_lora false"),
        ("权重", "基座 /data/shangguanrz/pic-edit-models/Qwen-Image-2.1；Union 权重 SHA256 65d6b66d…21dcfcd，7,550,979,904 B，180/180 控制键恢复、逐值差 0"),
        ("可训练参数", "14,646,784（基座与原 Union 全冻结）"),
        ("训练", "画布 512×512、target grid 32×32、batch 1、accum 1、200 次更新；AdamW lr 1e-4、warmup 32、clip 1；seed init/train 3407、sampler 20261015"),
        ("推理", "统一 40 步、seed 43、CFG 1.0、use_kv_cache false；标注为 engineering_probe（quality_claim: false、engineering_only: true）"),
        ("显存/耗时", "峰值 reserved 30,463,229,952 B；主训练用时 2854.34 s；推理 budget 16–32 GiB + sync_group_offload"),
        ("输入", "locked overfit6（iPER/TikTok/Hi4D 各 2 对 = 6 对 8 人）；val12 条件；几何输入 NPZ 在 runs/geometry_cpu_20261018T113627 等三个 run"),
        ("代码签名", "1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e；冻结根 code_task7b_20261018T161055/"),
        ("日期", "2026-10-18 ~ 2026-10-21"),
    ],
    widths=[3.6, 20.4], font=8.5,
)
H(2, "5.3 出图结果")
P("训练集 6 条样本：step0（零分支初始化）vs step200 同条件对照。", bold=True)
step0 = {f.split("__")[0]: os.path.join(A4, "train6_step0", f) for f in os.listdir(os.path.join(ASSETS, A4, "train6_step0"))}
step200 = {f.split("__")[0]: os.path.join(A4, "train6_step200", f) for f in os.listdir(os.path.join(ASSETS, A4, "train6_step200"))}
items = []
for k in sorted(step0):
    items.append((step0[k], "step0 · %s" % k))
    items.append((step200[k], "step200 · %s" % k))
GRID(items, cols=4, max_edge=760)
NOTE("左列 step0、右列 step200 交替成对。这批图标注为 engineering_probe，只用于证明分支能出图、能改变输出，不用于宣称画质。")

P("验证集配对：同 seed、同条件，union0（零分支纯官方 Union 对照）vs main200。", bold=True)
v0 = {f.split("__")[0]: os.path.join(A4, "valpair_union0", f) for f in os.listdir(os.path.join(ASSETS, A4, "valpair_union0"))}
v2 = {f.split("__")[0]: os.path.join(A4, "valpair_main200", f) for f in os.listdir(os.path.join(ASSETS, A4, "valpair_main200"))}
items = []
for k in sorted(v0):
    items.append((v0[k], "union0 对照 · %s" % k))
    items.append((v2[k], "main200 · %s" % k))
GRID(items, cols=4, max_edge=760)

P("checkpoint_000025 的 seed 消融（seed 42 vs 43，同条件）：", bold=True)
items = []
for k in sorted(os.listdir(os.path.join(ASSETS, A4, "pilot_valpair_seed42"))):
    items.append((os.path.join(A4, "pilot_valpair_seed42", k), "seed42 · %s" % k.split("__")[0]))
    items.append((os.path.join(A4, "pilot_valpair_seed43", k.replace("coarse", "coarse")), "seed43 · %s" % k.split("__")[0]))
GRID(items, cols=4, max_edge=760)

P("CFG=4.0 消融（steps=2，face 与 paired_hands 两例，mask 外逐位不变）：", bold=True)
GRID([(os.path.join(A4, "inference_cfg4", f), "CFG 4.0 · %s" % f.split("__")[0]) for f in sorted(os.listdir(os.path.join(ASSETS, A4, "inference_cfg4")))],
     cols=4, max_edge=760)

H(2, "5.4 量化结果")
TABLE(
    ["指标", "step0", "step200", "变化", "门槛", "结论"],
    [
        ("全局 MSE", "0.15983020", "0.15076623", "-5.67%", "≥ 20%", "未通过"),
        ("global + 区域总 loss", "0.94809722", "0.71681113", "-24.39%", "—", "区域项主导下降"),
        ("face 区域", "—", "—", "-31.86%", "—", "—"),
        ("contact 区域", "—", "—", "-64.37%", "—", "—"),
        ("hand 区域", "—", "—", "-29.44%", "—", "—"),
        ("body 区域", "—", "—", "-15.75%", "—", "—"),
        ("分域 iPER / TikTok / Hi4D", "—", "—", "-1.99% / -6.65% / -9.22%", "—", "均未过门槛"),
    ],
    widths=[5.0, 3.0, 3.0, 5.4, 2.0, 5.6], font=8.5,
)
P("")
P("25 步分支消融（runs/pilot_summary_20261021T110000/report.zh-CN.md）：", bold=True)
TABLE(
    ["消融组", "注入层", "可训练参数", "loss 变化"],
    [
        ("full4", "[0,4,8,12]", "14,646,784", "-1.6512%"),
        ("no_dense", "去 dense 分支", "—", "-1.6121%"),
        ("no_appearance", "去 appearance 分支", "—", "-1.6854%"),
        ("no_contact", "去 contact 分支", "—", "-1.6785%"),
        ("full8", "[0,2,…,14]", "23,051,776", "-1.8491%"),
    ],
    widths=[4.0, 4.0, 4.0, 4.0], font=8.5,
)
H(2, "5.5 实验四结论")
BULLETS([
    "分支能出图、能改变输出，工程门禁（CPU 回归 550 passed / 9 skipped，pilot 工具 34 项）全绿。",
    "但效果门禁未过：200 步全局 MSE 只降 5.67%，五组 25 步消融全部落在 -1.6% ~ -1.8%，远低于 20% 门槛。"
    "区域 loss（contact -64.37%）确有下降，说明分支在学它被监督的东西，但换不成全局收益。",
    "本线只有 latent flow 误差，没有 PCK / MPJPE / ArcFace 指标；1536 正式训练、val96、test1024 均未执行。",
    "与方案 B 同构的教训：旧 B200 步 -17.7868% 同样失败，两轮独立设计都在 20% 可学习性门槛前止步。",
])
PAGEBREAK()

# =============================================================== 6 总结
H(1, "六、总结论与下一步")
TABLE(
    ["问题", "回答", "最强证据"],
    [
        ("官方 Union 的姿态控制能力可用吗？", "可用。零样本就能把骨架跟随率从 0.3333 提到 0.6111~0.8889。", "实验二 T2a/T4b 对照 Native 基线"),
        ("接入层有没有引入混淆变量？", "没有。native 与 union_s0 在 24/24 配对上逐字节相同。", "实验一 15_report.py 的 SHA256 直比"),
        ("满强度控制有代价吗？", "有，且代价不在姿态上：骨架线被画进画面、身份分从 0.92 掉到 0.58。", "实验一 union_s10 面板 + 消融探针"),
        ("双人能直接用吗？", "不能。加控制通道后两人被并成一人，检出人数从 1.5/1.0/3.5 掉到 1.0，严格 PCK 按定义归零。", "实验一双人三组"),
        ("训练能不能救？", "两轮独立尝试（方案 B LoRA、Pro Adapter 任务分支）都没过 20% 可学习性门槛。", "17.7868% 与 -5.67%"),
    ],
    widths=[5.4, 12.0, 6.6], font=8.5,
)
P("")
P("下一步（按优先级）：", bold=True)
BULLETS([
    "① 控制强度上限取 0.5：单人在 0.5 档已经是「姿态改善 + 身份不掉 + 无火柴人」，1.0 必须先解决漏线。",
    "② 换控制图渲染形式为官方那种稀疏骨架（去掉密集脸/手点，线宽与配色对齐官方素材）——成本最低、最可能直接消除漏线。",
    "③ 双人必须处理人物可分性：按人分别注入，或在控制图里保留可区分的身份线索。本轮明确没做参考区域屏蔽与残差门控，正是这里欠的。",
    "④ Mask 精细化：当前用骨段膨胀启发式，后续结合 SMPL-X 2D 重投影轮廓，进一步降低背景泄漏。",
    "⑤ 双轨融合：Native Qwen 的文本/图像注意力（负责身份）+ Union 姿态控制分支（负责几何），"
    "实验二已经量化出两者的能力边界（ID 0.8259 vs 0.07~0.24；PCK 0.3333 vs 0.6111）。",
    "⑥ 以上做完仍拿不到收益，才转「另立训练实验」——且训练前必须先把可学习性门槛在 overfit6 上过掉。",
])

H(2, "附录 A：产物落点索引（服务器）")
TABLE(
    ["实验线", "代码（受 Git 管理）", "大产物根"],
    [
        ("一 方案 A", "experiments/6.qwen-controlnet union/qwen21_union_migration_v1/", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/"),
        ("二 hard_05 六臂", "workflow/experiment/qwen-union/", "同目录 output/ 与 data/prepared_controls/"),
        ("三 方案 B", "experiments/6.qwen-controlnet union/qwen21_union_dataset_adaptation_v1/", "/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/"),
        ("四 Pro Adapter", "experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1/ + union-adapter/pro-adapter/", "/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/runs/"),
        ("架构审计", "research/union_adapter_architecture_20261017/、research/qwen21_controlnet_transfer_20261015/", "/data/shangguanrz/pic-edit-research/qwen21_controlnet_transfer_20261015_005429"),
    ],
    widths=[3.4, 10.6, 8.0], font=8.5,
)
H(2, "附录 B：第 1 关的 8 步低分辨率预检产物（runs/official_smoke_20261015T131036）")
P("这批图是正式 40 步跑之前的链路预检，只用于确认 Pose / Depth / Inpaint 三条控制通道都能出图，不作为效果证据。"
  "收录在此是为了让「第 1 关怎么过的」这条链路可追溯。")
GRID([(os.path.join(A1, "runs_official_smoke_20261015T131036", f), "8 步预检 · %s" % f[:-4])
      for f in sorted(os.listdir(os.path.join(ASSETS, A1, "runs_official_smoke_20261015T131036")))], cols=4, max_edge=760)
NOTE("本报告由上述目录只读复制并缩放生成，服务器原始图片未做任何修改、移动或删除。本报告共收录 %d 张图片。" % IMG_COUNT)

doc.save(OUT)
print("SAVED", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1e6))
