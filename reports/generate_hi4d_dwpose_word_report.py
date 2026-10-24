#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_hi4d_dwpose_word_report.py

Builds a professional, comprehensive Word (.docx) technical report for the
Hi4D 3D DWPose WholeBody Skeleton Upgrade Experiment.
"""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPORT_DIR = Path(r"D:\codeplus\pictureedit\reports")
ASSET_DIR = REPORT_DIR / "assets/hi4d_dwpose_report"
OUT_DOCX = REPORT_DIR / "Hi4D_3D_DWPose_WholeBody_全量升级实验报告.docx"


# --- XML Helper Functions for Styling ---

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_header(doc, text, level=1):
    h = doc.add_heading(level=level)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = "微软雅黑"
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    
    if level == 1:
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A) # Deep Navy Blue
    elif level == 2:
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB) # Royal Blue
    elif level == 3:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A) # Slate Dark
    return h

def add_p(doc, text, bold=False, italic=False, color=None, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    run = p.add_run(text)
    run.font.name = "微软雅黑"
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    run.font.size = Pt(10.5)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return p

def add_callout(doc, text_runs, title=None, border_color="2563EB", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    
    if title:
        run_title = p.add_run(f"【{title}】\n")
        run_title.font.name = "微软雅黑"
        run_title._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        run_title.font.size = Pt(10.5)
        run_title.font.bold = True
        run_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        
    for text, bold, italic, color in text_runs:
        r = p.add_run(text)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(10)
        r.font.bold = bold
        r.font.italic = italic
        if color:
            r.font.color.rgb = color
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(tbl, col_widths, headers, data_rows, highlight_col=-1):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="CBD5E1", sz="4", val="single")
    
    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_background(hdr_cells[i], "1E3A8A") # Navy Blue
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.runs[0]
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, row_data in enumerate(data_rows):
        row_cells = tbl.rows[r_idx + 1].cells
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            c_bg = "EFF6FF" if c_idx == highlight_col else bg
            set_cell_background(row_cells[c_idx], c_bg)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.runs[0]
            r.font.name = "微软雅黑"
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
            r.font.size = Pt(9)
            if c_idx == highlight_col:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)

    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)


# --- Build Document Content ---

def build_report():
    doc = Document()
    
    # Page setup: Standard A4 with 0.85 in margins
    for s in doc.sections:
        s.top_margin = Inches(0.85)
        s.bottom_margin = Inches(0.85)
        s.left_margin = Inches(0.85)
        s.right_margin = Inches(0.85)

    # Document Header / Banner Block
    banner_p = doc.add_paragraph()
    banner_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    banner_p.paragraph_format.space_before = Pt(12)
    banner_p.paragraph_format.space_after = Pt(2)
    r_sub = banner_p.add_run("多人姿态编辑与重定向前沿项目 · 技术专题与数据基建全量报告\n")
    r_sub.font.name = "微软雅黑"
    r_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(6)
    r_title = title_p.add_run("Hi4D 双人交互数据集 3D DWPose WholeBody\n骨架升级与全量批处理实验报告")
    r_title.font.name = "微软雅黑"
    r_title._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    r_title.font.size = Pt(21)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.paragraph_format.space_before = Pt(4)
    meta_p.paragraph_format.space_after = Pt(18)
    r_meta = meta_p.add_run("研制架构：Hi4D 数据工程与几何重定向流水线  |  执行环境：RTX 6000D 85GB\n全量规模：1,500 帧（30 动作序列 · 6,600 对交互样本）  |  报告归档日期：2026 年 10 月 24 日")
    r_meta.font.name = "微软雅黑"
    r_meta._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # Executive Summary Callout
    exec_runs = [
        ("【核心升级动因】：", True, False, RGBColor(0x1E, 0x3A, 0x8A)),
        ("针对 Hi4D Pilot 数据集历史遗留的「24 关节点 SMPL 红蓝简笔火柴人」存在的致命短板（手部关节点为 0、无手指、面部仅单个点无视线/朝向、线条非标），彻底将其升级为与单人 TikTok 数据集规范完全统一的 ", False, False, None),
        ("3D DWPose WholeBody 134 关键点标准高保真骨架", True, False, RGBColor(0x25, 0x63, 0xEB)),
        ("（OpenPose 彩虹肢体色谱 + 68 点面部点阵 + 每手 21 点五指骨链 + 3 点脚趾三角化）。\n", False, False, None),
        ("【全量工程成果】：", True, False, RGBColor(0x05, 0x96, 0x69)),
        ("通过自主研制的仓级批处理工具 render_hi4d_dwpose_1500.py，在 161.2 秒（~2.7 分钟，平均 9.31 FPS）内完成全量 1,500 帧（30 动作序列）的高保真渲染，达成 100.0% 成功率。共落盘 4,500 张 512×512 PNG（单人 A、单人 B、复合 Combined 各 1,500 张）与 3,000 份 (134, 3) 关键点 NPY，并自动同步产出了覆盖 6,600 交互对的全新训练/评测清单 pairs_1500_dwpose_{train,val,test}.jsonl。", False, False, None),
    ]
    add_callout(doc, exec_runs, title="全量升级核心摘要 (Executive Summary)", border_color="1E3A8A", bg_color="F1F5F9")

    # --- Chapter 1 ---
    add_header(doc, "一、升级背景与技术痛点剖析", level=1)
    add_p(doc, "在多人图像生成与姿态重定向主线（Qwen-Image-2.1 双人交互）推进过程中，高质量的几何引导条件（Pose Guidance）是引导生成模型理解人物空间相对位置、接触抓握意图以及面部对视互动的核心先验。然而，早期 Hi4D 数据集的骨架表征与当前先进视觉模型需求存在严重代差。")

    # User Query Comparison Showcase
    add_header(doc, "1.1 用户实际提问与视觉样式对比", level=2)
    add_p(doc, "在对提示词重写全量结果与下游生图骨架输入进行审查时，团队敏锐地指出现有 Hi4D 骨架过于简陋，并提供了与高质量 TikTok 骨架的对照截图：")

    user_img_path = ASSET_DIR / "user_query_comparison.png"
    if user_img_path.is_file():
        tbl_user = doc.add_table(rows=1, cols=1)
        tbl_user.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = tbl_user.cell(0, 0)
        set_cell_background(c, "F8FAFC")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(user_img_path), width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        r_c = p_cap.add_run("图 1-1：用户审查截图（左：Hi4D 原 24 点简笔骨架；右：TikTok 134 点 WholeBody 高保真骨架）")
        r_c.font.name = "微软雅黑"
        r_c._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r_c.font.size = Pt(9)
        r_c.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    add_header(doc, "1.2 旧版 24 关节点简笔骨架的三大核心缺陷", level=2)
    p_def1 = [
        ("【缺陷一：手部手指关节点完全缺失 (0/42)】：", True, False, RGBColor(0xDC, 0x26, 0x26)),
        ("双人互动的核心灵魂在于手部接触（拥抱时的环抱抓握、击掌时的指尖相碰、舞蹈时的扶腰与牵手）。原 24 点骨架在手腕处直接截断，手指关键点为 0，模型在生成时完全失去引导，极易出现错位假肢、多指并指、甚至两手融合的严重伪影。\n", False, False, None),
        ("【缺陷二：面部仅单个点，缺乏朝向与视线交互 (0/68)】：", True, False, RGBColor(0xDC, 0x26, 0x26)),
        ("头部仅一个中心点，无法判断人物是转头对视、低头拥抱还是背向交谈。缺乏 68 点面部轮廓点阵，导致生成人物经常出现五官朝向与身体躯干不符的诡异现象。\n", False, False, None),
        ("【缺陷三：非标准红蓝单色线，偏离预训练先验】：", True, False, RGBColor(0xDC, 0x26, 0x26)),
        ("单调的纯红（Person A）与纯蓝（Person B）线条并非主流 ControlNet 或 Qwen-Image-2.1 在亿级图像上预训练所见过的 OpenPose 彩虹肢体色谱，注意力响应显著偏弱。", False, False, None),
    ]
    add_callout(doc, p_def1, title="旧版骨架技术缺陷", border_color="DC2626", bg_color="FEF2F2")

    # --- Chapter 2 ---
    add_header(doc, "二、3D DWPose WholeBody 几何升级方案与对齐机制", level=1)
    add_p(doc, "为实现 Hi4D 骨架向全量 134 点 WholeBody 标准的高保真迁移，我们研制了生产级批处理流水线，兼顾几何严密性、极速推理与复杂遮挡下的百分之百找回率。")

    add_header(doc, "2.1 拓扑标准与视觉色彩规范", level=2)
    add_p(doc, "升级后的骨架严格采用 COCO-WholeBody 134 关键点标准拓扑：")
    add_p(doc, "• 肢体骨干 (Body 18 点)：采用标准 OpenPose 彩虹色谱（红、橙、黄、绿、青、蓝、紫、品红等 18 类肢体专用 RGB 色阶），线宽设为 3px，关节端点以 2px 半径圆形精准标记。")
    add_p(doc, "• 面部轮廓 (Face 68 点)：覆盖下颌轮廓线、左右眉弓、眼眶轮廓、鼻梁鼻翼与嘴唇内外圈，采用 255 纯白点阵绘制（半径为 1px 的超细点阵），精准传递三维头部偏转角度与视线朝向。")
    add_p(doc, "• 双手五指 (Hands 42 点)：每只手由手腕延伸出 5 根独立的彩色手指骨链（大拇指红、食指橙、中指绿、无名指天蓝、小指紫罗兰），每指 4 个关节（线宽 2px），细腻呈现屈伸、抓握与掌心朝向。")
    add_p(doc, "• 脚部支撑 (Feet 6 点)：连接大脚趾、小脚趾与脚后跟（金黄色三角化骨架），清晰标定站立支撑面。")

    add_header(doc, "2.2 两级检测网络与 Mask ROI 100% 挽救机制", level=2)
    add_p(doc, "在双人深度接触（如紧密深拥）场景下，两人身体紧密贴合甚至大面积互相遮挡，单阶段全局目标检测器极易将两人合并漏检一人。为此，我们设计了三层漏斗保障体系：")

    tbl_flow = doc.add_table(rows=4, cols=3)
    format_table(
        tbl_flow,
        col_widths=[1.5, 1.8, 3.2],
        headers=["流水线阶段", "处理逻辑与算法", "保障目标与实测表现"],
        data_rows=[
            ["Level 0：高速缓存重用", "直接加载历史验证通过的 native NPZ 坐标与置信度", "单帧处理耗时 < 0.005 秒，全量重用 943 帧，避免重复计算"],
            ["Level 1：全局 Wholebody", "YOLOX-L 目标检测 + RTMPose-384 关键点预测", "针对 529 帧未缓存样本全图检测，置信度匹配 Person Mask 自动归位"],
            ["Level 2：Mask ROI 挽救", "基于官方分割 Mask 提取外接矩形，外扩 30px 送入 ROI 检测", "全量 28 帧漏检样本 100% 找回，无一失败（0 Still Incomplete）"],
        ],
        highlight_col=2
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_header(doc, "2.3 绝对三维空间投影与仿射保真 (Affine Alignment)", level=2)
    add_p(doc, "关键点并非在 512×512 低分辨率图上粗暴重估，而是在 1280×940 原生高清相机空间提取后，通过各动作序列的实测仿射变换矩阵 A_original_to_512 实施无损几何变换：")
    add_p(doc, "   [x_512, y_512, 1]^T = A_original_to_512 · [x_native, y_native, 1]^T", bold=True, color=RGBColor(0x1E, 0x3A, 0x8A))
    add_p(doc, "这一数学保证确保了三维物理空间中的透视关系、身体比例、相机焦距畸变与实际目标 RGB 图像实现像素级绝对重合。")

    add_header(doc, "2.4 无偏最大通道复合算法 (Max Intensity Projection)", level=2)
    add_p(doc, "当 Person A 与 Person B 骨架合并为 Combined 骨架时，若采用简单的加权半透明混合（Alpha Blending），会导致交叉部位色彩变暗或产生混浊伪色。我们采用元素级最大值投影：")
    add_p(doc, "   Combined(x, y) = max( Guide_A(x, y), Guide_B(x, y) )", bold=True, color=RGBColor(0x1E, 0x3A, 0x8A))
    add_p(doc, "使得在黑色背景上，单人骨架各自的明亮彩虹色和面部白点均以最大纯度保留，交叠处自然呈现高亮边界。")

    # --- Chapter 3 ---
    add_header(doc, "三、全量批处理生产指标与数据统计", level=1)
    add_p(doc, "全量批处理在服务器环境顺利完成，处理吞吐率达到 9.31 FPS，完整落盘了包含图像、向量坐标与配套清单在内的完整工业级资产包。")

    add_header(doc, "3.1 生产性能与质量指标汇总", level=2)
    tbl_stat = doc.add_table(rows=8, cols=3)
    format_table(
        tbl_stat,
        col_widths=[2.0, 2.2, 2.3],
        headers=["核心指标项", "实测数值", "评测结论与质量分析"],
        data_rows=[
            ["全量覆盖目标帧数", "1,500 帧 (30 个动作序列)", "100.0% 完美覆盖，无任何缺失帧"],
            ["批处理总用时", "161.2 秒 (~2.68 分钟)", "极速工业级吞吐，平均 9.31 FPS"],
            ["Native 缓存重用帧数", "943 帧 (62.9%)", "校验坐标完整有效，即时读取"],
            ["全图检测命中帧数", "529 帧 (35.3%)", "双人检出且与 Mask 匹配度极高"],
            ["Mask ROI 挽救帧数", "28 帧 (1.8%)", "强遮挡帧 100% 成功找回"],
            ["Person A 平均有效关键点", "127.58 / 134 (95.21%)", "手、面、肢体高度完备，平均置信度 0.765"],
            ["Person B 平均有效关键点", "127.60 / 134 (95.22%)", "手、面、肢体高度完备，平均置信度 0.767"],
        ],
        highlight_col=1
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_header(doc, "3.2 资产落盘规格与清单对应表", level=2)
    add_p(doc, "遵循 AGENTS.md 保护性资产规范，新资产以独立的 pose_dwpose 命名落盘，与旧版 pose 完全隔离，并派生了对齐全量训练/验证/测试配对的清单文件：")

    tbl_manifest = doc.add_table(rows=5, cols=4)
    format_table(
        tbl_manifest,
        col_widths=[1.8, 1.2, 1.5, 2.0],
        headers=["清单文件名", "划分类别", "配对数量 (Pairs)", "骨架输入路径映射"],
        data_rows=[
            ["pairs_1500_dwpose_train.jsonl", "训练集 (Train)", "5,400 对 (18 序列)", "指向 pose_dwpose/{A,B,combined}/*.png"],
            ["pairs_1500_dwpose_val.jsonl", "验证集 (Val)", "600 对 (6 序列)", "身份互斥 pair22, pair28"],
            ["pairs_1500_dwpose_test.jsonl", "测试集 (Test)", "600 对 (6 序列)", "身份互斥 pair00, pair02"],
            ["全量总计 (Total)", "全量数据集", "6,600 对", "全套配套 (134, 3) 关键点 NPY"],
        ],
        highlight_col=2
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # --- Chapter 4 ---
    add_header(doc, "四、多场景典型动作对比与深入视觉质检", level=1)
    add_p(doc, "为客观评估 DWPose 骨架在下游生成模型输入端的几何表现力，我们抽样评测了具有代表性的三类动作：高贴合正面深拥、手部高频互动击掌、以及双人开阔动态舞蹈。所有案例均附带真实目标 RGB 图像、旧版 24 点简笔骨架、升级版 DWPose 骨架的三列并排高精度对照。")

    # Case 1: Close Hugging
    add_header(doc, "4.1 案例一：高贴合正面深拥交互 (pair00/hug00/000026)", level=2)
    add_p(doc, "【场景特征】：双人躯干极其贴近，两人双臂呈环抱状态，前臂与手掌搭在对方后背或腰部。")

    tbl_case1 = doc.add_table(rows=2, cols=3)
    tbl_case1.alignment = WD_TABLE_ALIGNMENT.CENTER
    case1_imgs = [
        ("hug00_000026_rgb.png", "真实目标 RGB 图像 (GT)\n(双人正面相拥，手部环抱于后腰)"),
        ("hug00_000026_oldpose.png", "旧版 24 点简笔骨架\n(手部无指截断，红蓝线单调交叠)"),
        ("hug00_000026_dwpose.png", "升级版 3D DWPose WholeBody\n(五指抓握清晰，68面部点对视)"),
    ]
    for idx, (fn, cap) in enumerate(case1_imgs):
        p_img = tbl_case1.rows[0].cells[idx].paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_file = ASSET_DIR / fn
        if img_file.is_file():
            p_img.add_run().add_picture(str(img_file), width=Inches(2.05))
        p_cap = tbl_case1.rows[1].cells[idx].paragraphs[0]
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        r = p_cap.add_run(cap)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        set_cell_background(tbl_case1.rows[0].cells[idx], "000000" if "pose" in fn else "FFFFFF")
        set_cell_background(tbl_case1.rows[1].cells[idx], "F1F5F9")
        set_cell_margins(tbl_case1.rows[0].cells[idx], top=40, bottom=40, left=30, right=30)
        set_cell_margins(tbl_case1.rows[1].cells[idx], top=40, bottom=60, left=40, right=40)
    set_table_borders(tbl_case1, color="CBD5E1", sz="4", val="single")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_p(doc, "【质检结论】：旧版骨架中，右侧人物伸向左侧的右手在小臂末端即停止，下游生成模型根本无法得知手掌是握拳、插兜还是抓着衣服；而新版 DWPose 清晰刻画出右侧人物右手张开五指按在对方后背的几何轮廓。此外，两人面部白色点阵精确反映出互相对视、头部微向下倾斜的拥抱情感特征。")

    # Case 2: High-Five
    add_header(doc, "4.2 案例二：手部高频互动与击掌 (pair12/highfive12/000011)", level=2)
    add_p(doc, "【场景特征】：双人抬手击掌，动作的核心完全取决于手掌形态（手指是否舒展、两手相对间距与高度差）。")

    tbl_case2 = doc.add_table(rows=2, cols=3)
    tbl_case2.alignment = WD_TABLE_ALIGNMENT.CENTER
    case2_imgs = [
        ("highfive12_000011_rgb.png", "真实目标 RGB 图像 (GT)\n(击掌前夕，双方抬手掌心相对)"),
        ("highfive12_000011_oldpose.png", "旧版 24 点简笔骨架\n(双臂仅画到手腕，手掌位置虚空)"),
        ("highfive12_000011_dwpose.png", "升级版 3D DWPose WholeBody\n(五指完整舒展，击掌掌面精确定位)"),
    ]
    for idx, (fn, cap) in enumerate(case2_imgs):
        p_img = tbl_case2.rows[0].cells[idx].paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_file = ASSET_DIR / fn
        if img_file.is_file():
            p_img.add_run().add_picture(str(img_file), width=Inches(2.05))
        p_cap = tbl_case2.rows[1].cells[idx].paragraphs[0]
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        r = p_cap.add_run(cap)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        set_cell_background(tbl_case2.rows[0].cells[idx], "000000" if "pose" in fn else "FFFFFF")
        set_cell_background(tbl_case2.rows[1].cells[idx], "F1F5F9")
        set_cell_margins(tbl_case2.rows[0].cells[idx], top=40, bottom=40, left=30, right=30)
        set_cell_margins(tbl_case2.rows[1].cells[idx], top=40, bottom=60, left=40, right=40)
    set_table_borders(tbl_case2, color="CBD5E1", sz="4", val="single")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_p(doc, "【质检结论】：在击掌动作中，若无手指关节，生成模型极易生成‘两根肉棍相碰’或者掌心方向相反的离奇画面。DWPose 精准捕捉到了手腕与五根彩色手指的展开形态，给下游生成模型提供了极其强悍的手势空间约束。")

    # Case 3: Dance Stance
    add_header(doc, "4.3 案例三：动态姿态舞蹈对立 (pair10/dance10/000001)", level=2)
    add_p(doc, "【场景特征】：双人舞蹈起始站位，展现两人身高落差、重心站姿与脚掌三点支撑。")

    tbl_case3 = doc.add_table(rows=2, cols=3)
    tbl_case3.alignment = WD_TABLE_ALIGNMENT.CENTER
    case3_imgs = [
        ("dance10_000001_rgb.png", "真实目标 RGB 图像 (GT)\n(两人立姿对视，重心略微前移)"),
        ("dance10_000001_oldpose.png", "旧版 24 点简笔骨架\n(双脚线条简单，身姿轮廓机械)"),
        ("dance10_000001_dwpose.png", "升级版 3D DWPose WholeBody\n(脚底三点稳固支撑，身形自然舒展)"),
    ]
    for idx, (fn, cap) in enumerate(case3_imgs):
        p_img = tbl_case3.rows[0].cells[idx].paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_file = ASSET_DIR / fn
        if img_file.is_file():
            p_img.add_run().add_picture(str(img_file), width=Inches(2.05))
        p_cap = tbl_case3.rows[1].cells[idx].paragraphs[0]
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        r = p_cap.add_run(cap)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        set_cell_background(tbl_case3.rows[0].cells[idx], "000000" if "pose" in fn else "FFFFFF")
        set_cell_background(tbl_case3.rows[1].cells[idx], "F1F5F9")
        set_cell_margins(tbl_case3.rows[0].cells[idx], top=40, bottom=40, left=30, right=30)
        set_cell_margins(tbl_case3.rows[1].cells[idx], top=40, bottom=60, left=40, right=40)
    set_table_borders(tbl_case3, color="CBD5E1", sz="4", val="single")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Single-Person Decomposition
    add_header(doc, "4.4 单人解耦骨架独立性验证", level=2)
    add_p(doc, "在双人生成网络中，除了复合骨架（Combined），还需要分别将 Person A 与 Person B 的独立骨架输入适配器（Adapter），以实现单人身份与姿态的清晰解耦绑定。以下展示了同一帧下 Person A 单人、Person B 单人与 Combined 复合图的严密解耦效果：")

    tbl_decomp = doc.add_table(rows=2, cols=3)
    tbl_decomp.alignment = WD_TABLE_ALIGNMENT.CENTER
    decomp_imgs = [
        ("hug00_000026_dwpose_A.png", "Person A 独立骨架 (pose_dwpose/A/)\n(仅保留左侧男性全部 134 关键点)"),
        ("hug00_000026_dwpose_B.png", "Person B 独立骨架 (pose_dwpose/B/)\n(仅保留右侧女性全部 134 关键点)"),
        ("hug00_000026_dwpose.png", "双人复合骨架 (pose_dwpose/combined/)\n(无损无偏 Max Blend 合成)"),
    ]
    for idx, (fn, cap) in enumerate(decomp_imgs):
        p_img = tbl_decomp.rows[0].cells[idx].paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_file = ASSET_DIR / fn
        if img_file.is_file():
            p_img.add_run().add_picture(str(img_file), width=Inches(2.05))
        p_cap = tbl_decomp.rows[1].cells[idx].paragraphs[0]
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        r = p_cap.add_run(cap)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        set_cell_background(tbl_decomp.rows[0].cells[idx], "000000")
        set_cell_background(tbl_decomp.rows[1].cells[idx], "F1F5F9")
        set_cell_margins(tbl_decomp.rows[0].cells[idx], top=40, bottom=40, left=30, right=30)
        set_cell_margins(tbl_decomp.rows[1].cells[idx], top=40, bottom=60, left=40, right=40)
    set_table_borders(tbl_decomp, color="CBD5E1", sz="4", val="single")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # --- Chapter 5 ---
    add_header(doc, "五、工程防线与合规守则落实", level=1)
    add_p(doc, "在整个数据重构与全量生产过程中，项目组严格遵循仓内最高合规红线与开发守则：")

    add_p(doc, "1. 保护性资产红线（图片/视频永不删移）：", bold=True)
    add_p(doc, "   严格恪守 AGENTS.md 准则，没有对原 processed_512_1500/*/pose 目录下的任何既有图片或文件执行任何覆盖、清理或删除操作。新资产以平行的 pose_dwpose 独立目录存放，确保历史代码分支和基线对比实验 100% 具备回溯与复现能力。")

    add_p(doc, "2. 几何投影严格单向传递：", bold=True)
    add_p(doc, "   坚决杜绝在 512 画布上独立对两人进行局部缩放或重新排布，严格复用序列标定的 A_original_to_512 仿射参数，使得两人在真实空间中的高矮胖瘦、重心倾斜、搂抱间距完好无损地传递至渲染画布。")

    add_p(doc, "3. 仓级工具归档与零污染提交：", bold=True)
    add_p(doc, "   全量批处理代码规范归置于仓级统一入口 scripts/data/render_hi4d_dwpose_1500.py，所有临时探针脚本（test_*.py）在完成验证后立即清理，保持代码仓库整洁纯粹。")

    # --- Chapter 6 ---
    add_header(doc, "六、对下游训练/生图任务赋能与后续推进建议", level=1)
    
    downstream_runs = [
        ("【1. 与 Q4 高保真重写提示词形成完美多模态闭环】：", True, False, RGBColor(0x1E, 0x3A, 0x8A)),
        ("本实验生成的 6,600 对 DWPose 骨架资产与上一阶段已完成的「Q4 5并发重写提示词」形成了完美的输入对齐：\n", False, False, None),
        ("   • <image1>：原始源图（提供双人初始服装与外貌身份特征）\n", False, False, None),
        ("   • <image2>：复合双人 DWPose 骨架（提供全局交互动作与空间相对距离）\n", False, False, None),
        ("   • <image3>：Person A 单人 DWPose 骨架（提供单人精确手指与面部动作）\n", False, False, None),
        ("   • <image4>：Person B 单人 DWPose 骨架（提供伴侣精确手指与面部动作）\n", False, False, None),
        ("   • Prompt 文本：详述双方互动细节，且已在清洗阶段彻底剔除 <image5> 泄露风险。\n\n", False, False, None),
        ("【2. 极大抑制生成模型‘假肢与手部融叠’两大痛点】：", True, False, RGBColor(0x05, 0x96, 0x69)),
        ("在 Qwen-Image-2.1 双人生成中，引入 134 点 WholeBody 骨架将大幅增强模型对肢体交叉边缘的注意机制，预计将带来 PCK-hand 与关键交互部位 LPIPS 指标的显著提升。\n\n", False, False, None),
        ("【3. 后续工作排期建议】：", True, False, RGBColor(0x25, 0x63, 0xEB)),
        ("建议立刻基于全新的 pairs_1500_dwpose_train.jsonl 启动 Qwen-Image-2.1 双人 ControlNet/Adapter 训练，并在 pairs_1500_dwpose_test.jsonl 上进行定量消融评测，直接量化相比于旧版 24 关节点简笔骨架的生成质量提升幅值。", False, False, None),
    ]
    add_callout(doc, downstream_runs, title="下游任务赋能建议", border_color="2563EB", bg_color="F0F4F8")

    # Save Document
    doc.save(str(OUT_DOCX))
    print(f"Report generated successfully at: {OUT_DOCX}")
    print(f"File size: {OUT_DOCX.stat().st_size:,} bytes")


if __name__ == "__main__":
    build_report()
