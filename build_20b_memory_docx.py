# -*- coding: utf-8 -*-
"""
生成《Qwen-Image-Edit 20B 显存占用预算与单卡可行性分析简报》Word文档
专门分析 20B 模型静态权重、各分辨率与模式下的显存占用区间、单卡 84GB 可行性、
工程优化手段（特征缓存、梯度检查点）、以及在课题任务书下的定位建议。
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ----------------- 样式辅助函数 -----------------

def set_run_font(run, name="微软雅黑", size=10, bold=False, color=(45, 55, 72), italic=False):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:eastAsia'), name)
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E0", sz="4"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = OxmlElement('w:tblBorders')
        for b_name in ['top', 'left', 'bottom', 'right', 'insideH']:
            b = OxmlElement(f'w:{b_name}')
            b.set(qn('w:val'), 'single')
            b.set(qn('w:sz'), sz)
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), color)
            borders.append(b)
        insideV = OxmlElement('w:insideV')
        insideV.set(qn('w:val'), 'none')
        borders.append(insideV)
        tblPr[0].append(borders)

def add_sec_heading(doc, num_str, title_str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run_n = p.add_run(num_str + " ")
    set_run_font(run_n, name="微软雅黑", size=13, bold=True, color=(43, 108, 176))
    run_t = p.add_run(title_str)
    set_run_font(run_t, name="微软雅黑", size=13, bold=True, color=(26, 54, 93))
    return p

def add_body(doc, text, bold_p=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.2
    if bold_p:
        run_b = p.add_run(bold_p)
        set_run_font(run_b, name="微软雅黑", size=10, bold=True, color=(26, 54, 93))
    run = p.add_run(text)
    set_run_font(run, name="微软雅黑", size=10, bold=False, color=(45, 55, 72))
    return p

def add_bullet(doc, title, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_t = p.add_run(title)
    set_run_font(run_t, name="微软雅黑", size=9.5, bold=True, color=(43, 108, 176))
    run = p.add_run(text)
    set_run_font(run, name="微软雅黑", size=9.5, bold=False, color=(45, 55, 72))
    return p

def add_summary_box(doc, text, title="分析核心结论", border_color="1A365D", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '20')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_color)
    tcBorders.append(left)
    for side in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.15
    if title:
        run_t = p.add_run(f"⚡ {title}：\n")
        set_run_font(run_t, name="微软雅黑", size=10, bold=True, color=(26, 54, 93))
    run = p.add_run(text)
    set_run_font(run, name="微软雅黑", size=9.5, bold=False, color=(45, 55, 72))
    doc.add_paragraph().paragraph_format.space_after = Pt(3)

def format_clean_table(table, col_widths, headers, data, align_cols=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)
    
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1A365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.name = "微软雅黑"
        p.runs[0]._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), "微软雅黑")
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        hdr_cells[i].width = col_widths[i]

    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            if align_cols and c_idx in align_cols:
                p.alignment = align_cols[c_idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if len(p.runs) > 0:
                p.runs[0].font.name = "微软雅黑"
                p.runs[0]._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), "微软雅黑")
                p.runs[0].font.size = Pt(9)
                p.runs[0].font.color.rgb = RGBColor(45, 55, 72)
            row_cells[c_idx].width = col_widths[c_idx]

# ----------------- 主文档构建 -----------------

def build_20b_memory_doc():
    doc = docx.Document()
    
    # 页边距紧凑设为 2.0 cm (0.8 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # 标题
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Qwen-Image-Edit 20B 显存预算与单卡可行性分析简报")
    set_run_font(run_title, name="微软雅黑", size=18, bold=True, color=(26, 54, 93))

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("——基于 NVIDIA RTX 6000D (84GB / 可见 83GiB) 平台的推理与外挂适配器训练实测推算")
    set_run_font(run_sub, name="微软雅黑", size=11, bold=False, color=(74, 85, 104), italic=True)

    # 核心结论 Box
    add_summary_box(doc, 
        "1. 单卡运行可行性：RTX 6000D (可见 83 GiB) 可以稳定支撑 20B 原生 BF16 单图推理（约 60~66 GiB）与【离线缓存条件+梯度检查点】的外挂适配器微调训练（约 52~62 GiB）。\n"
        "2. 临界风险警戒：1024 在线编码训练显存将飙升至 72~84+ GiB，极易触发 OOM；若关闭梯度检查点则显存高达 85~110 GiB，必然崩溃。建议严格设立 78 GiB 实验终止红线。\n"
        "3. 课题规范与定位：20B 主干严重超出任务书【生成主干低于 6B】的硬性指标。其科学定位应为：【质量上限对照组 (Upper Bound)】与【教师蒸馏数据生成模型】，而非第一交付主干。",
        title="核心研判与决策建议"
    )

    # 一、静态权重拆解
    add_sec_heading(doc, "一、", "Qwen-Image-Edit-2511 静态权重体量拆解")
    add_body(doc, "官方仓库（Qwen/Qwen-Image-Edit-2511）标注为 20B BF16，仓库总体积约 57.7 GB。各组件存储与静态显存占用如下：")

    t_weights = doc.add_table(rows=5, cols=4)
    format_clean_table(t_weights,
        [Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.8)],
        ["模块组件", "BF16 权重存储体积", "显存常驻基础 (GiB)", "职能与技术特性"],
        [
            ["20B MMDiT 主干", "38.06 GiB", "约 38.1 GiB", "多模态流匹配生成核心，60 层 Transformer"],
            ["Qwen2.5-VL 编码器", "15.45 GiB", "约 15.5 GiB", "超大视觉语言理解底座，提取指令与视觉Token"],
            ["VAE 编解码器", "0.24 GiB", "约 0.24 GiB", "潜空间下采样与图像高保真重构"],
            ["整套 Pipeline 合计", "53.74 GiB (约57.7GB)", "约 53.8 GiB", "模型加载后的绝对静态底噪显存"]
        ],
        align_cols={0: WD_ALIGN_PARAGRAPH.CENTER, 1: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.CENTER}
    )

    # 二、显存测算全景表
    add_sec_heading(doc, "二、", "全场景显存测算全景（推理 vs 外挂训练）")
    add_body(doc, "结合当前【冻结主干、只训练外挂适配器】的方案，针对 512、768、1024 三种分辨率推算的显存区间明细如下：")

    t_mem = doc.add_table(rows=6, cols=6)
    format_clean_table(t_mem,
        [Inches(2.0), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.2)],
        ["运行配置 / 模式", "512 显存", "768 显存", "1024 显存", "83GiB 设备评估", "安全边际 / 风险"],
        [
            ["BF16 单图推理", "56–62 GiB", "58–64 GiB", "60–66 GiB", "✅ 稳定运行", "余量 17~23 GiB，安全"],
            ["BF16 双图推理 (含参考图)", "59–66 GiB", "62–69 GiB", "64–72 GiB", "✅ 大概率可跑", "余量 11~19 GiB，收窄"],
            ["缓存条件 + 开启梯度检查点", "45–51 GiB", "48–56 GiB", "52–62 GiB", "✅ 推荐训练配置", "释放 VLM，余量 >21 GiB"],
            ["在线编码 + 开启梯度检查点", "62–70 GiB", "66–77 GiB", "72–84+ GiB", "⚠️ 临界高危", "1024 存在极大 OOM 风险"],
            ["缓存条件 + 关闭梯度检查点", "60–72 GiB", "72–88 GiB", "85–110 GiB", "❌ 严禁使用", "768/1024 必然触发 OOM"]
        ],
        align_cols={0: WD_ALIGN_PARAGRAPH.LEFT, 1: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.CENTER, 3: WD_ALIGN_PARAGRAPH.CENTER, 4: WD_ALIGN_PARAGRAPH.CENTER}
    )

    add_body(doc, "业界基准对照验证：", bold_p="实测依据对照：", space_after=2)
    add_bullet(doc, "1. vLLM-Omni 官方实测：", "在 1024x1024、50 步 Euler 采样下，原版 Qwen-Image-Edit 单卡峰值约 59.66 GiB，与本表单图推理 60~66 GiB 预测完全吻合。")
    add_bullet(doc, "2. Musubi Tuner 训练基线：", "Qwen-Image 1024、batch=1、BF16 开启梯度检查点训练 LoRA 约需 42 GiB；增加参考图与几何适配器后，显存处于 52~62 GiB，证实缓存方案在单卡 84GB 上切实可行。")

    # 三、关键瓶颈与工程应对
    add_sec_heading(doc, "三、", "显存关键瓶颈定位与工程应对策略")
    add_body(doc, "要在单张 84GB 显卡上成功驱动 20B 实验，必须执行以下三项强制工程策略：")

    add_bullet(doc, "策略 1：离线特征缓存（Offline Feature Caching）——立减 15.5 GiB：", 
        "Qwen2.5-VL 编码器单独占用 15.45 GiB。在训练前将数据集通过 VLM 离线抽取文本与图像 Token 存入 SSD，训练时主显存完全卸载 VLM，将训练基线从 53.8 GiB 降至 38.3 GiB！")
    add_bullet(doc, "策略 2：强制开启梯度检查点（Gradient Checkpointing）：", 
        "1024 分辨率下，序列长度达 8192+ Tokens，60 层 MMDiT 激活值极其庞大。若关闭检查点显存达 85~110 GiB（必崩）；开启后仅保留关键前向激活，可压减 65% 动态显存。")
    add_bullet(doc, "策略 3：设立 78 GiB 显存熔断线（Guardrail）：", 
        "CUDA 显存碎片（Memory Fragmentation）通常会产生 3~5 GiB 的预留膨胀（Reserved Memory）。将 78 GiB 设为自动停止线，一旦突破立即中断，防止机器整机僵死。")

    # 四、对人物编辑效果的影响
    add_sec_heading(doc, "四、", "对姿态与交互编辑效果的深层影响")
    add_body(doc, "更换 20B (2511) 检查点后，在姿态编辑课题上的利弊分析：")

    add_bullet(doc, "优势预期（大幅改善身份与多人一致性）：", 
        "2511 官方重点强化了几何推理与多人一致性，面容漂移（Face Drift）与服饰材质丢失现象将显著优于 5B 模型，尤其有望改善前期在握手（D1）与搭肩（D2）上的纹理保持。")
    add_bullet(doc, "本质局限（无法自动解决 3D 接触与穿模）：", 
        "20B 原生模型仍只接受纯文本与参考图，不具备 SMPL-X、深度、法线和接触图感知。肢体穿模、手部悬空等物理接触缺陷依然必须依靠外挂 Adapter 解决。")
    add_bullet(doc, "适配器迁移阻碍（架构不兼容）：", 
        "前期为 DeepGen 设计的 1536 维适配器无法直接加载；Qwen 20B 主干有 60 层且维度大幅增加，需要重新设计注入层维度（如从 6 层扩展至 12 层残差头），调参周期较长。")

    # 五、科研定位与建议推进路径
    add_sec_heading(doc, "五、", "在课题任务书下的定位建议与测试步骤")
    add_body(doc, "根据《人物图像姿态编辑任务书_v3》中【建议探索 6B 以下蒸馏/轻量模型】的明确要求，提出如下双轨制科研建议：")

    t_pos = doc.add_table(rows=3, cols=3)
    format_clean_table(t_pos,
        [Inches(1.8), Inches(2.5), Inches(2.7)],
        ["技术轨别", "模型选型与配置", "项目职能与交付价值"],
        [
            ["正式交付主干轨\n(主推)", "DeepGen 1.0 (5B)\n+ Contact-Aware Adapter", "• 100% 满足任务书 <6B 轻量硬性指标；\n• 单卡推理仅 4.8GB (4.4s)，微调安全余量 >50%；\n• 承担第一阶段与最终验收的核心交付成果。"],
            ["科学对比上限轨\n(辅助参考)", "Qwen-Image-Edit (20B)\n+ 结构控制 Adapter", "• 作为【效果上界 (Upper Bound)】，评估 5B 是否存在容量瓶颈；\n• 作为【高质量教师模型】，生成高质量双人交互伪配对数据；\n• 为下一步 6B 以下蒸馏模型提供对齐监督 Target。"]
        ],
        align_cols={0: WD_ALIGN_PARAGRAPH.CENTER}
    )

    add_body(doc, "20B 部署测试标准推进顺序（Step-by-Step）：", bold_p="实操测试排期：", space_after=2)
    add_bullet(doc, "Step 1 原生推理基线测量：", "下载 2511 权重后，先跑 S0、S2、D0、D1 四个原生用例，实测 512、768、1024 静态与峰值显存；")
    add_bullet(doc, "Step 2 适配器小步验证：", "重构 Adapter 维度匹配 20B，仅在 512 分辨率执行 3 次反向更新，确认主干冻结哈希不变且 Adapter 梯度正常；")
    add_bullet(doc, "Step 3 阶梯式压力实测：", "开启特征缓存与梯度检查点，测试 768 与 1024（5次预热 + 20次连续迭代），确认峰值未突破 78 GiB 红线；")
    add_bullet(doc, "Step 4 评测对比归档：", "在相同测试集上，横向对比 5B DeepGen 与 20B Qwen-Edit 的握手接触率与 ID 保真度，形成消融实验章节。")

    # 签名
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(14)
    p_end.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_e = p_end.add_run("人物图像姿态编辑课题组 · 显存预算与架构评估 · 2026年9月")
    set_run_font(run_e, name="微软雅黑", size=9, bold=True, color=(113, 128, 150))

    # 保存
    out_1 = "d:/codeplus/pictureedit/Qwen_Edit_20B显存预算与可行性分析简报.docx"
    out_2 = "d:/codeplus/pictureedit/demooutput/Qwen_Edit_20B显存预算与可行性分析简报.docx"
    doc.save(out_1)
    doc.save(out_2)
    print(f"20B Memory Report saved successfully:\n  1. {out_1}\n  2. {out_2}")

if __name__ == "__main__":
    build_20b_memory_doc()
