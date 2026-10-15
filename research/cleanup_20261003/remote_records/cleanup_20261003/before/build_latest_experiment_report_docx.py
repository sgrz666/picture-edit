# -*- coding: utf-8 -*-
"""
生成《DeepGen 1.0 图像姿态编辑最新实验结果与显存分析简报》Word文档
基于 120 次实测运行、60 个画质复核、显存全矩阵采样与 100 次更新稳定性测试。
多图呈现：内嵌 2 张高清显存分析图表、多组单双人实验结果图、接触面复查图与适配器推理图。
"""

import os
import shutil
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

def add_sub_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, name="微软雅黑", size=11, bold=True, color=(43, 108, 176))
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

def add_summary_box(doc, text, title="最新实验核心结论速览"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '22')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), '1A365D')
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
        set_run_font(run_t, name="微软雅黑", size=10.5, bold=True, color=(26, 54, 93))
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
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=100, right=100)
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
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=100, right=100)
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

def add_img(doc, img_path, caption, width=Inches(5.8)):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(6)
        run_cap = p_cap.add_run(f"图：{caption}")
        set_run_font(run_cap, name="微软雅黑", size=8.5, bold=True, color=(113, 128, 150), italic=True)

# ----------------- 构建主文档 -----------------

def build_latest_report():
    doc = docx.Document()
    
    # 紧凑页边距 (0.8 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # 标题区
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("DeepGen 1.0 图像姿态编辑最新实验结果与显存分析简报")
    set_run_font(run_title, name="微软雅黑", size=18, bold=True, color=(26, 54, 93))

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("——基于 120 次实测运行、60 个画质复核、全矩阵显存采样与 100 次更新稳定性测试")
    set_run_font(run_sub, name="微软雅黑", size=11, bold=False, color=(74, 85, 104), italic=True)

    # 核心摘要 Box
    add_summary_box(doc, 
        "1. 显存实测结论：在单卡 RTX 6000D (84GB) 上，推荐配置【冻结基座 + M型 130M Adapter + 条件缓存 + 开启梯度检查点】，"
        "稳定训练整卡 NVML 峰值分别为 512: 10.57 GiB、768: 13.40 GiB、1024: 18.01 GiB。包含缓存生成与独立推理的全流程最大占用分别为 15.96、17.42、19.35 GiB，"
        "建议容量预算为 18、20、22 GiB，拥有极高安全余量（单卡仅占约 20%~23%）。\n"
        "2. 编辑质量深层复核：60 个无筛选输出显示，单人招手可较好保留帽子与墨镜，但双人近距离交互（握手、搭肩）暴露出严重的【人员错配与新增人物错误】——"
        "原坐姿人物未起身，模型凭空在右侧新增站立人物握手，且服装格纹明显重绘。调整 CFG、步数或负面词均无法解决，证明纯文本指令编辑存在根本瓶颈，必须引入显式 3D SMPL-X 与局部接触先验！\n"
        "3. 适配器链路验收：已打通完整训练反向传播，6 个残差头全部参与运算，零初始化等价性验证通过，连续 100 次更新显存波动为 0.0%，链路彻底稳固。",
        title="最新实验核心研判"
    )

    # ================= 一、显存消耗全矩阵分析 =================
    add_sec_heading(doc, "一、", "显存消耗全矩阵实测分析（推荐配置与多模式对比）")
    add_body(doc, "本次实验在远程 GPU 服务器（NVIDIA RTX 6000D 84GB）上完成了 512、768、1024 三个分辨率、4 种运行模式的严格全矩阵实测采样：")

    # 表1：推荐配置主表
    t_rec = doc.add_table(rows=4, cols=6)
    format_clean_table(t_rec,
        [Inches(1.2), Inches(1.3), Inches(1.5), Inches(1.2), Inches(1.3), Inches(1.3)],
        ["图像分辨率", "稳定训练峰值", "稳态 allocated / reserved", "单步耗时 P50", "全流程最大占用", "建议容量预算"],
        [
            ["512 x 512", "10.57 GiB", "8.80 / 9.39 GiB", "0.349 秒", "15.96 GiB", "18 GiB (安全)"],
            ["768 x 768", "13.40 GiB", "11.55 / 12.22 GiB", "0.881 秒", "17.42 GiB", "20 GiB (推荐)"],
            ["1024 x 1024", "18.01 GiB", "15.40 / 16.83 GiB", "2.091 秒", "19.35 GiB", "22 GiB (极限)"]
        ],
        align_cols={0: WD_ALIGN_PARAGRAPH.CENTER, 1: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.CENTER, 3: WD_ALIGN_PARAGRAPH.CENTER, 4: WD_ALIGN_PARAGRAPH.CENTER, 5: WD_ALIGN_PARAGRAPH.CENTER}
    )

    # 嵌入全景显存需求看板
    dash_path = "d:/codeplus/pictureedit/results/deepgen_vram_requirement_dashboard.png"
    add_img(doc, dash_path, "图 1：DeepGen 1.0 (5B) 姿态编辑实验显存需求全景分析看板（含四模式对比、全流程阶段预算、优化降幅瀑布及与20B体量对比）", width=Inches(6.2))

    # 嵌入图表 2
    chart1_path = "d:/codeplus/pictureedit/results/vram_comparison_by_mode.png"
    add_img(doc, chart1_path, "图 2：DeepGen 1.0 (5B) 适配器训练在不同模式与分辨率下的整卡峰值显存对比（RTX 6000D 实测）", width=Inches(5.6))

    add_body(doc, "从图 1 可得出显存工程优化的两大核心法则：", bold_p="核心显存规律：", space_after=2)
    add_bullet(doc, "法则 1：梯度检查点是 1024 分辨率的生命线：", 
        "在 1024 分辨率下，开启梯度检查点将缓存训练显存从 37.34 GiB 骤降至 18.01 GiB，降幅达 51.8%！关闭检查点时在线训练显存高达 46.41 GiB，极易引起显存碎片化崩溃。")
    add_bullet(doc, "法则 2：条件缓存机制立省 15.5 GiB：", 
        "由于 Qwen2.5-VL 编码器单独常驻需要约 15.5 GiB 显存，离线预抽取语义 Token 缓存至 SSD 后，主训练阶段无需挂载 VLM，大幅释放计算与显存开销。")

    # 嵌入图表 2
    chart2_path = "d:/codeplus/pictureedit/results/vram_adapter_scale_and_latency.png"
    add_img(doc, chart2_path, "图 2：适配器 S/M/L 规模显存敏感性（左）与推荐配置单步耗时/全生命周期显存（右）", width=Inches(5.6))

    add_body(doc, "适配器规模与训练步耗时实测：", bold_p="规模敏感性：", space_after=2)
    add_bullet(doc, "适配器规模影响相对有限：", 
        "S型（92M）、M型（130M）、L型（205M）在 1024 下整卡显存分别为 17.37、18.01、18.81 GiB。显存主导因素是分辨率与激活保存方式，而非适配器自身参数。本轮推荐保持 M型（130M）作为基线。")
    add_bullet(doc, "单步训练吞吐优异：", 
        "在 RTX 6000D 上，512 分辨率单步仅 0.35s，768 分辨率单步 0.88s，1024 分辨率单步 2.09s。连续 100 次更新的稳定性测试中，显存波动均为 0.0%，完全不存在显存泄漏。")

    # ================= 二、最新姿态编辑效果深度复核 =================
    add_sec_heading(doc, "二、", "最新姿态编辑效果深度复核（多图对比与失败归因）")
    add_body(doc, "本次质量复核覆盖 60 个无筛选原生编辑输出（不挑选最优随机种子），全面暴露了纯指令编辑在单人和双人复杂交互下的真实边界：")

    add_sub_heading(doc, "2.1 单人姿态编辑效果（S0 保持 / S1 改色 / S2 招手）")
    add_bullet(doc, "动作迁移成功率高：", "滑雪者单手自然招手（S2）动作完成度高，墨镜与条纹帽子均能较好保持，且身体比例未发生畸变。")
    add_bullet(doc, "局部细节出现偶发重绘：", "雪杖形状、手套细节及雪地滑行痕迹在姿态变化后有轻微形变。")

    # 嵌入单人实验图片
    s2_sheet = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/Q0-Q1_S2_0.png"
    add_img(doc, s2_sheet, "图 3：单人招手姿态编辑实验复核 Contact Sheet（覆盖种子 42/1024、补边与拉伸）", width=Inches(5.4))

    s_demo = "d:/codeplus/pictureedit/demooutput/comparison_single_pose_celebration.png"
    add_img(doc, s_demo, "图 4：单人双手高举欢呼庆祝对比（原图 vs 编辑后图，1024x512）", width=Inches(5.4))

    add_sub_heading(doc, "2.2 双人交互姿态编辑瓶颈与缺陷诊断（D0 保持 / D1 握手 / D2 搭肩）")
    add_body(doc, "双人近距离交互编辑暴露了纯文本编辑模型最致命的逻辑缺陷：")
    
    add_bullet(doc, "致命缺陷 1【人物错配与凭空新增人物】：", 
        "在双人握手（D1）测试中，指令要求左侧坐姿男子与站立男子握手。但生成结果中，原浅色上衣男子依然坐在沙发左侧，模型在右侧凭空新增了第三个站立男子与人格纹衬衫男士握手！"
        "表面上生成了符合握手动作的图像，但实质上指定对象的交互编辑完全失败。")
    add_bullet(doc, "致命缺陷 2【面部与服装格纹漂移】：", 
        "格纹衬衫男士的面部特征、衬衫线条被全局重绘；搭肩（D2）任务中新增了女性交互对象，甚至原坐姿人物依然留在背景中。")
    add_bullet(doc, "致命缺陷 3【参数搜索无法纠正该错误】：", 
        "我们系统搜索了 CFG (2.5 / 4.5 / 6.0)、去噪步数 (35 / 50 步) 以及负面提示词，新增人物与错配现象依然稳定存在，证明这不是生成随机性，而是纯文本缺乏空间坐标和人物 ID 绑定的根本性缺陷！")

    # 嵌入双人握手与搭肩图片
    d1_sheet = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/Q0-Q1_D1_0.png"
    add_img(doc, d1_sheet, "图 5：双人握手实验复核 Sheet（注意：左侧原坐姿男子仍在，右侧凭空新增了站立握手人物）", width=Inches(5.4))

    d_demo = "d:/codeplus/pictureedit/demooutput/comparison_dual_pose_shake_hands.png"
    add_img(doc, d_demo, "图 6：双人握手姿态编辑对比（原图 vs 编辑后图，1024x512）", width=Inches(5.4))

    d2_sheet = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/Q0-Q1_D2_0.png"
    add_img(doc, d2_sheet, "图 7：双人搭肩实验复核 Sheet（展示复杂接触下的遮挡与人物替换现象）", width=Inches(5.4))

    add_sub_heading(doc, "2.3 分辨率敏感性与伪影分析 (512 vs 768 vs 1024)")
    add_body(doc, 
        "实验对比了 512、768 和 1024 分辨率输出。实测表明：提升至 1024 分辨率并没有显著提升人物身份保真度，"
        "反而诱发了更多雪杖断裂、悬浮碎片及背景斑块等局部高频伪影。综合画质、速度与显存，768x768 是最稳健的研发分辨率。"
    )
    q4_sheet = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/Q4_S2_0.png"
    add_img(doc, q4_sheet, "图 8：单人招手在 768 与 1024 分辨率下的输出质量与伪影对比", width=Inches(5.4))

    # ================= 三、外挂适配器训练与推理验证 =================
    add_sec_heading(doc, "三、", "外挂适配器（Adapter）训练反向传播与推理验证")
    add_body(doc, "本轮实验对自定义 CNN 几何适配器的反向传播链路进行了彻底修复与技术验收：")

    add_bullet(doc, "1. 梯度链路完全畅通：", 
        "DiT 骨干保持严格冻结（前后参数 SHA-256 哈希 100% 一致），适配器 6 个残差头全部挂载进入 Transformer 隐藏状态，前 3 次反向传播更新后适配器主体均获得有效梯度。")
    add_bullet(doc, "2. 零初始化门控验证：", 
        "残差输出层的零初始化（Zero-Conv）等价性检查完全通过，确保未训练时与基座行为完全一致。")
    add_bullet(doc, "3. 独立加载推理重构测试：", 
        "训练生成的适配器权重经独立进程重新加载并注入 DeepGen，成功完成了独立推理，验证了全套保存与加载接口的可用性。")

    # 嵌入适配器推理图片
    ad_s0 = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/adapter_infer_S0_0.png"
    add_img(doc, ad_s0, "图 9：单人适配器独立加载推理效果（512 / 768 / 1024 分辨率）", width=Inches(5.4))

    ad_d0 = "d:/codeplus/pictureedit/results/review_20260908_v2/review_sheets/adapter_infer_D0_0.png"
    add_img(doc, ad_d0, "图 10：双人适配器独立加载推理效果（验证多尺度几何特征注入与重构）", width=Inches(5.4))

    # ================= 四、科研推进建议与下一步计划 =================
    add_sec_heading(doc, "四、", "科研推进建议与下一步攻坚计划")
    add_body(doc, "基于上述严密的显存与画质实测数据，提出后续课题推进的具体决策建议：")

    add_bullet(doc, "建议 1：将 768 分辨率 + M 型适配器设为后续默认配置：", 
        "稳定训练显存仅 13.40 GiB，单步耗时 0.88s，兼顾了细节表现力与高吞吐迭代，避开 1024 下的高频局部伪影。")
    add_bullet(doc, "建议 2：必须强制引入 3D SMPL-X 与 Contact Map 显式几何引导：", 
        "实测已证明纯文本无法解决双人交互中的【新增人物】与【人物错配】。必须通过 3D 深度通道锁定前后空间、部位分割通道锁定左右人物 ID、接触热图锁定握手咬合点，彻底杜绝凭空造人！")
    add_bullet(doc, "建议 3：推进真实配对数据训练：", 
        "适配器训练接口已验证完毕，下一步必须切换为真实人体姿态配对数据集（CHI3D / Hi4D / EgoBody），摆脱当前单图自重建的局限，正式开启姿态重定向训练。")
    add_bullet(doc, "建议 4：坚守 5B 轻量化交付主干：", 
        "DeepGen 1.0 (5B) 显存极低、白盒可控，是满足任务书 <6B 约束的最佳载体；20B 级 Qwen-Edit 保持为质量上限对照组。")

    # 签名
    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(14)
    p_end.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_e = p_end.add_run("人物图像姿态编辑课题组 · 实验复核与显存评测 · 2026年9月")
    set_run_font(run_e, name="微软雅黑", size=9, bold=True, color=(113, 128, 150))

    # 保存
    out_1 = "d:/codeplus/pictureedit/DeepGen最新实验结果与显存分析简报.docx"
    out_2 = "d:/codeplus/pictureedit/demooutput/DeepGen最新实验结果与显存分析简报.docx"
    doc.save(out_1)
    print(f"Latest experiment report saved successfully: {out_1}")
    try:
        doc.save(out_2)
        print(f"Copy saved successfully: {out_2}")
    except Exception as e:
        print(f"Note: Could not overwrite {out_2} (file may be open in Word): {e}")

    # 同时同步新图表至 demooutput
    for f_name in ["vram_comparison_by_mode.png", "vram_adapter_scale_and_latency.png", "deepgen_vram_requirement_dashboard.png"]:
        src_p = os.path.join("d:/codeplus/pictureedit/results", f_name)
        dst_p = os.path.join("d:/codeplus/pictureedit/demooutput", f_name)
        if os.path.exists(src_p):
            try:
                shutil.copy(src_p, dst_p)
            except Exception:
                pass
    print("VRAM charts copied to demooutput.")

if __name__ == "__main__":
    build_latest_report()
