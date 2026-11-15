#!/usr/bin/env python3
"""Build the SFT/RL Word report from the verified report assets.

Layout:
  portrait A4   cover, summary, protocol, quantitative results, conclusion
  landscape A4  annotated sample, 50 six-column case pages, RL300 strip, appendix
  portrait A4   reproducibility and source notes
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path('D:/codeplus/pictureedit/reports/bilibili-pose-id-sft-rl-comparison-20261114')
REPORT = Path('D:/temp/codex/worktrees/native-spatial-weights/pictureedit/experiments/'
              'bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1')
ASSETS = ROOT / 'assets'
OUT = ROOT / 'Bilibili_Pose_ID_SFT_RL_对比报告_20261114_v1.docx'

INK = RGBColor(0x20, 0x2A, 0x28)
MUTED = RGBColor(0x68, 0x71, 0x6C)
GREEN = RGBColor(0x2D, 0x6B, 0x52)
RUST = RGBColor(0xA4, 0x48, 0x35)
AMBER = RGBColor(0x8A, 0x62, 0x1E)
LINE = 'DCE1DB'
GROUND = 'F5F6F2'
GREEN_BG = 'EDF5EF'
RUST_BG = 'FBEFED'
AMBER_BG = 'FBF4E6'
WHITE = 'FFFFFF'

STATUS_LABEL = {'both': 'Pose / ID 同升', 'tradeoff': 'Pose / ID 权衡',
                'regression': 'Pose / ID 双降', 'id_na': 'ID 不适用'}
STATUS_COLOR = {'both': GREEN, 'tradeoff': AMBER, 'regression': RUST, 'id_na': MUTED}
STATUS_BG = {'both': GREEN_BG, 'tradeoff': AMBER_BG, 'regression': RUST_BG, 'id_na': GROUND}

quant = json.loads((ASSETS / 'quant.json').read_text(encoding='utf-8'))
base_per = json.loads((ASSETS / 'base' / 'per_image.json').read_text(encoding='utf-8'))
base_summary = json.loads((ASSETS / 'base' / 'summary.json').read_text(encoding='utf-8'))

checkpoints = {row['step']: row for row in quant['checkpoints']}
c0, c200, c300 = checkpoints[0], checkpoints[200], checkpoints[300]
s200, s300 = quant['stats']['200'], quant['stats']['300']

id_rows = [row for row in base_per if row['id_applicable']]
base_ids = [row['identity_cosine'] for row in id_rows if row['identity_cosine'] is not None]
base_buf = [row['buffalo_cosine'] for row in id_rows if row['buffalo_cosine'] is not None]
base_nme = [row['nme_diagonal'] for row in base_per if row['nme_diagonal'] is not None]
base = {
    'pose_reward': base_summary['pose_reward'],
    'pck_002_diagonal': base_summary['pck_002_diagonal'],
    'pck_005_diagonal': base_summary['pck_005_diagonal'],
    'nme_diagonal': sum(base_nme) / len(base_nme),
    'identity_cosine': sum(base_ids) / len(base_ids),
    'buffalo_cosine': sum(base_buf) / len(base_buf),
    'identity_reward': base_summary['identity_reward'],
    'buffalo_reward': base_summary['buffalo_reward'],
    'pose_failures': base_summary['pose_failures'],
    'id_failures': base_summary['id_failures'],
    'id_applicable': base_summary['id_applicable'],
}

pairs = quant['pairs']
pair_by_id = {row['pair_id']: row for row in pairs}
clip_order = []
for row in pairs:
    if row['clip_id'] not in clip_order:
        clip_order.append(row['clip_id'])
gallery = []
for clip in clip_order:
    gallery.extend([row for row in pairs if row['clip_id'] == clip])
case_number = {row['pair_id']: index + 1 for index, row in enumerate(gallery)}


# ---------------------------------------------------------------- primitives
def set_run_font(run, size=None, bold=None, color=None, name='Microsoft YaHei'):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_style_font(style, size, bold=False, color=INK, name='Microsoft YaHei'):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = color
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), name)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    for old in tc_pr.findall(qn('w:shd')):
        tc_pr.remove(old)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    tc_pr.append(shd)


def cell_margin(cell, top=40, bottom=40, left=70, right=70):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement('w:tcMar')
    for key, value in (('top', top), ('bottom', bottom), ('left', left), ('right', right)):
        node = OxmlElement(f'w:{key}')
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')
        mar.append(node)
    tc_pr.append(mar)


def table_borders(table, color=LINE, size=4, inside=True):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    edges = ['top', 'left', 'bottom', 'right']
    if inside:
        edges += ['insideH', 'insideV']
    for edge in edges:
        node = OxmlElement(f'w:{edge}')
        node.set(qn('w:val'), 'single' if inside or edge in edges[:4] else 'nil')
        node.set(qn('w:sz'), str(size))
        node.set(qn('w:space'), '0')
        node.set(qn('w:color'), color)
        borders.append(node)
    for old in tbl_pr.findall(qn('w:tblBorders')):
        tbl_pr.remove(old)
    tbl_pr.append(borders)


def no_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        node = OxmlElement(f'w:{edge}')
        node.set(qn('w:val'), 'nil')
        borders.append(node)
    for old in tbl_pr.findall(qn('w:tblBorders')):
        tbl_pr.remove(old)
    tbl_pr.append(borders)


def fixed_layout(table, widths_cm):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tbl_pr.append(layout)
    for index, width in enumerate(widths_cm):
        table.columns[index].width = Cm(width)
        for cell in table.columns[index].cells:
            cell.width = Cm(width)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement('w:tblHeader')
    node.set(qn('w:val'), 'true')
    tr_pr.append(node)


def cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement('w:cantSplit'))


def cell_text(cell, text, size=8.5, bold=False, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(str(text))
    set_run_font(run, size, bold, color)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, 8, False, MUTED)
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = ' PAGE '
    separate = OxmlElement('w:fldChar')
    separate.set(qn('w:fldCharType'), 'separate')
    text = OxmlElement('w:t')
    text.text = '1'
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    run._r.append(text)
    run._r.append(end)


def configure_header_footer(section, first_page_blank=False):
    section.different_first_page_header_footer = first_page_blank
    header = section.header
    header.is_linked_to_previous = False
    paragraph = header.paragraphs[0]
    paragraph.text = ''
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run('Bilibili Pose / ID · SFT → RL 对比 · 2026-11-14')
    set_run_font(run, 7.5, False, MUTED)
    run2 = paragraph.add_run('    sft-rl-comparison-20261114-v1')
    set_run_font(run2, 7.5, False, RGBColor(0x9A, 0xA2, 0x9D))
    footer = section.footer
    footer.is_linked_to_previous = False
    add_page_number(footer.paragraphs[0])
    if first_page_blank:
        section.first_page_header.paragraphs[0].text = ''
        section.first_page_footer.paragraphs[0].text = ''


def h1(text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(text)
    set_run_font(run, 19, True, INK)
    border = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '8')
    bottom.set(qn('w:space'), '4')
    bottom.set(qn('w:color'), LINE)
    border.append(bottom)
    paragraph._p.get_or_add_pPr().append(border)
    return paragraph


def h2(text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(10)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, 13, True, GREEN)
    return paragraph


def body(text, size=10, color=INK, space_after=5, first_indent=False):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.32
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Cm(0.74)
    run = paragraph.add_run(text)
    set_run_font(run, size, False, color)
    return paragraph


def bullet(text, size=10):
    paragraph = doc.add_paragraph(style='List Bullet')
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.28
    run = paragraph.add_run(text)
    set_run_font(run, size, False, INK)
    return paragraph


def note(text, size=8.5):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.2
    run = paragraph.add_run(text)
    set_run_font(run, size, False, MUTED)
    return paragraph


def caption(text):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    set_run_font(run, 8.5, False, MUTED)
    return paragraph


def add_figure(path, width_cm, cap, max_height_cm=None):
    doc.add_picture(str(path), width=Cm(width_cm))
    picture = doc.paragraphs[-1]
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.paragraph_format.space_before = Pt(4)
    picture.paragraph_format.space_after = Pt(0)
    if max_height_cm:
        # python-docx has already inserted at fixed width; scale down if needed.
        from PIL import Image
        with Image.open(path) as image:
            ratio = image.height / image.width
        if width_cm * ratio > max_height_cm:
            picture.runs[0]._r.get_or_add_drawing().xpath('./wp:extent')[0].attrib['cy'] = str(int(Cm(max_height_cm)))
    caption(cap)


def callout(title, lines, fill=GREEN_BG, border_color='2D6B52'):
    table = doc.add_table(rows=1, cols=1)
    no_borders(table)
    fixed_layout(table, [17.0])
    cell = table.cell(0, 0)
    shade(cell, fill)
    cell_margin(cell, 100, 100, 140, 140)
    tc_pr = cell._tc.get_or_add_tcPr()
    tcb = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '18')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_color)
    tcb.append(left)
    tc_pr.append(tcb)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(title)
    set_run_font(run, 10, True, GREEN if border_color == '2D6B52' else INK)
    for line in lines:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.24
        r = p.add_run(line)
        set_run_font(r, 9, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def data_table(headers, rows, widths, font=8.2, header_fill=GROUND, row_fills=None,
               align_first_left=False):
    table = doc.add_table(rows=1, cols=len(headers))
    table_borders(table)
    fixed_layout(table, widths)
    header = table.rows[0]
    cant_split(header)
    repeat_header(header)
    for index, text in enumerate(headers):
        cell = header.cells[index]
        shade(cell, header_fill)
        cell_margin(cell, 55, 55, 55, 55)
        cell_text(cell, text, font, True, MUTED)
    for r_idx, values in enumerate(rows):
        row = table.add_row()
        cant_split(row)
        fill = row_fills[r_idx] if row_fills else (WHITE if r_idx % 2 == 0 else 'FBFCFA')
        body_margin = 26 if font <= 7.2 else 45
        for c_idx, value in enumerate(values):
            cell = row.cells[c_idx]
            if fill:
                shade(cell, fill)
            cell_margin(cell, body_margin, body_margin, 50, 50)
            align = WD_ALIGN_PARAGRAPH.LEFT if (align_first_left and c_idx == 0) else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(cell, value, font, False, INK, align)
    return table


def fmt4(value):
    return f'{value:.4f}'


def fmt_pct(value):
    return f'{value * 100:.2f}%'


def fmt_delta(value, digits=4):
    return f'{value:+.{digits}f}'


def fmt_pp(value):
    return f'{value * 100:+.2f} pp'


def delta_text(value, digits=4):
    return f'{value:+.{digits}f}'


# ---------------------------------------------------------------- document
doc = Document()
style = doc.styles['Normal']
set_style_font(style, 10, False, INK)
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.line_spacing = 1.3

section = doc.sections[0]
section.orientation = WD_ORIENT.PORTRAIT
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.top_margin = Cm(2.0)
section.bottom_margin = Cm(1.9)
section.left_margin = Cm(2.0)
section.right_margin = Cm(2.0)
configure_header_footer(section, first_page_blank=True)

# Cover
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(46)
p.paragraph_format.space_after = Pt(10)
r = p.add_run('BILIBILI POSE / ID · 固定验证集 · SFT → RL')
set_run_font(r, 9.5, True, GREEN)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(4)
r = p.add_run('SFT 之后追加 RL：')
set_run_font(r, 27, True, INK)
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(10)
r = p.add_run('200 步更均衡，300 步更偏 ID')
set_run_font(r, 27, True, GREEN)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(24)
r = p.add_run('50 个固定验证 pair 的六列图像对比（原图 / 骨架 / GT / Base Qwen / SFT / RL200）\n'
              '以及 Pose、PCK、NME、glintr100、buffalo_l 的配对量化复算')
set_run_font(r, 11.5, False, MUTED)

# cover KPI strips
kpi = doc.add_table(rows=1, cols=3)
no_borders(kpi)
fixed_layout(kpi, [5.6, 5.6, 5.6])
kpis = [
    ('+0.0373', f'Pose Reward 增量（+5.07%）\n{fmt4(c0["pose_reward"])} → {fmt4(c200["pose_reward"])}'),
    ('+3.35 pp', f'PCK@0.02 增量\n{fmt_pct(c0["pck_002_diagonal"])} → {fmt_pct(c200["pck_002_diagonal"])}'),
    ('+0.0499', f'glintr100 raw cosine 增量\n{fmt4(c0["identity_cosine"])} → {fmt4(c200["identity_cosine"])}'),
]
for index, (big, small) in enumerate(kpis):
    cell = kpi.cell(0, index)
    shade(cell, GREEN_BG if index == 0 else GROUND)
    cell_margin(cell, 120, 120, 140, 100)
    p1 = cell.paragraphs[0]
    p1.paragraph_format.space_after = Pt(2)
    r1 = p1.add_run(big)
    set_run_font(r1, 20, True, GREEN if index == 0 else INK)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.2
    r2 = p2.add_run(small)
    set_run_font(r2, 7.5, False, MUTED)

doc.add_paragraph().paragraph_format.space_after = Pt(10)

meta = doc.add_table(rows=0, cols=2)
no_borders(meta)
fixed_layout(meta, [4.4, 12.6])
meta_rows = [
    ('报告日期', '2026-11-14（America/New_York）'),
    ('数据来源', 'sft-rl-comparison-20261114-v1（本地独立报告）'),
    ('评测集', '固定 50 个 val pair，5 个视频片段；固定 33 个 ID 可评估 pair'),
    ('采样协议', '384 × 672，seed 42，30-step ODE，CFG 1；7 个 checkpoint 使用同一批 pair'),
    ('当前候选', 'SFT + RL 200 updates（在 7 个 checkpoint 中按同一验证集选取）'),
    ('补充素材', 'Base Qwen 50 张：2026-11-14 ET（服务器 11-15 中午）同协议补跑，仅作视觉参考，不参与训练或选点'),
]
for key, value in meta_rows:
    row = meta.add_row()
    cell = row.cells[0]
    cell_margin(cell, 45, 45, 0, 80)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(key)
    set_run_font(r, 8.5, True, MUTED)
    cell2 = row.cells[1]
    cell_margin(cell2, 45, 45, 0, 0)
    p2 = cell2.paragraphs[0]
    p2.paragraph_format.space_after = Pt(0)
    r2 = p2.add_run(value)
    set_run_font(r2, 8.5, False, INK)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(26)
r = p.add_run('本报告只重排和复算既有评测产物；生成图未裁切、未加滤镜、未改写。'
              '所有结论都受固定验证集、片段相关性与 checkpoint 选择偏差限制，详见第 3 节。')
set_run_font(r, 7.5, False, MUTED)

# Page break then summary
h1('结论摘要').paragraph_format.page_break_before = True
body('在同一批 50 个验证 pair 上，从 SFT 9520 继续追加 RL 到 200 步，Pose 与 ID 同时有正收益：'
     f'Pose Reward 从 {fmt4(c0["pose_reward"])} 升到 {fmt4(c200["pose_reward"])}，'
     f'增加 {c200["pose_reward"] - c0["pose_reward"]:.4f}（+5.07%）；'
     f'PCK@0.02 从 {fmt_pct(c0["pck_002_diagonal"])} 升到 {fmt_pct(c200["pck_002_diagonal"])}，'
     f'增加 {(c200["pck_002_diagonal"] - c0["pck_002_diagonal"]) * 100:.2f} 个百分点。'
     f'50 对中 {quant["joint"]["pose_wins"]} 对的连续 Pose Reward 上升，'
     f'{quant["joint"]["pose_losses"]} 对下降。')

body(f'固定 33 个 ID 可评估 pair 的 glintr100 raw cosine 从 {fmt4(c0["identity_cosine"])} 升到 '
     f'{fmt4(c200["identity_cosine"])}，{quant["joint"]["id_wins"]} 对上升、{quant["joint"]["id_losses"]} 对下降；'
     f'buffalo_l raw cosine 从 {fmt4(c0["buffalo_cosine"])} 升到 {fmt4(c200["buffalo_cosine"])}，'
     f'{quant["joint"]["buffalo_wins"]} 对上升、{quant["joint"]["buffalo_losses"]} 对下降。'
     f'两套脸部特征模型方向一致，但仍有 12 对存在 Pose / ID 取舍、1 对双降，必须逐张检查。')

body(f'300 步的 ID 更高（glintr100 {fmt4(c300["identity_cosine"])}、buffalo_l {fmt4(c300["buffalo_cosine"])}），'
     f'但 Pose 与 PCK 均低于 200 步（Pose {fmt4(c300["pose_reward"])}、PCK@0.02 {fmt_pct(c300["pck_002_diagonal"])}），'
     f'且 Pose 的片段 bootstrap 区间跨 0。因此把 200 步作为当前默认候选，300 步保留为“更偏 ID”的对照。')

callout('一句话结论', [
    '本验证集支持“追加 RL 200 步相对现有 SFT 基线有观测收益”，但不证明泛化、也不证明 RL 优于等预算继续 SFT。',
    '200 步是在同一验证集的 7 个 checkpoint 中挑出的；下一步必须在未参与选择的新视频留出集上固定权重复核。',
])

h2('主要数字（RL200 相对 SFT 9520）')
summary_rows = [
    ['Pose Reward ↑', fmt4(c0['pose_reward']), fmt4(c200['pose_reward']), fmt4(c300['pose_reward']),
     f'{fmt_delta(c200["pose_reward"] - c0["pose_reward"])}'],
    ['PCK@0.02 ↑', fmt_pct(c0['pck_002_diagonal']), fmt_pct(c200['pck_002_diagonal']),
     fmt_pct(c300['pck_002_diagonal']), fmt_pp(c200['pck_002_diagonal'] - c0['pck_002_diagonal'])],
    ['PCK@0.05 ↑', fmt_pct(c0['pck_005_diagonal']), fmt_pct(c200['pck_005_diagonal']),
     fmt_pct(c300['pck_005_diagonal']), fmt_pp(c200['pck_005_diagonal'] - c0['pck_005_diagonal'])],
    ['NME / 对角线 ↓', f'{c0["nme_diagonal"]:.5f}', f'{c200["nme_diagonal"]:.5f}',
     f'{c300["nme_diagonal"]:.5f}', f'{c200["nme_diagonal"] - c0["nme_diagonal"]:+.5f}'],
    ['glintr100 raw cosine ↑', fmt4(c0['identity_cosine']), fmt4(c200['identity_cosine']),
     fmt4(c300['identity_cosine']), fmt_delta(c200['identity_cosine'] - c0['identity_cosine'])],
    ['buffalo_l raw cosine ↑', fmt4(c0['buffalo_cosine']), fmt4(c200['buffalo_cosine']),
     fmt4(c300['buffalo_cosine']), fmt_delta(c200['buffalo_cosine'] - c0['buffalo_cosine'])],
]
data_table(['指标', 'SFT 9520 · RL0', 'RL200', 'RL300', 'RL200 − SFT'],
           summary_rows, [4.4, 3.2, 3.0, 3.0, 3.2], font=8.4,
           row_fills=[WHITE, GREEN_BG, AMBER_BG, WHITE, WHITE, WHITE], align_first_left=True)
note('ID Reward = (raw cosine + 1) / 2；glintr100 / buffalo_l 只在固定 33 对比较，另外 17 对不记作 0 分。'
     '所有 checkpoint 的 Pose 检测失败为 0/50，ID 适用组生成漏脸为 0/33。')

# Protocol
h1('1. 实验与评测口径').paragraph_format.page_break_before = True
h2('1.1 实验链')
body('本轮从现成的 Qwen-Image-2.1 Edit LoRA（Bilibili 双图 Edit，SFT 9,520 updates）继续训练 300 次 '
     'optimizer update。每次 update 使用 4 个 pair × 8 张候选，共 150 轮、4,800 张训练候选；'
     '输入固定为 [A 的 RGB, B 的原始 DWPose]，画布 384 × 672。训练目标是 GRPO + GDPO 的 Pose:ID = 1:1，'
     'LR 1e-4、前 10 次 update 线性 warmup、clip 1e-4、v-based KL beta 1e-3。'
     '本报告不重新训练，只整理固定验证集上的 7 个 checkpoint。')
body('7 个 checkpoint 的映射：RL0 = SFT 9520；RL50/100/150/200/250/300 分别对应训练中每 50 次 update 的'
     '完整状态。200 步是事后从这 7 个点里挑出的当前候选，因此存在同验证集选点偏差；300 步是训练终点，'
     '不因为它是最后一步就当作最好模型。')

h2('1.2 固定验证集与配对')
body('验证集覆盖全部 5 个验证视频，每个视频均衡抽取 10 对，共 50 对；其中 33 对的目标头部可关联、'
     'A 的人脸 embedding 有效，纳入固定 ID 评估，另外 17 对保持“不适用”，不记作 0 分或漏检。'
     'Pose 使用全部 50 对，ID 使用固定 33 对；两个分母在所有 checkpoint 之间完全一致。')
body('5 个片段各有 10 对，但同一片段的多对样本共享人物、背景和动作来源，彼此相关。'
     '因此把 50 张图当作 50 个独立实验会高估证据量；本报告的分段 bootstrap 只是小样本描述，'
     '不宣称独立显著性。')

h2('1.3 指标定义与读法')
bullet('Pose Reward：0–1 连续分数，越大越好。由生成图检测到的 COCO17 关节与目标骨架比较得到；'
       '检测失败或多检会影响分数，本次 7 个 checkpoint 均为 0/50 失败。')
bullet('PCK@0.02 / PCK@0.05：目标有效关节使用固定分母，漏检记 0，再逐图平均；阈值按图像对角线归一化。'
       'PCK 的“上升”可能来自少数关节，必须结合 NME 和原图检查。')
bullet('NME / 对角线：只统计匹配到的关节，是诊断项，不是 Reward；匹配集合会变化，因此不能单独当作质量结论。')
bullet('glintr100 / buffalo_l raw cosine：越大越好，使用固定 33 对；buffalo_l 未参与训练梯度，'
       '但参与了 checkpoint 资格门槛，也不是完全未参与选择的外部 test。')
bullet('ID Reward = (raw cosine + 1) / 2。不要把 Reward 0.8313 与 raw cosine 0.6626 混用，也不把 Pose 与 ID '
       '的 raw 值加成一个总分。')

h2('1.4 Base Qwen 补跑说明')
base_pose_gap = base['pose_reward'] - c0['pose_reward']
base_id_gap = base['identity_cosine'] - c0['identity_cosine']
body('为了补齐用户要求的“Base Qwen”列，本次在 50 个相同 val pair 上补跑原始 Qwen-Image-2.1（无 LoRA）'
     '50 张：384 × 672、seed 42、30-step ODE、CFG 1，输入与 SFT/RL 相同的 [A, B DWPose] 和提示词。'
     '补跑写入独立目录，不覆盖任何既有产物；指标用与 RL 报告相同的 CPU evaluator（DWPose、SCRFD、'
     'glintr100、buffalo_l）计算。')
body(f'补跑结果：Base Qwen 的 Pose Reward 为 {fmt4(base["pose_reward"])}、PCK@0.02 为 '
     f'{fmt_pct(base["pck_002_diagonal"])}、glintr100 raw cosine 为 {fmt4(base["identity_cosine"])}、'
     f'buffalo_l raw cosine 为 {fmt4(base["buffalo_cosine"])}。相对 SFT 基线，Pose 为 '
     f'{base["pose_reward"] - c0["pose_reward"]:+.4f}、PCK@0.02 为 '
     f'{(base["pck_002_diagonal"] - c0["pck_002_diagonal"]) * 100:+.2f} pp，但 glintr100 为 '
     f'{base["identity_cosine"] - c0["identity_cosine"]:+.4f}、buffalo_l 为 '
     f'{base["buffalo_cosine"] - c0["buffalo_cosine"]:+.4f}。')
body(f'这组对照说明：SFT 9520 相对原始基座明显改善了骨架跟随，但 ID 反而低于 Base；'
     f'RL200 把 glintr100 拉回 Base 之上 {c200["identity_cosine"] - base["identity_cosine"]:+.4f}、'
     f'buffalo_l 基本持平（{c200["buffalo_cosine"] - base["buffalo_cosine"]:+.4f}），同时 Pose 仍高于 SFT。'
     '因此“RL 的 ID 提升”更准确的说法是：它补回了 SFT 阶段的身份代价，并在 glintr100 上略有超出；'
     '不能据此说 RL 已全面超过原始基座的身份保持。Base 列只用于看“未微调模型”的失败模式与量级，'
     '不参与 7 个 checkpoint 的结论、选点或显著性判断。')
note('实现细节：Base Qwen 走上游 QwenImage21Pipeline（Bilibili Edit 实验的 30-step 推理路径），'
     '其 KV cache 默认开启；RL runtime 的评测配置为 use_kv_cache=false。两者采样步数、seed、CFG 和画布一致，'
     '但数值实现并非逐位相同，因此 Base 只作视觉与量级参考。')

# Quantitative results
h1('2. 定量结果').paragraph_format.page_break_before = True
h2('2.1 七个 checkpoint 的主表')
main_rows = []
fills = []
for step in (0, 50, 100, 150, 200, 250, 300):
    row = checkpoints[step]
    label = '0（SFT 起点）' if step == 0 else str(step)
    tag = '当前候选' if step == 200 else ('训练终点' if step == 300 else '')
    main_rows.append([
        label,
        fmt4(row['pose_reward']), fmt_pct(row['pck_002_diagonal']), fmt_pct(row['pck_005_diagonal']),
        f'{row["nme_diagonal"]:.5f}', fmt4(row['identity_cosine']), fmt4(row['buffalo_cosine']), tag,
    ])
    fills.append(GREEN_BG if step == 200 else (AMBER_BG if step == 300 else (WHITE if step % 100 == 0 else 'FBFCFA')))
data_table(['RL update', 'Pose ↑', 'PCK@0.02 ↑', 'PCK@0.05 ↑', 'NME ↓', 'glintr100 ↑', 'buffalo_l ↑', '备注'],
           main_rows, [2.2, 2.0, 2.1, 2.1, 2.0, 2.3, 2.3, 2.0], font=8.0,
           row_fills=fills)
note('50 对 Pose / PCK，33 对 ID；检测失败 0/50，ID 漏脸 0/33。RL200 行高亮为当前候选；RL300 为训练终点对照。')

h2('2.2 Base Qwen、SFT、RL200、RL300 的同表对比')
base_rows = [
    ['Base Qwen（补跑）', fmt4(base['pose_reward']), fmt_pct(base['pck_002_diagonal']),
     fmt_pct(base['pck_005_diagonal']), f'{base["nme_diagonal"]:.5f}', fmt4(base['identity_cosine']),
     fmt4(base['buffalo_cosine']), f'{base["pose_failures"]}/{base["id_failures"]}'],
    ['SFT 9520（RL0）', fmt4(c0['pose_reward']), fmt_pct(c0['pck_002_diagonal']),
     fmt_pct(c0['pck_005_diagonal']), f'{c0["nme_diagonal"]:.5f}', fmt4(c0['identity_cosine']),
     fmt4(c0['buffalo_cosine']), f'{c0["pose_failures"]}/{c0["id_failures"]}'],
    ['SFT + RL200', fmt4(c200['pose_reward']), fmt_pct(c200['pck_002_diagonal']),
     fmt_pct(c200['pck_005_diagonal']), f'{c200["nme_diagonal"]:.5f}', fmt4(c200['identity_cosine']),
     fmt4(c200['buffalo_cosine']), f'{c200["pose_failures"]}/{c200["id_failures"]}'],
    ['SFT + RL300', fmt4(c300['pose_reward']), fmt_pct(c300['pck_002_diagonal']),
     fmt_pct(c300['pck_005_diagonal']), f'{c300["nme_diagonal"]:.5f}', fmt4(c300['identity_cosine']),
     fmt4(c300['buffalo_cosine']), f'{c300["pose_failures"]}/{c300["id_failures"]}'],
]
data_table(['模型', 'Pose ↑', 'PCK@0.02 ↑', 'PCK@0.05 ↑', 'NME ↓', 'glintr100 ↑', 'buffalo_l ↑', 'Pose/ID 失败'],
           base_rows, [3.2, 1.9, 2.0, 2.0, 1.9, 2.1, 2.1, 1.8], font=8.0,
           row_fills=[GROUND, WHITE, GREEN_BG, AMBER_BG], align_first_left=True)
note('Base Qwen 的 Pose/ID 失败列为 Pose 检测失败 / ID 适用组漏脸；33 对 ID 之外不评估。')

h2('2.3 配对胜负、区间与片段')
body(f'RL200 的 Pose 有 {quant["joint"]["pose_wins"]}/50 上升、{quant["joint"]["pose_losses"]} 下降；'
     f'PCK@0.02 为 {quant["joint"]["pck002_wins"]} 升、{quant["joint"]["pck002_losses"]} 降、'
     f'{quant["joint"]["pck002_ties"]} 持平；33 个 ID 有效例中 glintr100 {quant["joint"]["id_wins"]} 升 '
     f'{quant["joint"]["id_losses"]} 降，buffalo_l {quant["joint"]["buffalo_wins"]} 升 '
     f'{quant["joint"]["buffalo_losses"]} 降。33 个 ID 有效例中，'
     f'{quant["joint"]["both_up"]} 例 Pose/ID 同升、{quant["joint"]["tradeoff"]} 例取舍、'
     f'{quant["joint"]["both_down"]} 例双降。')
stat_rows = []
for key, label in (('pose_reward', 'Pose Reward'), ('pck_002_diagonal', 'PCK@0.02'),
                   ('nme_diagonal', 'NME ↓'), ('identity_cosine', 'glintr100 cos'),
                   ('buffalo_cosine', 'buffalo_l cos')):
    values = s200[key]
    wins, losses, ties = values['wins'], values['losses'], values['ties']
    ci = values['cluster_bootstrap_ci95_delta']
    stat_rows.append([label, f'{values["delta"]:+.4f}',
                      f'{wins} / {losses} / {ties}', f'[{ci[0]:+.4f}, {ci[1]:+.4f}]',
                      f'{values["cluster_signflip_p_two_sided"]:.3f}'])
data_table(['指标（RL200 − SFT）', 'Δ', '胜 / 负 / 平', '片段 bootstrap 95% CI', 'sign-flip p'],
           stat_rows, [3.8, 2.4, 3.0, 5.0, 2.8], font=8.2, align_first_left=True)
note('区间按视频片段聚类、穷举重采样；只有 4–5 个片段，p 值和区间是描述性的，不作独立显著性结论。')

stat_rows_300 = []
for key, label in (('pose_reward', 'Pose Reward'), ('pck_002_diagonal', 'PCK@0.02'),
                   ('nme_diagonal', 'NME ↓'), ('identity_cosine', 'glintr100 cos'),
                   ('buffalo_cosine', 'buffalo_l cos')):
    values = s300[key]
    ci = values['cluster_bootstrap_ci95_delta']
    stat_rows_300.append([label, f'{values["delta"]:+.4f}',
                          f'{values["wins"]} / {values["losses"]} / {values["ties"]}',
                          f'[{ci[0]:+.4f}, {ci[1]:+.4f}]',
                          f'{values["cluster_signflip_p_two_sided"]:.3f}'])
data_table(['指标（RL300 − SFT）', 'Δ', '胜 / 负 / 平', '片段 bootstrap 95% CI', 'sign-flip p'],
           stat_rows_300, [3.8, 2.4, 3.0, 5.0, 2.8], font=8.2, align_first_left=True)
note('RL300 的 Pose 区间跨 0；ID 仍上升。这个分化是“200 更均衡、300 更偏 ID”的直接依据。')

clip_rows = []
for clip in clip_order:
    row200 = next(row for row in quant['by_clip']['200'] if row['clip_id'] == clip)
    row300 = next(row for row in quant['by_clip']['300'] if row['clip_id'] == clip)
    clip_rows.append([
        clip, f'{row200["pairs"]} / {row200["id_n"]}',
        f'{row200["pose_delta"]:+.4f}', f'{row200["pck002_delta"] * 100:+.2f} pp',
        ('—' if row200['id_delta'] is None else f'{row200["id_delta"]:+.4f}'),
        f'{row300["pose_delta"]:+.4f}',
        ('—' if row300['id_delta'] is None else f'{row300["id_delta"]:+.4f}'),
        ('Pose ↑ / ID ↑' if row200['pose_delta'] > 0 and (row200['id_delta'] or 0) > 0 else
         ('Pose ↑ / ID —' if row200['pose_delta'] > 0 and row200['id_delta'] is None else 'Pose ↓ / ID ↑')),
    ])
clip_rows.append(['总体（50 / 33）', '50 / 33',
                  f'{s200["pose_reward"]["delta"]:+.4f}',
                  f'{s200["pck_002_diagonal"]["delta"] * 100:+.2f} pp',
                  f'{s200["identity_cosine"]["delta"]:+.4f}',
                  f'{s300["pose_reward"]["delta"]:+.4f}',
                  f'{s300["identity_cosine"]["delta"]:+.4f}', '5/5；4/4（RL200）'])
data_table(['片段', 'n / ID n', 'ΔPose 200', 'ΔPCK@.02 200', 'ΔID 200', 'ΔPose 300', 'ΔID 300', 'RL200 方向'],
           clip_rows, [2.8, 1.6, 2.1, 2.4, 1.9, 2.1, 1.9, 2.2], font=8.0,
           row_fills=[WHITE if index % 2 == 0 else 'FBFCFA' for index in range(len(clip_rows) - 1)] + [GREEN_BG],
           align_first_left=True)
note(f'RL200 的 5 个片段 Pose 增量均为正，范围 {quant["clip_pose_delta_range_200"][0]:+.4f} ～ '
     f'{quant["clip_pose_delta_range_200"][1]:+.4f}；RL300 只有 {quant["clips_positive_pose_300"]}/5 个片段为正，'
     '其中 BV1Aynt6zEQd、BV1XssX6AEsR 的 ΔPose 300 已为负。每个片段 10 对，ID 有效对数见 n/ID n 列。')

body(f'删除任意一个视频片段后，RL200 的 Pose 均值增量仍在 '
     f'{min(s200["pose_reward"]["leave_one_clip_out_deltas"].values()):+.4f} ～ '
     f'{max(s200["pose_reward"]["leave_one_clip_out_deltas"].values()):+.4f}，因此总体提升不是由单个片段单独造成；'
     '但这仍是 5 个相关片段的描述，不是独立视频的泛化证据。')

# Figures
h2('2.4 曲线与逐样本分布').paragraph_format.page_break_before = True
for path, cap, width in (
    ('figures/checkpoint_curves.png', '图 1｜7 个 checkpoint 的固定验证集曲线。横轴为追加 RL 的 update，0 = SFT 9520；三图纵轴各自使用原指标单位。', 16.2),
    ('figures/paired_wins.png', '图 2｜RL200 相对 SFT 的逐 pair 胜负数量。', 15.0),
    ('figures/paired_pose_identity.png', '图 3｜逐 pair 的 Pose 与 ID 变化分布。右下象限为两项同时下降，左上为同时上升。', 14.5),
    ('figures/clip_gains.png', '图 4｜5 个视频片段的平均增量。每个片段 10 对，片段内样本相关，不能当作 5 个独立实验。', 16.2),
):
    add_figure(REPORT / path, width, cap)

# Conclusion
h1('3. 结论、限制与下一步').paragraph_format.page_break_before = True
h2('3.1 可以下的结论')
bullet(f'在本固定验证集上，追加 RL 到 200 步相对 SFT 9520 有观测收益：Pose '
       f'{fmt_delta(c200["pose_reward"] - c0["pose_reward"])}（+5.07%）、PCK@0.02 '
       f'{fmt_pp(c200["pck_002_diagonal"] - c0["pck_002_diagonal"])}、glintr100 '
       f'{fmt_delta(c200["identity_cosine"] - c0["identity_cosine"])}、buffalo_l '
       f'{fmt_delta(c200["buffalo_cosine"] - c0["buffalo_cosine"])}。')
bullet('改善不是由单个视频片段造成：5/5 个片段的平均 Pose 上升，4/4 个有 ID 的片段平均 ID 上升；'
       '删除任意片段后 Pose 增量仍为正。')
bullet('200 步比 300 步更均衡：300 步的 ID 更高，但 Pose/PCK 回落，Pose 的片段区间跨 0；'
       '因此当前候选选 200，不选训练终点。')

h2('3.2 不能下的结论')
bullet('不能证明泛化：200 步是在同一验证集的 7 个 checkpoint 中挑出的，没有使用独立 test 集；'
       '50 张图来自 5 个相关片段，不是 50 个独立视频。')
bullet('不能证明 RL 方法优于“等额外计算量继续 SFT”：本轮没有同起点、等计算预算的继续 SFT 对照。')
bullet('不能从 Pose/ID 分数推断背景、衣物、手部全部改善。人工抽看 8 组（覆盖 5 个片段）显示差异多为局部手腕、'
       '腿部、面部调整；披肩、衣袖和手指细节仍有问题。分数上升不等于全图更忠实于 A。')
bullet('buffalo_l 未参与训练梯度，但参与了 checkpoint 资格门槛，因此也不是完全未参与选择的外部 test。')

h2('3.3 下一步')
body('固定 RL200 权重与采样协议，在未参与选择的新视频留出集上做配对复核；同时补一个同起点、等计算预算的'
     '继续 SFT 对照，才能回答“RL 方法本身是否更优”。视觉侧应做逐张盲评，重点检查身份漂移、手部/手指、'
     '衣袖与披肩、背景伪影，并同时报告 Pose、PCK、glintr100 与 buffalo_l。只有留出集收益和视觉质量检查都支持，'
     '才把 200 步从“当前候选”升级为正式结论。')

# Landscape section: annotated sample + gallery
new = doc.add_section(WD_SECTION.NEW_PAGE)
new.orientation = WD_ORIENT.LANDSCAPE
new.page_width = Cm(29.7)
new.page_height = Cm(21.0)
new.top_margin = Cm(1.2)
new.bottom_margin = Cm(1.2)
new.left_margin = Cm(1.2)
new.right_margin = Cm(1.2)
configure_header_footer(new, first_page_blank=False)

h1('4. 50 例六列图像对比')
body('每页 1 例，六列从左到右固定为：原图 A（参考人物）、目标骨架 B（DWPose）、GT B（目标帧）、'
     'Base Qwen（原始基座，补跑）、SFT 9520、RL200（追加 200 步）。所有图片均为 384 × 672 的原始 PNG，'
     '未裁切、未加滤镜、未做美颜；页面显示宽度约 4.35 cm，像素级检查请打开原始文件。', 9.5)
body(f'图库按视频片段分组，每个片段 10 例；同片段内保持验证集原始顺序。状态标签只使用 RL200 − SFT 的'
     f'Pose 与 glintr100 变化：同升 {quant["category_counts"]["both"]} 例、权衡 {quant["category_counts"]["tradeoff"]} 例、'
     f'双降 {quant["category_counts"]["regression"]} 例、ID 不适用 {quant["category_counts"]["id_na"]} 例。'
     '标签不改变任何数值，也不代表人工质量结论。', 9.5)
h2('图库索引与诊断入口')
index_rows = []
for clip in clip_order:
    selected = [row for row in pairs if row['clip_id'] == clip]
    index_rows.append([clip, str(len(selected)), str(sum(row['id_applicable'] for row in selected)),
                       f'{case_number[selected[0]["pair_id"]]:02d}–{case_number[selected[-1]["pair_id"]]:02d}'])
data_table(['片段', 'pair 数', 'ID 有效', '例号范围'], index_rows,
           [7.0, 3.0, 3.0, 5.0], font=9.0, align_first_left=True)
doc.add_paragraph().paragraph_format.space_after = Pt(0)
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(3)
r = p.add_run('诊断案例（4 个定位入口）')
set_run_font(r, 10, True, GREEN)
diag = [
    ('a3067229', 'Pose 与 ID 共同提升', 'Pose +0.1158、glintr100 +0.1227；手部与披肩仍需检查。'),
    ('67c42ecc', 'Pose 最大增幅', 'Pose +0.1896、glintr100 +0.0252；脚部、手部位置变化主要在局部。'),
    ('753f20d9', 'ID 最大增幅 / 权衡', 'glintr100 +0.1768、Pose −0.0066；用于检查偏脸部的取舍。'),
    ('edb890ee', '双退化反例', 'Pose −0.1539、glintr100 −0.0121；保留此例检查失败模式。'),
]
diag_rows = []
for prefix, title, text in diag:
    target = next(row for row in pairs if row['pair_id'].startswith(prefix))
    diag_rows.append([f'例 {case_number[target["pair_id"]]:02d}', prefix, title, text])
data_table(['位置', 'pair 前缀', '观察', '数值提示'], diag_rows,
           [2.2, 3.0, 5.0, 16.5], font=9.0, align_first_left=True)
note('这 4 例是定位入口，不是全量 50 例的人评结论，也不是盲评。先看每页的状态标签与底部指标，'
     '再放大看手部、脸、衣袖和背景；Pose/ID 双降或取舍的页面优先检查。')

# Gallery pages
for index, row in enumerate(gallery):
    number = index + 1
    category = row['category']
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f'例 {number:02d} / 50')
    set_run_font(r, 11, True, INK)
    r = p.add_run(f'    {row["clip_id"]}')
    set_run_font(r, 8.5, False, MUTED)
    r = p.add_run(f'    {STATUS_LABEL[category]}')
    set_run_font(r, 8.5, True, STATUS_COLOR[category])
    r = p.add_run(f'    pair {row["pair_id"][:12]}…')
    set_run_font(r, 7.5, False, MUTED)

    photos = [
        (REPORT / f'context/{row["pair_id"]}/A.png', '原图 A', '参考人物'),
        (REPORT / f'context/{row["pair_id"]}/pose.png', '骨架 B', 'DWPose'),
        (REPORT / f'context/{row["pair_id"]}/GT.png', 'GT B', '目标帧'),
        (ASSETS / 'base' / f'{row["pair_id"]}.png', 'Base Qwen', '原始基座 · 补跑'),
        (REPORT / f'images/0/{row["pair_id"]}.png', 'SFT 9520', 'RL0'),
        (REPORT / f'images/200/{row["pair_id"]}.png', 'RL200', '当前候选'),
    ]
    for path, _, _ in photos:
        if not path.exists():
            raise SystemExit(f'missing image: {path}')
    table = doc.add_table(rows=1, cols=6)
    no_borders(table)
    fixed_layout(table, [4.35] * 6)
    for col, (path, label, sub) in enumerate(photos):
        cell = table.cell(0, col)
        cell_margin(cell, 0, 0, 30, 30)
        label_p = cell.paragraphs[0]
        label_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label_p.paragraph_format.space_before = Pt(0)
        label_p.paragraph_format.space_after = Pt(0)
        rr = label_p.add_run(label)
        set_run_font(rr, 8.0, True, INK)
        sub_p = cell.add_paragraph()
        sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_p.paragraph_format.space_before = Pt(0)
        sub_p.paragraph_format.space_after = Pt(1)
        rr = sub_p.add_run(sub)
        set_run_font(rr, 6.5, False, MUTED)
        image_p = cell.add_paragraph()
        image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_p.paragraph_format.space_before = Pt(0)
        image_p.paragraph_format.space_after = Pt(0)
        image_p.add_run().add_picture(str(path), width=Cm(4.25))

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Pose  ')
    set_run_font(r, 7.8, True, MUTED)
    r = p.add_run(f'{fmt4(row["pose_base"])} → {fmt4(row["pose_200"])}  ')
    set_run_font(r, 8.6, False, INK)
    r = p.add_run(f'({row["pose_delta_200"]:+.4f})    ')
    set_run_font(r, 8.6, True, GREEN if row['pose_delta_200'] > 0 else RUST)
    r = p.add_run('PCK@0.02  ')
    set_run_font(r, 7.8, True, MUTED)
    r = p.add_run(f'{fmt_pct(row["pck002_base"])} → {fmt_pct(row["pck002_200"])}  ')
    set_run_font(r, 8.6, False, INK)
    r = p.add_run(f'({row["pck002_delta_200"] * 100:+.2f} pp)    ')
    set_run_font(r, 8.6, True, GREEN if row['pck002_delta_200'] > 0 else (MUTED if row['pck002_delta_200'] == 0 else RUST))

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(3)
    if row['id_applicable']:
        r = p2.add_run('glintr100  ')
        set_run_font(r, 7.8, True, MUTED)
        r = p2.add_run(f'{fmt4(row["id_base"])} → {fmt4(row["id_200"])}  ')
        set_run_font(r, 8.6, False, INK)
        r = p2.add_run(f'({row["id_delta_200"]:+.4f})    ')
        set_run_font(r, 8.6, True, GREEN if row['id_delta_200'] > 0 else RUST)
        r = p2.add_run('buffalo_l  ')
        set_run_font(r, 7.8, True, MUTED)
        r = p2.add_run(f'{fmt4(row["buffalo_base"])} → {fmt4(row["buffalo_200"])}  ')
        set_run_font(r, 8.6, False, INK)
        r = p2.add_run(f'({row["buffalo_delta_200"]:+.4f})')
        set_run_font(r, 8.6, True, GREEN if row['buffalo_delta_200'] > 0 else RUST)
    else:
        r = p2.add_run('ID 不适用：该 pair 不在固定 33 对 ID 评估内，不记作 0 分或漏检。')
        set_run_font(r, 8.0, False, MUTED)
    r = p2.add_run('    RL300 参考：')
    set_run_font(r, 7.2, True, MUTED)
    r = p2.add_run(f'Pose {fmt4(row["pose_300"])} ({row["pose_delta_300"]:+.4f})；')
    set_run_font(r, 7.6, False, INK)
    if row['id_applicable']:
        r = p2.add_run(f'glintr100 {fmt4(row["id_300"])} ({row["id_delta_300"]:+.4f})')
        set_run_font(r, 7.6, False, INK)
    else:
        r = p2.add_run('ID 不适用')
        set_run_font(r, 7.6, False, MUTED)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run('原图 A / B DWPose / B RGB GT / Base Qwen / SFT 9520 / RL200；原始 PNG，未裁切或美化。'
                  f'  pair: {row["pair_id"]}')
    set_run_font(r, 6.5, False, RGBColor(0x9A, 0xA2, 0x9D))

# RL300 evolution strip
h1('5. RL300：七个 checkpoint 的进化条带').paragraph_format.page_break_before = True
body('同一组 3 个案例、同一 seed，按 update 0 → 50 → 100 → 150 → 200 → 250 → 300 排开，最右为 GT。'
     '条带用于看“200 步更均衡、300 步更偏 ID”的具体形态：300 步的脸/身份有关节更贴，但 Pose 与 PCK 在部分样本回落。'
     '条带中的指标是该例的 Pose / PCK，不是 50 例均值；这是抽看入口，不是全量统计。', 9.0)
add_figure(ASSETS / 'figures/04_checkpoint_evolution_strip.png',
           19.0, '图 6｜3 个案例的 7-checkpoint 进化条带（A / pose / SFT / RL50–300 / GT）。')

# Appendix A: per-pair tables
h1('附录 A｜50 对逐 pair 明细').paragraph_format.page_break_before = True
body('以下两张表使用 analysis.json 的已核对配对记录，固定 pair_id 一对一连接；ID 不适用的 17 对保持 “—”，'
     '不记作 0。逐例表用于查数值，不替代原图检查。', 9.0)
h2('A.1 Pose、PCK 与分类（RL200 − SFT）')
rows_a1 = []
for index, row in enumerate(gallery):
    rows_a1.append([
        f'{index + 1:02d}', row['clip_id'], row['pair_id'][:8],
        '是' if row['id_applicable'] else '否',
        fmt4(row['pose_base']), fmt4(row['pose_200']), f'{row["pose_delta_200"]:+.4f}',
        fmt_pct(row['pck002_base']), fmt_pct(row['pck002_200']),
        f'{row["pck002_delta_200"] * 100:+.2f} pp', STATUS_LABEL[row['category']],
    ])
data_table(['#', '片段', 'pair', 'ID', 'Pose SFT', 'Pose RL200', 'ΔPose', 'PCK SFT', 'PCK RL200', 'ΔPCK', '分类'],
           rows_a1, [1.0, 3.4, 2.8, 1.3, 2.4, 2.5, 2.2, 2.4, 2.5, 2.2, 2.8], font=7.0,
           row_fills=[WHITE if i % 2 == 0 else 'FBFCFA' for i in range(50)])

h2('A.2 ID 与 RL300（逐 pair）').paragraph_format.page_break_before = True
rows_a2 = []
for index, row in enumerate(gallery):
    if row['id_applicable']:
        id_cells = [fmt4(row['id_base']), fmt4(row['id_200']), f'{row["id_delta_200"]:+.4f}',
                    fmt4(row['buffalo_base']), fmt4(row['buffalo_200']), f'{row["buffalo_delta_200"]:+.4f}',
                    fmt4(row['id_300']), f'{row["id_delta_300"]:+.4f}']
    else:
        id_cells = ['—'] * 8
    rows_a2.append([f'{index + 1:02d}', row['clip_id'], row['pair_id'][:8], *id_cells,
                    f'{row["pose_300"]:.4f} ({row["pose_delta_300"]:+.4f})'])
data_table(['#', '片段', 'pair', 'glintr SFT', 'glintr RL200', 'Δglintr200', 'buffalo SFT', 'buffalo RL200',
            'Δbuffalo200', 'glintr RL300', 'Δglintr300', 'Pose RL300 (Δ)'],
           rows_a2, [1.0, 2.8, 2.2, 2.2, 2.2, 2.0, 2.2, 2.2, 2.0, 2.2, 2.2, 3.0], font=7.0,
           row_fills=[WHITE if i % 2 == 0 else 'FBFCFA' for i in range(50)])
note('RL300 的 PCK / NME / buffalo_l 逐 pair 值、Base Qwen 逐 pair 值与所有原始路径，见同包 '
     'assets/per_pair_metrics.csv 与 assets/base/per_image.json。')

# Portrait appendix B
new2 = doc.add_section(WD_SECTION.NEW_PAGE)
new2.orientation = WD_ORIENT.PORTRAIT
new2.page_width = Cm(21.0)
new2.page_height = Cm(29.7)
new2.top_margin = Cm(2.0)
new2.bottom_margin = Cm(1.9)
new2.left_margin = Cm(2.0)
new2.right_margin = Cm(2.0)
configure_header_footer(new2, first_page_blank=False)

h1('附录 B｜来源、补跑与复现')
h2('B.1 数据来源')
body('本报告的事实数据来自本地独立报告 sft-rl-comparison-20261114-v1 的 analysis.json、'
     'source/update_*/per_image.json、source_manifest.json 与 report.html；'
     '原始训练产物在服务器 /data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1。'
     '报告中的 7 个 checkpoint、50 对验证清单、33 对 ID 适用标记在分析时均重新核对。')
body('Base Qwen 50 张是本次新增的参考推理，输出目录为 runs/20261112_v1/base_qwen_eval_20261114_v1/：'
     'data_subset 固定 50 对；cache/base_cache_20261114_v2 保存条件编码；'
     'output/base_infer_20261114_v2 保存 50 张原始图；eval_base_qwen_20261114_v1 保存同口径指标。'
     '既有 SFT/RL 产物未被覆盖或改写。')
callout('复现命令（服务器）', [
    'BASE=/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1/base_qwen_eval_20261114_v1',
    'QPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python',
    'cd /data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora',
    '"$QPY" scripts/run.py --config "$BASE/configs/base_infer.json" cache --name base_cache_20261114_v2 --data "$BASE/data_subset"',
    '"$QPY" scripts/run.py --config "$BASE/configs/base_infer.json" infer --name base_infer_20261114_v2 --data "$BASE/data_subset" --cache "$BASE/cache/base_cache_20261114_v2" --limit 50',
    'cd "$BASE"',
    'RL_PY=/data/shangguanrz/LoRA/envs/flow-factory-v050/bin/python',
    'PYTHONPATH=/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/source/runtime_20261112_v9 "$RL_PY" eval_base_qwen.py',
], fill=GROUND, border_color='68716C')

h2('B.2 关键口径与边界')
bullet('7 个 checkpoint 使用相同 50 对、相同参考路径、固定 33 个 ID 适用标记；384 × 672、seed 42、'
       '30-step ODE、CFG 1；SFT 基线在同一 RL runtime 的 RL0 重新生成。')
bullet('Pose 只监督 COCO17，没有手指精细指标；NME 只看匹配到的关节，是诊断项。背景、衣物和整体图像质量'
       '没有全量定量保证。')
bullet('5 个 Pose 片段、4 个 ID 片段；50/33 张图不是独立的 50/33 个实验。片段 bootstrap 与 sign-flip '
       '只作小样本描述，不宣称泛化或独立显著性。')
bullet('200 步在同一个 val 上从 7 个 checkpoint 中选出，存在选点偏差；没有同起点、等计算预算的继续 SFT 对照。'
       'test split 未用于本次结论。')
bullet('Base Qwen 补跑只作视觉与量级参考：上游 pipeline 默认 KV cache，RL runtime 为 use_kv_cache=false；'
       '两者采样步数、seed、CFG、画布一致，但不是逐位相同的实现。')

h2('B.3 文件清单')
files = [
    ('report.html / report.css / report.js', '原独立报告；本报告只重排，不修改原文。'),
    ('analysis.json', '按 pair_id 配对复算的 7 checkpoint、分片段、bootstrap 与逐 pair 记录。'),
    ('source/update_000000 … 000300/per_image.json', '7 个 checkpoint 的逐图原始指标。'),
    ('images/0、images/200、images/300', 'SFT、RL200、RL300 的 50 张原始 PNG。'),
    ('context/<pair>/{A,pose,GT}.png', '每对的参考人物、目标骨架、目标帧。'),
    ('assets/base/<pair>.png', '本次补跑的 Base Qwen 50 张。'),
    ('assets/base/base_manifest.json', 'Base Qwen 的来源、协议、逐图 SHA256 与逐图指标。'),
    ('assets/base/summary.json', 'Base Qwen 的同口径汇总指标。'),
    ('assets/quant.json', '本报告表格的聚合数据（从上述文件复算）。'),
    ('assets/per_pair_metrics.csv', '50 行逐 pair 全字段明细，含 Base 与 RL300。'),
]
data_table(['文件', '用途'], [[a, b] for a, b in files], [6.2, 10.8], font=8.0, align_first_left=True)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(18)
r = p.add_run('报告整理：2026-11-14 · 仅重排与复算既有评测；所有生成图保持原始状态。')
set_run_font(r, 7.5, False, MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print('saved', OUT.stat().st_size, 'bytes')
