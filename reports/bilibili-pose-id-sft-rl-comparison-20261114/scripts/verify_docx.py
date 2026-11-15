#!/usr/bin/env python3
"""Verify the generated Word report: image order, pair count, key numbers."""
import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path('D:/codeplus/pictureedit/reports/bilibili-pose-id-sft-rl-comparison-20261114')
REPORT = Path('D:/temp/codex/worktrees/native-spatial-weights/pictureedit/experiments/'
              'bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1')
DOCX = ROOT / 'Bilibili_Pose_ID_SFT_RL_对比报告_20261114_v1.docx'
quant = json.loads((ROOT / 'assets/quant.json').read_text(encoding='utf-8'))
pairs = {row['pair_id']: row for row in quant['pairs']}

doc = Document(DOCX)
print('sections', len(doc.sections), 'tables', len(doc.tables), 'shapes', len(doc.inline_shapes),
      'size_MB', round(DOCX.stat().st_size / 2**20, 1))

# Hash helper for embedded image parts.
def part_hash(part):
    return hashlib.sha256(part.blob).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Map expected source hashes to labels for useful errors.
expected_cache = {}
for pair_id in pairs:
    for label, path in (
        ('A', REPORT / f'context/{pair_id}/A.png'),
        ('pose', REPORT / f'context/{pair_id}/pose.png'),
        ('GT', REPORT / f'context/{pair_id}/GT.png'),
        ('base', ROOT / f'assets/base/{pair_id}.png'),
        ('sft', REPORT / f'images/0/{pair_id}.png'),
        ('rl200', REPORT / f'images/200/{pair_id}.png'),
    ):
        expected_cache[(pair_id, label)] = file_hash(path)

body = list(doc.element.body.iterchildren())
case_count = 0
image_matches = 0
errors = []
seen_pairs = set()
for index, element in enumerate(body):
    if element.tag != qn('w:p'):
        continue
    text = ''.join(node.text or '' for node in element.iter(qn('w:t')))
    match = re.match(r'例 (\d+) / 50\s+(\S+)\s+', text)
    if not match:
        continue
    number = int(match.group(1))
    clip = match.group(2)
    # The next body element must be the six-column table.
    table_el = body[index + 1] if index + 1 < len(body) else None
    if table_el is None or table_el.tag != qn('w:tbl'):
        errors.append(f'case {number}: no table')
        continue
    # Find the matching pair by case order in quant (gallery is grouped by clip).
    candidates = [row for row in quant['pairs'] if row['clip_id'] == clip]
    # Use the case number to select from the grouped gallery order.
    gallery = []
    clip_order = []
    for row in quant['pairs']:
        if row['clip_id'] not in clip_order:
            clip_order.append(row['clip_id'])
    for c in clip_order:
        gallery.extend([row for row in quant['pairs'] if row['clip_id'] == c])
    expected_pair = gallery[number - 1]
    if expected_pair['clip_id'] != clip:
        errors.append(f'case {number}: clip order mismatch {clip} vs {expected_pair["clip_id"]}')
    if expected_pair['pair_id'] in seen_pairs:
        errors.append(f'case {number}: duplicate pair')
    seen_pairs.add(expected_pair['pair_id'])
    cells = table_el.findall('.//' + qn('w:tc'))
    if len(cells) != 6:
        errors.append(f'case {number}: {len(cells)} cells')
        continue
    labels = ['A', 'pose', 'GT', 'base', 'sft', 'rl200']
    for cell, label in zip(cells, labels):
        blips = cell.findall('.//' + qn('a:blip'))
        if len(blips) != 1:
            errors.append(f'case {number}/{label}: {len(blips)} blips')
            continue
        r_id = blips[0].get(qn('r:embed'))
        part = doc.part.related_parts[r_id]
        actual = part_hash(part)
        expected = expected_cache[(expected_pair['pair_id'], label)]
        if actual != expected:
            errors.append(f'case {number}/{label}: hash mismatch')
        else:
            image_matches += 1
    case_count += 1

print('cases', case_count, 'image_matches', image_matches, 'expected', 300)
print('errors', len(errors))
for error in errors[:20]:
    print(' ', error)

# Key metric text checks.
all_text = '\n'.join(''.join(node.text or '' for node in p._p.iter(qn('w:t'))) for p in doc.paragraphs)
checks = [
    ('Pose +0.0373', '+0.0373' in all_text),
    ('PCK +3.35pp', '+3.35 pp' in all_text),
    ('glintr +0.0499', '+0.0499' in all_text),
    ('base pose 0.7085', '0.7085' in all_text),
    ('base glintr 0.6561', '0.6561' in all_text),
]
print('text_checks', checks)
print('pdf_pages', end=' ')
try:
    import fitz
    pdf = fitz.open(ROOT / 'Bilibili_Pose_ID_SFT_RL_对比报告_20261114_v1.pdf')
    print(pdf.page_count)
except Exception as exc:
    print('unavailable', exc)
