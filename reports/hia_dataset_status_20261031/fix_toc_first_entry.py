#!/usr/bin/env python3
"""Fuse the TOC field's begin/instrText/separate runs into the first real TOC entry
paragraph and delete the stale field paragraph that carries an empty-PAGEREF
duplicate of the first entry.

Before: <p TOC1>[begin][instr TOC][separate]<hyperlink>一、...(empty PAGEREF)</hyperlink></p>
        <p TOC1><hyperlink>一、...1</hyperlink></p>   <- script-inserted entries
After:  <p TOC1>[begin][instr TOC][separate]<hyperlink>一、...1</hyperlink></p>
"""
import re
import shutil
import sys
import zipfile

src = sys.argv[1]
tmp = src + '.tocfuse'

with zipfile.ZipFile(src, 'r') as zin:
    names = zin.namelist()
    contents = {n: zin.read(n) for n in names}

xml = contents['word/document.xml'].decode('utf-8')

# 1) the paragraph holding the TOC field code
instr_i = xml.find('>TOC \\h')
if instr_i == -1:
    raise SystemExit('TOC instrText not found')
p_start = max(xml.rfind('<w:p ', 0, instr_i), xml.rfind('<w:p>', 0, instr_i))
p_end = xml.find('</w:p>', instr_i) + len('</w:p>')
field_para = xml[p_start:p_end]

m = re.search(r'<w:r><w:fldChar w:fldCharType="begin"[^/]*/></w:r>'
              r'<w:r><w:instrText[^>]*>TOC [^<]*</w:instrText></w:r>'
              r'<w:r><w:fldChar w:fldCharType="separate"/></w:r>', field_para)
if not m:
    raise SystemExit('field control runs not found inside the TOC paragraph')
control_runs = m.group(0)
stale_hyperlink = re.search(r'<w:hyperlink\b.*?</w:hyperlink>', field_para[m.end():])
print('stale entry text:', re.findall(r'<w:t[^>]*>([^<]*)</w:t>', stale_hyperlink.group(0)) if stale_hyperlink else None)

# 2) the next paragraph (first real entry) must directly follow
next_start = xml.find('<w:p', p_end)
next_para_end = xml.find('</w:p>', next_start) + len('</w:p>')
next_para = xml[next_start:next_para_end]
if '一、数据集概览与一页结论' not in next_para:
    raise SystemExit('next paragraph is not the first real TOC entry; aborting')

# 3) rebuild: drop the stale hyperlink from the field paragraph, keep only control runs,
#    and merge everything into the next paragraph right after its pPr.
merged = next_para
ppr_end = merged.find('</w:pPr>')
insert_at = ppr_end + len('</w:pPr>') if ppr_end != -1 else merged.find('>') + 1
merged = merged[:insert_at] + control_runs + merged[insert_at:]

xml = xml[:p_start] + merged + xml[next_para_end:]
# sanity: exactly one TOC instrText, one fewer hyperlink-with-empty-pageref
assert xml.count('>TOC \\h') == 1, 'TOC field count changed'
print('merged field control into first real entry paragraph')

contents['word/document.xml'] = xml.encode('utf-8')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, contents[n])
shutil.move(tmp, src)
print('DONE', src)
