#!/usr/bin/env python3
"""Post-process the generated docx per documents:docx skill toc.md rules:
1. footer instrText: bare PAGE -> PAGE \* ROMAN (front matter) / PAGE \* arabic (body)
2. remove empty <w:pgNumType/> from document.xml (cover section)
Only rewrites the zip entries that change; the source docx is our own generated file.
"""
import re
import shutil
import sys
import zipfile

src = sys.argv[1]
tmp = src + '.patched'

with zipfile.ZipFile(src, 'r') as zin:
    names = zin.namelist()
    footers = sorted(n for n in names if re.match(r'word/footer\d+\.xml', n))
    contents = {n: zin.read(n) for n in names}

# Determine section order in document.xml: sectPr blocks reference footer rIds.
doc_xml = contents['word/document.xml'].decode('utf-8')

# 1) remove empty pgNumType (cover section)
before = doc_xml.count('<w:pgNumType/>')
doc_xml = doc_xml.replace('<w:pgNumType/>', '')
print(f'removed empty pgNumType: {before}')

# Map footers to sections: walk sectPr chunks (split on '<w:sectPr'), read each chunk's
# pgNumType fmt and footerReference r:id.
rels = contents['word/_rels/document.xml.rels'].decode('utf-8')
rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))

sect_footers = []
for chunk in doc_xml.split('<w:sectPr')[1:]:
    m = re.search(r'footerReference[^>]*r:id="(rId\d+)"', chunk)
    fmt = re.search(r'<w:pgNumType[^>]*w:fmt="([^"]+)"', chunk)
    sect_footers.append((rid_to_target.get(m.group(1)) if m else None,
                         fmt.group(1) if fmt else None))
print('sections ->', sect_footers)

for target, fmt in sect_footers:
    if not target:
        continue
    key = 'word/' + target
    xml = contents[key].decode('utf-8')
    switch = 'ROMAN' if fmt == 'upperRoman' else 'arabic'
    xml2, n = re.subn(r'(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)',
                      rf'\1 PAGE \\* {switch} \\* MERGEFORMAT \2', xml)
    if n:
        contents[key] = xml2.encode('utf-8')
        print(f'{target}: patched {n} PAGE field(s) -> \\* {switch}')
    else:
        print(f'{target}: no bare PAGE field found (maybe already patched)')

contents['word/document.xml'] = doc_xml.encode('utf-8')

with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, contents[n])
shutil.move(tmp, src)
print('PATCHED', src)
