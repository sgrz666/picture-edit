#!/usr/bin/env python3
"""Remove the docx-js TOC field's cached stub entry (the single empty-pageref line
between the 'separate' and 'end' fldChars) so it does not duplicate the first
literal placeholder entry inserted by add_toc_placeholders.py."""
import re
import shutil
import sys
import zipfile

src = sys.argv[1]
tmp = src + '.tocfix'

with zipfile.ZipFile(src, 'r') as zin:
    names = zin.namelist()
    contents = {n: zin.read(n) for n in names}

xml = contents['word/document.xml'].decode('utf-8')
sep = xml.find('w:fldCharType="separate"')
end = xml.find('w:fldCharType="end"', sep)
if sep == -1 or end == -1:
    raise SystemExit('TOC field markers not found')
seg = xml[sep:end]
paras = re.findall(r'<w:p\b.*?</w:p>', seg, re.S)
print(f'field result: {len(paras)} cached paragraph(s)')
if len(paras) == 1:
    stub = paras[0]
    if 'PAGEREF' in stub or '<w:t' in stub:
        new_seg = seg.replace(stub, '')
        xml = xml[:sep] + new_seg + xml[end:]
        print('removed cached stub entry:', re.findall(r'<w:t[^>]*>([^<]*)</w:t>', stub))
    else:
        print('cached segment has no text entry; nothing to do')
else:
    print('unexpected cached structure; leaving as-is')

contents['word/document.xml'] = xml.encode('utf-8')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, contents[n])
shutil.move(tmp, src)
print('DONE', src)
