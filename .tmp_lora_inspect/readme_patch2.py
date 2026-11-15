"""Update the pictures bullet of the LoRA workspace README (own earlier text only)."""
import hashlib

p = '/data/shangguanrz/LoRA/README.md'
s = open(p, encoding='utf-8').read()
old = '`manifest.json` 逐张记录源路径、字节数与 sha256，121/121 校验通过。'
new = ('`manifest.json` 逐张记录源路径、字节数与 sha256，126/126 校验通过；swap_pose/blank_pose 与探针的 '
       'panel 为后补派生图，按实际喂入的条件重建。`index.html` 是带逐 case 指标的浏览索引。')
assert s.count(old) == 1, s.count(old)
s2 = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s2)
back = open(p, encoding='utf-8').read()
assert back == s2
print('README %d -> %d bytes, prefix sha %s' % (len(s), len(back), hashlib.sha256(s.encode()).hexdigest()[:16]))
print('DONE')