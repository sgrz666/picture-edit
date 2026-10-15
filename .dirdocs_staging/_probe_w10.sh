cd /home/shangguanrz/project/pic-edit || exit 1
PY=~/miniconda3/envs/deepgen/bin/python
echo "=== outputs git status ==="
git status --short outputs/ | head; ls -la outputs/
find / -maxdepth 6 -name 'adapter_v6_smoke_dual_champ.json' 2>/dev/null | head -3
echo "=== .pt under data ==="
find data -name '*.pt' -printf '%s %TY-%Tm-%Td %p\n' | head
echo "=== npy shapes (deepgen env) ==="
$PY - <<'PY'
import numpy as np, json
a=np.load('/home/shangguanrz/project/pic-edit/data/debug_samples/controls/sample_000_control8ch.npy')
print('debug ctrl', a.shape, a.dtype, float(a.min()), float(a.max()))
b=np.load('/home/shangguanrz/project/pic-edit/data/iper_debug_001/controls/control_0100.npy')
print('iper ctrl', b.shape, b.dtype, float(b.min()), float(b.max()))
from PIL import Image
for p in ('data/iper_debug_001/images/src_0000.jpg','data/iper_debug_001/images/tgt_0100.jpg'):
    im=Image.open('/home/shangguanrz/project/pic-edit/'+p); print(p, im.size)
j=json.load(open('/home/shangguanrz/project/pic-edit/data/debug_samples/pairs.json'))
print('pairs count', len(j), 'two_person', sum(1 for x in j if x.get('is_two_person')))
print('prompts', [x['prompt'] for x in j][:16])
PY
echo "=== inputs usage grep ==="
grep -rIn --include='*.py' --include='*.sh' --include='*.json' --include='*.md' -e 'person1.jpg' -e 'dual_person.jpg' src scripts workflow configs tests deepgen 2>/dev/null | head -25
echo "=== champ_overfit usage ==="
grep -rIn --include='*.py' --include='*.sh' -e 'champ_overfit' src scripts workflow configs tests 2>/dev/null | head -15
grep -rIn -e 'iper_debug_001' --include='*.py' --include='*.sh' src scripts workflow configs tests 2>/dev/null | head -10
grep -rIn -e 'debug_samples' --include='*.py' --include='*.sh' src scripts workflow configs tests 2>/dev/null | head -10
