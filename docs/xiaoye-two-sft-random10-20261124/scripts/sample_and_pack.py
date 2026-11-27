import json
import random
import shutil
import tarfile
from pathlib import Path

ROOT = Path('/data/shangguanrz/xiaohongshu/author/3.小野一舟/two/pairs-final/sft')
OUT = Path('/tmp/xiaoye_sheet')
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
valid = []
for d in dirs:
    a = sorted(d.glob('a_*.jpg'))
    b = sorted(d.glob('b_*.jpg'))
    if len(a) == 1 and len(b) == 1:
        valid.append((d, a[0], b[0]))
print('valid pairs', len(valid))

random.seed(20261124)
picks = random.sample(valid, 10)

meta = []
for i, (d, a_src, b_src) in enumerate(picks, 1):
    a_dst = OUT / f'{i:02d}_a.jpg'
    b_dst = OUT / f'{i:02d}_b.jpg'
    shutil.copy2(a_src, a_dst)
    shutil.copy2(b_src, b_dst)
    meta.append({'idx': i, 'pair_id': d.name, 'source_tag': 'xiaoye-two-sft',
                 'a': a_dst.name, 'b': b_dst.name, 'a_name': a_src.name, 'b_name': b_src.name})

(OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf8')
with tarfile.open('/tmp/xiaoye_sheet.tar.gz', 'w:gz') as tar:
    tar.add(OUT, arcname='xiaoye_sheet')
print('packed', len(meta))
for m in meta:
    print(m['idx'], m['pair_id'], m['a_name'], m['b_name'])
