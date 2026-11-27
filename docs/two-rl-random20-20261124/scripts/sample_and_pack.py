import json
import os
import random
import shutil
import tarfile
from pathlib import Path

ROOT = Path('/data/shangguanrz/datasets/xiaohognshu(two-rl)')
OUT = Path('/tmp/two_rl_sheet')
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

rows = [json.loads(line) for line in (ROOT / 'pairs/pairs.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
rows = [r for r in rows if r.get('disk_status') == 'complete' and Path(r['source_rgb_path']).exists() and Path(r['target_rgb_path']).exists()]
print('available', len(rows))

random.seed(20261124)
picks = random.sample(rows, 20)

meta = []
for i, r in enumerate(picks, 1):
    a_src = Path(r['source_rgb_path'])
    b_src = Path(r['target_rgb_path'])
    a_dst = OUT / f'{i:02d}_a{a_src.suffix}'
    b_dst = OUT / f'{i:02d}_b{b_src.suffix}'
    shutil.copy2(a_src, a_dst)
    shutil.copy2(b_src, b_dst)
    meta.append({'idx': i, 'pair_id': r['pair_id'], 'note': r['note_id'], 'source_tag': r.get('source_tag'),
                 'a': a_dst.name, 'b': b_dst.name, 'a_name': a_src.name, 'b_name': b_src.name})

(OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf8')
with tarfile.open('/tmp/two_rl_sheet.tar.gz', 'w:gz') as tar:
    tar.add(OUT, arcname='two_rl_sheet')
print('packed', len(meta))
for m in meta:
    print(m['idx'], m['pair_id'], m['source_tag'], m['a'], m['b'])
