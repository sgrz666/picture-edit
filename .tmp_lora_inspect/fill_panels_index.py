"""Fill panel gaps in LoRA/pictures and build a browsable index.

Adds five-column panels for the cases panels() never covered (swap_pose,
blank_pose diagnostics of both runs, and the 1-step probe), using the ACTUAL
requested conditions (cache_row) rather than original_row so the swap panel
shows the skeleton that was really fed. Then writes pictures/index.html and
extends pictures/manifest.json. Copy/derive only; never overwrites.
"""
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

WS = Path('/data/shangguanrz/LoRA')
EXP = WS / 'experiments/bilibili-dwpose-edit-lora'
PIC = WS / 'pictures'
SIZE = (384, 672)
LABELS = ['A', 'B DWPose (requested)', 'B RGB (requested GT)', 'Base Qwen', 'Edit LoRA']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def jl(p):
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]


def build_panel(row_b, row_l, dest):
    assert not dest.exists(), dest
    cr = row_b['cache_row']
    paths = [*cr['condition_images'], cr['target_image'], row_b['image'], row_l['image']]
    panel = Image.new('RGB', (SIZE[0] * 5, SIZE[1] + 32), 'white')
    draw = ImageDraw.Draw(panel)
    for i, (p, label) in enumerate(zip(paths, LABELS)):
        with Image.open(p) as im:
            panel.paste(im.convert('RGB'), (i * SIZE[0], 32))
        draw.text((i * SIZE[0] + 5, 8), label, fill='black')
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.save(dest)
    return dest


added = []
runs = [('full_20261111_v1', 'output/full_comparison_20261111_v1', 'report/full_report_20261111_v1'),
        ('pilot_20261111_v1', 'output/comparison_20261111_v1', 'report/pilot_report_20261111_v1')]
metrics = {}
for run, comp, rep in runs:
    rows = jl(EXP / comp / 'outputs.jsonl')
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case'], {})[r['method']] = r
    for r in jl(EXP / rep / 'per_image.jsonl'):
        metrics[(run, r['method'], r['case'])] = r
    for case in ('swap_pose', 'blank_pose'):
        m = by_case[case]
        split = m['baseline']['original_row']['split']
        dest = PIC / run / 'panels' / ('%s_%s_comparison.png' % (case, split))
        if dest.exists():
            print('exists', dest)
            continue
        build_panel(m['baseline'], m['lora'], dest)
        added.append((run, 'panel_derived', dest, [m['baseline']['image'], m['lora']['image']]))
        print('built', dest)

probe = jl(EXP / 'output/probe_inference_20261111_v1/outputs.jsonl')
pm = {r['method']: r for r in probe}
dest = PIC / 'pilot_20261111_v1/probe_inference/probe_normal_00_comparison.png'
if dest.exists():
    print('exists', dest)
else:
    build_panel(pm['baseline'], pm['lora'], dest)
    added.append(('pilot_20261111_v1', 'panel_derived', dest, [pm['baseline']['image'], pm['lora']['image']]))
    print('built', dest)

# extend manifest
mp = PIC / 'manifest.json'
man = json.load(open(mp, encoding='utf-8'))
have = {r['dest'] for r in man['files']}
for run, kind, dest, sources in added:
    rel = dest.relative_to(WS).as_posix()
    assert rel not in have, rel
    man['files'].append({'dest': rel, 'source': None, 'derived_from': sources, 'run': run,
                         'kind': kind, 'method': None, 'bytes': dest.stat().st_size,
                         'sha256': sha(dest), 'verified': True})
man['layout']['<run>/panels/'] = ('5-column panels (A | requested B skeleton | requested B GT | base | LoRA); '
                                 'normal cases from the report, swap_pose/blank_pose derived here from the '
                                 'actual requested conditions')
man['layout']['index.html'] = 'browsable contact sheet of every panel with per-case metrics'
with open(mp, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(man, f, ensure_ascii=False, indent=2)
    f.write('\n')
print('manifest now', len(man['files']), 'records')


def caption(run, case, split):
    b = metrics.get((run, 'baseline', case))
    l = metrics.get((run, 'lora', case))
    if not b or not l:
        return case

    def fmt(v):
        return 'n/a' if v is None else '%.3f' % v
    note = '（诊断: 对原 B 计分）' if case in ('swap_pose', 'blank_pose') else ''
    return ('%s [%s]%s  PCK@.02 base %s / LoRA %s  PCK@.05 %s / %s  FaceID %s / %s'
            % (case, split, note, fmt(b['pck_002_diagonal']), fmt(l['pck_002_diagonal']),
               fmt(b['pck_005_diagonal']), fmt(l['pck_005_diagonal']),
               fmt(b['identity_a_cosine']), fmt(l['identity_a_cosine'])))


cards = []
for run, comp, rep in runs:
    rows = jl(EXP / comp / 'outputs.jsonl')
    cases = []
    seen = set()
    for r in rows:
        if r['case'] not in seen:
            seen.add(r['case'])
            cases.append((r['case'], r['original_row']['split']))
    cards.append('<h2>%s</h2><div class="grid">' % run)
    for case, split in cases:
        if case in ('swap_pose', 'blank_pose'):
            rel = '%s/panels/%s_%s_comparison.png' % (run, case, split)
        else:
            rel = '%s/panels/%s_%s_comparison.png' % (run, case, split)
        cards.append('<figure><a href="%s"><img loading="lazy" src="%s" alt="%s"></a>'
                     '<figcaption>%s</figcaption></figure>' % (rel, rel, case, caption(run, case, split)))
    cards.append('</div>')
cards.append('<h2>pilot probe (1 step)</h2><div class="grid">')
rel = 'pilot_20261111_v1/probe_inference/probe_normal_00_comparison.png'
cards.append('<figure><a href="%s"><img loading="lazy" src="%s" alt="probe"></a>'
             '<figcaption>probe normal_00: 1-step pilot, native LoRA reload check</figcaption></figure></div>')
cards.append('<h2>test_20261112_v1（运行中）</h2><p>全量 1146 对 test 正在串行门后等待 GPU；'
             '完成后每对一张 panel，归档到 test_20261112_v1/panels/，本页届时更新。</p>')

html = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA 生图对照 panel 索引</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:20px} h2{font-size:16px;margin:28px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
figure{margin:0;background:#fff;border:1px solid #ddd;padding:8px}
img{width:100%;display:block;background:#eee}
figcaption{font-size:12px;margin-top:6px;color:#444;word-break:break-all}
a{color:inherit}
</style></head><body>
<h1>LoRA 生图对照 panel 索引</h1>
<p>列序：A | 请求的 B 骨架 | 请求的 B GT | Base Qwen | Edit LoRA。正常 case 的 panel 来自实验报告；
swap_pose / blank_pose 为诊断，按实际喂入的条件重建，计分仍对原 B（只作诊断）。
full = 9520 步 checkpoint-009520；pilot = 300 步 checkpoint-000300。</p>
__CARDS__
</body></html>
"""
html = html.replace('__CARDS__', '\n'.join(cards))
(PIC / 'index.html').write_text(html, encoding='utf-8')
print('index.html written', (PIC / 'index.html').stat().st_size, 'bytes')
print('DONE')