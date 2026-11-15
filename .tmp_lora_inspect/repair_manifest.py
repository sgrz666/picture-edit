"""Repair double 'pictures/' prefix in manifest dests, then verify every record."""
import hashlib
import json
import os

WS = '/data/shangguanrz/LoRA'
P = WS + '/pictures'
mp = P + '/manifest.json'
m = json.load(open(mp, encoding='utf-8'))
fixed = 0
for r in m['files']:
    while r['dest'].startswith('pictures/pictures/'):
        r['dest'] = r['dest'][len('pictures/'):]
        fixed += 1
if fixed:
    with open(mp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
        f.write('\n')
print('prefix repairs:', fixed)
bad = []
for r in m['files']:
    p = os.path.join(WS, r['dest'])
    if not os.path.isfile(p):
        bad.append('MISSING ' + r['dest'])
    elif hashlib.sha256(open(p, 'rb').read()).hexdigest() != r['sha256']:
        bad.append('SHA ' + r['dest'])
print('manifest records:', len(m['files']), 'problems:', len(bad), bad[:5])
for d in ('full_20261111_v1/panels', 'pilot_20261111_v1/panels', 'pilot_20261111_v1/probe_inference'):
    print(d, len([f for f in os.listdir(os.path.join(P, d)) if f.endswith('.png')]))
print('index.html bytes:', os.path.getsize(P + '/index.html'))
R = WS + '/experiments/bilibili-dwpose-edit-lora'
for rep in ('report/full_report_20261111_v1', 'report/pilot_report_20261111_v1'):
    rows = [json.loads(l) for l in open(os.path.join(R, rep, 'per_image.jsonl'), encoding='utf-8')]
    none = [(r['method'], r['case'], k) for r in rows
            for k in ('pck_002_diagonal', 'pck_005_diagonal', 'identity_a_cosine') if r[k] is None]
    print(rep, 'None metric cells:', none)
print('DONE')