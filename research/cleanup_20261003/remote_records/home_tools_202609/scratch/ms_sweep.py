#!/usr/bin/env python3
"""Sweep ModelScope dataset search for anything usable for single-person pose retargeting."""
import json
import time
import urllib.parse
import urllib.request

API = 'https://www.modelscope.cn/api/v1/dolphin/datasets'
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36'

QUERIES = [
    ('pose transfer', 'EN'), ('human pose', 'EN'), ('person image', 'EN'),
    ('pose', 'EN'), ('body', 'EN'), ('human', 'EN'),
    ('DeepFashion', 'EN'), ('Market1501', 'EN'), ('Market-1501', 'EN'),
    ('iPER', 'EN'), ('SMPL', 'EN'), ('Human3.6M', 'EN'), ('COCO', 'EN'),
    ('人体姿态', 'ZH'), ('人体', 'ZH'), ('姿态迁移', 'ZH'), ('体型', 'ZH'),
    ('换装', 'ZH'), ('虚拟试衣', 'ZH'), ('重定向', 'ZH'),
]


def search(q, size=20):
    url = API + '?' + urllib.parse.urlencode({'PageSize': size, 'PageNumber': 1, 'Query': q})
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode('utf-8'))


seen = {}
for q, tag in QUERIES:
    try:
        d = search(q)
    except Exception as exc:
        print(f'[{q}] FAIL {type(exc).__name__}')
        continue
    rows = d.get('Data') or []
    hits = 0
    for it in rows:
        full = f"{it.get('Namespace')}/{it.get('Name')}"
        if full in seen:
            continue
        seen[full] = it
        hits += 1
    print(f'[{q:14s}] {len(rows)} 返回, {hits} 新增')
    time.sleep(0.4)

print(f'\n=== 去重后共 {len(seen)} 个数据集 ===\n')
KEY = ('pose', 'person', 'human', 'body', 'deepfashion', 'market', 'iper', 'smpl',
       'coco', '人体', '姿态', '体型', '换装', '试衣', '重定向', '动作')
out = []
for full, it in seen.items():
    blob = (full + ' ' + str(it.get('ChineseName', '')) + ' ' + str(it.get('Description', ''))).lower()
    if any(k.lower() in blob for k in KEY):
        out.append((it.get('DownloadCount') or 0, full,
                    it.get('ChineseName') or '',
                    (it.get('License') or '')[:26],
                    (it.get('Description') or '')[:90]))
out.sort(reverse=True)
for cnt, full, cn, lic, desc in out[:45]:
    print(f'  {full:52s} | {lic:26s} | {cn[:26]}')
    if desc:
        print(f'      {desc}')
