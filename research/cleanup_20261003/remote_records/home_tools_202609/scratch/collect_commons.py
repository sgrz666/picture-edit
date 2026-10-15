#!/usr/bin/env python3
"""Collect HD full-body candidates from Wikimedia Commons for cross-body-type retargeting tests."""
import json
import os
import time
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
UA = 'Mozilla/5.0 (X11; Linux x86_64) pic-edit-research/1.0 (contact: research)'

OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/candidates'

HEAVY = [
    'Category:Obese men',
    'Category:Obese women',
    'Category:Overweight',
    'Category:Ashley Graham (model)',
    'Category:Paloma Elsesser',
    'Category:Fluvia Lacerda',
    'Category:Toccara Jones',
    'Category:Ella Halikas',
    'Category:Simcha Gueta',
]
SLIM = [
    'Category:21st-century portrait photographs of standing women at full length',
    'Category:21st-century portrait photographs of standing men at full length',
    'Category:20th-century portrait photographs of standing women at full length',
    'Category:Catwalks (fashion)',
    'Category:Barbara Palvin',
    'Category:Sofia Mechetner',
]

MAX_PER_CAT = 30
MIN_W = 1100
MIN_H = 1200          # 全身照：高度应显著大于宽度


def api(params, tries=4):
    p = dict(params)
    p['format'] = 'json'
    p['formatversion'] = '2'
    url = API + '?' + urllib.parse.urlencode(p)
    delay = 1.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception:
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return {}


def files_in_category(cat, limit=MAX_PER_CAT):
    r = api({'action': 'query', 'generator': 'categorymembers',
             'gcmtitle': cat, 'gcmtype': 'file', 'gcmlimit': str(limit),
             'prop': 'imageinfo', 'iiprop': 'url|size|mime'})
    out = []
    for page in r.get('query', {}).get('pages', []):
        ii = (page.get('imageinfo') or [{}])[0]
        if not ii.get('url'):
            continue
        if ii.get('mime') not in ('image/jpeg', 'image/png'):
            continue
        out.append({'title': page['title'], 'url': ii['url'],
                    'width': ii.get('width', 0), 'height': ii.get('height', 0)})
    return out


def collect(cats, bucket):
    dest = os.path.join(OUT, bucket)
    os.makedirs(dest, exist_ok=True)
    records = []
    for cat in cats:
        try:
            items = files_in_category(cat)
        except Exception as exc:
            print(f'  [{cat}] FAIL {exc}', flush=True)
            continue
        ok = 0
        for it in items:
            if it['width'] < MIN_W or it['height'] < MIN_H:
                continue
            if it['height'] <= it['width']:      # 要竖版全身
                continue
            name = it['title'].replace('File:', '').replace(' ', '_')
            name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                name += '.jpg'
            path = os.path.join(dest, name)
            if os.path.exists(path):
                ok += 1
                continue
            try:
                req = urllib.request.Request(it['url'], headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                with open(path, 'wb') as fh:
                    fh.write(data)
                records.append({'bucket': bucket, 'category': cat, 'file': name,
                                'width': it['width'], 'height': it['height'],
                                'bytes': len(data), 'source': it['url'],
                                'title': it['title']})
                ok += 1
                print(f'  + {bucket}/{name}  {it["width"]}x{it["height"]}', flush=True)
            except Exception as exc:
                print(f'  ! {bucket}/{name} DL_FAIL {exc}', flush=True)
            time.sleep(0.25)
        print(f'[{bucket}] {cat}: {ok} kept', flush=True)
        time.sleep(0.6)
    with open(os.path.join(dest, '_manifest.json'), 'w') as fh:
        json.dump(records, fh, indent=2)
    print(f'== {bucket}: {len(records)} new files -> {dest}', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('=== HEAVY bucket ===', flush=True)
    collect(HEAVY, 'heavy')
    print('=== SLIM bucket ===', flush=True)
    collect(SLIM, 'slim')
    print('DONE', flush=True)
