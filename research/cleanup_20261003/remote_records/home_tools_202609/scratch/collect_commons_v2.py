#!/usr/bin/env python3
"""Collect HD full-body candidates from Wikimedia Commons using API thumbnail URLs.

Wikimedia rate-limits direct original-file downloads; the sanctioned path is the
thumbnail URL returned by iiurlwidth. We throttle requests and back off on 429.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
UA = 'pic-edit-research-bot/1.0 (research use; contact: research@example.org)'

OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/candidates'
THUMB_W = 1920
MIN_W = 1000
MIN_H = 1100
MAX_PER_CAT = 24
MIN_INTERVAL = 1.1          # 秒；全局限流
_last = [0.0]

HEAVY = [
    'Category:Obese men',
    'Category:Obese women',
    'Category:Overweight',
    'Category:Ashley Graham (model)',
    'Category:Toccara Jones',
    'Category:Paloma Elsesser',
    'Category:Fluvia Lacerda',
]
SLIM = [
    'Category:21st-century portrait photographs of standing women at full length',
    'Category:21st-century portrait photographs of standing men at full length',
    'Category:Catwalks (fashion)',
    'Category:Barbara Palvin',
]


def throttle():
    gap = time.time() - _last[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    _last[0] = time.time()


def api(params, tries=5):
    p = dict(params)
    p['format'] = 'json'
    p['formatversion'] = '2'
    url = API + '?' + urllib.parse.urlencode(p)
    delay = 2.0
    for attempt in range(tries):
        throttle()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as exc:
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return {}


def fetch(url, tries=5):
    delay = 3.0
    for attempt in range(tries):
        throttle()
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': UA,
                'Referer': 'https://commons.wikimedia.org/',
                'Accept': 'image/avif,image/webp,image/*,*/*;q=0.8',
            })
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
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
    raise RuntimeError('fetch exhausted')


def files_in_category(cat):
    r = api({'action': 'query', 'generator': 'categorymembers',
             'gcmtitle': cat, 'gcmtype': 'file', 'gcmlimit': str(MAX_PER_CAT),
             'prop': 'imageinfo', 'iiprop': 'url|size|mime',
             'iiurlwidth': str(THUMB_W)})
    out = []
    for page in r.get('query', {}).get('pages', []):
        ii = (page.get('imageinfo') or [{}])[0]
        if ii.get('mime') not in ('image/jpeg', 'image/png'):
            continue
        out.append({'title': page['title'],
                    'thumb': ii.get('thumburl') or ii.get('url'),
                    'url': ii.get('url'),
                    'width': ii.get('width', 0),
                    'height': ii.get('height', 0)})
    return out


def collect(cats, bucket):
    dest = os.path.join(OUT, bucket)
    os.makedirs(dest, exist_ok=True)
    records = []
    for cat in cats:
        try:
            items = files_in_category(cat)
        except Exception as exc:
            print(f'  [{cat}] LIST_FAIL {exc}', flush=True)
            continue
        kept = 0
        for it in items:
            if it['width'] < MIN_W or it['height'] < MIN_H:
                continue
            if it['height'] <= it['width']:
                continue
            name = it['title'].replace('File:', '').replace(' ', '_')
            name = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in name)
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                name += '.jpg'
            path = os.path.join(dest, name)
            if os.path.exists(path) and os.path.getsize(path) > 20000:
                kept += 1
                continue
            try:
                data = fetch(it['thumb'])
            except Exception as exc:
                print(f'  ! {bucket}/{name} DL_FAIL {exc}', flush=True)
                continue
            if len(data) < 20000:
                print(f'  ~ {bucket}/{name} too small ({len(data)}B)', flush=True)
                continue
            with open(path, 'wb') as fh:
                fh.write(data)
            records.append({'bucket': bucket, 'category': cat, 'file': name,
                            'width': it['width'], 'height': it['height'],
                            'bytes': len(data), 'thumb_url': it['thumb'],
                            'original_url': it['url'], 'title': it['title']})
            kept += 1
            print(f'  + {bucket}/{name}  {it["width"]}x{it["height"]}  {len(data)//1024}KB', flush=True)
        print(f'[{bucket}] {cat}: {kept} kept', flush=True)
    mf = os.path.join(dest, '_manifest.json')
    old = []
    if os.path.exists(mf):
        try:
            old = json.load(open(mf))
        except Exception:
            old = []
    seen = {r['file'] for r in records}
    merged = records + [r for r in old if r['file'] not in seen]
    with open(mf, 'w') as fh:
        json.dump(merged, fh, indent=2)
    print(f'== {bucket}: {len(records)} new / {len(merged)} total', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('=== HEAVY ===', flush=True)
    collect(HEAVY, 'heavy')
    print('=== SLIM ===', flush=True)
    collect(SLIM, 'slim')
    print('DONE', flush=True)
