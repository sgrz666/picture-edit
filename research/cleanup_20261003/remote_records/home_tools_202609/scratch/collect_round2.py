#!/usr/bin/env python3
"""Round 2: target genuinely full-body heavy-set subjects on Wikimedia Commons."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
UA = 'pic-edit-research-bot/1.0 (research use)'
OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/candidates'
THUMB_W = 1920
MIN_W = 1000
MIN_H = 1100
MAX_PER_CAT = 24
MIN_INTERVAL = 1.1
_last = [0.0]

# 目标：现代、全身、体型显著偏大的人群
CATS = {
    'heavy': [
        'Category:Sumo wrestlers',
        'Category:Professional sumo wrestlers',
        'Category:Strongmen',
        'Category:Heavyweight boxers',
        'Category:Plus-size models',
        'Category:Big Beautiful Women',
        'Category:Fat acceptance movement',
    ],
    'slim': [
        'Category:Full length portraits of women',
        'Category:Full length portraits of men',
        'Category:Ballet dancers',
        'Category:Figure skaters',
    ],
}


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
        except Exception:
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
    raise RuntimeError('exhausted')


def run(bucket, cats):
    dest = os.path.join(OUT, bucket)
    os.makedirs(dest, exist_ok=True)
    added = 0
    for cat in cats:
        try:
            r = api({'action': 'query', 'generator': 'categorymembers',
                     'gcmtitle': cat, 'gcmtype': 'file', 'gcmlimit': str(MAX_PER_CAT),
                     'prop': 'imageinfo', 'iiprop': 'url|size|mime',
                     'iiurlwidth': str(THUMB_W)})
        except Exception as exc:
            print(f'  [{cat}] LIST_FAIL {exc}', flush=True)
            continue
        pages = r.get('query', {}).get('pages', [])
        kept = 0
        for page in pages:
            ii = (page.get('imageinfo') or [{}])[0]
            if ii.get('mime') not in ('image/jpeg', 'image/png'):
                continue
            w, h = ii.get('width', 0), ii.get('height', 0)
            if w < MIN_W or h < MIN_H or h <= w:
                continue
            name = page['title'].replace('File:', '').replace(' ', '_')
            name = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in name)
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                name += '.jpg'
            path = os.path.join(dest, name)
            if os.path.exists(path) and os.path.getsize(path) > 20000:
                kept += 1
                continue
            try:
                data = fetch(ii.get('thumburl') or ii['url'])
            except Exception as exc:
                print(f'  ! {bucket}/{name} FAIL {exc}', flush=True)
                continue
            if len(data) < 20000:
                continue
            with open(path, 'wb') as fh:
                fh.write(data)
            kept += 1
            added += 1
            print(f'  + {bucket}/{name}  {w}x{h}', flush=True)
        print(f'[{bucket}] {cat}: {kept} kept', flush=True)
    print(f'== {bucket}: {added} new', flush=True)


if __name__ == '__main__':
    for bucket, cats in CATS.items():
        print(f'=== {bucket.upper()} ===', flush=True)
        run(bucket, cats)
    print('DONE', flush=True)
