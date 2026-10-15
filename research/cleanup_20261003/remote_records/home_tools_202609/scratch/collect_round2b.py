#!/usr/bin/env python3
"""Round 2 (fixed): pick a thumbnail width strictly below the original so we never
fall back to the rate-limited original-file URL."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
UA = 'pic-edit-research-bot/1.0 (research use)'
OUT = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data/candidates'
ALLOWED = (2560, 1920, 1280, 1024, 800, 640, 320)
MIN_INTERVAL = 1.6
_last = [0.0]

CATS = {
    'heavy': [
        'Category:Sumo wrestlers',
        'Category:Professional sumo wrestlers',
        'Category:Strongmen',
        'Category:Heavyweight boxers',
        'Category:Plus-size models',
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
    delay = 3.0
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


def thumb_url(original, width):
    """https://upload.wikimedia.org/wikipedia/commons/a/ab/F.jpg
       -> .../wikipedia/commons/thumb/a/ab/F.jpg/1280px-F.jpg"""
    marker = '/wikipedia/commons/'
    if marker not in original:
        return None
    head, tail = original.split(marker, 1)
    fname = tail.rsplit('/', 1)[-1]
    return f'{head}{marker}thumb/{tail}/{width}px-{fname}'


def pick_width(w):
    for s in ALLOWED:
        if s < w:
            return s
    return None


def fetch(url, tries=5):
    delay = 3.0
    for attempt in range(tries):
        throttle()
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': UA, 'Referer': 'https://commons.wikimedia.org/'})
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
                     'gcmtitle': cat, 'gcmtype': 'file', 'gcmlimit': '24',
                     'prop': 'imageinfo', 'iiprop': 'url|size|mime'})
        except Exception as exc:
            print(f'  [{cat}] LIST_FAIL {exc}', flush=True)
            continue
        kept = 0
        for page in r.get('query', {}).get('pages', []):
            ii = (page.get('imageinfo') or [{}])[0]
            if ii.get('mime') not in ('image/jpeg', 'image/png'):
                continue
            w, h = ii.get('width', 0), ii.get('height', 0)
            if w < 900 or h < 1000 or h <= w:
                continue
            tw = pick_width(w)
            if not tw:
                continue
            tu = thumb_url(ii['url'], tw)
            if not tu:
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
                data = fetch(tu)
            except Exception as exc:
                print(f'  ! {bucket}/{name} FAIL {type(exc).__name__}', flush=True)
                continue
            if len(data) < 20000:
                continue
            with open(path, 'wb') as fh:
                fh.write(data)
            kept += 1
            added += 1
            print(f'  + {bucket}/{name}  orig {w}x{h} thumb {tw}px  {len(data)//1024}KB', flush=True)
        print(f'[{bucket}] {cat}: {kept} kept', flush=True)
    print(f'== {bucket}: {added} new', flush=True)


if __name__ == '__main__':
    for bucket, cats in CATS.items():
        print(f'=== {bucket.upper()} ===', flush=True)
        try:
            run(bucket, cats)
        except Exception as exc:
            print(f'!! {bucket} aborted: {exc}', flush=True)
    print('DONE', flush=True)
