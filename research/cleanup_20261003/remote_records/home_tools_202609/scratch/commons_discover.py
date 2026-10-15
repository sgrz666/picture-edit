import json
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
UA = 'Mozilla/5.0 (X11; Linux x86_64) pic-edit-research/1.0'


def api(params):
    params = dict(params)
    params['format'] = 'json'
    params['formatversion'] = '2'
    url = API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


KEYWORDS = [
    'plus size model', 'plus-size fashion', 'overweight person', 'obese person',
    'full-length portrait', 'full body portrait', 'fashion model standing',
    'runway model', 'body positivity', 'curvy model', 'heavyweight person',
    'slim model', 'thin person full body', 'tall man standing full body',
]

print('=' * 72)
print('Commons 分类发现（namespace 14 = Category）')
print('=' * 72)
seen = set()
for kw in KEYWORDS:
    try:
        r = api({'action': 'query', 'list': 'search', 'srsearch': kw,
                 'srnamespace': '14', 'srlimit': '8'})
    except Exception as exc:
        print(f'[{kw}] ERROR {exc}')
        continue
    hits = r.get('query', {}).get('search', [])
    print(f'\n--- 关键词: {kw}  ({len(hits)} 命中) ---')
    for h in hits:
        t = h['title']
        if t in seen:
            continue
        seen.add(t)
        print('   ', t)

print()
print('=' * 72)
print('Category:Plus-size models 的直接子分类')
print('=' * 72)
try:
    r = api({'action': 'query', 'list': 'categorymembers',
             'cmtitle': 'Category:Plus-size models', 'cmtype': 'subcat',
             'cmlimit': '60'})
    for m in r.get('query', {}).get('categorymembers', []):
        print('   ', m['title'])
except Exception as exc:
    print('ERROR', exc)

print()
print('=' * 72)
print('Category:Plus-size models 的直接文件')
print('=' * 72)
try:
    r = api({'action': 'query', 'list': 'categorymembers',
             'cmtitle': 'Category:Plus-size models', 'cmtype': 'file',
             'cmlimit': '60'})
    for m in r.get('query', {}).get('categorymembers', []):
        print('   ', m['title'])
except Exception as exc:
    print('ERROR', exc)
