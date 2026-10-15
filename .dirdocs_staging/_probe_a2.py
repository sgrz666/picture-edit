import csv, collections, json, os
V='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20261003/experiments'
V='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2'
rows=list(csv.DictReader(open(os.path.join(V,'summary.csv'),encoding='utf-8-sig')))
print('rows',len(rows))
print('kinds',collections.Counter(r['kind'] for r in rows))
print('status',collections.Counter(r['status'] for r in rows))
print('group',collections.Counter(r['group'] for r in rows))
print('resolution',collections.Counter(r['resolution'] for r in rows))
print('steps',collections.Counter(r['steps'] for r in rows))
print('cfg',collections.Counter(r['cfg'] for r in rows))
print('preprocess',collections.Counter(r['preprocess'] for r in rows))
print('reviewer',collections.Counter(r['reviewer'] for r in rows))
# peak nvml by resolution for training runs
d=collections.defaultdict(list)
for r in rows:
    if r['peak_nvml_gib']:
        d[(r['kind'],r['resolution'])].append(float(r['peak_nvml_gib']))
for k in sorted(d):
    v=d[k]; print(k,'n=%d max=%.2f min=%.2f'%(len(v),max(v),min(v)))
q=[r for r in rows if r['kind']=='quality']
print('quality rows',len(q),'with scores',sum(1 for r in q if r['action']))
sc=[r for r in rows if r['action']]
print('scored runs',len(sc))
for r in sc[:8]:
    print('  ',r['id'],'action',r['action'],'identity',r['identity'],'clothing',r['clothing'],'bg',r['background'],'anatomy',r['anatomy'],'added',r['added_person'],'note',r['note'][:60])
print('--- scores.json ---')
s=json.load(open(os.path.join(V,'scores.json')))
print(type(s), (list(s.keys())[:8] if isinstance(s,dict) else len(s)))
print(json.dumps(s if not isinstance(s,dict) else {k:s[k] for k in list(s)[:2]},ensure_ascii=False)[:1200])
print('--- dirs ---')
for x in ['cache','review_sheets','configs','logs']:
    p=os.path.join(V,x); ch=sorted(os.listdir(p))
    print(x,len(ch),ch[:6])
