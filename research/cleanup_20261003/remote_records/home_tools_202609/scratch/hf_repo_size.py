import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/qwen21.json'
d = json.load(open(path))
sib = d.get('siblings', [])
rows = []
tot = 0
for s in sib:
    sz = s.get('size') or 0
    tot += sz
    rows.append((sz, s['rfilename']))
rows.sort(reverse=True)
print(f"repo_id   : {d.get('id')}")
print(f"sha       : {d.get('sha')}")
print(f"pipeline  : {d.get('config', {}).get('diffusers', {}).get('_class_name')}")
print(f"git-lfs bytes reported: {d.get('usedStorage', 'n/a')}")
print("-" * 62)
for sz, n in rows:
    if sz:
        print(f"{sz/1e9:9.3f} GB   {n}")
    else:
        print(f"{'  (lfs?)':>9}      {n}")
print("-" * 62)
print(f"FILES = {len(sib)}")
print(f"TOTAL (sum of size fields) = {tot/1e9:.2f} GB = {tot/2**30:.2f} GiB")
