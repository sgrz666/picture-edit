#!/usr/bin/env bash
# Verify downloaded Qwen-Image-2.1 weights against HuggingFace-published LFS sha256 (lfs.oid)
set -u
cd /home/shangguanrz || exit 1
PY=/home/shangguanrz/miniconda3/envs/deepgen/bin/python
"$PY" - <<'PYEOF'
import json, hashlib, os

D = '/data/shangguanrz/pic-edit-models/Qwen-Image-2.1'
meta = '/tmp/qwen21.json'
d = json.load(open(meta))
print(f"repo   : {d['id']}")
print(f"sha    : {d['sha']}")
print("-" * 66)

ok = bad = plain = 0
tot = 0
for s in d['siblings']:
    rel = s['rfilename']
    p = os.path.join(D, rel)
    lfs = s.get('lfs')
    if not os.path.isfile(p):
        print(f"MISSING   {rel}")
        bad += 1
        continue
    st = os.path.getsize(p)
    tot += st
    if not lfs:
        print(f"plain     {rel:60s} {st:>12,}")
        plain += 1
        continue
    want = lfs['sha256']
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    got = h.hexdigest()
    good = (got == want) and (st == lfs['size'])
    print(f"{'OK' if good else 'MISMATCH':9s} {rel:60s} {st:>12,}")
    if good:
        ok += 1
    else:
        bad += 1
        if got != want:
            print(f"          want={want}")
            print(f"          got ={got}")
print("-" * 66)
print(f"LFS verified : {ok}")
print(f"MISMATCH     : {bad}")
print(f"non-LFS files: {plain}")
print(f"TOTAL bytes  : {tot:,} = {tot/1e9:.3f} GB = {tot/2**30:.3f} GiB")
print("RESULT:", "ALL_OK" if bad == 0 else "FAILED")
PYEOF
