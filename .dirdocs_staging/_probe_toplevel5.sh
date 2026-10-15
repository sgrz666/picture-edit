#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== full asset_registry =="
cat configs/asset_registry.json
echo "== adapter_v6 key fields =="
python3 -c "import json;d=json.load(open('configs/adapter_v6_architecture.json'));print(json.dumps({k:d[k] for k in ['version','scope','max_people']},ensure_ascii=False));print('injection:',json.dumps(d.get('condition_injection',{}))[:600])"
echo "== research dirs =="
for d in research/*; do [ -d "$d" ] || continue; echo "-- $d size=$(du -sh --si $d|cut -f1) f=$(find $d -maxdepth 1 -type f|wc -l) d2=$(find $d -mindepth 1 -maxdepth 1 -type d|wc -l)"; ls "$d" | head -12; done
echo "== results deeper =="
for d in results/legacy_demo_202609 results/review_20260908 results/review_20260908_v2; do echo "-- $d"; ls "$d" | head -15; done
echo "== archive depth2 =="
for d in archive/*; do [ -d "$d" ] || continue; echo "-- $d"; ls "$d" | head -15; done
echo "== docs linecounts =="
for f in docs/*.md; do echo "$(wc -l <$f) $f"; done
echo "== tests grouping =="
python3 - <<'PY'
import os,re,collections
c=collections.Counter()
for f in os.listdir('tests'):
    if f.startswith('test_'):
        m=re.match(r'test_(qwen21|qwen21_full|tiktok|v65|v66|v6|iper|hi4d|deepgen|adapter|condition|champ|analysis|lab|report)',f)
        c[m.group(1) if m else 'other']+=1
print(c, 'total py=', sum(1 for f in os.listdir('tests') if f.endswith('.py')))
PY
cat pytest.ini
echo "== scripts root heads =="
for f in scripts/*.py scripts/*.sh scripts/*.txt; do [ -f "$f" ] && echo "-- $f"; head -3 "$f" | grep -v '^$'; done 2>/dev/null
echo "== scripts/run/home =="
ls -la scripts/run/home | head
echo "== experiments count =="
find experiments -mindepth 1 -maxdepth 1 -type d | wc -l
echo "== v6 vs qwen21_full diff sample =="
md5sum src/pose_control/v6/control_core.py src/pose_control/qwen21_full/control_core.py src/pose_control/v6/deepgen_adapter.py src/pose_control/qwen21_full/deepgen_adapter.py src/pose_control/v6/face/adapter.py src/pose_control/qwen21_full/face/adapter.py
