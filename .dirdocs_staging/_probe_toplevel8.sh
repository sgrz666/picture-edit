#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== deepgen_injection + parameter_policy =="
python3 -c "import json;d=json.load(open('configs/adapter_v6_architecture.json'));print(json.dumps(d.get('deepgen_injection'),ensure_ascii=False)[:700]);print(json.dumps(d.get('parameter_policy'),ensure_ascii=False)[:300]);print('tokens:',json.dumps(d.get('tokens'))[:300])"
echo "== src py per-file lines =="
wc -l src/*/*.py src/pose_control/*.py src/pose_control/*/*.py 2>/dev/null | grep -v 总计 | grep -v __pycache__ | sort -k2
echo "== src py count =="
find src -name '*.py' -not -path '*__pycache__*' | wc -l
echo "== datasets iPER & 3dpw layout =="
ls datasets/iPER; ls datasets/3dpw_single_final_v2 | head -12; ls datasets/champ_motions_example
echo "== results/legacy_demo filetypes =="
ls results/legacy_demo_202609 | awk -F. '{print $NF}' | sort | uniq -c
echo "== scripts/run full ls =="
ls scripts/run scripts/train scripts/weights
echo "== logs/legacy largest =="
ls -S logs/legacy | head -6
echo "== workflow dir exists at top =="
ls workflow | head -8
