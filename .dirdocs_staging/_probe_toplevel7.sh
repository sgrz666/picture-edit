#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== research json samples =="
python3 - <<'PY'
import json
def show(p, n=350):
    try:
        d=json.load(open(p)); print("--",p, str(d)[:n])
    except Exception as e: print("--",p,"ERR",e)
show('research/tiktok_training_20261003/readiness.json')
show('research/smplx_condition_audit_20261008/body_inputs.json')
show('research/bfs_transfer_audit_20261004/results.json')
show('research/legacy_202609/checkpoint_audit.json')
PY
echo "== research qwen21_dual_upgrade dirs =="
ls research/qwen21_dual_upgrade_20261011 | head -25; ls research/qwen21_dual_upgrade_20261011/repositories 2>/dev/null; head -20 research/qwen21_dual_upgrade_20261011/2026-10-11-qwen21-dual-v8-design.md
echo "== hi4d pilot README head =="
head -14 research/hi4d_smplx_pilot_20261011/README.md
echo "== results review v2 key files =="
head -c 300 results/review_20260908_v2/EXPERIMENT_CONCLUSION.md; echo; python3 -c "import json;d=json.load(open('results/review_20260908_v2/ACCEPTANCE.json'));print(str(d)[:300])"
ls results/review_20260908_v2
echo "== legacy demo count =="
ls results/legacy_demo_202609 | wc -l
echo "== archive readme heads =="
for d in archive/cleanup_20261003 archive/cleanup_20261008 archive/cleanup_20261011 archive/home_tools_202609 archive/add_picture_edit_rescue_20260921; do echo "-- $d"; head -6 "$d/README.md"; done
echo "== docs experiments sizes =="
wc -l docs/experiments/*.md; ls docs/superpowers/plans docs/superpowers/specs | wc -l
echo "== root bfs_explore head =="
head -8 bfs_explore.py
echo "== scripts/gen_folder_docs marker =="
grep -n '本文件由' scripts/gen_folder_docs.py | head -2
echo "== who imports v6/qwen21/qwen21_full (counts) =="
grep -rlE 'from src.pose_control.v6|pose_control\.v6' --include='*.py' src scripts tests 2>/dev/null | wc -l
grep -rlE 'pose_control\.qwen21_full' --include='*.py' src scripts tests 2>/dev/null | wc -l
grep -rlE 'pose_control\.qwen21[^_]' --include='*.py' src scripts tests 2>/dev/null | wc -l
echo "== vram_lab protected README head =="
head -8 vram_lab/README.md
echo "== add/README, workflow etc existence (for cross-refs) =="
for f in add/README.md data/README.md models/README.md scripts/README.md reports/README.md workflow/README.md deepgen/README.md; do [ -f $f ] && echo "$f $(grep -q gen_folder_docs $f && echo GEN || echo HAND)"; done
