set -u
cd /home/shangguanrz/project/pic-edit/workflow || exit 1
GEO=/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python
QW=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$GEO -m py_compile evaluate2.py scorecard.py && $QW -m py_compile evaluate2.py scorecard.py && echo COMPILE_OK_BOTH
echo '=== 1) 几何环境：只拟合 A ==='
$GEO evaluate2.py --run wp1_accept_hard02 --fit-only 2>&1 | tail -4
echo '=== 2) 工作语言环境：全量重跑 ==='
$QW evaluate2.py --all --force 2>&1 | tail -7
echo '=== 3) beta 是否真算出来了 ==='
$QW - <<'PY'
import json, os
for run in sorted(os.listdir('runs')):
    p=f'runs/{run}/evaluation/eval2.json'
    if not os.path.isfile(p): continue
    b=json.load(open(p)).get('beta') or {}
    print('%-26s %-9s l2=%s max=%s' % (run, b.get('status'), b.get('beta_l2'), b.get('beta_max_abs')))
PY
