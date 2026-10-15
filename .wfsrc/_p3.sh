set -u
cd /home/shangguanrz/project/pic-edit || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY -m py_compile workflow/evaluate2.py && cd workflow && $PY evaluate2.py --all --force 2>&1 | tail -8
echo '=== 汇总表 ==='
$PY - <<'PY'
import json, os
rows=[]
for run in sorted(os.listdir('runs')):
    p=f'runs/{run}/evaluation/eval2.json'
    if not os.path.isfile(p): continue
    e=json.load(open(p))
    s=(e.get('pose_generated_vs_guide') or {}).get('summary') or {}
    ch=(e.get('pose_generated_vs_source') or {}).get('summary') or {}
    ag=e.get('estimator_agreement') or {}
    ev=None
    if os.path.isfile(f'runs/{run}/evaluation/eval.json'):
        ev=(json.load(open(f'runs/{run}/evaluation/eval.json')).get('summary') or {}).get('body_pck@0.2')
    rows.append((run, ev, s.get('body_pck@0.2'), s.get('body_pck@0.2_strict'),
        (e.get('scale_fidelity') or {}).get('torso_ratio_generated_over_guide'),
        ch.get('body_mpjpe_norm'), e['identity'].get('identity_cosine'),
        (e.get('preservation') or {}).get('background_mae'),
        (e.get('preservation') or {}).get('person_mae'),
        ((e.get('shape') or {}).get('drift') or {}).get('max_rel_change'),
        ag.get('body_agree@0.2'), ag.get('verdict')))
print('%-26s %7s %7s %7s %7s %7s %6s %6s %6s %6s %6s %s' % ('run','pck_SMPLX','pck_DW','strict','torsoG','move','idcos','bgMAE','perMAE','shape','agree','verdict'))
for r in rows:
    print('%-26s %7s %7s %7s %7s %7s %6s %6s %6s %6s %6s %s' % tuple(
        ('-' if v is None else ('%.3f'%v if isinstance(v,float) else v)) for v in r))
PY
