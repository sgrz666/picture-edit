set -u
cd /home/shangguanrz/project/pic-edit/workflow/runs || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
echo '=== wp1_accept_hard02：第一读者 evaluate.py vs 第二读者 evaluate2 ==='
$PY - <<'PY'
import json
ev=json.load(open('wp1_accept_hard02/evaluation/eval.json'))
print('evaluate.py  summary:', json.dumps(ev['summary'], ensure_ascii=False))
e2=json.load(open('wp1_accept_hard02/evaluation/eval2.json'))
s=e2['pose_generated_vs_guide']
print('evaluate2.py summary:', json.dumps(s['summary'], ensure_ascii=False))
print('evaluate2.py joints: requested', s['joints_requested'], 'compared', s['joints_compared'], 'torso', s['torso_normalisation_px'])
print('evaluate.py  joints: requested', ev['joints_requested'], 'compared', ev['joints_compared'], 'torso', ev['torso_normalisation_px'])
print('agreement:', e2.get('estimator_agreement',{}).get('verdict'), e2.get('estimator_agreement',{}).get('body_agree@0.2'))
PY
echo '=== 各 run 的尺寸与 content_box 约定 ==='
$PY - <<'PY'
import json, os
from PIL import Image
import numpy as np
for run in sorted(os.listdir('.')):
    fin=os.path.join(run,'generation','final_image.png')
    if not os.path.isfile(fin): continue
    pp=json.load(open(os.path.join(run,'inputs','source_preprocess.json')))
    with Image.open(fin) as im: gen=im.size
    a=os.path.join(run,'inputs','A_rgb.png')
    with Image.open(a) as im: src=im.size
    kp=np.load(os.path.join(run,'geometry','B_keypoints.npz'))
    g=kp['keypoints_generation_px']; v=kp['valid_generation']
    inside_x=((g[v,0]>=0)&(g[v,0]<gen[0])).mean() if v.any() else None
    print('%-26s gen=%s src=%s box=%s scale=%s guide_x_in_gen_range=%.2f' % (
        run, gen, src, pp.get('content_box'), [round(x,3) for x in pp.get('scale_xy',[])], inside_x))
PY
