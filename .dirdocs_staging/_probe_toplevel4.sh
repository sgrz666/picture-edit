#!/bin/bash
R=/home/shangguanrz/project/pic-edit
cd $R
echo "== configs json keys =="
for f in configs/*.json; do echo "-- $f ($(wc -c <$f) bytes)"; python3 -c "
import json;d=json.load(open('$f'))
def walk(o,p=''):
  if isinstance(o,dict):
    for k,v in list(o.items())[:25]: print(p+k, type(v).__name__)
walk(d)"; done
echo "== caches target =="
ls /data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls | head -8
ls /data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls | wc -l
du -sh --si /data/shangguanrz/pic-edit-caches 2>/dev/null
echo "== inputs files =="
file inputs/*.jpg
echo "== patches head =="
head -12 patches/deepgen_pose_control.patch; grep -c '^@@' patches/deepgen_pose_control.patch
echo "== logs top json =="
python3 -c "import json;d=json.load(open('logs/hi4d_dwpose_maskbbox_failures.json'));print(type(d), len(d));print(str(d)[:400])"
echo "== legacy logs range =="
ls -la logs/legacy | head -6
ls logs/legacy/*.log logs/legacy/*.out | wc -l
tail -3 logs/legacy/qwen21_hi4d_dual_train.log
tail -3 logs/legacy/inference_202609.log
echo "== src py docstrings =="
for f in src/adapter/pose_adapter.py src/controls/generator.py src/data/tiktok_dataset.py src/data/iper_dataset.py src/data/hi4d_qwen21_dataset.py src/data/letterbox.py src/evaluation/joint_pilot_metrics.py src/integration/champ_deepgen.py src/pose_control/qwen21/body_adapter.py src/pose_control/qwen21/face_adapter.py src/pose_control/qwen21/dual_adapter.py src/pose_control/qwen21/dual_inference.py src/pose_control/qwen21/runtime.py src/pose_control/qwen21/action_prompts.py src/pose_control/qwen21_full/qwen_adapter.py src/pose_control/qwen21_full/token_encoders.py src/pose_control/qwen21_full/face_supervision.py src/pose_control/v6/control_core.py third_party/champ_guidance/guidance_encoder.py; do echo "-- $f"; head -6 "$f"; done
echo "== pose_control subdirs stats =="
for d in src/pose_control/qwen21_full/face src/pose_control/qwen21_full/detail src/pose_control/qwen21_full/hand src/pose_control/qwen21_full/interface src/pose_control/qwen21_full/reasoning src/pose_control/v6/face src/pose_control/v6/detail src/pose_control/v6/hand src/pose_control/v6/interface src/pose_control/v6/reasoning; do echo "$d: $(ls $d | grep -v pyc | tr '\n' ' ') | lines=$(find $d -maxdepth 1 -name '*.py' | xargs wc -l 2>/dev/null|tail -1)"; done
echo "== eval dwpose_onnx =="
ls src/evaluation/dwpose_onnx
echo "== v6/qwen21_full top py sizes =="
wc -l src/pose_control/qwen21_full/*.py | tail -30
wc -l src/pose_control/v6/*.py
