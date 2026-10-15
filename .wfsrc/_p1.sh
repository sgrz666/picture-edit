set -u
cd /home/shangguanrz/project/pic-edit || exit 1
R=workflow/runs/wp1_accept_hard02
echo '=== geometry 目录 ==='; ls -1 $R/geometry/ | head -20
echo '=== inputs 目录 ==='; ls -1 $R/inputs/
echo '=== generation 目录 ==='; ls -1 $R/generation/
echo '=== final_smplx.pt 的键 ==='
add/envs/smplestx_bw/bin/python -c "
import torch
d=torch.load('$R/evaluation/final_smplx.pt', map_location='cpu', weights_only=False)
for k,v in d.items():
    try: print('  %-18s %-10s %s' % (k, tuple(v.shape), v.dtype))
    except Exception: print('  %-18s %s' % (k, type(v)))
" 2>&1 | head -20
echo '=== SMPL-X faces 是否可直接读 ==='
ls -1 add/tools/SMPLest-X/human_models/human_model_files/smplx/ 2>/dev/null | head
add/envs/smplestx_bw/bin/python -c "
import numpy as np, glob
for f in glob.glob('add/tools/SMPLest-X/human_models/human_model_files/smplx/*NPZ') + glob.glob('add/tools/SMPLest-X/human_models/human_model_files/smplx/*.npz'):
    z=np.load(f, allow_pickle=True)
    print(f, 'faces' in z, getattr(z,'shape',None) if 'faces' in z else '', np.asarray(z['faces']).shape if 'faces' in z else '')
    break
" 2>&1 | head -5
echo '=== dwpose_onnx API ==='
grep -n 'def inference_detector\|def inference_pose\|def post_process' src/evaluation/dwpose_onnx/onnxpose.py src/evaluation/dwpose_onnx/onnxdet.py 2>/dev/null | head -10
echo '=== joint_pilot_metrics 里 detect 的调用约定 ==='
sed -n '30,40p' src/evaluation/joint_pilot_metrics.py
