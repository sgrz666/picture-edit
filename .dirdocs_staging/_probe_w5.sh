cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== grep DWPose ==="
grep -rIl --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'DWPose' . 2>/dev/null | head -30
echo "--- dw-ll / yolox path strings ---"
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'dw-ll_ucoco' -e 'yolox_l' . 2>/dev/null | grep -v '^./archive/' | head -25
echo "=== grep antelopev2 ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'antelopev2' . 2>/dev/null | head -25
echo "=== grep dinov2-giant ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'dinov2-giant' -e 'dinov2' . 2>/dev/null | grep -v '^./archive/' | head -25
echo "=== grep face_eval ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'face_eval' -e 'identity_encoder' -e 'landmark_encoder' -e 'perceptual_encoder' . 2>/dev/null | grep -v '^./archive/' | head -30
echo "=== grep SMPL_NEUTRAL / models/smpl ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'SMPL_NEUTRAL' -e 'models/smpl' -e 'SMPLX' -e 'smplx' . 2>/dev/null | grep -v '^./archive/' | head -30
echo "=== grep DeepGen-1.0-diffusers refs ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'DeepGen-1.0-diffusers' . 2>/dev/null | grep -v '^./archive/' | head -25
echo "=== grep model-qwen-edit refs ==="
grep -rIn --exclude-dir=.git --exclude-dir=__pycache__ --exclude='目录说明.md' -e 'model-qwen-edit' -e 'Qwen-Image-Edit-20B' . 2>/dev/null | grep -v '^./archive/' | head -20
