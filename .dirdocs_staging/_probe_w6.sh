cd /home/shangguanrz/project/pic-edit || exit 1
run(){ echo "=== $1 ==="; grep -rIn --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.md' -e "$1" src scripts configs tests workflow docs reports vram_lab patches deepgen *.py 2>/dev/null | head -20; }
run 'dw-ll_ucoco'
run 'yolox_l'
run 'models/DWPose'
run 'antelopev2'
run 'dinov2-giant'
run 'face_eval'
run 'identity_encoder'
run 'lpips.ts'
run 'SMPL_NEUTRAL'
run 'models/smpl'
