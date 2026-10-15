cd /home/shangguanrz/project/pic-edit || exit 1
PY=~/miniconda3/envs/deepgen/bin/python
echo "=== hi4d control .pt content ==="
$PY - <<'PY'
import torch
p='/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls/control_000d4c3aa7eb1504623c181b.pt'
d=torch.load(p, map_location='cpu')
print(type(d))
if isinstance(d, dict):
    for k,v in d.items():
        try: print(' ', k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
        except Exception: print(' ', k, type(v), v if not hasattr(v,'shape') else '')
PY
echo "=== who writes qwen21_hi4d_controls ==="
grep -rIn --include='*.py' --include='*.sh' -e 'qwen21_hi4d_controls' src scripts configs workflow 2>/dev/null | head -12
echo "=== file count / du ==="
find /data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls -maxdepth 1 -name '*.pt' | wc -l
echo "=== deepgen dir ==="
ls -la deepgen/; du -sh --si deepgen
echo "=== deepgen git? ==="
git -C deepgen log --oneline -3 2>&1 | head -5
git -C deepgen status --short 2>&1 | head -20
git check-ignore -v deepgen 2>&1 | head -3
echo "=== deepgen tree maxdepth2 ==="
find deepgen -maxdepth 2 -printf '%y %s %TY-%Tm-%Td %p\n' | sort -k4 | head -60
echo "=== patches ==="
ls -la patches/
head -40 patches/deepgen_pose_control.patch 2>/dev/null
echo "=== patch apply check (dry, read-only) ==="
cd deepgen && git apply --check ../patches/deepgen_pose_control.patch 2>&1 | head -10; cd ..
