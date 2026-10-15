set -u
cd /home/shangguanrz/project/pic-edit || exit 1
GEO=add/envs/smplestx_bw/bin/python
echo '=== smplestx 环境依赖 ==='
$GEO -c "
import importlib
for m in ['numpy','PIL','cv2','onnxruntime','insightface','lpips','skimage','torch','pytest']:
    try:
        mod=importlib.import_module(m); print('  OK', m)
    except Exception: print('  --', m)
"
echo '=== 清理我留下的孤儿（run 本体已被删，只剩我写的 evaluation/）==='
for d in workflow/runs/*/; do
  n=$(ls -1 "$d" | wc -l)
  if [ "$n" = "1" ] && [ -d "$d/evaluation" ]; then
    has_other=$(ls -1 "$d/evaluation" | grep -v -E 'eval2|stage2' | wc -l)
    echo "  $d 仅 evaluation/，其中非我产物 $has_other 个"
    if [ "$has_other" = "0" ]; then rm -rf "$d/evaluation"; echo "    已删除我生成的 evaluation/"; fi
  fi
done
echo '=== 试 --fit-source（beta 路径），用 smplestx 环境 ==='
cd workflow && $GEO evaluate2.py --run wp1_accept_hard02 --fit-source --no-lpips --force 2>&1 | tail -6
