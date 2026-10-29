#!/usr/bin/env bash
# Sequential GPU arms: never two at once, the card is shared.
set -u
R=/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1
P=/home/shangguanrz/project/pic-edit
cd "$R" || exit 1
run() {
  name=$1; shift
  echo "=== START $name $(date -Is) ==="
  "$@" >> "logs/${name}.log" 2>&1
  rc=$?
  echo "=== END $name rc=${rc} $(date -Is) ==="
}
run sapiens_detr070 /home/shangguanrz/miniconda3/envs/deepgen/bin/python "$R/scripts/infer_2d.py" --request "$R/data/request.json" --mode sapiens_detr070
run sapiens_yolox  /home/shangguanrz/miniconda3/envs/deepgen/bin/python "$R/scripts/infer_2d.py" --request "$R/data/request.json" --mode sapiens_yolox
run multihmr       "$P/add/envs/multihmr/bin/python" "$R/scripts/infer_3d.py" --request "$R/data/request.json" --mode multihmr
run multihmr_fov60 "$P/add/envs/multihmr/bin/python" "$R/scripts/infer_3d.py" --request "$R/data/request.json" --mode multihmr_fov60
run smplest_gtbox  "$P/add/envs/smplestx_bw/bin/python" "$R/scripts/infer_3d.py" --request "$R/data/request.json" --mode smplest_gtbox
run smplest_yolox  "$P/add/envs/smplestx_bw/bin/python" "$R/scripts/infer_3d.py" --request "$R/data/request.json" --mode smplest_yolox
echo "=== ALL_DONE $(date -Is) ==="
