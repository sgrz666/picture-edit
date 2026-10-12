#!/usr/bin/env bash
set -euo pipefail

cd /home/shangguanrz/project/pic-edit

PYTHON="/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python"

mkdir -p logs

echo "============================================================"
echo "START: TikTok V3 fill missing geometry"
echo "TIME : $(date '+%F %T')"
echo "HOST : $(hostname)"
echo "PWD  : $(pwd)"
echo "PY   : $PYTHON"
echo "============================================================"

"$PYTHON" -u fill_tiktok_v3_missing_geometry.py \
  2>&1 | tee logs/fill_tiktok_v3_missing_geometry.log

echo
echo "============================================================"
echo "DONE : $(date '+%F %T')"
echo "============================================================"
