#!/bin/bash
set -euo pipefail
EXP="/data/shangguanrz/pic-edit-experiments/姿态估计/face"
cd "$EXP"
export PYTHONPATH="$EXP/build/rpe"
export OMP_NUM_THREADS=4
for COND in align_j5 align_j10 blur1.5 lowres48 lowres32 jpeg20 occl genlike; do
  if [ -f "output/emb/$COND.npz" ]; then
    echo "skip $COND"
    continue
  fi
  echo "=== $COND ==="
  .venv/bin/python scripts/extract.py --crops data/scan-v1/crops.npz \
    --out "output/emb/$COND.npz" --condition "$COND" 2>&1 | grep -Ev 'VerifyOutputSizes|FutureWarning|warnings.warn'
done
echo "all conditions done"
