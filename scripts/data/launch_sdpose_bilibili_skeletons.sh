#!/usr/bin/env bash
# Launch the full Bilibili SDPose-Wholebody skeleton + keypoint generation as a detached job.
#
#   bash scripts/data/launch_sdpose_bilibili_skeletons.sh            # full run (resume-safe)
#   BVID=BV14NHK6SE9d LIMIT=5 bash scripts/data/launch_sdpose_bilibili_skeletons.sh   # smoke slice
#
# Background on CUDA_HOME: `mmengine._strategy` imports `deepspeed`, whose version probe calls
# `installed_cuda_version()` and raises MissingCUDAException when no CUDA toolkit is visible.
# This box has no /usr/local/cuda, but the qwen21 env ships a real nvcc 12.8.93, so point
# CUDA_HOME there. No extension is ever compiled for SDPose inference.
set -euo pipefail

REPO=/home/shangguanrz/project/pic-edit
PY=/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python
OUT_ROOT=${OUT_ROOT:-/data/shangguanrz/datasets/bilibili/geometry/skeleton}
IMAGES_ROOT=${IMAGES_ROOT:-/data/shangguanrz/datasets/bilibili/frames/images}
LOG_DIR=${LOG_DIR:-/data/shangguanrz/pic-edit-experiments/姿态估计/sdpose_bilibili/logs}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/run_${STAMP}.log"

export CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
cd "$REPO"

EXTRA=()
if [[ -n "${BVID:-}" ]]; then EXTRA+=(--bvid "$BVID"); fi
if [[ -n "${LIMIT:-}" ]]; then EXTRA+=(--limit "$LIMIT"); fi
if [[ -n "${OVERLAY_STRIDE:-}" ]]; then EXTRA+=(--overlay-stride "$OVERLAY_STRIDE"); fi

nohup setsid "$PY" scripts/data/generate_sdpose_skeletons.py \
  --images-root "$IMAGES_ROOT" \
  --out-root "$OUT_ROOT" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  > "$LOG" 2>&1 &

echo "$!" > "$LOG_DIR/run.pid"
echo "pid=$! log=$LOG"
