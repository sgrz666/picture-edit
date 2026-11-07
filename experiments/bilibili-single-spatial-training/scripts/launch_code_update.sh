#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${BILIBILI_RUN_ROOT:-/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1}"
PY="${BILIBILI_PYTHON:-/home/shangguanrz/nihaochen/envs/qwen21/bin/python}"
mkdir -p -- "$RUN/logs"
LOG="$RUN/logs/code_update_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}.log"
nohup setsid "$PY" "$HERE/update_training_code.py" "$@" >"$LOG" 2>&1 </dev/null &
printf 'code_update_pid=%s\ncode_update_log=%s\n' "$!" "$LOG"
