#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
LOG="$RUN/logs/resource_recovery_$(date -u +%Y%m%dT%H%M%SZ)_$$_$RANDOM.log"
nohup setsid "$PY" "$HERE/recover_typed_oom.py" --run-root "$RUN" >"$LOG" 2>&1 </dev/null &
printf 'resource_recovery_pid=%s\nresource_recovery_log=%s\n' "$!" "$LOG"
