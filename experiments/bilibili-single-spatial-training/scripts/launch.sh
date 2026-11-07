#!/usr/bin/env bash
# The detached Linux supervisor is independent of SSH/Codex. Reboot recovery is manual resume.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$ACTION" != run && "$ACTION" != resume ]]; then
  echo 'Usage: launch.sh run|resume [--run-root DIR --repo-root DIR --python PY --gpu N]' >&2
  exit 2
fi
RUN="${BILIBILI_RUN_ROOT:-/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1}"
PY="${BILIBILI_PYTHON:-/home/shangguanrz/nihaochen/envs/qwen21/bin/python}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == --run-root ]]; then RUN="${ARGS[i+1]}"; fi
  if [[ "${ARGS[i]}" == --python ]]; then PY="${ARGS[i+1]}"; fi
done
mkdir -p -- "$RUN/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
LOG="$RUN/logs/launch_${STAMP}.log"
nohup setsid "$PY" "$HERE/supervisor.py" "$ACTION" --run-root "$RUN" --python "$PY" "$@" >"$LOG" 2>&1 </dev/null &
printf 'supervisor_pid=%s\nlaunch_log=%s\n' "$!" "$LOG"
