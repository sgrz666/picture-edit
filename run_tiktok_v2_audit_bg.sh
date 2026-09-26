#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/shangguanrz/project/pic-edit"
PYTHON="/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python"

LOG_DIR="$PROJECT/logs"
LOG="$LOG_DIR/audit_and_select_tiktok_v2_bg.log"
PID_FILE="$LOG_DIR/audit_and_select_tiktok_v2.pid"

cd "$PROJECT"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "TikTok V2 audit"
echo "start: $(date)"
echo "host : $(hostname)"
echo "python: $PYTHON"
echo "============================================================"

"$PYTHON" -u audit_and_select_tiktok_v2.py

echo
echo "============================================================"
echo "TikTok V2 audit FINISHED"
echo "end: $(date)"
echo "============================================================"
