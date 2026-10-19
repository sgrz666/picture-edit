#!/usr/bin/env bash
# 第 1 关（官方 smoke）与第 2 关（接入契约 A/B/C/D）串行排队跑，日志带时间戳。
set -uo pipefail
W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
C=/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1
LOG="$W/logs/stage1_$(date +%Y%m%dT%H%M%S).log"
mkdir -p "$W/logs"
exec >>"$LOG" 2>&1
cd "$C/scripts" || exit 1
echo "[stage1] start $(date -Is) log=$LOG"
echo "[stage1] official smoke"
"$W/env/bin/python" 04_official_samples.py --mode smoke
echo "[stage1] official smoke rc=$?"
echo "[stage1] contract gate"
"$W/env/bin/python" 06_contract_gate.py
echo "[stage1] contract gate rc=$?"
echo "[stage1] done $(date -Is)"
