#!/usr/bin/env bash
# 串行推进剩余三关：接入契约 → 官方正式样例 → 48 张矩阵。
# 前一关不过就停，不带病跑下一关（方案第 5 节）。矩阵的 records.jsonl 逐张追加，中断可续。
set -uo pipefail
W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
C=/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1
PY="$W/env/bin/python"
LOG="$W/logs/queue_all_$(date +%Y%m%dT%H%M%S).log"
mkdir -p "$W/logs" "$W/cache"
exec >>"$LOG" 2>&1
cd "$C/scripts" || exit 1
echo "[queue] start $(date -Is) log=$LOG"

echo "[queue] gate phase=official $(date -Is)"
"$PY" 06_contract_gate.py --phase official
rc=$?; echo "[queue] gate official rc=$rc"
[ $rc -ne 0 ] && { echo "[queue] STOP: 官方 latent 未能落盘"; exit $rc; }

echo "[queue] gate phase=adapter $(date -Is)"
"$PY" 06_contract_gate.py --phase adapter
rc=$?; echo "[queue] gate adapter rc=$rc"
if [ $rc -ne 0 ]; then
  echo "[queue] STOP: 接入契约未通过，先定位原因，不跑正式样例与矩阵"
  exit $rc
fi

echo "[queue] official formal samples $(date -Is)"
"$PY" 04_official_samples.py --mode formal
rc=$?; echo "[queue] official formal rc=$rc"
if [ $rc -ne 0 ]; then
  echo "[queue] STOP: 官方正式样例未通过，不启动矩阵"
  exit $rc
fi

echo "[queue] matrix 48 images $(date -Is)"
"$PY" 07_run_matrix.py
rc=$?; echo "[queue] matrix rc=$rc"
echo "[queue] done $(date -Is)"
exit $rc
