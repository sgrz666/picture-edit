#!/usr/bin/env bash
# 跑 180 键加载审计（CPU，约 21 GB 权重），日志与 JSON 落在 report/ 下。
set -uo pipefail
W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
C=/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1
mkdir -p "$W/report"
cd "$C/scripts" || exit 1
"$W/env/bin/python" 03_weight_audit.py > "$W/report/weight_audit_run.log" 2>&1
echo "audit rc=$?" >> "$W/report/weight_audit_run.log"
