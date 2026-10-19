#!/usr/bin/env bash
# 等矩阵跑完，自动出指标/面板 + 指标报告 + 把交付物复制进实验目录。
# 单独起一个进程做这件事，避免人工反复查看进度。
set -u

W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
C=/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1
PY="$W/env/bin/python"
RUN="${1:-matrix_20261015T134829}"
LOG="$C/logs/finish_${RUN}_$(date +%Y%m%dT%H%M%S).log"

echo "[finish] 等待矩阵运行结束（$RUN）... " | tee -a "$LOG"
while pgrep -f '[0]7_run_matrix' > /dev/null; do sleep 60; done
echo "[finish] $(date) 矩阵进程已退出，开始出报告" | tee -a "$LOG"

cd "$C/scripts" || exit 1
"$PY" 08_evaluate.py --run "$RUN" --panels >> "$LOG" 2>&1
echo "[finish] 08 rc=$?" | tee -a "$LOG"
"$PY" 15_report.py --run "$RUN" >> "$LOG" 2>&1
echo "[finish] 15 rc=$?" | tee -a "$LOG"
"$PY" 19_collect_deliverables.py --run "$RUN" >> "$LOG" 2>&1
echo "[finish] 19 rc=$?" | tee -a "$LOG"
echo "[finish] 完成 $(date)" | tee -a "$LOG"
