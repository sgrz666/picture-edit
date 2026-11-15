#!/bin/bash
set -euo pipefail
BASE=/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1/base_qwen_eval_20261114_v1
UP=/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
export LD_LIBRARY_PATH=/data/shangguanrz/仪表盘/compat:${LD_LIBRARY_PATH:-}
export HF_HOME=/data/shangguanrz/LoRA/hf-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$UP"
echo "[$(date -Is)] cache start"
"$PY" -u scripts/run.py --config "$BASE/configs/base_infer.json" cache \
  --name base_cache_20261114_v2 --data "$BASE/data_subset"
echo "[$(date -Is)] infer start"
"$PY" -u scripts/run.py --config "$BASE/configs/base_infer.json" infer \
  --name base_infer_20261114_v2 --data "$BASE/data_subset" \
  --cache "$BASE/cache/base_cache_20261114_v2" --limit 50
echo "[$(date -Is)] DONE"
