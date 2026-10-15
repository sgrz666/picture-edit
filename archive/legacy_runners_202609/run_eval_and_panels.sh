#!/usr/bin/env bash
set -euo pipefail
cd /home/shangguanrz/project/pic-edit

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
PYTHON=/home/shangguanrz/miniconda3/envs/deepgen/bin/python

TRAIN_DIR="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_200"
EVAL_DIR="$TRAIN_DIR/final_eval"
mkdir -p "$EVAL_DIR"

echo "=== STARTING FULL EVALUATION AND PANEL GENERATION ==="
date

$PYTHON scripts/eval_tiktok_full_overfit_and_generate_panels.py \
  --sequence 00001 \
  --source-stem 0014 \
  --target-stem 0074 \
  --training-dir "$TRAIN_DIR" \
  --output-dir "$EVAL_DIR" \
  --num-inference-steps 28 \
  --guidance-scale 4.0 \
  --seed 42 \
  --device cuda \
  --dtype bf16 \
  2>&1 | tee "$EVAL_DIR/eval.log"

echo "=== EVALUATION AND PANEL GENERATION COMPLETED ==="
date
