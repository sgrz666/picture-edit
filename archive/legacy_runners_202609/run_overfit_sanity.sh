#!/usr/bin/env bash
# Two-phase overfit sanity test (full run): Phase A 200 步固定配置管线自检 +
# Phase B 1300 步全时间步采样真实轨迹过拟合；每 100 步做一次 28 步真实生成评测。
set -euo pipefail
cd /home/shangguanrz/project/pic-edit

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
PYTHON=/home/shangguanrz/miniconda3/envs/deepgen/bin/python

OUTPUT_DIR="experiments/tiktok_overfit_sanity_full"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== STARTING TWO-PHASE OVERFIT SANITY TEST ==="
date

$PYTHON train_tiktok_overfit_sanity.py \
  --sequence 00001 \
  --source-stem 0014 \
  --target-stem 0074 \
  --v64-checkpoint experiments/iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 1500 \
  --phase-a-steps 200 \
  --timestep-min 0 \
  --timestep-max 999 \
  --lr 1e-5 \
  --checkpoint-every 500 \
  --preview-every 100 \
  --eval-gen-every 100 \
  --eval-gen-steps 28 \
  --guidance-scale 4.0 \
  --resolution 512 \
  --device cuda \
  --dtype bf16 \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "=== SANITY TEST COMPLETED ==="
date
