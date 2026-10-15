#!/usr/bin/env bash
set -euo pipefail
cd /home/shangguanrz/project/pic-edit

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
PYTHON=/home/shangguanrz/miniconda3/envs/deepgen/bin/python

OUTPUT_DIR="datasets/TikTokDataset/showcase_single/sample_01_00001_0014_to_0074/overfit_full_200"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== STARTING FULL ADAPTER OVERFIT TRAINING (200 STEPS) ==="
date

$PYTHON train_tiktok_v65_face.py \
  --sequence 00001 \
  --source-stem 0014 \
  --target-stem 0074 \
  --v64-checkpoint experiments/iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 200 \
  --warmup-steps 0 \
  --checkpoint-every 50 \
  --preview-every 25 \
  --resolution 512 \
  --device cuda \
  --dtype bf16 \
  --prompt "Change the person's pose to match the target pose. Preserve identity, face, hair, clothing, hands, body proportions, lighting, and background." \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "=== TRAINING COMPLETED ==="
date
