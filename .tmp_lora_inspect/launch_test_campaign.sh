#!/bin/bash
# Detached launcher for the one-shot full-test LoRA evaluation.
cd /data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora || exit 1
export LD_LIBRARY_PATH="/data/shangguanrz/仪表盘/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/shangguanrz/nihaochen/envs/qwen21/bin/python -u scripts/test_campaign.py \
  --config configs/test.json --tag 20261112_v1