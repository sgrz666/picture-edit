#!/bin/bash
set -euo pipefail
EXP="/data/shangguanrz/pic-edit-experiments/姿态估计/face"
CVL="/home/shangguanrz/project/pic-edit/models/face/cvlface_adaface_vit_base_kprpe_webface4m"
mkdir -p "$EXP/build/rpe" /tmp/rpe_build
cd "$CVL/models/vit_kprpe/RPE/rpe_ops"
export CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21
export PATH=/home/shangguanrz/nihaochen/envs/qwen21/bin:/usr/local/bin:/usr/bin:/bin
export MAX_JOBS=8
"$EXP/.venv/bin/python" setup.py build_ext --build-lib "$EXP/build/rpe" --build-temp /tmp/rpe_build
ls -la "$EXP/build/rpe"
