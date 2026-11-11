#!/usr/bin/env bash
set -euo pipefail
RUN=/data/shangguanrz/pic-edit-experiments/bilibili-native-v5-training-20261106-v1
QPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
TAG=$(date +%Y%m%dT%H%M%S)
printf '%s\n' '{"cuda_allocator":"cudaMallocAsync","scope":"this_preparation_process_only","training_started":false}' > "$RUN/report/allocation_$TAG.json"
"$QPY" -u "$RUN/scripts/prepare_native_v4.py" --method sapiens_v1 \
  --output-root "$RUN/native_sapiens_v1" \
  --report "$RUN/report/native_preparation_$TAG.json" \
  2>&1 | tee "$RUN/report/native_preparation_$TAG.log"
