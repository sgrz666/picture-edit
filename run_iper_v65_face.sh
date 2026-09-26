#!/bin/bash
set -e
cd /home/shangguanrz/project/pic-edit
export PYTHONPATH=.
mkdir -p experiments/iper_v65_face_overfit_024_6

/home/shangguanrz/miniconda3/envs/deepgen/bin/python -u train_iper_v65_face.py \
  --appearance 024_6 \
  --source-stem source_motion_f000375 \
  --target-stem target_f001492 \
  --v64-checkpoint experiments/iper_v64_detail_overfit_024_6/checkpoints/checkpoint_step_1500.pt \
  --output-dir experiments/iper_v65_face_overfit_024_6 \
  --max-steps 1500 \
  --warmup-steps 100 \
  --checkpoint-every 250 \
  --preview-every 250 \
  --device cuda \
  --dtype bf16
