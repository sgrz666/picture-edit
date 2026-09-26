#!/usr/bin/env bash
set -euo pipefail

BASE=/home/shangguanrz/project/pic-edit
ADD="$BASE/add"
SMPL="$ADD/tools/SMPLest-X"

V4="$ADD/data/iper_sampled_v4_6src64tgt"
MASTER="$ADD/data/iper_master_v4"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ADD/envs/smplestx_bw"

python "$SMPL/scripts/build_iper_v4_master_only.py" \
  --sampled_root "$V4" \
  --output_root "$MASTER" \
  --mode symlink \
  --force \
  --strict
