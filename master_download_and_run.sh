#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
REPO="deepgenteam/DeepGen-1.0-diffusers"
DIR="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
CLI="/home/shangguanrz/miniconda3/envs/deepgen/bin/huggingface-cli"
PY="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"

download_with_retry() {
    file=$1
    dest="$DIR/$file"
    if [ -f "$dest" ]; then
        echo "[Already Exists] $file"
        return 0
    fi
    echo "=== [Start Download] $file ==="
    until $CLI download "$REPO" "$file" --local-dir "$DIR"; do
        echo "[Retry] Resuming $file..."
        sleep 2
    done
    echo "=== [Completed] $file ==="
}

echo "=== Ensuring all weights are fully downloaded ==="
download_with_retry "vlm/model-00002-of-00002.safetensors"
download_with_retry "vlm/model-00001-of-00002.safetensors"
download_with_retry "transformer/diffusion_pytorch_model.safetensors"

echo "=== All weights verified! Launching Demo Pose Editing ==="
$PY /home/shangguanrz/project/pic-edit/run_demo_pose_edit.py