#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
REPO="deepgenteam/DeepGen-1.0-diffusers"
DIR="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
CLI="/home/shangguanrz/miniconda3/envs/deepgen/bin/huggingface-cli"

echo "Starting transformer download with auto-resume..."
until $CLI download "$REPO" transformer/diffusion_pytorch_model.safetensors --local-dir "$DIR"; do
    echo "[Retry] Connection dropped. Auto-resuming in 2s..."
    sleep 2
done
echo "=== Transformer safetensors downloaded successfully! ==="