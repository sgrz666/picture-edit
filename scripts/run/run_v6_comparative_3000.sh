#!/usr/bin/env bash
# V6.3 Comparative Training Launcher: 3,000 micro-steps on exact baseline sample stream.
set -euo pipefail

PROJECT_DIR="/home/shangguanrz/project/pic-edit"
cd "$PROJECT_DIR" || exit 1

PYTHON="/home/shangguanrz/miniconda3/envs/deepgen/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR"

EXP_DIR="$PROJECT_DIR/experiments/iper_v6_comparative_3000"
LOG_FILE="$EXP_DIR/train.log"
mkdir -p "$EXP_DIR"

echo "=================================================================="
echo " Starting V6.3 Native Adapter Comparative Training (3,000 micro-steps)"
echo " Exact samples from baseline: baseline_3000_microstep_pairs.jsonl"
echo " Target directory: $EXP_DIR"
echo " Log file: $LOG_FILE"
echo "=================================================================="

$PYTHON train_iper_v6_native.py \
    --model_path "$PROJECT_DIR/models/DeepGen-1.0-diffusers" \
    --train_pairs "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt/splits/baseline_3000_microstep_pairs.jsonl" \
    --val_pairs "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt/splits/test_pairs.jsonl" \
    --sampled_root "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt" \
    --assets_root "$PROJECT_DIR/datasets/iPER/iper_assets_512_nvdiffrast" \
    --cache_dir "$PROJECT_DIR/experiments/iper_adapter_finetune_v1/condition_cache" \
    --output_dir "$EXP_DIR" \
    --resolution 512 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_micro_steps 3000 \
    --save_micro_steps 500 \
    --val_micro_steps 500 \
    --val_samples 50 \
    --log_steps 10 \
    --no_shuffle \
    --seed 42 \
    --max_rolling_checkpoints 10 \
    --lr_zero_heads 1e-4 \
    --lr_condition 5e-5 \
    --lr_bridge 5e-5 \
    --lr_control_block 1e-5 \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT=$?
echo "Training completed with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Running Multi-Stage Evolution Evaluation (Steps 500, 1000, 1500, 2000, 2500, 3000) ==="
    $PYTHON scripts/multi_stage_eval_v6.py \
        --model_path "$PROJECT_DIR/models/DeepGen-1.0-diffusers" \
        --exp_dir "$EXP_DIR" \
        --baseline_exp_dir "$PROJECT_DIR/experiments/iper_adapter_finetune_v1" \
        --sampled_root "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt" \
        --assets_root "$PROJECT_DIR/datasets/iPER/iper_assets_512_nvdiffrast" \
        --test_pairs "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt/splits/test_pairs.jsonl" \
        --cache_dir "$PROJECT_DIR/experiments/iper_adapter_finetune_v1/condition_cache" \
        --steps 500 1000 1500 2000 2500 3000 \
        --inference_steps 30 \
        2>&1 | tee -a "$EXP_DIR/multi_stage_eval.log"

    echo "=== Running Final Checkpoint 50-sample Evaluation ==="
    $PYTHON eval_iper_v6.py \
        --model_path "$PROJECT_DIR/models/DeepGen-1.0-diffusers" \
        --adapter_checkpoint "$EXP_DIR/adapter_step_3000.pt" \
        --pairs_jsonl "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl" \
        --sampled_root "$PROJECT_DIR/datasets/iPER/iper_sampled_6src64tgt" \
        --assets_root "$PROJECT_DIR/datasets/iPER/iper_assets_512_nvdiffrast" \
        --cache_dir "$PROJECT_DIR/experiments/iper_adapter_finetune_v1/condition_cache" \
        --output_dir "$EXP_DIR/eval_results" \
        --resolution 512 \
        --max_samples 50 \
        --inference_steps 30 \
        2>&1 | tee -a "$EXP_DIR/eval.log"

    echo "=== V6.3 Comparative Experiment Run & Evaluation Finished Successfully ==="
fi
