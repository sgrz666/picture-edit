#!/usr/bin/env bash
# Reproduce the bounded chain or prepare a separate pilot run on sg.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QWEN21_PYTHON:-/home/shangguanrz/nihaochen/envs/qwen21/bin/python}"
MODE="${1:-help}"
OUT="${2:-}"
STEPS="${3:-200}"
case "$MODE" in
  help|-h|--help)
    echo 'Usage: bash scripts/run_tiktok_qwen21_full.sh MODE ABSOLUTE_RUN_DIR [TOTAL_STEPS]'
    echo 'Modes: prepare, smoke, body, face-prepare, face, resume-body, resume-face, eval'
    echo 'Use a new run directory for smoke or a new pilot. Resume keeps the original settings.'
    exit 0 ;;
esac
[[ "$OUT" == /* ]] || { echo 'An absolute run directory is required' >&2; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'TOTAL_STEPS must be positive' >&2; exit 2; }
cd "$ROOT"
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
BODY_INIT="${TIKTOK_BODY_INIT:-/home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt}"
COMMON=(--dataset-type tiktok_v3 --resolution 512 --cache "$OUT/native_cache" --require-active-gradients)
FACE_OPTIONS=(--features "$OUT/face_data/face_cache" --face-supervision "$OUT/face_data/face_supervision.json"
              --face-evaluator-range normalized --face-flow-loss roi --face-aux-alignment five_point --lr-scale 0.01)

prepare() {
  "$PYTHON" scripts/prepare_tiktok_qwen21_training.py --output "$OUT/data" | tee "$OUT/logs/prepare.log"
}
face_prepare() {
  "$PYTHON" scripts/prepare_tiktok_qwen21_face.py --train-pairs "$1/train_pairs.jsonl" \
    --val-pairs "$1/val_pairs.jsonl" --output "$OUT/face_data" --resolution 512 | tee "$OUT/logs/face_prepare.log"
  "$PYTHON" -m src.pose_control.qwen21_full.face_supervision \
    --pairs "$OUT/face_data/train_pairs.jsonl" "$OUT/face_data/val_pairs.jsonl" \
    --features "$OUT/face_data/face_cache" --output "$OUT/face_data/face_supervision.json" | tee "$OUT/logs/supervision.log"
}
body_train() {
  local data="$1" output="$2" total="$3" warmup="$4" initialization="$5" checkpoint="$6"
  "$PYTHON" -u train_iper_qwen21_full.py --branch body "${COMMON[@]}" \
    --train-pairs "$data/train_pairs.jsonl" --val-pairs "$data/val_pairs.jsonl" \
    --output "$output" --max-steps "$total" --warmup-steps "$warmup" --save-every 20 \
    --save-steps "$total" --verify-resume "$initialization" "$checkpoint" | tee "$OUT/logs/body_${total}.log"
}
face_train() {
  local output="$1" total="$2" warmup="$3" interval="$4" initialization="$5" checkpoint="$6"
  "$PYTHON" -u train_iper_qwen21_full.py --branch face "${COMMON[@]}" "${FACE_OPTIONS[@]}" \
    --train-pairs "$OUT/face_data/train_pairs.jsonl" --val-pairs "$OUT/face_data/val_pairs.jsonl" \
    --output "$output" --max-steps "$total" --warmup-steps "$warmup" --face-aux-every "$interval" \
    --save-every 20 --save-steps "$total" --diagnostic-probes --verify-resume \
    "$initialization" "$checkpoint" | tee "$OUT/logs/face_${total}.log"
}
evaluate() {
  "$PYTHON" -u eval_iper_qwen21_full.py --dataset-type tiktok_v3 --resolution 512 \
    --pairs "$OUT/face_data/val_pairs.jsonl" --features "$OUT/face_data/face_cache" \
    --cache "$OUT/native_cache" --body-checkpoint "$1" --face-checkpoint "$2" \
    --modes initial body body_face --steps 30 --progress-mode sigma --face-scatter-grid checkpoint \
    --max-samples 1 --output "$OUT/generation" | tee "$OUT/logs/generation.log"
}

case "$MODE" in
  prepare) prepare ;;
  smoke)
    [[ ! -e "$OUT/data/readiness.json" ]] || { echo 'Smoke requires a new run directory' >&2; exit 2; }
    prepare
    body_train "$OUT/data/smoke" "$OUT/body_smoke" 2 0 --init-checkpoint "$BODY_INIT"
    body_train "$OUT/data/smoke" "$OUT/body_smoke" 3 0 --resume "$OUT/body_smoke/checkpoint_last.pt"
    face_prepare "$OUT/data/smoke"
    face_train "$OUT/face_stable_smoke" 2 0 1 --init-checkpoint "$OUT/body_smoke/checkpoint_last.pt"
    face_train "$OUT/face_stable_smoke" 4 0 1 --resume "$OUT/face_stable_smoke/checkpoint_last.pt"
    evaluate "$OUT/body_smoke/checkpoint_last.pt" "$OUT/face_stable_smoke/checkpoint_last.pt" ;;
  body) body_train "$OUT/data" "$OUT/body" "$STEPS" 20 --init-checkpoint "$BODY_INIT" ;;
  face-prepare) face_prepare "$OUT/data" ;;
  face) face_train "$OUT/face" "$STEPS" 20 4 --init-checkpoint "$OUT/body/checkpoint_last.pt" ;;
  resume-body) body_train "$OUT/data" "$OUT/body" "$STEPS" 20 --resume "$OUT/body/checkpoint_last.pt" ;;
  resume-face) face_train "$OUT/face" "$STEPS" 20 4 --resume "$OUT/face/checkpoint_last.pt" ;;
  eval) evaluate "$OUT/body/checkpoint_last.pt" "$OUT/face/checkpoint_last.pt" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
