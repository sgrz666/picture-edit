#!/bin/bash
# Full pass over every pair in pairs_strict_v1 (train 8574 + val 1120 + test 1156).
# Resumable: each completed pair writes records/<split>/<pair_id>.json and is skipped
# on the next start. Reassemble the bank at the end.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ARGS=${ARGS:-}
for split in train val test; do
    echo "=== SPLIT $split $(date -Is) ==="
    $PY -B scripts/pe_rewrite_tiktok.py run --split "$split" $ARGS || echo "SPLIT_FAILED $split"
done
$PY -B scripts/pe_rewrite_tiktok.py assemble || exit 1
$PY -B scripts/pe_rewrite_tiktok.py qc --count 24
