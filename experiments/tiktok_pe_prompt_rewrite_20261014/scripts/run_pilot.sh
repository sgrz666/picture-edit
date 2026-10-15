#!/bin/bash
# Pilot: 20 train + 5 val + 5 test pairs, then assemble and build the QC sample.
# Re-running is safe: pairs already recorded as complete are skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ARGS=${ARGS:-}
$PY -B scripts/pe_rewrite_tiktok.py run --split train --limit 20 $ARGS || exit 1
$PY -B scripts/pe_rewrite_tiktok.py run --split val --limit 5 $ARGS || exit 1
$PY -B scripts/pe_rewrite_tiktok.py run --split test --limit 5 $ARGS || exit 1
$PY -B scripts/pe_rewrite_tiktok.py assemble || exit 1
$PY -B scripts/pe_rewrite_tiktok.py qc --count 12
