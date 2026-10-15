#!/bin/bash
# Decisive throughput test inside ONE time window, because absolute seconds drift by 3x
# between windows on this shared card. Arms:
#   A thinking+native  K=1  (reference, 2 pairs)
#   B thinking+512     K=1  (downscale effect, same 2 pairs)
#   C thinking+native  K=3  (concurrency effect, 6 pairs = 2 per worker)
# All three keep thinking ON: the no-thinking arm was disqualified on contract failures.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
PAIRS2="train_0004176 train_0004238"
PAIRS6="train_0004176 train_0004238 train_0004254 train_0004385 train_0004697 train_0004704"
mkdir -p logs
echo "=== WINDOW_START $(date -Is) ==="
$PY -B scripts/pe_rewrite_tiktok.py --out ablation2/A_think_native_k1 run --split train --pair-id $PAIRS2
$PY -B scripts/pe_rewrite_tiktok.py --out ablation2/A_think_native_k1 stats
T0=$(date +%s)
$PY -B scripts/pe_rewrite_tiktok.py --out ablation2/B_think_512_k1 run --split train --pair-id $PAIRS2 --max-image-side 512
echo "B_WALL_SECONDS=$(( $(date +%s) - T0 ))"
$PY -B scripts/pe_rewrite_tiktok.py --out ablation2/B_think_512_k1 stats
T0=$(date +%s)
for index in 0 1 2; do
    setsid nohup $PY -B scripts/pe_rewrite_tiktok.py --out ablation2/C_think_native_k3 \
        run --split train --shard-index $index --shard-count 3 --pair-id $PAIRS6 \
        > "logs/ablation2_C_shard$index.log" 2>&1 < /dev/null &
done
wait
echo "C_WALL_SECONDS=$(( $(date +%s) - T0 )) for 6 pairs at K=3"
$PY -B scripts/pe_rewrite_tiktok.py --out ablation2/C_think_native_k3 stats
echo "=== WINDOW_DONE $(date -Is) ==="
