#!/bin/bash
# Throughput ablation: identical pairs, four serial configs, then one 3-worker run.
# Serial arms first so each measurement is alone on the GPU; the concurrency arm then
# shows what K=3 actually buys. Results land in ablation/<arm>/stats.json.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
PAIRS6="train_0001022 train_0001416 train_0001487 train_0001975 train_0002288 train_0003394"
PAIRS12="$PAIRS6 train_0003723 train_0004176 train_0004238 train_0004254 train_0004385 train_0004697"
mkdir -p logs
arm () {
    local name=$1; shift
    echo "=== ARM $name $* $(date -Is) ==="
    $PY -B scripts/pe_rewrite_tiktok.py --out "ablation/$name" \
        run --split train --pair-id $PAIRS6 "$@" || echo "ARM_FAILED $name"
    $PY -B scripts/pe_rewrite_tiktok.py --out "ablation/$name" stats || true
}
arm baseline
arm notk_native --no-thinking
arm tk_512 --max-image-side 512
arm notk_512 --no-thinking --max-image-side 512

echo "=== ARM notk_512_k3 (3 co-resident workers, 12 pairs) $(date -Is) ==="
T0=$(date +%s)
for index in 0 1 2; do
    setsid nohup $PY -B scripts/pe_rewrite_tiktok.py --out ablation/notk_512_k3 \
        run --split train --shard-index $index --shard-count 3 --pair-id $PAIRS12 \
        --no-thinking --max-image-side 512 > "logs/ablation_k3_shard$index.log" 2>&1 < /dev/null &
done
wait
echo "K3_WALL_SECONDS=$(( $(date +%s) - T0 ))"
$PY -B scripts/pe_rewrite_tiktok.py --out ablation/notk_512_k3 stats || true
echo "ABLATION_DONE $(date -Is)"
