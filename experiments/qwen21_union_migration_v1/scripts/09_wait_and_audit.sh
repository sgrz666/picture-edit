#!/usr/bin/env bash
# 等 Union 权重下载完成后立刻跑加载检查（180 键 + 反证加载后被置零）。
# 单独排队是为了不让 ssh 会话挂在下载上；产物写在 report/ 下，带时间戳日志。
set -uo pipefail
W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
C=/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1
TARGET=7550979904
LOG="$W/logs/wait_and_audit_$(date +%Y%m%dT%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[wait] start $(date -Is) target=$TARGET"

for _ in $(seq 1 120); do
  INC=$(find "$W/models" -name '*.incomplete' 2>/dev/null | head -1)
  if [ -z "$INC" ]; then echo "[wait] incomplete gone $(date -Is)"; break; fi
  SZ=$(stat -c %s "$INC" 2>/dev/null || echo 0)
  echo "[wait] $(date -Is) $SZ / $TARGET"
  [ "$SZ" -ge "$TARGET" ] && { echo "[wait] full size reached"; break; }
  sleep 20
done

# 等 bootstrap 自己写完 _checksums.json 与 versions.json
for _ in $(seq 1 90); do
  pgrep -f '01_bootstrap.sh' >/dev/null || { echo "[wait] bootstrap finished"; break; }
  sleep 10
done
ls -la "$W/models/Qwen-Image-2.1-Fun-Controlnet-Union/" | head -20
echo "[audit] start $(date -Is)"
cd "$C/scripts" && "$W/env/bin/python" 03_weight_audit.py
echo "[audit] rc=$? $(date -Is)"
