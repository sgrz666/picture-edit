#!/usr/bin/env bash
# 迁移基线双端 sha256 再生（阶段 C 之后执行，产出供阶段 D 三方终检的基线）
# 2026-10-25；对应 docs/模型跨盘迁移执行记录_20261025.md
# 源侧为改名后的 .old 目录，目标侧为 /data 实体；两侧均只读。
set -u
B=/home/shangguanrz/migration_baselines_20261025
mkdir -p "$B"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
hash_dir(){ ( cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -P 1 -I{} sha256sum {} ) > "$2" 2>>"$2.err"; }
PAIRS=(
"DeepGen-1.0-diffusers|/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers.old|/data/shangguanrz/pic-edit-models/repo-models/DeepGen-1.0-diffusers"
"antelopev2|/home/shangguanrz/project/pic-edit/models/antelopev2.old|/data/shangguanrz/pic-edit-models/repo-models/antelopev2"
"DWPose|/home/shangguanrz/project/pic-edit/models/DWPose.old|/data/shangguanrz/pic-edit-models/repo-models/DWPose"
"face_eval|/home/shangguanrz/project/pic-edit/models/face_eval.old|/data/shangguanrz/pic-edit-models/repo-models/face_eval"
"smpl|/home/shangguanrz/project/pic-edit/models/smpl.old|/data/shangguanrz/pic-edit-models/repo-models/smpl"
"home-models|/home/shangguanrz/models.old|/data/shangguanrz/models/home-models"
"tiktok_qwen21_full_20261003|/home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003.old|/data/shangguanrz/nihaochen/tiktok_qwen21_full_20261003"
"iper1024|/home/shangguanrz/nihaochen/iper1024.old|/data/shangguanrz/nihaochen/iper1024"
)
log "BASELINE_START"
for entry in "${PAIRS[@]}"; do
  IFS='|' read -r n s t <<<"$entry"
  [ -d "$s" ] || { log "MISSING_SRC $n ($s)"; continue; }
  [ -d "$t" ] || { log "MISSING_TGT $n ($t)"; continue; }
  log "START $n"
  hash_dir "$s" "$B/$n.src.sha256"
  hash_dir "$t" "$B/$n.dst.sha256"
  if diff -q "$B/$n.src.sha256" "$B/$n.dst.sha256" >/dev/null 2>&1; then
    log "OK $n files=$(wc -l <"$B/$n.src.sha256") agg=$(sha256sum "$B/$n.src.sha256" | cut -c1-16)"
  else
    log "MISMATCH $n"
    diff "$B/$n.src.sha256" "$B/$n.dst.sha256" | head -8
  fi
done
log "BASELINE_DONE"
