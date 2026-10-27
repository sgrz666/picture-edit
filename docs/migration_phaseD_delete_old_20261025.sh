#!/usr/bin/env bash
# 阶段 D：门控式删除 .old（唯一不可逆步骤）。默认 DRY_RUN=1 只演练不删。
# 用法：DRY_RUN=0 bash migration_phaseD_delete_old_20261025.sh
# 前置：migration_baseline_hash_20261025.sh 已成功跑完（基线存在于 ~/migration_baselines_20261025/）
# 2026-10-25；对应 docs/模型跨盘迁移执行记录_20261025.md
set -u
DRY_RUN="${DRY_RUN:-1}"
B=/home/shangguanrz/migration_baselines_20261025
P=/home/shangguanrz/project/pic-edit
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
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
hash_dir(){ ( cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -P 1 -I{} sha256sum {} ) > "$2" 2>/dev/null; }

echo "=== PHASE_D START $(date -Is) DRY_RUN=$DRY_RUN ==="
[ "$DRY_RUN" = "0" ] || log "DRY-RUN 模式：不执行任何删除"
FAIL=0
for entry in "${PAIRS[@]}"; do
  IFS='|' read -r n s t <<<"$entry"
  old="$s"; lnk="${s%.old}"; tgt="$t"
  log "---- $n ----"
  # 1. 结构断言（拦"删错对象"）
  if [ -L "$old" ]; then log "ABORT $n: OLD 是软链"; FAIL=1; continue; fi
  if [ ! -d "$old" ]; then log "ABORT $n: OLD 不是目录"; FAIL=1; continue; fi
  if [ ! -L "$lnk" ]; then log "ABORT $n: LNK 不是软链"; FAIL=1; continue; fi
  if [ "$(realpath "$lnk")" != "$tgt" ]; then log "ABORT $n: 软链指向不符 ($(realpath "$lnk"))"; FAIL=1; continue; fi
  if [ ! -d "$tgt" ]; then log "ABORT $n: 目标缺失"; FAIL=1; continue; fi
  log "PREFLIGHT_OK $n"
  # 2. 删除前全量哈希终检（源 vs 目标 vs 基线，三方吻合）
  A=/tmp/pd_$n.src; Dd=/tmp/pd_$n.dst
  hash_dir "$old" "$A"; hash_dir "$tgt" "$Dd"
  if ! diff -q "$A" "$Dd" >/dev/null 2>&1; then log "ABORT $n: 源!=目标"; FAIL=1; continue; fi
  if [ -s "$B/$n.src.sha256" ]; then
    if ! diff -q "$B/$n.src.sha256" "$A" >/dev/null 2>&1; then log "ABORT $n: 与基线不一致"; FAIL=1; continue; fi
    log "HASH_OK(三方) $n files=$(wc -l <"$A")"
  else
    log "HASH_OK(两方,无基线) $n files=$(wc -l <"$A")"
  fi
  # 3. 逐顶层条目删除 + 日志
  if [ "$DRY_RUN" = "0" ]; then
    ok=1; ec=0; fc=0
    cd "$old" || { log "ABORT $n: 无法进入 OLD"; FAIL=1; continue; }
    while IFS= read -r -d '' e; do
      ec=$((ec+1)); nf=$(find "$e" -type f | wc -l)
      if rm -rf -- "$e"; then fc=$((fc+nf)); log "deleted ($ec) $e [files=$nf]"; else log "DELETE_FAIL $e"; ok=0; break; fi
    done < <(find . -mindepth 1 -maxdepth 1 -print0)
    cd / || true
    if [ "$ok" = "1" ]; then
      if rmdir "$old"; then log "DELETED $n entries=$ec files=$fc"; else log "RMDIR_FAIL $n"; FAIL=1; fi
    else FAIL=1; fi
  else
    log "DRY: would delete $old ($(find "$old" -mindepth 1 -maxdepth 1 | wc -l) entries, $(find "$old" -type f | wc -l) files, $(du -sh "$old" | cut -f1))"
  fi
done
# 4. POST 复核
log "POST: .old remaining=$(ls -d "$P"/models/*.old /home/shangguanrz/models.old /home/shangguanrz/nihaochen/*.old 2>/dev/null | wc -l)"
log "POST: models broken links=$(find "$P/models" -xtype l 2>/dev/null | wc -l) (预期 2，既有失效链)"
log "POST: nihaochen broken links=$(find /home/shangguanrz/nihaochen -xtype l 2>/dev/null | wc -l)"
log "POST: df:"; df -h / /data | sed 's/^/    /'
echo "=== PHASE_D END FAIL=$FAIL ==="
[ "$FAIL" = "0" ] || exit 1
