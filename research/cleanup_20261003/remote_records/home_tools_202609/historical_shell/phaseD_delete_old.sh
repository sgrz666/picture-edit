#!/usr/bin/env bash
# Phase D: final verification, then delete Hi4D-origin.old
# Gate: PREFLIGHT + FULL SHA256 (source .old vs /data target) must pass, else ABORT.
set -u

D=/home/shangguanrz/project/pic-edit/datasets
OLD="$D/Hi4D-origin.old"
LNK="$D/Hi4D-origin"
TGT=/data/shangguanrz/pic-edit-datasets/Hi4D-origin
BASE=/home/shangguanrz/hi4d_origin_sha256_baseline.txt
PY=/home/shangguanrz/miniconda3/envs/deepgen/bin/python
LOG=/home/shangguanrz/delete_hi4d_old.log

log(){ echo "[$(date +%H:%M:%S)] $*"; }

echo "=== PHASE_D START $(date -Is) ==="

log "--- PREFLIGHT ---"
if [ -L "$OLD" ]; then log "ABORT: OLD is a symlink"; exit 1; fi
if [ ! -d "$OLD" ]; then log "ABORT: OLD is not a directory"; exit 1; fi
if [ ! -L "$LNK" ]; then log "ABORT: LNK is not a symlink"; exit 1; fi
RP=$(realpath "$LNK" 2>/dev/null)
if [ "$RP" != "$TGT" ]; then log "ABORT: realpath mismatch [$RP]"; exit 1; fi
N=$(ls -1 "$TGT" 2>/dev/null | wc -l)
if [ "$N" -ne 34 ]; then log "ABORT: target entries=$N (expect 34)"; exit 1; fi
if [ ! -d "$TGT" ]; then log "ABORT: target missing"; exit 1; fi
log "PREFLIGHT=OK link_target=$RP target_entries=$N"

log "--- FINAL FULL SHA256 VERIFY ---"
A=/tmp/d_src.sha256
B=/tmp/d_dst.sha256
( cd "$OLD" && ls -1 | LC_ALL=C sort | xargs -d '\n' -I{} sha256sum {} ) > "$A" 2>/dev/null
log "source hashed: $(wc -l < "$A") files"
( cd "$TGT" && ls -1 | LC_ALL=C sort | xargs -d '\n' -I{} sha256sum {} ) > "$B" 2>/dev/null
log "target hashed: $(wc -l < "$B") files"

if [ ! -s "$A" ] || [ ! -s "$B" ]; then log "ABORT: empty hash file"; exit 1; fi
if ! diff -q <(LC_ALL=C sort -k2 "$A") <(LC_ALL=C sort -k2 "$B") >/dev/null; then
  log "ABORT: HASH MISMATCH between .old and /data -- refusing to delete"
  diff <(LC_ALL=C sort -k2 "$A") <(LC_ALL=C sort -k2 "$B") | head -20
  exit 1
fi
SRC_AGG=$(LC_ALL=C sort -k2 "$A" | sha256sum | awk '{print $1}')
DST_AGG=$(LC_ALL=C sort -k2 "$B" | sha256sum | awk '{print $1}')
log "HASH_OK src_agg=$SRC_AGG dst_agg=$DST_AGG"
if [ -f "$BASE" ]; then
  B_AGG=$(LC_ALL=C sort -k2 "$BASE" | sha256sum | awk '{print $1}')
  if [ "$B_AGG" = "$DST_AGG" ]; then log "BASELINE_AGG=$B_AGG MATCH"; else log "BASELINE_AGG=$B_AGG DIFFER(!)"; exit 1; fi
else
  log "BASELINE file not found, skipped"
fi

log "--- DF BEFORE ---"
df -h / | tail -1
df -h /data | tail -1

log "--- DELETE (per-file, sequential) ---"
cd "$OLD" || { log "ABORT: cannot cd into OLD"; exit 1; }
total=$(ls -1 | wc -l)
i=0
for f in $(ls -1); do
  i=$((i+1))
  if ! rm -f -- "./$f"; then log "DELETE_FAIL ($i/$total) $f"; exit 2; fi
  log "deleted ($i/$total) $f"
done
cd / || exit 3
if rmdir "$OLD"; then log "rmdir OK"; else log "rmdir FAIL (maybe leftover)"; ls -la "$OLD" 2>/dev/null | head; fi

log "--- POST VERIFY ---"
if [ -e "$OLD" ]; then log "POST: OLD still exists(!)"; else log "POST: OLD removed"; fi
if [ -L "$LNK" ]; then log "POST: LNK intact -> $(realpath "$LNK")"; else log "POST: LNK BROKEN(!)"; fi
N2=$(ls -1 "$TGT" 2>/dev/null | wc -l)
log "POST: target_entries=$N2"

log "--- SMOKE TEST through symlink ---"
ls -1 "$LNK" 2>/dev/null | head -3
log "raw ls count via symlink = $(ls -1 "$LNK" 2>/dev/null | wc -l)"
if [ -x "$PY" ]; then
  "$PY" - <<'PYEOF'
import os
p = '/home/shangguanrz/project/pic-edit/datasets/Hi4D-origin'
print('isdir   :', os.path.isdir(p))
print('listdir :', len(os.listdir(p)))
print('realpath:', os.path.realpath(p))
print('sample  :', sorted(os.listdir(p))[:3])
PYEOF
fi

log "--- DF AFTER ---"
df -h / | tail -1
df -h /data | tail -1
echo "=== PHASE_D END $(date -Is) ==="
