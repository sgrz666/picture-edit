#!/usr/bin/env bash
# 在 sg 上引导 qwen21_union_migration_v1 的隔离运行条件：
#   1) 浅取固定 commit 的 VideoX-Fun 官方源码（不拉全历史）
#   2) 从现有 qwen21 解释器建 venv（--system-site-packages），只补少的包
#   3) 按固定 revision 下载 Union 权重并逐个记 sha256
# 现有 qwen21 环境不写入任何包（红线 7）。
set -uo pipefail

W=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1
QPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
VIDEOX_URL=https://github.com/aigc-apps/VideoX-Fun.git
VIDEOX_SHA=13bb3c625e8b1e457d7124d92207b49e00e4a6fe
UNION_REPO=alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union
UNION_REV=8a4702014d4dabb5f896fcba917e2ee0a961465f

TS=$(date +%Y%m%dT%H%M%S)
LOG="$W/logs/bootstrap_$TS.log"
mkdir -p "$W/models" "$W/source" "$W/env" "$W/cache" "$W/runs" "$W/logs" "$W/report"
exec >>"$LOG" 2>&1
echo "[bootstrap] start $(date -Is) log=$LOG"

# --- 1. 固定版本源码 ---
VX="$W/source/VideoX-Fun"
if [ ! -e "$VX/.git" ]; then
  mkdir -p "$VX"
  git -C "$VX" init -q
  git -C "$VX" remote add origin "$VIDEOX_URL"
fi
HEAD_NOW=$(git -C "$VX" rev-parse HEAD 2>/dev/null || echo none)
if [ "$HEAD_NOW" != "$VIDEOX_SHA" ]; then
  git -C "$VX" fetch -q --depth 1 origin "$VIDEOX_SHA"
  git -C "$VX" checkout -q FETCH_HEAD
fi
echo "[bootstrap] videox HEAD=$(git -C "$VX" rev-parse HEAD) (expect $VIDEOX_SHA)"
[ "$(git -C "$VX" rev-parse HEAD)" = "$VIDEOX_SHA" ] || { echo "[bootstrap] FAIL videox commit mismatch"; exit 1; }

# --- 2. 独立环境（键复用 qwen21 的 torch/diffusers/transformers） ---
if [ ! -x "$W/env/bin/python" ]; then
  "$QPY" -m venv --system-site-packages "$W/env"
fi
VPY="$W/env/bin/python"
echo "[bootstrap] venv python=$("$VPY" -c 'import sys;print(sys.version.split()[0])')"
BEFORE=$("$VPY" -c 'import torch,diffusers,transformers as t;print(torch.__version__,diffusers.__version__,t.__version__)')
echo "[bootstrap] core versions before: $BEFORE"

# 只装缺失项：pip 在 --system-site-packages 下会把基座已装的版本视为已满足
MISSING=$("$VPY" - <<'PY'
import importlib.util
pairs = [("huggingface_hub", "huggingface_hub"), ("safetensors", "safetensors"),
         ("accelerate", "accelerate"), ("einops", "einops"), ("omegaconf", "omegaconf"),
         ("opencv-python-headless", "cv2"), ("pillow", "PIL"), ("numpy", "numpy"),
         ("tqdm", "tqdm"), ("requests", "requests"), ("regex", "regex")]
miss = [p for p, m in pairs if importlib.util.find_spec(m) is None]
print(" ".join(miss))
PY
)
echo "[bootstrap] missing packages: [$MISSING]"
for pkg in $MISSING; do
  "$VPY" -m pip install -q "$pkg" && echo "[bootstrap] installed $pkg into venv"
done

AFTER=$("$VPY" -c 'import torch,diffusers,transformers as t;print(torch.__version__,diffusers.__version__,t.__version__)')
echo "[bootstrap] core versions after : $AFTER"
[ "$BEFORE" = "$AFTER" ] || echo "[bootstrap] WARN core versions changed -> venv is shadowing base packages"
"$VPY" -m pip freeze --all > "$W/env/pip_freeze.txt" 2>/dev/null
"$VPY" -m pip list --format=freeze > "$W/env/pip_list.txt" 2>/dev/null

# --- 3. Union 权重（固定 revision） ---
U="$W/models/Qwen-Image-2.1-Fun-Controlnet-Union"
mkdir -p "$U"
"$VPY" - "$UNION_REPO" "$UNION_REV" "$U" <<'PY'
import sys, json, hashlib, pathlib
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1], sys.argv[2], sys.argv[3]
p = snapshot_download(repo, revision=rev, local_dir=dst, max_workers=8,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.png", "*.jpg", "*.md", "*.yaml"])
print("snapshot_download resolved commit:", p)
files = sorted(pathlib.Path(dst).rglob("*"))
out = []
for f in files:
    if f.is_file() and ".cache" not in f.parts:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        out.append({"path": str(f.relative_to(dst)), "bytes": f.stat().st_size, "sha256": h})
        print(f"  {f.stat().st_size/1048576:12.2f} MB  {out[-1]['path']}  {h[:16]}", flush=True)
json.dump(out, open(pathlib.Path(dst) / "_checksums.json", "w"), indent=1)
print("files:", len(out), "total GiB: %.3f" % (sum(e["bytes"] for e in out) / 1073741824))
PY

# --- 4. 版本与依赖记录 ---
"$VPY" - "$VIDEOX_SHA" "$UNION_REV" "$W" <<'PY'
import json, sys, subprocess, pathlib, torch
vx, rev, W = sys.argv[1], sys.argv[2], sys.argv[3]
import diffusers, transformers
u = pathlib.Path(W) / "models/Qwen-Image-2.1-Fun-Controlnet-Union"
ck = json.load(open(u / "_checksums.json")) if (u / "_checksums.json").exists() else []
rec = {
  "videoxfun_commit": vx,
  "union_repo": "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
  "union_revision": rev,
  "union_files": len(ck),
  "union_bytes": sum(e["bytes"] for e in ck),
  "torch": torch.__version__,
  "diffusers": diffusers.__version__,
  "transformers": transformers.__version__,
  "cuda_available": torch.cuda.is_available(),
  "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "env_python": str(pathlib.Path(W) / "env/bin/python"),
  "base_qwen": "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1",
}
json.dump(rec, open(pathlib.Path(W) / "versions.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
PY

echo "[bootstrap] done $(date -Is)"
