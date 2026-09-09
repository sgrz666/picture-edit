import os
import sys
from huggingface_hub import HfApi

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo = "deepgenteam/DeepGen-1.0-diffusers"
local_dir = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"

api = HfApi()
try:
  repo_files = list(api.list_repo_tree(repo, recursive=True))
except Exception as e:
  print(f"Error fetching repo tree: {e}")
  # Fallback to local files check
  sys.exit(1)

all_ok = True
total_actual_bytes = 0
total_expected_bytes = 0

print(f"=== FULL AUDIT OF ALL FILES IN {repo} ===", flush=True)
for f in repo_files:
  if hasattr(f, "size") and f.size is not None:
    expected_size = f.size
    total_expected_bytes += expected_size
    local_path = os.path.join(local_dir, f.path)

    if not os.path.exists(local_path):
      print(
          f"[MISSING] {f.path:45} (expected {expected_size/(1024*1024):.2f} MB)",
          flush=True,
      )
      all_ok = False
    else:
      actual_size = os.path.getsize(local_path)
      total_actual_bytes += actual_size
      if actual_size != expected_size:
        print(
            f"[SIZE MISMATCH] {f.path:45} expected: {expected_size} B, got:"
            f" {actual_size} B",
            flush=True,
        )
        all_ok = False
      else:
        if actual_size >= 1024 * 1024 * 1024:
          size_str = f"{actual_size / (1024*1024*1024):.2f} GB"
        elif actual_size >= 1024 * 1024:
          size_str = f"{actual_size / (1024*1024):.2f} MB"
        else:
          size_str = f"{actual_size} B"
        print(f"[OK] {f.path:45} ({size_str})", flush=True)

print(
    f"\nTotal expected: {total_expected_bytes / (1024*1024*1024):.2f} GB",
    flush=True,
)
print(
    f"Total verified: {total_actual_bytes / (1024*1024*1024):.2f} GB",
    flush=True,
)

# Check for lingering incomplete downloads
cache_dir = os.path.join(local_dir, ".cache")
incomplete_files = []
if os.path.exists(cache_dir):
  for root, dirs, files in os.walk(cache_dir):
    for fl in files:
      if fl.endswith(".incomplete") or fl.endswith(".download"):
        p = os.path.join(root, fl)
        incomplete_files.append((p, os.path.getsize(p)))

if incomplete_files:
  print(
      f"\nLingering incomplete temp files in .cache: {len(incomplete_files)}",
      flush=True,
  )
  for p, sz in incomplete_files:
    print(f"  - {os.path.basename(p)} ({sz/(1024*1024):.2f} MB)", flush=True)
else:
  print("\nNo lingering incomplete download files in .cache.", flush=True)

if all_ok:
  print(
      "\n*** VERDICT: ALL MODEL WEIGHTS AND CONFIG FILES ARE 100% COMPLETELY"
      " DOWNLOADED! ***",
      flush=True,
  )
else:
  print(
      "\n*** VERDICT: Some files are missing or have mismatched sizes! ***",
      flush=True,
  )
