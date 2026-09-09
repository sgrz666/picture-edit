import os
import sys
import time
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

local_dir = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
print(f"Starting download to {local_dir}...")
start_time = time.time()

snapshot_download(
    repo_id="deepgenteam/DeepGen-1.0-diffusers",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

elapsed = time.time() - start_time
print(f"Download finished successfully in {elapsed:.1f}s!")