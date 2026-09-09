import os
import sys
import time
from huggingface_hub import hf_hub_download

# Use hf-mirror endpoint directly for high-speed download in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Avoid HF_HUB_ENABLE_HF_TRANSFER if hf_transfer isn't installed or configured, but if available, enable it
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "deepgenteam/DeepGen-1.0-diffusers"
local_dir = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"

files_to_download = [
    ("connector/model.safetensors", "Connector weights (~1.7GB)"),
    ("vlm/model-00001-of-00002.safetensors", "VLM weights part 1 (~4.9GB)"),
    ("vlm/model-00002-of-00002.safetensors", "VLM weights part 2 (~2.6GB)"),
    ("transformer/diffusion_pytorch_model.safetensors", "DiT Transformer weights (~4.9GB)"),
]

print("=== DEEPGEN MODEL WEIGHTS DOWNLOAD START ===", flush=True)
print(f"Target directory: {local_dir}")
print(f"Endpoint: {os.environ['HF_ENDPOINT']}\n", flush=True)

for i, (fname, desc) in enumerate(files_to_download, 1):
    dest_path = os.path.join(local_dir, fname)
    if os.path.exists(dest_path):
        size_gb = os.path.getsize(dest_path) / (1024**3)
        print(f"[{i}/{len(files_to_download)}] [ALREADY EXISTS] {fname} ({size_gb:.2f} GB) - {desc}", flush=True)
        continue
    
    print(f"[{i}/{len(files_to_download)}] [STARTING] {fname} ({desc}) ...", flush=True)
    t0 = time.time()
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.time() - t0
        size_gb = os.path.getsize(downloaded) / (1024**3)
        speed = (size_gb * 1024) / elapsed if elapsed > 0 else 0
        print(f"[{i}/{len(files_to_download)}] [COMPLETED] {fname}: {size_gb:.2f} GB in {elapsed:.1f}s ({speed:.2f} MB/s)\n", flush=True)
    except Exception as e:
        print(f"[{i}/{len(files_to_download)}] [FAILED] {fname}: {e}\n", flush=True)
        # Try fallback through local proxy if hf-mirror had an issue
        print("Retrying with local proxy http://127.0.0.1:5674 ...", flush=True)
        os.environ["http_proxy"] = "http://127.0.0.1:5674"
        os.environ["https_proxy"] = "http://127.0.0.1:5674"
        del os.environ["HF_ENDPOINT"]
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.time() - t0
        size_gb = os.path.getsize(downloaded) / (1024**3)
        speed = (size_gb * 1024) / elapsed if elapsed > 0 else 0
        print(f"[{i}/{len(files_to_download)}] [COMPLETED via PROXY] {fname}: {size_gb:.2f} GB in {elapsed:.1f}s ({speed:.2f} MB/s)\n", flush=True)

print("=== ALL MODEL WEIGHTS DOWNLOADED AND VERIFIED! ===", flush=True)
