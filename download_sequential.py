import os
import sys
import time
from huggingface_hub import hf_hub_download

os.environ["http_proxy"] = "http://127.0.0.1:5674"
os.environ["https_proxy"] = "http://127.0.0.1:5674"

repo_id = "deepgenteam/DeepGen-1.0-diffusers"
local_dir = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"

files = [
    "vae/diffusion_pytorch_model.safetensors",
    "connector/model.safetensors",
    "vlm/model-00002-of-00002.safetensors",
    "vlm/model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
]

print("=== STARTING SEQUENTIAL DOWNLOAD ===", flush=True)
for i, f in enumerate(files, 1):
    dest = os.path.join(local_dir, f)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024*1024)
        print(f"[{i}/{len(files)}] [ALREADY EXISTS] {f} ({size_mb:.1f} MB)", flush=True)
        continue
    
    print(f"[{i}/{len(files)}] [DOWNLOADING] {f} ...", flush=True)
    t0 = time.time()
    hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest) / (1024*1024)
    speed = size_mb / elapsed if elapsed > 0 else 0
    print(f"[{i}/{len(files)}] [COMPLETE] {f}: {size_mb:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)", flush=True)

print("=== ALL MODEL WEIGHTS DOWNLOADED SUCCESSFULLY! ===", flush=True)