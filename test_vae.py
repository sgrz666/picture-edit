import os
import time

os.environ["http_proxy"] = "http://127.0.0.1:5674"
os.environ["https_proxy"] = "http://127.0.0.1:5674"

from huggingface_hub import hf_hub_download

start = time.time()
path = hf_hub_download(
    repo_id="deepgenteam/DeepGen-1.0-diffusers",
    filename="vae/diffusion_pytorch_model.safetensors",
    local_dir="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
)
elapsed = time.time() - start
size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"Downloaded {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/elapsed:.2f} MB/s) to {path}")