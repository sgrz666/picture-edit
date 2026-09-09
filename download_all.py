import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

os.environ["http_proxy"] = "http://127.0.0.1:5674"
os.environ["https_proxy"] = "http://127.0.0.1:5674"

repo_id = "deepgenteam/DeepGen-1.0-diffusers"
local_dir = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"

api = HfApi()
repo_files = api.list_repo_files(repo_id)
print(f"Total files in repo: {len(repo_files)}")

# Separate small files and large files
large_files = [f for f in repo_files if f.endswith(".safetensors")]
small_files = [f for f in repo_files if not f.endswith(".safetensors")]

print(f"Downloading {len(small_files)} config/metadata files first...")
for f in small_files:
    hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
print("Config and metadata files downloaded successfully!")

def download_file(filename):
    t0 = time.time()
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        print(f"[Skip] {filename} already exists ({os.path.getsize(dest)/(1024*1024):.1f} MB)")
        return filename, 0
    print(f"[Start] Downloading {filename}...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest) / (1024*1024)
    print(f"[Done] {filename}: {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/elapsed:.2f} MB/s)")
    return filename, elapsed

print(f"\nDownloading {len(large_files)} large model weight files in parallel...")
t_start = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(download_file, f): f for f in large_files}
    for future in as_completed(futures):
        f = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error downloading {f}: {e}")

total_elapsed = time.time() - t_start
print(f"\nAll downloads completed in {total_elapsed:.1f}s!")