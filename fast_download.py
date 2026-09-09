import os
import sys
import time
import hashlib
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

origin_url = "https://hf-mirror.com/deepgenteam/DeepGen-1.0-diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors"
target_file = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/transformer/diffusion_pytorch_model.safetensors"
tmp_file = target_file + ".download"
expected_sha256 = "54078ee51ab477275e94b610249ea912d6f80d24256c4197fe48945eef706883"

print("Resolving CDN download URL...", flush=True)
r = requests.head(origin_url, allow_redirects=True, timeout=30)
if r.status_code != 200:
    print(f"Failed to get headers: HTTP {r.status_code}")
    sys.exit(1)

cdn_url = r.url
total_size = int(r.headers.get("Content-Length", 0))
print(f"Target size: {total_size} bytes ({total_size / (1024**3):.2f} GB)", flush=True)

NUM_WORKERS = 16
chunk_size = (total_size + NUM_WORKERS - 1) // NUM_WORKERS

# Pre-create file with exact size
with open(tmp_file, "wb") as f:
    f.truncate(total_size)

downloaded_bytes = 0
lock = threading.Lock()
start_time = time.time()
last_report = start_time

def download_part(idx, start_byte, end_byte):
    global downloaded_bytes, last_report
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    for attempt in range(5):
        try:
            with requests.get(cdn_url, headers=headers, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp_file, "r+b") as out_f:
                    out_f.seek(start_byte)
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out_f.write(chunk)
                            with lock:
                                downloaded_bytes += len(chunk)
                                now = time.time()
                                if now - last_report >= 2.0:
                                    elapsed = now - start_time
                                    speed = downloaded_bytes / (1024 * 1024 * elapsed)
                                    pct = downloaded_bytes * 100 / total_size
                                    print(f"[{pct:.1f}%] {downloaded_bytes / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB  Speed: {speed:.1f} MB/s", flush=True)
                                    last_report = now
            return True
        except Exception as e:
            print(f"Worker {idx} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)
    return False

print(f"Starting {NUM_WORKERS} parallel download workers...", flush=True)
futures = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for i in range(NUM_WORKERS):
        s = i * chunk_size
        e = min(total_size - 1, (i + 1) * chunk_size - 1)
        futures.append(executor.submit(download_part, i, s, e))
    for f in as_completed(futures):
        if not f.result():
            print("A worker failed permanently!")
            sys.exit(1)

elapsed = time.time() - start_time
print(f"\nDownload completed in {elapsed:.1f}s! Average speed: {total_size / (1024**2 * elapsed):.1f} MB/s", flush=True)
print("Verifying SHA256 checksum...", flush=True)

h = hashlib.sha256()
with open(tmp_file, "rb") as f:
    while chunk := f.read(16 * 1024 * 1024):
        h.update(chunk)
calc_sha256 = h.hexdigest()
print(f"Calculated SHA256: {calc_sha256}")
print(f"Expected   SHA256: {expected_sha256}")

if calc_sha256 == expected_sha256:
    print("SUCCESS: SHA256 MATCHES! Moving to final location...", flush=True)
    os.replace(tmp_file, target_file)
    print(f"Target ready at: {target_file}", flush=True)
else:
    print("ERROR: SHA256 mismatch! Keeping tmp_file for inspection.")
    sys.exit(1)