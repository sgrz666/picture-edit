import os
import sys
import time
import hashlib
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

origin_url = "https://hf-mirror.com/deepgenteam/DeepGen-1.0-diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors"
target_file = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/transformer/diffusion_pytorch_model.safetensors"
expected_sha256 = "54078ee51ab477275e94b610249ea912d6f80d24256c4197fe48945eef706883"

print("1. Resolving CDN URL...", flush=True)
r = requests.head(origin_url, allow_redirects=True, timeout=30)
cdn_url = r.url
total_size = int(r.headers.get("Content-Length", 4939433672))
print(f"Total size: {total_size} bytes ({total_size / (1024**3):.2f} GB)", flush=True)

NUM_WORKERS = 4
chunk_len = (total_size + NUM_WORKERS - 1) // NUM_WORKERS

# Remove any old file and pre-allocate cleanly
if os.path.exists(target_file):
    os.remove(target_file)
with open(target_file, "wb") as f:
    f.truncate(total_size)

progress = [0] * NUM_WORKERS
worker_ranges = []
for i in range(NUM_WORKERS):
    s = i * chunk_len
    e = min(total_size - 1, (i + 1) * chunk_len - 1)
    worker_ranges.append((s, e))

start_time = time.time()

def worker_task(worker_id, start_b, end_b):
    cur_b = start_b
    while cur_b <= end_b:
        headers = {"Range": f"bytes={cur_b}-{end_b}"}
        try:
            with requests.get(cdn_url, headers=headers, stream=True, timeout=30) as resp:
                if resp.status_code not in (200, 206):
                    time.sleep(2)
                    continue
                with open(target_file, "r+b") as out_f:
                    out_f.seek(cur_b)
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out_f.write(chunk)
                            cur_b += len(chunk)
                            progress[worker_id] = cur_b - start_b
        except Exception as e:
            time.sleep(1.5)
    return True

def monitor():
    while any(progress[i] < (worker_ranges[i][1] - worker_ranges[i][0] + 1) for i in range(NUM_WORKERS)):
        time.sleep(3)
        now = time.time()
        elapsed = now - start_time
        downloaded = sum(progress)
        speed = downloaded / (1024 * 1024 * max(0.1, elapsed))
        pct = downloaded * 100 / total_size
        parts_str = " | ".join(f"W{i}:{progress[i]/(1024**2):.0f}M" for i in range(NUM_WORKERS))
        print(f"[{pct:.1f}%] {downloaded/(1024**2):.1f}MB/{total_size/(1024**2):.1f}MB ({speed:.1f} MB/s) [{parts_str}]", flush=True)

mon_t = threading.Thread(target=monitor, daemon=True)
mon_t.start()

print(f"2. Launching {NUM_WORKERS} workers with persistent range resume...", flush=True)
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    futures = [pool.submit(worker_task, i, r[0], r[1]) for i, r in enumerate(worker_ranges)]
    for f in futures:
        f.result()

print("\n3. All 4 workers completed! Computing SHA256...", flush=True)
h = hashlib.sha256()
with open(target_file, "rb") as f:
    while b := f.read(16 * 1024 * 1024):
        h.update(b)
calc_sha256 = h.hexdigest()
print(f"Calculated: {calc_sha256}")
print(f"Expected:   {expected_sha256}")

if calc_sha256 == expected_sha256:
    print("SUCCESS: 100% PERFECT SHA256 MATCH!", flush=True)
else:
    print("FATAL: SHA256 mismatch!")
    sys.exit(1)