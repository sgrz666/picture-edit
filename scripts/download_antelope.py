import os
import sys
import time
from pathlib import Path
import requests

FILES = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
]

TARGET_DIR = Path("/home/shangguanrz/project/pic-edit/models/antelopev2")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main"

def download_file(filename: str):
    dest = TARGET_DIR / filename
    temp_dest = TARGET_DIR / f".{filename}.tmp"
    url = f"{BASE_URL}/{filename}"
    
    # Check if already fully downloaded
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[SKIP] {filename} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"[START] Downloading {filename} from {url}...")
    headers = {}
    downloaded = 0
    if temp_dest.exists():
        downloaded = temp_dest.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        print(f"Resuming {filename} from {downloaded / 1e6:.1f} MB...")

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
    mode = "ab" if downloaded > 0 and response.status_code == 206 else "wb"
    if mode == "wb":
        downloaded = 0

    total = int(response.headers.get("content-length", 0)) + downloaded
    start_time = time.time()
    last_print = start_time

    with open(temp_dest, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 3:
                    pct = (downloaded / total * 100) if total else 0
                    speed = (downloaded) / (now - start_time) / 1e6
                    print(f"  {filename}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                    last_print = now

    temp_dest.replace(dest)
    print(f"[DONE] {filename} completed ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    for f in FILES:
        download_file(f)
    print("\nAll antelopev2 models ready:")
    for p in sorted(TARGET_DIR.glob("*.onnx")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
