#!/usr/bin/env python3
"""Verify DeepGen weight integrity on remote server."""
import struct
import json
import os
import hashlib
import math

MODEL_DIR = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"

# Expected values from checkpoint_audit.json
EXPECTED = {
    "connector/model.safetensors": {"elements": 432449532, "tensors": 103},
    "transformer/diffusion_pytorch_model.safetensors": {"elements": 2469663936, "tensors": 909},
    "vae/diffusion_pytorch_model.safetensors": {"elements": 83819683, "tensors": 244},
    "vlm/model-00001-of-00002.safetensors": {"elements": 2498840576, "tensors": 628},
    "vlm/model-00002-of-00002.safetensors": {"elements": 1255782400, "tensors": 196},
}

results = {}
for rel_path, expect in EXPECTED.items():
    full_path = os.path.join(MODEL_DIR, rel_path)
    entry = {"path": full_path, "exists": os.path.exists(full_path)}
    if not entry["exists"]:
        entry["status"] = "MISSING"
        results[rel_path] = entry
        continue

    entry["file_bytes"] = os.path.getsize(full_path)

    try:
        with open(full_path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(hlen)
            header = json.loads(raw)

        keys = [k for k in header if k != "__metadata__"]
        total_elements = 0
        for k in keys:
            total_elements += math.prod(header[k]["shape"])

        entry["header_bytes"] = hlen
        entry["tensor_count"] = len(keys)
        entry["tensor_elements"] = total_elements
        entry["expected_tensors"] = expect["tensors"]
        entry["expected_elements"] = expect["elements"]
        entry["tensors_match"] = len(keys) == expect["tensors"]
        entry["elements_match"] = total_elements == expect["elements"]

        # Check if data region is complete
        max_offset = 0
        for k in keys:
            start, end = header[k]["data_offsets"]
            if end > max_offset:
                max_offset = end
        expected_file_size = 8 + hlen + max_offset
        entry["expected_file_size"] = expected_file_size
        entry["data_complete"] = entry["file_bytes"] >= expected_file_size

        if entry["tensors_match"] and entry["elements_match"] and entry["data_complete"]:
            entry["status"] = "OK"
        else:
            entry["status"] = "CORRUPT/INCOMPLETE"
    except Exception as e:
        entry["status"] = f"ERROR: {e}"

    results[rel_path] = entry

print("=" * 70)
print("DeepGen Weight Integrity Report")
print("=" * 70)
for rel_path, info in results.items():
    status = info["status"]
    icon = "✓" if status == "OK" else "✗"
    print(f"\n{icon} {rel_path}: {status}")
    if "file_bytes" in info:
        print(f"  File size: {info['file_bytes']:,} bytes")
    if "tensor_count" in info:
        print(f"  Tensors:   {info['tensor_count']} (expected {info.get('expected_tensors','?')})")
        print(f"  Elements:  {info['tensor_elements']:,} (expected {info.get('expected_elements','?'):,})")
    if "data_complete" in info:
        print(f"  Data region complete: {info['data_complete']} (file {info['file_bytes']:,} vs needed {info.get('expected_file_size',0):,})")

all_ok = all(v["status"] == "OK" for v in results.values())
print("\n" + "=" * 70)
if all_ok:
    print("RESULT: All weights verified successfully!")
else:
    missing = [k for k,v in results.items() if v["status"] == "MISSING"]
    broken = [k for k,v in results.items() if v["status"] not in ("OK", "MISSING")]
    if missing:
        print(f"MISSING files: {missing}")
    if broken:
        print(f"CORRUPT/ERROR files: {broken}")
    print("RESULT: Weight verification FAILED")
