"""Count stored tensor elements from public HF safetensors headers only.

No torch, model execution, or full weight download is required. Counts are
checkpoint tensor elements, not a live model's deduplicated parameter numel.
Example: python audit_checkpoint_headers.py --output checkpoint_audit.json
"""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.parse
import urllib.request

DEFAULT_REPOS = [
    "deepgenteam/DeepGen-1.0-diffusers",
    "black-forest-labs/FLUX.2-klein-base-4B",
    "meituan-longcat/LongCat-Image-Edit",
]


def read_exact(stream, size):
    parts = []
    remaining = size
    while remaining:
        data = stream.read(remaining)
        if not data:
            raise EOFError("Truncated safetensors header")
        parts.append(data)
        remaining -= len(data)
    return b"".join(parts)


def audit_file(args):
    repo, revision, filename = args
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
    # Open a bounded range, then close after the header even if Range is ignored.
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
    with urllib.request.urlopen(req, timeout=45) as stream:
        length = struct.unpack("<Q", read_exact(stream, 8))[0]
        if not 2 <= length <= 1048568:
            raise ValueError(f"Unexpected/oversized header: {filename}: {length}")
        raw = read_exact(stream, length)
    header = json.loads(raw)
    dtypes = collections.Counter()
    elements = storage_bytes = tensors = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        count = math.prod(value["shape"])
        elements += count
        dtypes[value["dtype"]] += count
        start, end = value["data_offsets"]
        storage_bytes += end - start
        tensors += 1
    return {"file": filename, "url": url, "tensor_elements": elements,
            "tensors": tensors, "dtype_elements": dict(dtypes),
            "stored_tensor_bytes": storage_bytes, "header_bytes": length,
            "header_sha256": hashlib.sha256(raw).hexdigest()}


def audit_repo(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as stream:
        metadata = json.load(stream)
    revision = metadata["sha"]
    filenames = sorted(x["rfilename"] for x in metadata["siblings"]
                       if x["rfilename"].endswith(".safetensors"))
    # FLUX distributes the same transformer as a root single file AND as a
    # Diffusers component. Count the component representation exactly once.
    excluded = []
    if any(f.startswith("transformer/") for f in filenames):
        excluded = [f for f in filenames if "/" not in f]
        filenames = [f for f in filenames if f not in excluded]
    if not filenames:
        raise ValueError(f"No safetensors found: {repo}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(audit_file, [(repo, revision, f) for f in filenames]))
    groups = collections.Counter()
    for entry in files:
        groups[entry["file"].split("/")[0]] += entry["tensor_elements"]
    total = sum(groups.values())
    return {"repo": repo, "revision": revision, "components": dict(groups),
            "excluded_root_alternative_weights": excluded,
            "total_tensor_elements": total, "bf16_equivalent_GiB": total * 2 / 2**30,
            "stored_tensor_bytes": sum(x["stored_tensor_bytes"] for x in files),
            "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", help="Repeat to override default repositories")
    parser.add_argument("--output", type=Path, default=Path("checkpoint_audit.json"))
    args = parser.parse_args()
    output = {"accessed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "method": "Public safetensors header shape products; no model or training run",
              "caveats": "Stored tensors may include buffers; tied parameters require live model verification. Root alternative weights are excluded when a transformer component exists. Other repository layouts/variants require manual review.",
              "repositories": [], "errors": []}
    for repo in args.repo or DEFAULT_REPOS:
        try:
            result = audit_repo(repo)
            output["repositories"].append(result)
            print(repo, result["components"], flush=True)
        except Exception as exc:
            output["errors"].append({"repo": repo, "error": str(exc)})
            print(repo, "ERROR", str(exc), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", args.output.resolve())
    if output["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
