#!/usr/bin/env python3
"""Precompute reusable Qwen dual-image shards, with verified continuation by sample."""

import argparse
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

from dataset_contract import file_sha256, fingerprint, load_export, read_export


def publish(path, data):
    """Publish a complete immutable file only after its contents reach disk."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete_on_close=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.link(tmp.name, path)


def write_record(path, record):
    publish(path, (json.dumps(record, indent=2) + "\n").encode())


def payload_sha(blob):
    header_end = 8 + int.from_bytes(blob[:8], "little")
    return hashlib.sha256(blob[header_end:]).hexdigest()


def cache_rows(root, ids, spec, encode, *, max_new=None):
    from safetensors import safe_open
    from safetensors.torch import save

    root.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(spec)
    with (root / ".lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec_path = root / "spec.json"
        if spec_path.is_file():
            if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
                raise ValueError("Existing cache has a different base, dataset, runtime, or seed")
        else:
            write_record(spec_path, spec)
        index_path = root / "index.json"
        committed = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
        if committed is not None and (committed["identity"] != identity or len(committed["rows"]) != len(ids)):
            raise ValueError("Cache index does not match its encoding contract")
        built, rows = 0, []
        for number, row_id in enumerate(ids):
            shard = root / f"{number:06d}.safetensors"
            if not shard.is_file():
                if committed is not None:
                    raise ValueError(f"Committed cache shard is missing: {shard}")
                if max_new is not None and built >= max_new:
                    break
                tensors, layout = encode(number, row_id)
                record = {"identity": identity, "id": row_id, "layout": json.dumps(layout)}
                record["payload_sha256"] = payload_sha(save(tensors))
                publish(shard, save(tensors, metadata=record))
                built += 1
            with safe_open(shard, framework="pt", device="cpu") as reader:
                fields = reader.metadata() or {}
            blob = shard.read_bytes()
            if (fields.get("id") != row_id or fields.get("identity") != identity
                    or payload_sha(blob) != fields.get("payload_sha256")):
                raise ValueError(f"Corrupt or incompatible cache shard: {shard}")
            digest = hashlib.sha256(blob).hexdigest()
            entry = {"id": row_id, "file": shard.name, "sha256": digest, **json.loads(fields["layout"])}
            if committed is not None and entry != committed["rows"][number]:
                raise ValueError(f"Cache shard differs from its committed index: {shard}")
            rows.append(entry)
        if len(rows) == len(ids) and committed is None:
            write_record(index_path, {"identity": identity, "spec": spec, "rows": rows})
        return {"pairs": len(ids), "ready": len(rows), "encoded": built, "reused": len(rows) - built,
                "complete": len(rows) == len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-sha256", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-pairs", type=int)
    parser.add_argument("--hf-cache")
    args = parser.parse_args()
    if args.max_new_pairs is not None and args.max_new_pairs < 0:
        parser.error("--max-new-pairs must be nonnegative")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import torch
    from qwen21_multiimg import CONDITION_COLUMNS, RESOLUTION, encode_pair

    export, paths = read_export(args.dataset, "train")
    if tuple(export["condition_columns"]) != CONDITION_COLUMNS:
        raise ValueError("Expected a source,pose dataset")
    # Compare the file list too; an incomplete SHA list cannot bind the pipeline.
    components = ("model_index.json", "processor", "scheduler", "text_encoder", "transformer", "vae")
    model_files = sorted(p.relative_to(args.base).as_posix() for name in components
                         for p in ([args.base / name] if (args.base / name).is_file()
                                   else (args.base / name).rglob("*")) if p.is_file())
    manifest = args.base_sha256.resolve()
    if [line[66:] for line in manifest.read_text().splitlines()] != model_files:
        raise ValueError("Base manifest must cover every Qwen pipeline file")
    subprocess.run(["sha256sum", "--strict", "--quiet", "--check", str(manifest)], cwd=args.base, check=True)
    rows = load_export(paths, "train", args.hf_cache)
    ids = list(rows.select_columns("id")["id"])
    if len(ids) != export["pairs"] or len(set(ids)) != len(ids):
        raise ValueError("Cache dataset has repeated pair IDs or disagrees with its declared size")
    spec = {"schema": 1, "base_manifest_sha256": file_sha256(manifest),
            "dataset_sha256": export["dataset_sha256"], "resolution": RESOLUTION,
            "condition_columns": list(CONDITION_COLUMNS), "dtype": "bf16", "seed": args.seed,
            "encoding_sha256": file_sha256(__file__),
            "encoder_helpers_sha256": file_sha256(Path(__file__).with_name("qwen21_multiimg.py")),
            "packages": {name: metadata.version(name) for name in ("diffusers", "torch", "transformers", "datasets")},
            "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    model = None

    def encode(index, row_id):
        nonlocal model
        if model is None:
            from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
            from transformers import Qwen3VLForConditionalGeneration

            opts = dict(torch_dtype=torch.bfloat16, local_files_only=True)
            vae = AutoencoderKLQwenImage21.from_pretrained(args.base, subfolder="vae", **opts).eval().to("cuda")
            text = Qwen3VLForConditionalGeneration.from_pretrained(
                args.base, subfolder="text_encoder", **opts).eval().to("cuda")
            pipe = QwenImage21Pipeline.from_pretrained(
                args.base, text_encoder=text, vae=None, transformer=None, scheduler=None, **opts)
            model = pipe, vae
        # Sampling the target is stable under interruption and independent of shard order.
        rng_seed = int(fingerprint([args.seed, row_id])[:16], 16)
        return encode_pair(rows[index], *model, torch.device("cuda:0"), rng_seed)

    print(json.dumps(cache_rows(args.out, ids, spec, encode, max_new=args.max_new_pairs)))


if __name__ == "__main__":
    main()
