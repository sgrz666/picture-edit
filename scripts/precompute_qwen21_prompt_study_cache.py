#!/usr/bin/env python3
"""Precompute per-prompt source cache and train target latents for the study.

Strict protocol isolation: cache directories are separated by protocol (p0/pg/pb/pbg).
Target latent cache is computed ONLY for train; val/test/diagnostic NEVER encode target latents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    PROTOCOLS, P0_PROMPT, atomic_json, sha256_file, directory_artifact, load_study_lock,
    text_sha256, verify_artifact,
)
from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
from src.pose_control.qwen21_full.data_runtime import pair_metadata
from src.pose_control.qwen21_full.qwen_runtime import (
    dataset_image_to_pil,
    Qwen21IPERCache,
    PROMPT,
)
from src.pose_control.qwen21_full.execution import MODEL
from scripts.precompute_qwen21_pei2i_prompts import split_manifest


def cache_split(pipe: Any, cache: Any, dataset: Any, bank: PromptBank, *, split: str) -> dict[str, int]:
    if pipe is None:
        raise ValueError("A loaded Qwen pipeline is required for cache precompute")
    if split not in ("train", "val", "test", "diagnostic"):
        raise ValueError(f"Unsupported cache split: {split}")
    counts = {"source_calls": 0, "target_calls": 0}
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample.get("pair_metadata", sample)
        pair_id = meta.get("pair_id")
        if not pair_id:
            pair_id = pair_metadata(meta)["pair_id"]
        cache.source(
            pipe,
            dataset_image_to_pil(sample["src_image"]),
            prompt=bank.prompt_for(pair_id),
        )
        counts["source_calls"] += 1
        if split == "train":
            cache.target(pipe, dataset_image_to_pil(sample["tgt_image"]))
            counts["target_calls"] += 1
    return counts


def load_pipeline(model_dir: Path):
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"Qwen model directory missing: {model_dir}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen source/latent cache precompute")
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
        module.eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    pipe.vae.to("cuda")
    return pipe


def validate_cache_manifest(run_dir: Path, protocol: str, splits, *, verify_entries: bool = True) -> dict[str, Any]:
    """CPU check of cache provenance and every recorded file for requested splits."""
    run_dir = Path(run_dir).resolve()
    cache_root = run_dir / "cache" / protocol
    try:
        if protocol not in PROTOCOLS:
            raise ValueError("unknown protocol")
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["schema"] != "qwen21_prompt_study_cache_v1" or manifest["protocol"] != protocol
                or manifest["status"] != "complete" or manifest["resolution"] != "native"
                or manifest["preprocessing"] != "native_v1"):
            raise ValueError("incomplete cache or incompatible policy")
        if text_sha256(json.dumps(manifest["model_files_sha256"], sort_keys=True)) != manifest["model_manifest_sha256"]:
            raise ValueError("model manifest hash")
        lock_path = run_dir / "data/lock.json"
        if lock_path.is_file():
            lock = load_study_lock(lock_path, verify_assets=False)
            artifact = lock["artifacts"]["model"]
            verify_artifact(artifact)
            if (manifest["model_manifest_sha256"] != artifact["manifest_sha256"]
                    or Path(manifest["model_dir"]).resolve() != Path(artifact["path"]).resolve()):
                raise ValueError("locked model provenance")
        for split in splits:
            info = manifest["splits"][split]
            split_path = split_manifest(run_dir, split)
            rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            pair_ids = [str(row.get("pair_id") or pair_metadata(row)["pair_id"]) for row in rows]
            prompt_path = run_dir / "prompts" / protocol / f"{split}.jsonl"
            bank = PromptBank(prompt_path, protocol, pair_ids)
            if (info["status"] != "complete" or info["split_sha256"] != sha256_file(split_path)
                    or info["prompt_manifest_sha256"] != sha256_file(prompt_path)
                    or info["source_calls"] != len(pair_ids)
                    or info["target_calls"] != (len(pair_ids) if split == "train" else 0)):
                raise ValueError(f"split/prompt/count provenance: {split}")
            files = info["files"]
            if [str(row["pair_id"]) for row in files] != pair_ids or len(set(pair_ids)) != len(pair_ids):
                raise ValueError("cache file coverage/order")
            for row in files:
                if split != "train" and "target" in row:
                    raise ValueError("evaluation target cache forbidden")
                for kind in (("source", "target") if split == "train" else ("source",)):
                    entry = row[kind]
                    relative = Path(entry["path"])
                    path = (run_dir / relative).resolve()
                    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(cache_root.resolve()):
                        raise ValueError("cache path escapes protocol root")
                    if not path.is_file() or sha256_file(path) != entry["sha256"]:
                        raise ValueError(f"missing/corrupt cache file: {path}")
            if verify_entries:
                from src.pose_control.qwen21_full.execution import QwenFullDataset
                from src.pose_control.qwen21_full.data_runtime import validate_study_cache
                dataset = QwenFullDataset(split_path, resolution="native", align_mode="native", dataset_type="tiktok_v3")
                cache = Qwen21IPERCache(cache_root, resolution="native", prompt=P0_PROMPT if protocol == "p0" else PROMPT,
                                       preprocessing="native_v1")
                validate_study_cache(cache, dataset, bank, indices=range(len(dataset)), require_target=split == "train")
        return manifest
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid cache provenance for {protocol}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Qwen prompt study source and train target caches")
    parser.add_argument("--run", type=Path, required=True, help="Experiment run root")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic"])
    parser.add_argument("--model", type=Path, default=MODEL, help="Qwen model directory")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if (len(args.splits) != len(set(args.splits)) or len(args.protocols) != len(set(args.protocols))
            or any(split not in ("train", "val", "test", "diagnostic") for split in args.splits)):
        raise ValueError("Unsupported or duplicate cache splits/protocols")
    if "test" in args.splits:
        from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
        require_eight_arms_completed(run_dir)
    split_files = {split: split_manifest(run_dir, split) for split in args.splits}
    from src.pose_control.qwen21_full.execution import QwenFullDataset
    lock_path = run_dir / "data/lock.json"
    lock = load_study_lock(lock_path, verify_assets=True) if lock_path.is_file() else None
    model_artifact = directory_artifact(args.model, revision=(lock["artifacts"]["model"]["revision"] if lock else None))
    if lock and (Path(model_artifact["path"]) != Path(lock["artifacts"]["model"]["path"]).resolve()
                 or model_artifact["manifest_sha256"] != lock["artifacts"]["model"]["manifest_sha256"]):
        raise ValueError("Cache model provenance differs from locked Qwen model")
    model_hashes = {name: value["sha256"] for name, value in model_artifact["files"].items()}
    prepared = {}
    manifests = {}
    for proto in args.protocols:
        cache_dir = run_dir / "cache" / proto
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        if manifest and (manifest.get("model_manifest_sha256") != model_artifact["manifest_sha256"]
                         or manifest.get("model_dir") != model_artifact["path"]
                         or manifest.get("protocol") != proto
                         or manifest.get("resolution") != "native"
                         or manifest.get("preprocessing") != "native_v1"):
            raise ValueError(f"Cache model provenance/policy differs for {proto}; use a new cache directory")
        if manifest is None and cache_dir.is_dir() and any(cache_dir.glob("*.pt")):
            raise ValueError(f"Existing cache has no model provenance: {cache_dir}")
        manifests[proto] = manifest or {"schema": "qwen21_prompt_study_cache_v1", "protocol": proto,
            "model_dir": model_artifact["path"], "model_revision": model_artifact["revision"],
            "model_manifest_sha256": model_artifact["manifest_sha256"], "model_files_sha256": model_hashes,
            "resolution": "native", "preprocessing": "native_v1", "splits": {}}
        for split in args.splits:
            prompt_path = run_dir / "prompts" / proto / f"{split}.jsonl"
            if not prompt_path.is_file():
                raise FileNotFoundError(f"Missing prompt manifest for {proto}/{split}: {prompt_path}")
            dataset = QwenFullDataset(split_files[split], resolution="native", align_mode="native", dataset_type="tiktok_v3")
            expected_ids = [pair_metadata(sample)["pair_id"] for sample in dataset.samples]
            bank = PromptBank(prompt_path, proto, expected_ids)
            prepared[proto, split] = (dataset, bank, prompt_path)
    # All CPU/data/provenance checks finish before allocating the pipeline.
    pipe = load_pipeline(args.model)
    if pipe is None:
        raise RuntimeError("Qwen pipeline failed to load")
    summary: dict[str, Any] = {}
    for proto in args.protocols:
        cache_dir = run_dir / "cache" / proto
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = Qwen21IPERCache(
            cache_dir,
            resolution="native",
            prompt=P0_PROMPT if proto == "p0" else PROMPT,
            preprocessing="native_v1",
        )
        proto_counts: dict[str, Any] = {}
        manifest = manifests[proto]
        manifest_path = cache_dir / "manifest.json"
        for split in args.splits:
            split_file = split_files[split]
            dataset, bank, prompt_path = prepared[proto, split]
            split_info = {"status": "running", "split_sha256": sha256_file(split_file),
                          "prompt_manifest_sha256": sha256_file(prompt_path), "source_calls": 0,
                          "target_calls": 0, "files": []}
            manifest["splits"][split] = split_info
            manifest["status"] = "running"
            atomic_json(manifest_path, manifest)
            try:
                counts = cache_split(pipe, cache, dataset, bank, split=split)
                from src.pose_control.qwen21_full.data_runtime import validate_study_cache
                validate_study_cache(cache, dataset, bank, indices=range(len(dataset)), require_target=split == "train")
                files = []
                for index in range(len(dataset)):
                    sample = dataset[index]
                    meta = sample.get("pair_metadata", sample)
                    pair_id = meta.get("pair_id") or pair_metadata(meta)["pair_id"]
                    entry = {"pair_id": pair_id}
                    source = cache._path(dataset_image_to_pil(sample["src_image"]), "source", prompt=bank.prompt_for(pair_id))
                    if not source.is_file():
                        raise RuntimeError(f"Source cache was not created: {source}")
                    entry["source"] = {"path": source.relative_to(run_dir).as_posix(), "sha256": sha256_file(source)}
                    if split == "train":
                        target = cache._path(dataset_image_to_pil(sample["tgt_image"]), "target")
                        if not target.is_file():
                            raise RuntimeError(f"Train target cache was not created: {target}")
                        entry["target"] = {"path": target.relative_to(run_dir).as_posix(), "sha256": sha256_file(target)}
                    files.append(entry)
                split_info.update(counts, status="complete", files=files)
                manifest["status"] = "complete" if all(item["status"] == "complete" for item in manifest["splits"].values()) else "incomplete"
                atomic_json(manifest_path, manifest)
            except Exception as exc:
                split_info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                manifest["status"] = "failed"
                atomic_json(manifest_path, manifest)
                raise
            proto_counts[split] = counts

        summary[proto] = proto_counts

    print("Precompute cache finished:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
