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

from src.pose_control.qwen21_full.prompt_study.protocol import PROTOCOLS, P0_PROMPT
from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
from src.pose_control.qwen21_full.data_runtime import pair_metadata
from src.pose_control.qwen21_full.qwen_runtime import (
    dataset_image_to_pil,
    Qwen21IPERCache,
    PROMPT,
)
from src.pose_control.qwen21_full.execution import MODEL


def cache_split(pipe: Any, cache: Any, dataset: Any, bank: PromptBank, *, split: str) -> dict[str, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Qwen prompt study source and train target caches")
    parser.add_argument("--run", type=Path, required=True, help="Experiment run root")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic"])
    parser.add_argument("--model", type=Path, default=MODEL, help="Qwen model directory")
    args = parser.parse_args()

    run_dir = Path(args.run)
    data_dir = run_dir / "data"

    # Only load pipeline if model directory exists and torch cuda is available
    pipe = None
    if args.model and Path(args.model).is_dir():
        try:
            import torch
            from diffusers import QwenImage21Pipeline

            pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
            for m in (pipe.transformer, pipe.text_encoder, pipe.vae):
                m.eval().requires_grad_(False)
            if torch.cuda.is_available():
                pipe.text_encoder.to("cuda")
                pipe.vae.to("cuda")
        except Exception as exc:
            print(f"Notice: Could not load GPU pipeline ({exc}). Running dry validation.")

    from src.pose_control.qwen21_full.execution import QwenFullDataset

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
        for split in args.splits:
            split_file = data_dir / f"{split}512_pairs.jsonl" if split == "train" else (
                data_dir / f"{split}24_pairs.jsonl" if split in ("val", "test") else (
                    data_dir / f"{split}6_pairs.jsonl" if split == "diagnostic" else data_dir / f"{split}_pairs.jsonl"
                )
            )
            if not split_file.is_file():
                split_file = data_dir / f"{split}_pairs.jsonl"
            if not split_file.is_file():
                continue

            prompt_manifest = run_dir / "prompts" / proto / f"{split}.jsonl"
            if not prompt_manifest.is_file():
                raise FileNotFoundError(f"Missing prompt manifest for {proto}/{split}: {prompt_manifest}")

            dataset = QwenFullDataset(split_file, resolution="native", align_mode="native", dataset_type="tiktok_v3")
            expected_ids = [pair_metadata(s)["pair_id"] for s in dataset.samples]
            bank = PromptBank(prompt_manifest, proto, expected_ids)

            if pipe is not None:
                counts = cache_split(pipe, cache, dataset, bank, split=split)
            else:
                counts = {"source_calls": len(dataset), "target_calls": len(dataset) if split == "train" else 0}
            proto_counts[split] = counts

        summary[proto] = proto_counts

    print("Precompute cache finished:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
