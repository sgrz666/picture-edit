#!/usr/bin/env python3
"""The aspect-preserving training entry for the Qwen-Image-2.1 Full adapter (plan §4).

Unlike the legacy `train_iper_qwen21_*` scripts, this one cannot be talked into a square canvas:
the policy is chosen here, the pair contract is verified before any GPU work, the latent cache is
bound to the policy, and a pre-contract checkpoint may only be inherited as an explicit warm start.

    python scripts/train/train_qwen21_native_full.py \
      --train-pairs <pairs.jsonl> --output <run_dir> \
      --canvas-policy aspect-preserving --max-pixels 1048576 --preflight-only
"""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data.canvas_contract import (
    ASPECT_PRESERVING, CANVAS_MULTIPLE, POLICY_FOR_MODE, RESIZE_SQUARE, SQUARE_CROP,
)
from src.data.geometry_pair_dataset_v2 import (
    SCHEMA, assert_pair_contract, bucket_by_canvas, canvas_fingerprint,
)
from src.pose_control.qwen21_full.canvas_provenance import require_canvas_compatible
from src.pose_control.qwen21_full.data_runtime import FaceConditionStore, QwenFullDataset
from src.pose_control.qwen21_full.execution import (
    MODEL, ROOT, batch_one, json_write, new_adapter, precompute, run_loss, save_checkpoint,
    set_training_phase, seed_all,
)
from src.pose_control.qwen21_full.qwen_runtime import PROMPT, Qwen21IPERCache

# The plan's CLI names, mapped onto the align_mode spellings the datasets accept.
POLICY_TO_MODE = {"native-pad": "native", "aspect-preserving": ASPECT_PRESERVING}
FORBIDDEN_POLICIES = {"resize-square": RESIZE_SQUARE, "square-crop": SQUARE_CROP}


DEEPGEN_CONFIG = ROOT / "models/DeepGen-1.0-diffusers/transformer/config.json"


def build_adapter(args, device):
    """warm-start reproduces the legacy initialisation; scratch exists for plumbing smoke tests.

    The V6 body/face warm-start checkpoints are not on every machine, and a canvas-correctness
    check does not need trained weights — it needs the real geometry front end, VAE, DiT and
    backward pass to agree on a non-square grid.
    """
    if args.adapter_init == "warm-start":
        return new_adapter(device)
    from types import SimpleNamespace

    from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter

    print("adapter initialised from the DeepGen config WITHOUT pretrained V6 weights: "
          "this validates the canvas plumbing only and must not be used for real training",
          flush=True)
    config = SimpleNamespace(**json.loads(DEEPGEN_CONFIG.read_text()))
    legacy = UnifiedSMPLXAdapterV6.from_deepgen_config(config, enable_face_adapter=True,
                                                       enable_hand_adapter=False)
    return Qwen21FullAdapter(legacy).to(device=device, dtype=torch.float32)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-pairs", type=Path, required=True)
    p.add_argument("--val-pairs", type=Path, required=True,
                   help="save_checkpoint records its sha256, so a formal run must name one")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canvas-policy", choices=(*POLICY_TO_MODE, *FORBIDDEN_POLICIES),
                   default="aspect-preserving",
                   help="resize-square and square-crop are listed only so they can be refused")
    p.add_argument("--max-pixels", type=int, default=1024 * 1024,
                   help="pixel budget for aspect-preserving; both axes shrink by one factor")
    p.add_argument("--pad-multiple", type=int, default=CANVAS_MULTIPLE)
    p.add_argument("--geometry-schema", default=SCHEMA,
                   help=f"must stay {SCHEMA}; the entry refuses a pair that reports otherwise")
    p.add_argument("--branch", choices=("body", "face"), default="body")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--features", type=Path, help="Face condition cache (only for --branch face)")
    p.add_argument("--exclude-padding-tokens", action="store_true",
                   help="drop fully-padding target tokens from attention keys (plan §5). Measured on "
                        "the real backbone: this removes a 5x-signal leak from padding into valid "
                        "tokens, but it also changes valid outputs by ~5x signal on its own, so "
                        "existing checkpoints must not enable it")
    p.add_argument("--resolution", default="native", choices=("native",),
                   help="canvas size always comes from the policy, never an integer; this exists "
                        "because save_checkpoint records it in the run metadata")
    p.add_argument("--adapter-init", choices=("warm-start", "scratch"), default="warm-start",
                   help="scratch skips the legacy V6 warm-start weights; smoke tests only")
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--warm-start-ok", action="store_true",
                   help="inherit a pre-contract checkpoint's weights, without claiming its features "
                        "already suit an aspect-preserving canvas")
    p.add_argument("--preflight-only", action="store_true",
                   help="verify the canvas and the pair contract on CPU and exit before any model load")
    p.add_argument("--contract-samples", type=int, default=8,
                   help="how many pairs to validate during preflight")
    return p.parse_args(argv)


def resolve_policy(args) -> str:
    if args.canvas_policy in FORBIDDEN_POLICIES:
        raise SystemExit(
            f"--canvas-policy {args.canvas_policy} is refused by this entry: it either stretches the "
            "person to a square or crops the field of view. Historical runs stay reproducible through "
            "the legacy trainers with --legacy-policy-ok; new training must not reintroduce it.")
    if args.geometry_schema != SCHEMA:
        raise SystemExit(f"--geometry-schema must be {SCHEMA}, got {args.geometry_schema!r}")
    if args.canvas_policy == "aspect-preserving" and args.max_pixels <= 0:
        raise SystemExit("--max-pixels must be positive for aspect-preserving")
    if args.pad_multiple % 16:
        raise SystemExit(f"--pad-multiple must be a multiple of the 16x Qwen VAE stride, got {args.pad_multiple}")
    return POLICY_TO_MODE[args.canvas_policy]


def build_dataset(args, mode: str) -> QwenFullDataset:
    return QwenFullDataset(args.train_pairs, resolution="native", dataset_type="tiktok_v3",
                           align_mode=mode, max_pixels=args.max_pixels)


def preflight(args, dataset, mode: str) -> dict:
    """Fail before the model load if any pair cannot prove its own canvas."""
    checked = min(args.contract_samples, len(dataset))
    facts, fingerprints = None, []
    for index in range(checked):
        pair = dataset[index]
        facts = assert_pair_contract(pair, where=f"preflight pair {index}")
        fingerprints.append(canvas_fingerprint(pair))
    if facts is None:
        raise SystemExit("preflight checked no pairs")
    target = facts["target_canvas"]
    expected = POLICY_FOR_MODE.get(mode, mode)
    if target.policy != expected:
        raise SystemExit(f"dataset produced policy {target.policy!r} but {expected!r} was requested")
    buckets = bucket_by_canvas([dataset[i] for i in range(checked)])
    report = {
        "geometry_schema": SCHEMA,
        "canvas_policy": args.canvas_policy,
        "resolved_mode": mode,
        "max_pixels": args.max_pixels if mode == ASPECT_PRESERVING else None,
        "pad_multiple": args.pad_multiple,
        "pairs_checked": checked,
        "sample_canvas_hw": list(target.canvas_hw),
        "sample_resized_hw": list(target.resized_hw),
        "sample_original_hw": list(target.original_hw),
        "target_token_hw": list(facts["target_token_hw"]),
        "target_tokens": target.token_count,
        "square_grid_would_be": [(target.canvas_hw[0] // 16)] * 2,
        "canvas_buckets": {f"{list(k[0])}x{list(k[1])}": len(v) for k, v in buckets.items()},
        "distinct_canvases": len(set(fingerprints)),
        "all_pairs_aspect_preserving": all(
            abs((s["resized_hw"][1] / s["resized_hw"][0]) - (s["original_hw"][1] / s["original_hw"][0]))
            < 2e-3 for s in [dataset[i]["target_canvas"] for i in range(checked)]),
    }
    return report


def main(argv=None):
    args = arguments(argv)
    mode = resolve_policy(args)
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    dataset = build_dataset(args, mode)
    report = preflight(args, dataset, mode)
    json_write(args.output / "canvas_preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0
    if args.steps <= args.warmup_steps:
        raise SystemExit(
            f"--steps {args.steps} <= --warmup-steps {args.warmup_steps}: every step would sit "
            "inside warmup, where the trainable phase is zeroed and grad_norm is exactly 0. A run "
            "like that looks green while learning nothing; set --warmup-steps below --steps.")

    from diffusers import QwenImage21Pipeline

    device = torch.device("cuda")
    cache = Qwen21IPERCache(args.output / "cache", resolution="native", prompt=args.prompt,
                            canvas_policy=mode)
    adapter = build_adapter(args, device)
    if args.init_checkpoint:
        from src.pose_control.qwen21_full.execution import load_adapter_checkpoint
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        provenance = require_canvas_compatible(saved.get("metadata") or saved,
                                               warm_start_ok=args.warm_start_ok,
                                               where="train_qwen21_native_full")
        load_adapter_checkpoint(adapter, args.init_checkpoint)
        print(f"init checkpoint inherited as {provenance}", flush=True)

    group = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(group, lr=args.lr)
    store = FaceConditionStore(args.features or (args.output / "face_cache"))

    print("loading pipeline...", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to(device)
    precompute(pipe, cache, [dataset])

    started = time.perf_counter()
    rows = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        if args.exclude_padding_tokens:
            from src.data.canvas_contract import CanvasSpec
            from src.pose_control.qwen21_full.padding_attention import (
                install_padding_key_mask, padding_key_masked)

            if len(report["canvas_buckets"]) != 1:
                raise SystemExit(
                    f"--exclude-padding-tokens needs one canvas per batch; this dataset yields "
                    f"{len(report['canvas_buckets'])} buckets - bucket by (H, W) first")
            install_padding_key_mask(pipe.transformer)
            info = stack.enter_context(padding_key_masked(
                pipe.transformer, CanvasSpec.from_record(dataset[0]["target_canvas"]),
                allow_semantic_change=True))
            print(f"padding tokens excluded from attention: {info}", flush=True)

        for step in range(args.steps):
            set_training_phase(adapter, args.branch, warmup=step < args.warmup_steps)
            index = int(torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + step))[0])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    row = run_loss(pipe, adapter, cache, dataset, store, index, args.branch,
                                   seed=args.seed + step)
                except torch.OutOfMemoryError as error:
                    grid = (dataset[index].get("target_token_hw") or ("?", "?"))
                    raise SystemExit(
                        f"CUDA OOM at step {step + 1} on a {grid[0]}x{grid[1]}="
                        f"{grid[0] * grid[1]}-token canvas: {str(error)[:120]}. native-pad keeps every "
                        "original pixel, which needs gradient checkpointing on this card; either lower "
                        "--max-pixels (aspect-preserving shrinks both axes together and never "
                        "distorts) or enable the checkpointed training path.") from error
            norm = float(torch.nn.utils.clip_grad_norm_(group, 1.0))
            if not torch.isfinite(torch.tensor([norm, row["loss"]])).all():
                raise FloatingPointError(f"non-finite loss/gradient at step {step + 1}: {row}")
            if step >= args.warmup_steps and norm == 0:
                # A zero gradient outside warmup means the geometry never reached the trainable
                # heads; without this check a run can look green while learning nothing.
                raise AssertionError(
                    f"grad_norm is exactly 0 at step {step + 1} (past warmup): the adapter is not "
                    f"receiving the condition signal; residual_rms="
                    f"{row.get('control_diagnostics', {}).get('residual_rms')}")
            optimizer.step()
            row.update(step=step + 1, grad_norm=norm, pair_index=index,
                       target_token_hw=list(row.get("control_diagnostics", {}).get("target_token_hw") or []),
                       elapsed_seconds=time.perf_counter() - started,
                       peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
            rows.append(row)
            with (args.output / "train_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, default=str) + "\n")
            print(json.dumps(row, default=str), flush=True)

    save_checkpoint(args.output / "checkpoint_last.pt", adapter, optimizer, step=args.steps,
                    branch=args.branch, args=args, sample_order=[r["pair_index"] for r in rows])
    json_write(args.output / "run_manifest.json", {
        "canvas": report, "pairs": len(dataset), "steps": args.steps, "seed": args.seed,
        "exclude_padding_tokens": args.exclude_padding_tokens,
        "train_pairs_sha256": hashlib.sha256(args.train_pairs.read_bytes()).hexdigest(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
