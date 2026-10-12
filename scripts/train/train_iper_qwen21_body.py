#!/usr/bin/env python3
"""Train a target-token body adapter on the frozen Qwen-Image-2.1 backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline

from src.data.iper_dataset import IPERPoseDataset
from src.data.tiktok_dataset import load_pose_dataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.runtime import MODEL_REVISION, PROMPT, Qwen21IPERCache, dataset_image_to_pil, transformer_inputs


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    parser.add_argument("--v6-checkpoint", type=Path, default=ROOT / "experiments/iper_v6_native_20k_v1/checkpoint_last.pt")
    parser.add_argument("--train-pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/dev_train_pairs.jsonl")
    parser.add_argument("--val-pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/dev_val_pairs.jsonl")
    parser.add_argument("--dataset-type", choices=("auto", "iper", "tiktok_v3"), default="auto")
    parser.add_argument("--align-mode", choices=("square_crop", "resize"), default="square_crop")
    parser.add_argument("--sampled-root", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "datasets/iPER/iper_assets_512_nvdiffrast")
    parser.add_argument("--tiktok-raw-root", type=Path, default=ROOT / "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset")
    parser.add_argument("--tiktok-assets-root", type=Path, default=ROOT / "datasets/TikTokDataset/TikTok_3d_assets_native")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/qwen21_iper_body")
    parser.add_argument("--cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_iper_cache"))
    parser.add_argument("--max-steps", type=int, default=3000, help="Optimizer steps, not micro-steps")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Pairs per transformer forward")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit training pairs for a small overfit run")
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--init-checkpoint", type=Path, help="Load adapter weights only; start a new optimizer run")
    parser.add_argument("--transferred-lr", type=float, default=5e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, pairs: Path):
    return load_pose_dataset(
        pairs_jsonl=pairs,
        dataset_type=getattr(args, "dataset_type", "auto"),
        sampled_root=args.sampled_root,
        assets_root=args.assets_root,
        tiktok_raw_root=getattr(args, "tiktok_raw_root", ROOT / "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset"),
        tiktok_assets_root=getattr(args, "tiktok_assets_root", ROOT / "datasets/TikTokDataset/TikTok_3d_assets_native"),
        resolution=args.resolution,
        align_mode=getattr(args, "align_mode", "square_crop"),
        augment=False,
    )


def batch_many(dataset: IPERPoseDataset, indices: list[int]) -> dict:
    samples = [dataset[index] for index in indices]
    return {
        name: torch.stack([sample[name] for sample in samples]) if torch.is_tensor(samples[0][name])
        else [sample[name] for sample in samples]
        for name in samples[0]
    }


def batch_one(dataset: IPERPoseDataset, index: int) -> dict:
    return batch_many(dataset, [index])


def validation_indices(dataset: IPERPoseDataset, count: int) -> list[int]:
    """Choose held-out validation pairs across appearances before repeats."""
    by_appearance: dict[str, list[int]] = {}
    for index, meta in enumerate(dataset.samples):
        by_appearance.setdefault(meta["appearance"], []).append(index)
    selected = []
    depth = 0
    while len(selected) < min(count, len(dataset)):
        added = False
        for indices in by_appearance.values():
            if depth < len(indices):
                selected.append(indices[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def precompute(pipe, cache: Qwen21IPERCache, dataset: IPERPoseDataset, *, limit: int = 0, indices: list[int] | None = None) -> None:
    sample_indices = indices if indices is not None else list(range(min(len(dataset), limit) if limit else len(dataset)))
    count = len(sample_indices)
    seen_sources: set[tuple[str, str]] = set()
    seen_targets: set[tuple[str, str]] = set()
    for position, index in enumerate(sample_indices):
        meta = dataset.samples[index]
        source_key = (meta["appearance"], meta["source"]["stem"])
        target_key = (meta["appearance"], meta["target"]["stem"])
        if source_key not in seen_sources:
            source_path = Path(meta["source_rgb"]) if "source_rgb" in meta else (Path(dataset.sampled_root) / meta["appearance"] / meta["source"]["rgb_1024"])
            source = dataset_image_to_pil(dataset.rgb_transform(dataset._load_image(str(source_path))))
            if not cache._path(source, "source").exists():
                cache.source(pipe, source)
        if target_key not in seen_targets:
            target_path = Path(meta["target_rgb"]) if "target_rgb" in meta else (Path(dataset.sampled_root) / meta["appearance"] / meta["target"]["rgb_1024"])
            target = dataset_image_to_pil(dataset.rgb_transform(dataset._load_image(str(target_path))))
            if not cache._path(target, "target").exists():
                cache.target(pipe, target)
        seen_sources.add(source_key)
        seen_targets.add(target_key)
        if position == 0 or (position + 1) % 100 == 0 or position + 1 == count:
            print(f"condition cache {position + 1}/{count}", flush=True)


def loss_for_batch(
    transformer, adapter: Qwen21BodyAdapter, cache: Qwen21IPERCache,
    pipe, batch: dict, device: torch.device, *, seed: int | None = None,
    backward_scale: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair = cache.pair(pipe, batch, device)
    target = pair["target_latent"]
    if seed is None:
        sigma = torch.rand(target.shape[0], device=device, dtype=target.dtype)
        noise = torch.randn_like(target)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        sigma = torch.rand(target.shape[0], device=device, dtype=target.dtype, generator=generator)
        noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    velocity = noise - target
    grid = cache.resolution // 16
    tokens = grid * grid
    geometry = adapter.encode_geometry(batch, (grid, grid))
    inputs = transformer_inputs(pair, noisy, sigma)
    with Qwen21ControlContext(transformer, adapter, geometry, sigma=sigma, target_tokens=tokens) as control:
        prediction = transformer(**inputs)[0][:, -tokens:]
        mask = F.interpolate(batch["human_mask"].to(device=device, dtype=torch.float32), size=(grid, grid), mode="bilinear", align_corners=False)
        weight = (1 + 2 * mask.flatten(2).transpose(1, 2)).clamp(1, 3)
        squared = (prediction.float() - velocity.float()).square()
        loss = (squared * weight).sum() / (weight.sum() * squared.shape[-1])
        if backward_scale is not None:
            (loss / backward_scale).backward()
    return loss, {"loss": float(loss.detach()), "sigma": float(sigma.float().mean()), "control_calls": sum(control.call_counts.values())}


def save_checkpoint(path: Path, adapter, optimizer, *, step: int, micro_step: int, args: argparse.Namespace, best_validation: float) -> None:
    payload = {
        "adapter_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "micro_step": micro_step, "args": vars(args),
        "best_validation": best_validation,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def verify_zero_control(transformer, adapter, cache, pipe, batch, device) -> float:
    pair = cache.pair(pipe, batch, device)
    sigma = torch.full((1,), 0.5, dtype=pair["target_latent"].dtype, device=device)
    noisy = pair["target_latent"] * 0.5
    inputs = transformer_inputs(pair, noisy, sigma)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = (cache.resolution // 16) ** 2
        native = transformer(**inputs)[0][:, -tokens:]
        geometry = adapter.encode_geometry(batch, (cache.resolution // 16,) * 2)
        with Qwen21ControlContext(transformer, adapter, geometry, sigma=sigma, target_tokens=tokens):
            controlled = transformer(**inputs)[0][:, -tokens:]
    difference = float((native.float() - controlled.float()).abs().max())
    if difference > 1e-3:
        raise AssertionError(f"zero-initialized control changes native prediction by {difference}")
    return difference


@torch.no_grad()
def validation_loss(transformer, adapter, cache, pipe, dataset, count, device) -> float:
    adapter.eval()
    losses = []
    for index in validation_indices(dataset, count):
        batch = batch_one(dataset, index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = loss_for_batch(transformer, adapter, cache, pipe, batch, device, seed=42000 + index)
        losses.append(float(loss))
    adapter.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = arguments()
    if args.max_steps <= 0 or args.accumulation <= 0 or args.micro_batch_size <= 0 or args.val_count < 0:
        raise ValueError("training step and accumulation counts must be positive")
    if args.transferred_lr <= 0 or args.new_lr <= 0:
        raise ValueError('learning rates must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_pairs_sha256": hashlib.sha256(args.train_pairs.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "versions": {name: version(name) for name in ("torch", "diffusers", "transformers", "huggingface_hub", "tokenizers")},
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "train_iper_qwen21_body.py",
                ROOT / "src/pose_control/qwen21/body_adapter.py",
                ROOT / "src/pose_control/qwen21/runtime.py",
            )
        },
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.transformer.eval()
    pipe.text_encoder.to(device)
    pipe.vae.to(device)
    cache = Qwen21IPERCache(args.cache, prompt=args.prompt, resolution=args.resolution)
    training = load_dataset(args, args.train_pairs)
    validation = load_dataset(args, args.val_pairs)
    train_count = min(args.max_pairs, len(training)) if args.max_pairs else len(training)
    precompute(pipe, cache, training, limit=train_count)
    if args.val_count:
        precompute(pipe, cache, validation, indices=validation_indices(validation, args.val_count))
    if args.precompute_only:
        return
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to(device)
    adapter = Qwen21BodyAdapter()
    if args.init_checkpoint:
        adapter.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)['adapter_state_dict'])
        print('Initialized Body from', args.init_checkpoint, flush=True)
    else:
        loaded = adapter.load_v6_geometry(str(args.v6_checkpoint))
        print("Transferred V6 tensors:", loaded, flush=True)
    adapter.to(device=device, dtype=torch.float32)
    adapter.train()
    if not args.resume and not args.init_checkpoint:
        parity = verify_zero_control(pipe.transformer, adapter, cache, pipe, batch_one(training, 0), device)
        print(f"Zero-control velocity parity max_abs={parity:.6g}", flush=True)
    transferred = [parameter for name, parameter in adapter.named_parameters() if name.startswith(("condition_injector.", "reasoner.", "condition_bridge."))]
    new = [parameter for name, parameter in adapter.named_parameters() if not name.startswith(("condition_injector.", "reasoner.", "condition_bridge."))]
    optimizer = torch.optim.AdamW([
        {"params": transferred, "lr": args.transferred_lr}, {"params": new, "lr": args.new_lr}
    ], weight_decay=0.01)
    transformer = pipe.transformer
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    start_step = 0
    micro_step = 0
    best_validation = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if int(saved.get('args', {}).get('resolution', 512)) != args.resolution:
            raise ValueError('cannot resume a checkpoint with a different resolution')
        adapter.load_state_dict(saved["adapter_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved["step"])
        micro_step = int(saved["micro_step"])
        previous_args = saved.get("args", {})
        same_validation = (
            int(previous_args.get("val_count", -1)) == args.val_count
            and str(previous_args.get("val_pairs", "")) == str(args.val_pairs)
        )
        best_validation = float(saved.get("best_validation", float("inf"))) if same_validation else float("inf")
        torch.random.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        previous_best = args.resume.with_name("checkpoint_best.pt")
        current_best = args.output / "checkpoint_best.pt"
        if same_validation and previous_best.exists() and previous_best.resolve() != current_best.resolve() and not current_best.exists():
            shutil.copy2(previous_best, current_best)
    metrics_path = args.output / "train_metrics.jsonl"
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, args.max_steps):
        micro_losses = []
        pair_indices = []
        for _ in range(args.accumulation):
            indices = []
            for _ in range(args.micro_batch_size):
                epoch, offset = divmod(micro_step, train_count)
                permutation = torch.randperm(train_count, generator=torch.Generator().manual_seed(args.seed + epoch))
                indices.append(int(permutation[offset]))
                micro_step += 1
            pair_indices.extend(indices)
            batch = batch_many(training, indices)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, diagnostics = loss_for_batch(
                    transformer, adapter, cache, pipe, batch, device,
                    backward_scale=args.accumulation,
                )
            micro_losses.append(float(loss.detach()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step + 1, "micro_step": micro_step, "loss": float(np.mean(micro_losses)),
            "pair_indices": pair_indices,
            "effective_batch": args.accumulation * args.micro_batch_size,
            "grad_norm": grad_norm, "sigma": diagnostics["sigma"],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            if args.val_count:
                row["val_loss"] = validation_loss(transformer, adapter, cache, pipe, validation, args.val_count, device)
                print(f"step {step + 1}: val_loss={row['val_loss']:.6f}", flush=True)
                if row["val_loss"] < best_validation:
                    best_validation = row["val_loss"]
                    save_checkpoint(args.output / "checkpoint_best.pt", adapter, optimizer, step=step + 1, micro_step=micro_step, args=args, best_validation=best_validation)
            save_checkpoint(args.output / "checkpoint_last.pt", adapter, optimizer, step=step + 1, micro_step=micro_step, args=args, best_validation=best_validation)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if (step + 1) % args.log_every == 0 or step == start_step:
            print(json.dumps(row), flush=True)
    print(f"Finished {args.max_steps} optimizer steps", flush=True)


if __name__ == "__main__":
    main()
