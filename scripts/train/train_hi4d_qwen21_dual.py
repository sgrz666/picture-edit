#!/usr/bin/env python3
"""Train Qwen21 dual-person control from Hi4D N2C pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline
from PIL import Image

from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset, Hi4DGeometryConverter, collate_hi4d
from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.qwen21.body_adapter import Qwen21ControlContext
from src.pose_control.qwen21.action_prompts import PROMPT_VERSION, build_dual_action_prompts
from src.pose_control.qwen21.dual_adapter import Qwen21DualAdapter
from src.pose_control.qwen21.runtime import MODEL_REVISION, PROMPT as IPER_PROMPT, Qwen21IPERCache, dataset_image_to_pil, transformer_inputs


ROOT = Path(__file__).resolve().parent
HI4D = ROOT / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests"
# Target VAE latents are prompt-independent. Keep this old cache namespace so
# precomputed target latents remain reusable; it is never a model instruction.
DUAL_TARGET_CACHE_NAMESPACE = ("Move both people to the target interaction pose while preserving person A and person B "
                               "identity, clothing, body proportions and the scene background. Respect their contact and relative positions.")
NEW_PREFIXES = (
    "dual_global_encoder.", "source_projection.", "person_binder.", "interaction_in.",
    "depth_in.", "interaction_fusion.", "appearance_fusion.", "interaction_heads.",
    "appearance_heads.", "reasoner.cross_person_reasoner.", "reasoner.contact_reasoner.",
    "reasoner.dual_fusion.", "reasoner.role_embedding.", "condition_injector.relative_encoder.",
    "condition_injector.contact_raster_encoder.", "condition_injector.contact_relation_encoder.",
    "condition_bridge.interaction_projection.", "condition_bridge.interaction_high_downsample.",
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    p.add_argument("--single-checkpoint", type=Path, default=ROOT / "experiments/qwen21_iper_body/full_20k/checkpoint_best.pt")
    p.add_argument("--train-pairs", type=Path, default=HI4D / "main_pairs_train.jsonl")
    p.add_argument("--val-pairs", type=Path, default=HI4D / "main_pairs_val.jsonl")
    p.add_argument("--iper-pairs", type=Path, default=ROOT / "datasets/iPER/iper_sampled_6src64tgt/splits/dev_train_pairs.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "experiments/qwen21_hi4d_dual/action_prompt_v2")
    p.add_argument("--control-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_controls"))
    p.add_argument("--latent-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_latents"))
    p.add_argument("--iper-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_iper_cache"))
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--stage1-steps", type=int, default=2000)
    p.add_argument("--iper-replay-rate", type=float, default=0.1)
    p.add_argument("--stage1-lr", type=float, default=1e-4)
    p.add_argument("--stage2-new-lr", type=float, default=3e-5)
    p.add_argument("--stage2-base-lr", type=float, default=3e-6)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--val-count", type=int, default=64)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--precompute-controls-only", action="store_true")
    p.add_argument("--precompute-only", action="store_true")
    p.add_argument("--control-limit", type=int, default=0, help="Only for smoke precompute")
    p.add_argument("--max-train-pairs", type=int, default=0, help="Only for overfit smoke")
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


def precompute_controls(args, train, val) -> None:
    converter = Hi4DGeometryConverter(ROOT / "models/smpl")
    if args.control_limit:
        old = train.rows
        train.rows = train.rows[:args.control_limit]
        train.precompute(converter)
        train.rows = old
    else:
        train.precompute(converter)
        val.precompute(converter)
    del converter
    torch.cuda.empty_cache()


@lru_cache(maxsize=2048)
def prompts_for_row(action: str, mask_a: str, mask_b: str) -> tuple[str, str]:
    masks = tuple(torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy())
                  for path in (mask_a, mask_b))
    return build_dual_action_prompts(action, masks)


def row_prompts(row: dict) -> tuple[str, str]:
    return prompts_for_row(row["target_action"], row["source_person_mask_A"], row["source_person_mask_B"])


def batch_short_prompts(batch: dict) -> list[str]:
    return [build_dual_action_prompts(row["target_action"],
            (batch["source_mask_a"][index], batch["source_mask_b"][index]))[1]
            for index, row in enumerate(batch["row"])]


def write_prompt_manifest(dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(dataset.rows):
            native, short = row_prompts(row)
            stream.write(json.dumps({"index": index, "pair": row["pair"],
                "source": row["source_rgb"], "target_action": row["target_action"],
                "prompt_version": PROMPT_VERSION, "native_detailed": native,
                "adapter_short": short}) + "\n")


def precompute_latents(pipe, cache, dataset, indices=None) -> None:
    indices = range(len(dataset)) if indices is None else indices
    seen_source, seen_target = set(), set()
    for position, index in enumerate(indices):
        row = dataset.rows[index]
        prompt = row_prompts(row)[1]
        source_key = (row["source_rgb"], prompt)
        if source_key not in seen_source or row["target_rgb"] not in seen_target:
            sample = dataset[index]
            if source_key not in seen_source:
                cache.source(pipe, dataset_image_to_pil(sample["src_image"]), prompt=prompt)
            if row["target_rgb"] not in seen_target:
                cache.target(pipe, dataset_image_to_pil(sample["tgt_image"]))
        seen_source.add(source_key)
        seen_target.add(row["target_rgb"])
        if position == 0 or (position + 1) % 100 == 0:
            print(f"Hi4D latent cache {position + 1}/{len(indices)}", flush=True)


def batch_one(dataset, index):
    return collate_hi4d([dataset[index]])


def batch_many(dataset, indices):
    return collate_hi4d([dataset[index] for index in indices])


def iper_batch_many(dataset, indices):
    samples = [dataset[index] for index in indices]
    return {name: torch.stack([sample[name] for sample in samples]) if torch.is_tensor(samples[0][name])
            else [sample[name] for sample in samples] for name in samples[0]}


def batch_loss(transformer, adapter, cache, pipe, batch, device, *, dual=True, seed=None,
               backward_scale=None):
    pair = cache.pair(pipe, batch, device, prompts=batch_short_prompts(batch) if dual else None)
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
    state = adapter.encode_dual(batch, pair["source_latent"]) if dual else adapter.encode_geometry(batch, (32, 32))
    with Qwen21ControlContext(transformer, adapter, state, sigma=sigma, target_tokens=target.shape[1]) as control:
        prediction = transformer(**transformer_inputs(pair, noisy, sigma))[0][:, -target.shape[1]:]
        union = F.interpolate(batch["human_mask"].to(device=device, dtype=torch.float32),
                              size=(32, 32), mode="nearest").flatten(2).transpose(1, 2)
        weight = 1 + 2 * union
        if dual:
            contact = F.interpolate(batch["contact_raster"].to(device=device, dtype=torch.float32),
                                    size=(32, 32), mode="bilinear", align_corners=False)
            contact = contact.amax(dim=1, keepdim=True).flatten(2).transpose(1, 2).clamp(0, 1)
            weight = weight + 2 * contact
        square = (prediction.float() - velocity.float()).square()
        loss = (square * weight).sum() / (weight.sum() * square.shape[-1])
        # Gradient checkpointing recomputes transformer blocks in backward.
        # The control hooks must remain attached until that recomputation ends.
        if backward_scale is not None:
            (loss / backward_scale).backward()
    return loss, control.call_counts


@torch.no_grad()
def validation_loss(transformer, adapter, cache, pipe, dataset, count, device):
    adapter.eval()
    losses = []
    for index in np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int):
        batch = batch_one(dataset, int(index))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = batch_loss(transformer, adapter, cache, pipe, batch, device,
                                 seed=42000 + int(index))
        losses.append(float(loss))
    adapter.train()
    return float(np.mean(losses))


def save(path, adapter, optimizer, args, step, micro_step, best):
    payload = {"adapter_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "step": step, "micro_step": micro_step, "best_validation": best, "args": vars(args),
               "torch_rng_state": torch.random.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
               "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = arguments()
    args.prompt_version = PROMPT_VERSION
    if (args.max_steps <= 0 or not 0 <= args.stage1_steps <= args.max_steps
            or args.accumulation <= 0 or args.micro_batch_size <= 0
            or not 0 <= args.iper_replay_rate <= 1
            or min(args.stage1_lr, args.stage2_new_lr, args.stage2_base_lr) <= 0
            or args.stage2_new_lr >= args.stage1_lr):
        raise ValueError("invalid training stage or accumulation count")
    args.output.mkdir(parents=True, exist_ok=True)
    train = Hi4DQwen21Dataset(args.train_pairs, args.control_cache)
    val = Hi4DQwen21Dataset(args.val_pairs, args.control_cache)
    write_prompt_manifest(train, args.output / "prompts_train.jsonl")
    write_prompt_manifest(val, args.output / "prompts_val.jsonl")
    precompute_controls(args, train, val)
    if args.precompute_controls_only:
        return
    if args.control_limit:
        raise ValueError("control-limit may only be used with --precompute-controls-only")
    device = torch.device("cuda")
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to(device)
    pipe.transformer.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    dual_cache = Qwen21IPERCache(args.latent_cache, prompt=DUAL_TARGET_CACHE_NAMESPACE)
    train_count = min(args.max_train_pairs, len(train)) if args.max_train_pairs else len(train)
    precompute_latents(pipe, dual_cache, train, range(train_count))
    precompute_latents(pipe, dual_cache, val)
    if args.precompute_only:
        return
    iper = IPERPoseDataset(str(args.iper_pairs),
        str(ROOT / "datasets/iPER/iper_sampled_6src64tgt"),
        str(ROOT / "datasets/iPER/iper_assets_512_nvdiffrast"), resolution=512, augment=False)
    iper_cache = Qwen21IPERCache(args.iper_cache, prompt=IPER_PROMPT)
    # iPER cache was generated by the existing single-person training run.
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    adapter = Qwen21DualAdapter()
    print("Single checkpoint transfer:", adapter.load_single_checkpoint(str(args.single_checkpoint)), flush=True)
    adapter.to(device=device, dtype=torch.float32).train()
    new_params, base_params = [], []
    for name, parameter in adapter.named_parameters():
        (new_params if name.startswith(NEW_PREFIXES) else base_params).append(parameter)
    optimizer = torch.optim.AdamW([{"params": new_params, "lr": args.stage1_lr},
                                   {"params": base_params, "lr": args.stage2_base_lr}], weight_decay=0.01)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    start, micro_step, best = 0, 0, float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("args", {}).get("prompt_version") != PROMPT_VERSION:
            raise ValueError("resume checkpoint prompt version differs from current run")
        adapter.load_state_dict(payload["adapter_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start, micro_step = int(payload["step"]), int(payload["micro_step"])
        best = float(payload["best_validation"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    manifest = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "model_revision": MODEL_REVISION, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "train_sha256": hashlib.sha256(args.train_pairs.read_bytes()).hexdigest(),
                "val_sha256": hashlib.sha256(args.val_pairs.read_bytes()).hexdigest(),
                "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("train_hi4d_qwen21_dual.py", "src/data/hi4d_qwen21_dataset.py",
                                 "src/pose_control/qwen21/dual_adapter.py",
                                 "src/pose_control/qwen21/action_prompts.py")}}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    metrics = args.output / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for step in range(start, args.max_steps):
        stage1 = step < args.stage1_steps
        optimizer.param_groups[0]["lr"] = args.stage1_lr if stage1 else args.stage2_new_lr
        optimizer.param_groups[1]["lr"] = args.stage2_base_lr
        for parameter in base_params:
            parameter.requires_grad_(not stage1)
        losses, replay_count = [], 0
        for _ in range(args.accumulation):
            replay = (not stage1) and random.random() < args.iper_replay_rate
            if replay:
                replay_count += args.micro_batch_size
                batch = iper_batch_many(iper, [random.randrange(len(iper)) for _ in range(args.micro_batch_size)])
                cache, dual = iper_cache, False
            else:
                indices = []
                for _ in range(args.micro_batch_size):
                    epoch, offset = divmod(micro_step, train_count)
                    indices.append(int(torch.randperm(train_count, generator=torch.Generator().manual_seed(args.seed + epoch))[offset]))
                    micro_step += 1
                batch = batch_many(train, indices)
                cache, dual = dual_cache, True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, calls = batch_loss(pipe.transformer, adapter, cache, pipe, batch, device,
                                         dual=dual, backward_scale=args.accumulation)
            if any(value < 1 for value in calls.values()):
                raise RuntimeError(f"a control hook did not fire: {calls}")
            losses.append(float(loss.detach()))
        norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {"step": step + 1, "stage": 1 if stage1 else 2,
               "new_lr": optimizer.param_groups[0]["lr"],
               "base_lr": optimizer.param_groups[1]["lr"],
               "loss": float(np.mean(losses)), "replay_microbatches": replay_count,
               "effective_batch": args.accumulation * args.micro_batch_size,
               "grad_norm": norm, "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            row["val_loss"] = validation_loss(pipe.transformer, adapter, dual_cache, pipe, val, args.val_count, device)
            if row["val_loss"] < best:
                best = row["val_loss"]
                save(args.output / "checkpoint_best.pt", adapter, optimizer, args, step + 1, micro_step, best)
            save(args.output / "checkpoint_last.pt", adapter, optimizer, args, step + 1, micro_step, best)
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if step == start or (step + 1) % args.log_every == 0 or "val_loss" in row:
            print(json.dumps(row), flush=True)
    best_payload = torch.load(args.output / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(best_payload["adapter_state_dict"])
    with torch.no_grad():
        full_validation = validation_loss(pipe.transformer, adapter, dual_cache, pipe, val, len(val), device)
    (args.output / "validation_full.json").write_text(json.dumps({
        "checkpoint_step": int(best_payload["step"]), "count": len(val),
        "flow_loss": full_validation, "manifest": str(args.val_pairs),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
