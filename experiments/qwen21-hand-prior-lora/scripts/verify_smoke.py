#!/usr/bin/env python3
"""Check a saved Qwen 2.1 LoRA and run a fixed-seed base / LoRA I2I comparison."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageOps

from dataset_contract import file_sha256, load_export, read_export


def audit_weights(path):
    import torch
    from safetensors import safe_open

    modules = [f"transformer.transformer_blocks.{layer}.attn.{projection}"
               for layer in range(32) for projection in ("to_q", "to_k", "to_v", "to_out.0")]
    expected = {f"{module}.lora_{side}.weight": shape
                for module in modules for side, shape in (("A", (16, 4096)), ("B", (4096, 16)))}
    nonzero_b = 0
    with safe_open(path, framework="pt", device="cpu") as saved:
        actual = set(saved.keys())
        if actual != set(expected):
            raise ValueError(f"Unexpected LoRA state. Missing: {set(expected) - actual}. Extra: {actual - set(expected)}")
        for key, shape in expected.items():
            tensor = saved.get_tensor(key)
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid tensor {key}: {tuple(tensor.shape)}, expected {shape}")
            if ".lora_B." in key and torch.count_nonzero(tensor).item():
                nonzero_b += 1
    if nonzero_b == 0:
        raise ValueError("All LoRA B tensors are still zero")
    return {
        "sha256": file_sha256(path), "modules": len(modules), "tensors": len(expected),
        "parameters": sum(a * b for a, b in expected.values()), "nonzero_lora_b": nonzero_b,
    }


def check_held_out(train_paths, comparison_paths):
    import pyarrow.parquet as pq

    columns = ["group", "source_sha256", "target_sha256"]
    train = pq.read_table(list(train_paths), columns=columns).to_pydict()
    comparison = pq.read_table(list(comparison_paths), columns=columns).to_pydict()
    if set(train["group"]).intersection(comparison["group"]):
        raise ValueError("Comparison subjects overlap the training subjects")
    train_images = set(train["source_sha256"] + train["target_sha256"])
    if train_images.intersection(comparison["source_sha256"] + comparison["target_sha256"]):
        raise ValueError("Comparison RGB SHA256 overlaps the training images")


SWAP_COLUMNS = ("source_sha256", "caption", "group", "hand_type", "pose_sha256", "pose_joints")


def select_pose_swap(rows, index, alternate=None):
    first = rows[index]
    original = json.loads(first["pose_joints"])
    candidates = range(len(rows)) if alternate is None else (alternate,)
    best = None
    for candidate in candidates:
        second = rows[candidate]
        if (second["pose_sha256"] == first["pose_sha256"]
                or any(second[key] != first[key] for key in SWAP_COLUMNS[:4])):
            continue
        other = json.loads(second["pose_joints"])
        shared = [(a, b) for h0, h1 in zip(original, other) for a, b in zip(h0, h1)
                  if a is not None and b is not None]
        if shared:
            pairs = np.asarray(shared)
            rms = float(np.sqrt(np.square(pairs[:, 0] - pairs[:, 1]).sum(axis=1).mean()))
            if rms > 0 and (best is None or rms > best[1]):
                best = (candidate, rms, len(shared))
    if best is None:
        raise ValueError("Pose swap requires two distinct target skeletons for the same source and prompt")
    return best


def image_delta(first, second):
    a, b = (np.asarray(image.convert("RGB"), dtype=np.float32) for image in (first, second))
    return float(np.abs(a - b).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Training directory created by train.sh")
    parser.add_argument("--base", type=Path, help="Relocated copy of the exact pinned base")
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--swap-index", type=int, help="Explicit alternate paired target for dual-image inference")
    parser.add_argument("--no-swap", action="store_true", help="Score independent rows with no second target for A")
    parser.add_argument("--out", required=True, type=Path, help="New comparison directory")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.index < 0 or args.steps < 1 or args.checkpoint_step < 1:
        parser.error("index must be nonnegative and step counts must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from diffusers import QwenImage21Pipeline

    settings = dict(line.split("=", 1) for line in
                    (args.run / "provenance.txt").read_text(encoding="utf-8").splitlines() if "=" in line)
    if file_sha256(args.run / "base.sha256") != settings["base_manifest_sha256"]:
        raise ValueError("Training base checksum manifest changed")
    base = (args.base or Path(settings["base"])).resolve(strict=True)
    components = ("model_index.json", "processor", "scheduler", "text_encoder", "transformer", "vae")
    model_files = sorted(p.relative_to(base).as_posix() for component in components
                         for p in ([base / component] if (base / component).is_file()
                                   else (base / component).rglob("*")) if p.is_file())
    if [line[66:] for line in (args.run / "base.sha256").read_text().splitlines()] != model_files:
        raise ValueError("Inference base files differ from the locked pipeline")
    export, comparison_paths = read_export(args.dataset, args.split)
    columns = settings.get("condition_columns", "source").split(",")
    dual = columns == ["source", "pose"]
    if columns != export.get("condition_columns", ["source"]):
        raise ValueError("Comparison condition schema must match the training run")
    if args.swap_index is not None and (not dual or args.swap_index < 0 or args.no_swap):
        parser.error("swap index requires dual-image inference and a nonnegative index")
    if args.split == "train" and export["dataset_sha256"] != settings["dataset_sha256"]:
        raise ValueError("Expected the original training dataset for a train split comparison")
    if args.split != "train":
        train, train_paths = read_export(settings["dataset"], "train")
        if train["dataset_sha256"] != settings["dataset_sha256"]:
            raise ValueError("Training dataset changed since the run")
        check_held_out(train_paths, comparison_paths)
    train_recipe = json.loads((args.run / "dataset.json").read_text(encoding="utf-8"))
    if export.get("pose_mode", "visible") != train_recipe.get("pose_mode", "visible"):
        raise ValueError("The inference pose rendering policy must match training")

    checkpoint = args.run / "weights" / f"checkpoint-{args.checkpoint_step}"
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    state_files = ["optimizer.bin", "scheduler.bin", "random_states_0.pkl", weights.name]
    if not all((checkpoint / name).stat().st_size for name in state_files):
        raise ValueError(f"Incomplete training checkpoint: {checkpoint}")
    torch.set_num_threads(1)
    checkpoint_audit = audit_weights(weights)
    audits = {
        "adapter": audit_weights(checkpoint.parent / weights.name),
        "checkpoint_adapter": checkpoint_audit,
        "comparison_adapter_sha256": checkpoint_audit["sha256"],
        "checkpoint_step": args.checkpoint_step,
        "base_manifest_sha256": settings["base_manifest_sha256"],
        "base": str(base),
        "dataset_sha256": export["dataset_sha256"],
        "split": args.split,
        "condition_columns": columns,
    }
    args.out.mkdir()
    with (args.out / "base-verify.log").open("x", encoding="utf-8") as handle:
        subprocess.run(["sha256sum", "--strict", "--check", str((args.run / "base.sha256").resolve())],
                       cwd=base, stdout=handle, check=True)
    dataset = load_export(comparison_paths, args.split, str(args.out / "hf-cache"))
    row = dataset[args.index]
    images = {}
    swap = dual and not args.no_swap
    if dual:
        images.update({role: row[role] for role in ("source", "target", "pose")})
        audits["pose_sha256"] = row["pose_sha256"]
        audits["target_joints"] = {"target": json.loads(row["pose_joints"])}
        if swap:
            candidate, rms, common = select_pose_swap(dataset.select_columns(list(SWAP_COLUMNS)),
                                                      args.index, args.swap_index)
            alternate = dataset[candidate]
            images.update(target_swap=alternate["target"], pose_swap=alternate["pose"])
            audits["target_joints"]["target_swap"] = json.loads(alternate["pose_joints"])
            audits["pose_swap"] = {
                "id": alternate["id"], "index": candidate, "target_sha256": alternate["target_sha256"],
                "pose_sha256": alternate["pose_sha256"], "pose_rms_px": rms, "common_joints": common,
            }
        if any(image.size != (512, 512) for image in images.values()):
            raise ValueError("Expected pre-aligned 512x512 dual-image canvases")
    else:
        for role in ("source", "target"):
            images[role] = ImageOps.fit(row[role].convert("RGBA"), (512, 512), Image.Resampling.BILINEAR)

    pipe = QwenImage21Pipeline.from_pretrained(base, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    seed = 42
    options = dict(prompt=row["caption"], width=512, height=512,
                   output_resolution=512, true_cfg_scale=1.0, num_inference_steps=args.steps)

    def generate(arm):
        if dual:
            for suffix in (("", "_swap") if swap else ("",)):
                refs = [images["source"], images[f"pose{suffix}"]]
                images[f"{arm}{suffix}"] = pipe(image=refs, **options,
                                                generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        else:
            images[arm] = pipe(image=images["source"], **options,
                               generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    generate("baseline")
    pipe.load_lora_weights(str(weights.parent), weight_name=weights.name, adapter_name="hand_prior")
    adapter = pipe.transformer.peft_config["hand_prior"]
    if adapter.r != 16 or adapter.lora_alpha != 16:
        raise ValueError(f"Reloaded rank/alpha mismatch: {adapter.r}/{adapter.lora_alpha}")
    generate("lora")

    png_sha = {}
    for name, image in images.items():
        path = args.out / f"{name}.png"
        with path.open("xb") as handle:
            image.save(handle, format="PNG")
        png_sha[name] = file_sha256(path)
    pixel_mae = image_delta(images["baseline"], images["lora"])
    if pixel_mae == 0:
        raise ValueError("Same-seed baseline and LoRA produced identical RGB")
    audits.update(id=row["id"], group=row["group"], caption=row["caption"], source_sha256=row["source_sha256"],
                  target_sha256=row["target_sha256"], seed=seed, inference_steps=args.steps, resolution=512,
                  png_sha256=png_sha, rgb_mae_0_255=pixel_mae)
    if swap:
        for arm in ("baseline", "lora"):
            audits["pose_swap"][f"{arm}_rgb_mae_0_255"] = image_delta(images[arm], images[f"{arm}_swap"])
        if audits["pose_swap"]["lora_rgb_mae_0_255"] == 0:
            raise ValueError("The loaded LoRA produced identical RGB after the pose swap")
    with (args.out / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(audits, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audits))


if __name__ == "__main__":
    main()
