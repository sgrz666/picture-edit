"""Ordered dual-image conditioning for the pinned Qwen-Image-2.1 trainer."""

import json
from pathlib import Path
import random

import numpy as np
import torch
from dataset_contract import file_sha256, fingerprint


CONDITION_COLUMNS = ("source", "pose")
RESOLUTION = 512


def image_slot_spans(mask, expected):
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError("Dual Qwen conditioning requires one image-pad mask per micro-batch")
    padded = torch.nn.functional.pad(mask[0].to(torch.int8), (1, 1))
    transitions = padded.diff()
    starts = torch.where(transitions == 1)[0].tolist()
    ends = torch.where(transitions == -1)[0].tolist()
    if tuple(end - start for start, end in zip(starts, ends)) != tuple(expected):
        raise ValueError(f"Wrong per-image vision slots: {list(zip(starts, ends))}, expected sizes {expected}")
    return list(zip(starts, ends))


def latent_shapes(latents):
    if any(value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != 1
           or any(side % 2 for side in value.shape[-2:]) for value in latents):
        raise ValueError("Expected batch-one images with one frame and even latent dimensions")
    return [tuple(value.shape[-3:]) for value in latents]


def pack_inputs(pack, conditions, noisy_target, prompt_image_mask):
    if len(conditions) != len(CONDITION_COLUMNS):
        raise ValueError("Expected ordered source RGB and pose latents")
    latents = (*conditions, noisy_target)
    shapes = latent_shapes(latents)
    slots = [height * width // 4 for _, height, width in shapes]
    image_slot_spans(prompt_image_mask, slots[:-1])
    packed = [pack(value, batch_size=1, num_channels_latents=value.shape[1], height=height, width=width)
              for value, (_, height, width) in zip(latents, shapes)]
    mask = torch.cat((prompt_image_mask, prompt_image_mask.new_ones(1, slots[-1])), dim=1)
    return torch.cat(packed, dim=1), [shapes], mask, packed[-1].shape[1]


class DiskConditionCache(torch.utils.data.Dataset):
    """One immutable safetensors shard per pair, loaded only when sampled."""

    def __init__(self, root, count):
        self.root = Path(root)
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        from safetensors.torch import load_file

        return load_file(self.root / f"{index:06d}.safetensors")


def image_tensor(image, device, dtype):
    pixels = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
    return pixels.div_(127.5).sub_(1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)


def single_cached_pair(examples):
    pair, = examples
    return pair


def open_condition_cache(args, run):
    root = Path(args.condition_cache_dir)
    record = json.loads((root / "index.json").read_text(encoding="utf-8"))
    spec = record["spec"]
    provenance = dict(line.split("=", 1) for line in (run / "provenance.txt").read_text().splitlines()
                      if "=" in line)
    if (record["identity"] != fingerprint(spec) or spec["schema"] != 1
            or tuple(spec["condition_columns"]) != CONDITION_COLUMNS or spec["resolution"] != RESOLUTION
            or spec["dtype"] != args.mixed_precision or spec["seed"] != args.seed
            or any(spec[key] != provenance[key] for key in ("base_manifest_sha256", "dataset_sha256"))
            or file_sha256(root / "index.json") != provenance["cache_index_sha256"]):
        raise ValueError("Condition cache does not match the locked run")
    return DiskConditionCache(root, len(record["rows"])), provenance


@torch.no_grad()
def encode_pair(row, pipeline, vae, device, seed):
    if any(row[key].size != (RESOLUTION, RESOLUTION) for key in (*CONDITION_COLUMNS, "target")):
        raise ValueError(f"Pair {row['id']} must be aligned to the shared 512x512 canvas")
    a, pose, target = (row[key].convert("RGBA") for key in (*CONDITION_COLUMNS, "target"))
    latent_a, latent_pose = (vae.encode(image_tensor(image, device, vae.dtype)).latent_dist.mode()
                             for image in (a, pose))
    rng = torch.Generator(device=device).manual_seed(seed)
    latent_target = vae.encode(image_tensor(target, device, vae.dtype)).latent_dist.sample(generator=rng)
    embeds, prompt_mask, pad = pipeline.encode_prompt(prompt=row["caption"], image=[a, pose], device=device)
    if prompt_mask is None:
        prompt_mask = torch.ones_like(pad, dtype=torch.long)
    shapes = latent_shapes((latent_a, latent_pose, latent_target))
    spans = image_slot_spans(pad, [h * w // 4 for _, h, w in shapes[:-1]])
    values = {"source": latent_a, "pose": latent_pose, "target": latent_target,
              "prompt_embeds": embeds, "prompt_mask": prompt_mask, "image_pad_mask": pad}
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError(f"Non-finite condition for {row['id']}")
    return {k: v.detach().cpu().contiguous() for k, v in values.items()}, {
        "latent_shapes": [list(shape) for shape in shapes], "condition_slot_spans": [list(span) for span in spans]}


class UpdateSampler:
    """Replay each epoch by seed and seek directly to a completed optimizer step."""

    def __init__(self, size, seed, rank, world, accumulation, first_step, last_step):
        if size < 1 or not 0 <= rank < world or not 0 <= first_step < last_step or accumulation < 1:
            raise ValueError("Invalid cache size, process topology, or update interval")
        self.size, self.seed, self.rank, self.world = size, seed, rank, world
        self.start, self.stop = first_step * accumulation, last_step * accumulation

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        epoch, permutation = None, None
        for batch in range(self.start, self.stop):
            new_epoch, offset = divmod(batch * self.world + self.rank, self.size)
            if new_epoch != epoch:
                epoch = new_epoch
                permutation = list(range(self.size))
                random.Random(self.seed + epoch).shuffle(permutation)
            yield [permutation[offset]]


def run_signature(args, provenance, accelerator):
    keys = ("base_manifest_sha256", "dataset_sha256", "cache_index_sha256", "init_lora_sha256",
            "effective_trainer_sha256", "helper_sha256", "deterministic")
    return fingerprint({
        **{key: provenance[key] for key in keys}, "world": accelerator.num_processes,
        "seed": args.seed, "grad_accum": args.gradient_accumulation_steps, "precision": args.mixed_precision,
        "rank": args.rank, "alpha": args.lora_alpha, "layers": args.lora_layers, "dropout": args.lora_dropout,
        "lr": args.learning_rate, "lr_scheduler": args.lr_scheduler, "warmup": args.lr_warmup_steps,
        "optimizer": args.optimizer, "adam": [args.adam_beta1, args.adam_beta2, args.adam_weight_decay, args.adam_epsilon],
        "clip": args.max_grad_norm, "loss": [args.weighting_scheme, args.logit_mean, args.logit_std, args.mode_scale],
        "activation_checkpointing": args.gradient_checkpointing,
        "runtimes": [line for line in (Path(args.output_dir).parent / "provenance.txt").read_text().splitlines()
                     if line.startswith(("Name:", "Version:"))],
    })


def rng_digest(device):
    import hashlib

    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()
                          + torch.cuda.get_rng_state(device).numpy().tobytes()).hexdigest()


def save_training_state(accelerator, path, step, signature):
    from accelerate.utils import gather_object
    from prepare_cache import write_record

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    accelerator.wait_for_everyone()
    states = gather_object([rng_digest(accelerator.device)])
    accelerator.save_state(str(path))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        files = {p.name: file_sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
        write_record(path / "resume.json", {"step": step, "signature": signature, "rng": states, "files": files})
    accelerator.wait_for_everyone()


def resume_training_state(accelerator, path, signature, max_steps):
    path = Path(path)
    record = json.loads((path / "resume.json").read_text(encoding="utf-8"))
    step = record["step"]
    names = {"pytorch_lora_weights.safetensors", "optimizer.bin", "scheduler.bin",
             *(f"random_states_{rank}.pkl" for rank in range(accelerator.num_processes))}
    if (record["signature"] != signature or not 0 < step < max_steps
            or len(record["rng"]) != accelerator.num_processes or not names <= record["files"].keys()):
        raise ValueError("Resume checkpoint signature, completeness, or step is incompatible")
    for name, checksum in record["files"].items():
        if Path(name).name != name or file_sha256(path / name) != checksum:
            raise ValueError(f"Resume checkpoint file changed: {name}")
    accelerator.load_state(str(path))
    if rng_digest(accelerator.device) != record["rng"][accelerator.process_index]:
        raise ValueError("Accelerate failed to restore the exact torch CPU/CUDA RNG")
    return step
