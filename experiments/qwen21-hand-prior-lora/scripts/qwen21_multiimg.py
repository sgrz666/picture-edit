"""Ordered dual-image conditioning for the pinned Qwen-Image-2.1 trainer."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


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


@torch.no_grad()
def cache_dataset(args, pipeline, vae, device):
    from datasets import load_dataset
    from safetensors.torch import save_file
    from tqdm.auto import tqdm

    rows = load_dataset(args.dataset_name, args.dataset_config_name, split="train", cache_dir=args.cache_dir)
    root = Path(args.condition_cache_dir)
    root.mkdir()
    index = []
    for number, row in enumerate(tqdm(rows, desc="Caching dual-image conditions")):
        if any(row[column].size != (RESOLUTION, RESOLUTION) for column in (*CONDITION_COLUMNS, args.image_column)):
            raise ValueError(f"Pair {row['id']} must be aligned to the shared 512x512 canvas")
        conditions = [row[column].convert("RGBA") for column in CONDITION_COLUMNS]
        target = row[args.image_column].convert("RGBA")
        caption = row[args.caption_column]
        pixels = [image_tensor(image, device, vae.dtype) for image in (*conditions, target)]
        latent_a = vae.encode(pixels[0]).latent_dist.mode()
        latent_pose = vae.encode(pixels[1]).latent_dist.mode()
        latent_target = vae.encode(pixels[2]).latent_dist.sample()
        embeds, prompt_mask, pad = pipeline.encode_prompt(prompt=caption, image=conditions, device=device)
        if prompt_mask is None:
            prompt_mask = torch.ones_like(pad, dtype=torch.long)
        shapes = latent_shapes((latent_a, latent_pose, latent_target))
        spans = image_slot_spans(pad, [h * w // 4 for _, h, w in shapes[:-1]])
        values = {"source": latent_a, "pose": latent_pose, "target": latent_target,
                  "prompt_embeds": embeds, "prompt_mask": prompt_mask, "image_pad_mask": pad}
        for name in ("source", "pose", "target", "prompt_embeds"):
            if not torch.isfinite(values[name]).all():
                raise ValueError(f"Non-finite cached {name} for {row['id']}")
        shard = root / f"{number:06d}.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, shard)
        with shard.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        index.append({"id": row["id"], "file": shard.name, "sha256": digest,
                      "latent_shapes": shapes, "condition_slot_spans": spans})
    if not index:
        raise ValueError("No dual-image training pairs")
    provenance = root.parent / "provenance.txt"
    hashes = dict(line.split("=", 1) for line in provenance.read_text().splitlines() if "=" in line)
    record = {"condition_columns": CONDITION_COLUMNS, "resolution": RESOLUTION, "seed": args.seed,
              "base_manifest_sha256": hashes["base_manifest_sha256"], "dataset_sha256": hashes["dataset_sha256"],
              "trainer_sha256": hashes["effective_trainer_sha256"], "helper_sha256": hashes["helper_sha256"],
              "rows": index}
    with (root / "index.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return DiskConditionCache(root, len(index))
