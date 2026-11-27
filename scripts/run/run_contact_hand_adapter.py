#!/usr/bin/env python3
"""Generate with a separately trained Contact Hand Adapter checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen-Image-2.1 directory")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True, help="Ordered RGB references")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "local", "global-local"), default="global")
    parser.add_argument("--context", type=Path, help="Target-canvas RGB for local-only refinement")
    parser.add_argument("--local-size", type=int, default=512)
    parser.add_argument("--slot", action="append", choices=("A-L", "A-R", "B-L", "B-R"))
    parser.add_argument("--local-prompt", help="Select an explicit instruction for the local pass")
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (args.mode == "local") != (args.context is not None):
        parser.error("--context is required for local-only mode")
    if args.out.exists():
        raise FileExistsError(args.out)

    import torch
    from diffusers import QwenImage21Pipeline

    from src.pose_control.contact_hand_adapter import HAND_SLOTS, install_hand_adapter
    from src.pose_control.contact_hand_adapter.runtime import base_fingerprint, generate, load_adapter, load_condition
    from src.pose_control.contact_hand_adapter.training import open_rgb
    from src.pose_edit.common import sha256_file

    condition = load_condition(args.condition)
    base_sha = base_fingerprint(args.model)
    adapter, checkpoint = load_adapter(args.adapter, base_sha=base_sha)
    print(f"Qwen {args.model.resolve()} SHA256={base_sha}")
    print(f"Adapter {args.adapter.resolve()} SHA256={sha256_file(args.adapter)} update={checkpoint['step']}", flush=True)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    pipe = QwenImage21Pipeline.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to(device)
    install_hand_adapter(pipe.transformer, adapter.to(device).eval())
    references = tuple(open_rgb(path) for path in args.reference)
    context = open_rgb(args.context) if args.context else None
    common = dict(references=references, seed=args.seed, steps=args.steps, strength=args.strength,
                  source_hashes=tuple(sha256_file(path) for path in args.reference))
    if args.mode in ("global", "global-local"):
        context = generate(pipe, condition, prompt=args.prompt, **common)
    if args.mode in ("local", "global-local"):
        slots = None if args.slot is None else tuple(HAND_SLOTS.index(slot) for slot in args.slot)
        context = generate(pipe, condition, prompt=args.local_prompt or args.prompt, local_context=context,
                           local_size=args.local_size, local_slots=slots, **common)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        context.save(output, format="PNG")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
