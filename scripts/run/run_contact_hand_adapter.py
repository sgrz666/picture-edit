#!/usr/bin/env python3
"""Generate with a separately trained Contact Hand Adapter checkpoint."""
import argparse
import json
import math
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
    parser.add_argument("--current-support", type=Path,
                        help="Torch-saved boolean or float [4,H,W] current-hand masks in full target canvas coordinates")
    parser.add_argument("--local-size", type=int, default=512)
    parser.add_argument("--local-dilation", type=int, help="Local edit mask dilation in full-canvas pixels (default: 8)")
    parser.add_argument("--slot", action="append", choices=("A-L", "A-R", "B-L", "B-R"))
    parser.add_argument("--local-prompt", help="Select an explicit instruction for the local pass")
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--hand-prior", type=Path, help="Separate official Qwen I2I Q/K/V/O hand LoRA safetensors")
    parser.add_argument("--hand-prior-sha", help="Required pinned SHA256 for --hand-prior")
    parser.add_argument("--hand-prior-strength", type=float,
                        help="Maximum late-denoising hand ROI strength (default: 1 with a prior, otherwise 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (args.mode == "local") != (args.context is not None):
        parser.error("--context is required for local-only mode")
    if args.mode == "global" and (args.current_support is not None or args.local_dilation is not None):
        parser.error("--current-support and --local-dilation require a local pass")
    if args.local_dilation is not None and args.local_dilation < 0:
        parser.error("--local-dilation must be nonnegative")
    if (args.hand_prior is None) != (args.hand_prior_sha is None):
        parser.error("--hand-prior and --hand-prior-sha must be supplied together")
    prior_strength = args.hand_prior_strength
    if prior_strength is None:
        prior_strength = float(args.hand_prior is not None)
    if not math.isfinite(prior_strength) or prior_strength < 0 or (prior_strength and args.hand_prior is None):
        parser.error("--hand-prior-strength must be nonnegative, and positive strength requires pinned prior weights")
    if args.out.exists():
        raise FileExistsError(args.out)

    import torch
    from diffusers import QwenImage21Pipeline

    from src.pose_control.contact_hand_adapter import HAND_SLOTS, install_hand_adapter
    from src.pose_control.contact_hand_adapter.prior import load_hand_prior
    from src.pose_control.contact_hand_adapter.runtime import base_fingerprint, generate, load_adapter, load_condition
    from src.pose_control.contact_hand_adapter.training import open_rgb
    from src.pose_edit.common import sha256_file

    condition = load_condition(args.condition)
    local_options = {}
    if args.current_support is not None:
        local_options["current_support"] = torch.load(args.current_support, map_location="cpu", weights_only=True)
    if args.local_dilation is not None:
        local_options["local_dilation"] = args.local_dilation
    base_sha = base_fingerprint(args.model)
    adapter, checkpoint = load_adapter(args.adapter, base_sha=base_sha)
    prior = None if args.hand_prior is None else load_hand_prior(args.hand_prior, sha256=args.hand_prior_sha)
    print(f"Qwen {args.model.resolve()} SHA256={base_sha}")
    print(f"Adapter {args.adapter.resolve()} SHA256={sha256_file(args.adapter)} update={checkpoint['step']}", flush=True)
    if prior is not None:
        provenance = {**prior.provenance(), "strength": prior_strength, "path": str(args.hand_prior.resolve())}
        print(f"Hand prior {json.dumps(provenance, sort_keys=True)}", flush=True)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    pipe = QwenImage21Pipeline.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to(device)
    install_hand_adapter(pipe.transformer, adapter.to(device).eval(),
                         hand_prior=None if prior is None else prior.to(device))
    references = tuple(open_rgb(path) for path in args.reference)
    context = open_rgb(args.context) if args.context else None
    common = dict(references=references, seed=args.seed, steps=args.steps, strength=args.strength,
                  source_hashes=tuple(sha256_file(path) for path in args.reference), prior_strength=prior_strength)
    if args.mode in ("global", "global-local"):
        context = generate(pipe, condition, prompt=args.prompt, **common)
    if args.mode in ("local", "global-local"):
        slots = None if args.slot is None else tuple(HAND_SLOTS.index(slot) for slot in args.slot)
        context = generate(pipe, condition, prompt=args.local_prompt or args.prompt, local_context=context,
                           local_size=args.local_size, local_slots=slots, **local_options, **common)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        context.save(output, format="PNG")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
