#!/usr/bin/env python3
"""Train only the independent Contact Hand Adapter from immutable source/target pairs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--check-data", action="store_true", help="Verify the manifest, images and hand geometry on CPU")
    parser.add_argument("--model", type=Path, help="Local Qwen-Image-2.1 directory")
    parser.add_argument("--out", type=Path, help="New independent Adapter checkpoint")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", type=Path, help="Resume the same training signature and optimizer")
    restart.add_argument("--init-adapter", type=Path, help="Warm-start a new stage with a fresh optimizer")
    parser.add_argument("--updates", type=int, default=100, help="Final optimizer update")
    parser.add_argument("--save-every", type=int, default=100, help="Write immutable intermediate checkpoints every N updates")
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-size", type=int, default=512)
    parser.add_argument("--local-fraction", type=float, help="Fraction of samples drawn from local rows")
    parser.add_argument("--local-context-weights", type=float, nargs=3, metavar=("SOURCE", "COARSE", "SYNTHETIC"),
                        help="Relative weights for the three context kinds when sampling local training rows")
    parser.add_argument("--person-swap-probability", type=float, default=.5,
                        help="Probability of exchanging whole A/B actors on an eligible training row (default: 0.5)")
    parser.add_argument("--surface-tokens", type=int, choices=(32, 64), default=64)
    parser.add_argument("--injection-layers", type=int, choices=(8, 16), default=8)
    parser.add_argument("--contact-level", choices=("hand", "part", "surface"), default="surface")
    parser.add_argument("--image-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-lora-rank", type=int, default=0)
    parser.add_argument("--rgb-evaluator", type=Path, help="Frozen TorchScript taking [1,3,H,W] and returning named heads")
    parser.add_argument("--rgb-weight", type=float, default=.1)
    parser.add_argument("--rgb-sigma-max", type=float, default=.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (min(args.updates, args.accumulate, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 < args.rgb_sigma_max <= 1 or not math.isfinite(args.rgb_weight) or args.rgb_weight < 0
            or (args.local_fraction is not None and not 0 <= args.local_fraction <= 1)
            or not 0 <= args.person_swap_probability <= 1):
        parser.error("Invalid training updates, learning rate, local fraction, person swap probability or RGB settings")
    if not args.check_data and (args.model is None or args.out is None):
        parser.error("Training requires --model and --out")

    import torch
    from src.pose_control.contact_hand_adapter.training import (
        LocalContextMixture, eligible_output_labels, load_manifest, load_training_pair,
    )

    records = load_manifest(args.manifest)
    train_rows = [row for row in records if row["split"] == "train"]
    local_rows = [row for row in train_rows if row["mode"] == "local"]
    try:
        mixture = LocalContextMixture(local_rows, args.local_context_weights) if args.local_context_weights else None
    except ValueError as error:
        parser.error(str(error))
    check_rng = torch.Generator().manual_seed(args.seed)
    swap_counts = Counter()
    for record in records:
        swap_people = record["split"] == "train" and args.person_swap_probability > 0
        pair = load_training_pair(record, local_size=args.local_size, swap_people=swap_people, rng=check_rng)
        record["rgb_eligible"] = eligible_output_labels(pair.rgb_labels or {})
        if swap_people:
            swap_counts[pair.person_swap_status] += 1
    counts = Counter((row["split"], row["mode"]) for row in records)
    print(json.dumps({f"{split}/{mode}": count for (split, mode), count in sorted(counts.items())}), flush=True)
    context_counts = Counter(row["context_kind"] for row in local_rows)
    print(f"local_context_rows={dict(context_counts)} person_swap_rows={dict(swap_counts)}", file=sys.stderr, flush=True)
    if args.check_data:
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    _train(args, train_rows, mixture)


def _train(args, records, mixture):
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers import transformer_qwenimage21
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21

    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter, install_hand_adapter
    from src.pose_control.contact_hand_adapter.runtime import base_fingerprint, initialize_adapter, load_adapter, save_adapter
    from src.pose_control.contact_hand_adapter.training import (
        REGION_WEIGHTS, check_output_gradients, decode_clean_latent, encode_pair, load_training_pair, output_supervision,
        predict_flow, region_flow_objective, token_regions,
    )
    from src.pose_edit.common import sha256_file

    pools = {mode: [row for row in records if row["mode"] == mode] for mode in ("global", "local")}
    if not records or (args.local_fraction is not None and
                       ((args.local_fraction > 0 and not pools["local"]) or
                        (args.local_fraction < 1 and not pools["global"]))):
        raise ValueError("Training manifest lacks the requested global/local sample pools")
    if args.rgb_evaluator and not any(row["rgb_eligible"] for row in records):
        raise ValueError("Output-side RGB supervision requires eligible training labels")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    base_sha = base_fingerprint(args.model)
    signature = dict(manifest_sha=sha256_file(args.manifest), seed=args.seed, accumulate=args.accumulate,
                     local_size=args.local_size, local_fraction=args.local_fraction, lr=args.lr,
                     local_context_weights=args.local_context_weights, person_swap_probability=args.person_swap_probability,
                     rgb_evaluator_sha=sha256_file(args.rgb_evaluator) if args.rgb_evaluator else None,
                     rgb_weight=args.rgb_weight, rgb_sigma_max=args.rgb_sigma_max, dtype=str(dtype),
                     qwen_transformer_sha=sha256_file(transformer_qwenimage21.__file__),
                     qwen_pipeline_sha=sha256_file(pipeline_qwenimage21.__file__))
    print(f"Qwen {args.model.resolve()} SHA256={base_sha}", flush=True)
    torch.manual_seed(args.seed)
    config = AdapterConfig(surface_tokens=args.surface_tokens, image_feedback=args.image_feedback,
                           contact_level=args.contact_level, hand_lora_rank=args.hand_lora_rank,
                           layers=tuple(range(32 // args.injection_layers - 1, 32, 32 // args.injection_layers)))
    step, saved = 0, None
    parent_path = args.resume or args.init_adapter
    parent_sha = sha256_file(parent_path) if parent_path else None
    if args.resume:
        adapter, record = load_adapter(args.resume, base_sha=base_sha)
        step, saved = record["step"], record["training"]
        if saved is None or saved["signature"] != signature or step >= args.updates or adapter.config != config:
            raise ValueError("Resume requires the same data/training signature and a later update")
        print(f"Parent {args.resume.resolve()} SHA256={parent_sha} update={step}", flush=True)
    else:
        adapter = ContactHandAdapter(config)
        if args.init_adapter:
            parent_step = initialize_adapter(adapter, args.init_adapter, base_sha=base_sha)
            print(f"Warm start {args.init_adapter.resolve()} SHA256={parent_sha} update={parent_step}", flush=True)
        else:
            print(f"Adapter initialization seed={args.seed}, architecture={config}", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to(device)
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    install_hand_adapter(pipe.transformer, adapter.to(device).train())
    pipe.transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sample_rng"])
        noise_rng.set_state(saved["noise_rng"])
    evaluator = (torch.jit.load(str(args.rgb_evaluator), map_location=device).eval().requires_grad_(False)
                 if args.rgb_evaluator else None)
    checked_rgb = set()

    def save(path, update):
        state = dict(signature=signature, optimizer=optimizer.state_dict(), sample_rng=rng.get_state(),
                     noise_rng=noise_rng.get_state(), parent_sha=parent_sha)
        sha = save_adapter(path, adapter, base_sha=base_sha, step=update, training=state)
        print(f"Adapter {path.resolve()} SHA256={sha} update={update}", flush=True)

    for update in range(step + 1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        window = []
        for micro in range(args.accumulate):
            index = (update - 1) * args.accumulate + micro
            pool = records
            if args.local_fraction is not None:
                ratio = args.local_fraction
                local = math.floor((index + 1) * ratio) > math.floor(index * ratio)
                pool = pools["local" if local else "global"]
            row = pool[int(torch.randint(len(pool), (), generator=rng))]
            if mixture is not None and row["mode"] == "local":
                row = mixture.sample(rng)
            swap_people = (args.person_swap_probability > 0
                           and torch.rand((), generator=rng).item() < args.person_swap_probability)
            pair = load_training_pair(row, local_size=args.local_size, swap_people=swap_people, rng=rng)
            sigma = torch.rand(1, generator=rng).clamp(.001, .999)
            rgb = row["rgb_eligible"] if evaluator is not None and sigma.item() <= args.rgb_sigma_max else ()
            masks = token_regions(pair.condition.canvas, pair.regions, "cpu")
            window.append((pair, sigma, rgb, masks))
        counts = {name: sum(bool(masks[name].any()) for _, _, _, masks in window) for name in REGION_WEIGHTS}
        rgb_counts = Counter(name for _, _, active, _ in window for name in active)
        context_counts = Counter(pair.context_kind for pair, _, _, _ in window if pair.context_kind is not None)
        swap_counts = Counter(pair.person_swap_status for pair, _, _, _ in window)
        total = 0.
        for pair, sigma, rgb, _ in window:
            batch = encode_pair(pipe, condition=pair.condition, references=pair.references,
                                target=pair.target, prompt=pair.prompt, source_hashes=pair.source_hashes)
            noise = torch.randn(batch.target.shape, device=device, generator=noise_rng)
            prediction, velocity, clean = predict_flow(pipe.transformer, batch, sigma.to(device), noise)
            loss, _ = region_flow_objective(prediction, velocity, pair.condition.canvas, pair.regions,
                                           denominators=counts)
            if rgb:
                decoded = decode_clean_latent(pipe, clean, pair.condition.canvas)
                selected = {name: pair.rgb_labels[name] for name in rgb}
                scores = output_supervision(evaluator(decoded), selected)
                new_heads = {name: loss for name, loss in scores.items() if name not in checked_rgb}
                check_output_gradients(clean, new_heads)
                checked_rgb.update(new_heads)
                loss = loss + args.rgb_weight * sum(value / rgb_counts[name] for name, value in scores.items())
            if not torch.isfinite(loss):
                raise RuntimeError("The hand training loss is non-finite")
            loss.backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        if not norm.item():
            raise RuntimeError("No Adapter gradient on this training update")
        optimizer.step()
        print(f"update={update} loss={total:.6f} grad={norm.item():.6f} rgb_eligible={dict(rgb_counts)} "
              f"contexts={dict(context_counts)} person_swap={dict(swap_counts)}", flush=True)
        if update < args.updates and update % args.save_every == 0:
            save(args.out.with_name(f"{args.out.stem}.step-{update:06d}{args.out.suffix}"), update)
    save(args.out, args.updates)
    print(f"trainable={sum(p.numel() for p in adapter.parameters())}")


if __name__ == "__main__":
    main()
