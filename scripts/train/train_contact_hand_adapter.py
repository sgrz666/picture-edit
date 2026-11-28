#!/usr/bin/env python3
"""Train only the independent Contact Hand Adapter from immutable source/target pairs."""
import argparse
from collections import Counter
from dataclasses import asdict
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
    parser.add_argument("--residual-mode", choices=("compressed", "low_rank", "direct"), default="compressed")
    parser.add_argument("--spatial-mode", choices=("instance", "average"), default="instance")
    parser.add_argument("--gate-mode", choices=("typed", "uniform"), default="typed")
    parser.add_argument("--appearance-mode", choices=("part", "global", "off"), default="part")
    parser.add_argument("--appearance-dino", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cached SHA-bound source DINO features when present")
    parser.add_argument("--hand-prior", type=Path, help="Pinned frozen official hand LoRA checkpoint")
    parser.add_argument("--hand-prior-sha", help="Expected SHA256 of the official hand prior")
    parser.add_argument("--hand-prior-strength", type=float, help="Frozen prior gain (default: 1 when supplied)")
    parser.add_argument("--rgb-evaluator", type=Path, help="Frozen TorchScript taking [1,3,H,W] and returning named heads")
    parser.add_argument("--rgb-evaluator-sha", help="Expected SHA256 of the frozen TorchScript evaluator")
    parser.add_argument("--rgb-feature-model", type=Path, help="Local frozen DINO ViT directory for target perceptual loss")
    parser.add_argument("--rgb-feature-sha", help="Expected aggregate DINO model/config/processor SHA256")
    parser.add_argument("--objective-weight", action="append", default=[], metavar="NAME=VALUE",
                        help="Repeat for flow, joint, pose_2d, pose_3d, part, anatomy, contact, occlusion, collision, appearance, perceptual")
    parser.add_argument("--rgb-weight", type=float, help="Legacy common weight for named V2.1 RGB labels")
    parser.add_argument("--rgb-sigma-max", type=float, default=.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (min(args.updates, args.accumulate, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 < args.rgb_sigma_max <= 1
            or (args.rgb_weight is not None and (not math.isfinite(args.rgb_weight) or args.rgb_weight <= 0))
            or (args.local_fraction is not None and not 0 <= args.local_fraction <= 1)
            or not 0 <= args.person_swap_probability <= 1):
        parser.error("Invalid training updates, learning rate, local fraction, person swap probability or RGB settings")
    if not args.check_data and (args.model is None or args.out is None):
        parser.error("Training requires --model and --out")
    if args.hand_prior_strength is None:
        args.hand_prior_strength = 1. if args.hand_prior is not None else 0.
    if (bool(args.hand_prior) != bool(args.hand_prior_sha) or not math.isfinite(args.hand_prior_strength)
            or args.hand_prior_strength < 0 or (args.hand_prior is None and args.hand_prior_strength)):
        parser.error("A frozen hand prior requires --hand-prior, --hand-prior-sha and a nonnegative strength")
    if (args.rgb_evaluator_sha and args.rgb_evaluator is None) or (args.rgb_weight is not None and
            (args.rgb_evaluator is None or args.objective_weight)):
        parser.error("--rgb-weight and --rgb-evaluator-sha require an evaluator; use --objective-weight for per-head weights")

    import torch
    from src.pose_control.contact_hand_adapter.objectives import LEGACY_RGB_OBJECTIVES, RGB_OBJECTIVES, objective_weights
    from src.pose_control.contact_hand_adapter.training import (
        LOCAL_CONTEXT_KINDS, LocalContextMixture, load_manifest, load_training_pair, training_output_data,
    )
    from src.pose_edit.common import sha256_file

    try:
        weights = {name: weight for name, weight in objective_weights(args.objective_weight).items() if weight > 0}
    except ValueError as error:
        parser.error(str(error))
    legacy_rgb = args.rgb_evaluator is not None and not args.objective_weight
    candidate_aux = LEGACY_RGB_OBJECTIVES if legacy_rgb else tuple(name for name in weights if name != "flow")
    if candidate_aux and args.rgb_sigma_max < .001:
        parser.error("Output objectives require --rgb-sigma-max >= 0.001 (minimum sampled sigma)")
    if bool(args.rgb_evaluator) != any(name in RGB_OBJECTIVES for name in candidate_aux):
        parser.error("Enabled evaluator heads require --rgb-evaluator; provide its named positive objective weights")
    if args.rgb_evaluator is not None:
        evaluator_sha = sha256_file(args.rgb_evaluator)
        if args.rgb_evaluator_sha is not None and args.rgb_evaluator_sha != evaluator_sha:
            parser.error("Frozen RGB evaluator SHA256 differs from --rgb-evaluator-sha")
        args.rgb_evaluator_sha = evaluator_sha
    if args.hand_prior is not None and sha256_file(args.hand_prior) != args.hand_prior_sha:
        parser.error("Frozen hand prior SHA256 differs from --hand-prior-sha")
    perceptual = "perceptual" in weights
    if bool(args.rgb_feature_model) != perceptual or bool(args.rgb_feature_sha) != perceptual:
        parser.error("Enable perceptual with --rgb-feature-model and --rgb-feature-sha together")
    if perceptual:
        from src.pose_control.contact_hand_adapter.appearance import dino_model_fingerprint

        if dino_model_fingerprint(args.rgb_feature_model) != args.rgb_feature_sha:
            parser.error("Local DINO feature model SHA256 differs from --rgb-feature-sha")

    records = load_manifest(args.manifest)
    train_rows = [row for row in records if row["split"] == "train"]
    local_rows = [row for row in train_rows if row["mode"] == "local"]
    global_rows = [row for row in train_rows if row["mode"] == "global"]
    if (args.local_fraction is not None and ((args.local_fraction > 0 and not local_rows)
                                             or (args.local_fraction < 1 and not global_rows))):
        parser.error("Training manifest lacks the requested global/local sample pools")
    try:
        mixture = LocalContextMixture(local_rows, args.local_context_weights) if args.local_context_weights else None
    except ValueError as error:
        parser.error(str(error))
    check_rng = torch.Generator().manual_seed(args.seed)
    swap_counts = Counter()
    eligible_counts = Counter()
    appearance_model_shas = set()
    selected_kinds = {kind for kind, weight in zip(LOCAL_CONTEXT_KINDS, args.local_context_weights or ()) if weight > 0}
    for record in records:
        swap_people = record["split"] == "train" and args.person_swap_probability > 0
        pair = load_training_pair(record, local_size=args.local_size, swap_people=swap_people, rng=check_rng)
        if args.appearance_dino and pair.condition.appearance_dino_model_sha is not None:
            appearance_model_shas.add(pair.condition.appearance_dino_model_sha)
        is_local = record["mode"] == "local"
        if (candidate_aux and record["split"] == "train"
                and ((is_local and args.local_fraction != 0 and (mixture is None or record["context_kind"] in selected_kinds))
                     or (not is_local and args.local_fraction != 1))):
            active, _, _ = training_output_data(pair, candidate_aux)
            eligible_counts.update(active)
        if swap_people:
            swap_counts[pair.person_swap_status] += 1
    if len(appearance_model_shas) > 1:
        parser.error("One stage must use a single frozen appearance DINO model SHA256")
    args.appearance_dino_model_sha = next(iter(appearance_model_shas), None)
    if legacy_rgb:
        weights.update({name: args.rgb_weight or .1 for name in LEGACY_RGB_OBJECTIVES if eligible_counts[name]})
        if len(weights) == 1:
            parser.error("Output-side RGB supervision requires eligible legacy training labels")
    missing = [name for name in weights if name != "flow" and not eligible_counts[name]]
    if missing:
        parser.error(f"No qualified reachable training rows for {', '.join(missing)}")
    args.objective_weights = dict(sorted(weights.items()))
    counts = Counter((row["split"], row["mode"]) for row in records)
    print(json.dumps({f"{split}/{mode}": count for (split, mode), count in sorted(counts.items())}), flush=True)
    context_counts = Counter(row["context_kind"] for row in local_rows)
    print(f"local_context_rows={dict(context_counts)} person_swap_rows={dict(swap_counts)}", file=sys.stderr, flush=True)
    print(f"objective_weights={weights} objective_eligible_rows={dict(eligible_counts)} "
          f"appearance_dino_model_sha={args.appearance_dino_model_sha}", file=sys.stderr, flush=True)
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
    from src.pose_control.contact_hand_adapter import training
    from src.pose_control.contact_hand_adapter.objectives import (
        check_output_gradients, image_output_supervision, load_perceptual_features, output_supervision,
    )
    from src.pose_control.contact_hand_adapter.training import (
        REGION_WEIGHTS, decode_clean_latent, encode_pair, load_training_pair, predict_flow, region_flow_objective,
        token_regions, training_output_data,
    )
    from src.pose_edit.common import sha256_file

    pools = {mode: [row for row in records if row["mode"] == mode] for mode in ("global", "local")}
    if not records or (args.local_fraction is not None and
                       ((args.local_fraction > 0 and not pools["local"]) or
                        (args.local_fraction < 1 and not pools["global"]))):
        raise ValueError("Training manifest lacks the requested global/local sample pools")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    base_sha = base_fingerprint(args.model)
    prior = None
    if args.hand_prior is not None:
        from src.pose_control.contact_hand_adapter.prior import load_hand_prior

        prior = load_hand_prior(args.hand_prior, sha256=args.hand_prior_sha).eval().requires_grad_(False)
    package = Path(training.__file__).parent
    runtime_sources = ("appearance", "augmentation", "conditions", "geometry", "integration", "model",
                       "objectives", "prior", "routing", "runtime", "training")
    source_hashes = {name: sha256_file(package / f"{name}.py") for name in runtime_sources}
    signature = dict(manifest_sha=sha256_file(args.manifest), seed=args.seed, accumulate=args.accumulate,
                     local_size=args.local_size, local_fraction=args.local_fraction, lr=args.lr,
                     local_context_weights=args.local_context_weights, person_swap_probability=args.person_swap_probability,
                     objective_weights=args.objective_weights, rgb_evaluator_sha=args.rgb_evaluator_sha,
                     rgb_feature_sha=args.rgb_feature_sha, rgb_sigma_max=args.rgb_sigma_max, dtype=str(dtype),
                     appearance_dino_model_sha=args.appearance_dino_model_sha,
                     hand_prior=None if prior is None else prior.provenance(), hand_prior_strength=args.hand_prior_strength,
                     adapter_code_sha=source_hashes, train_entry_sha=sha256_file(__file__), torch_version=str(torch.__version__),
                     qwen_transformer_sha=sha256_file(transformer_qwenimage21.__file__),
                     qwen_pipeline_sha=sha256_file(pipeline_qwenimage21.__file__))
    print(f"Qwen {args.model.resolve()} SHA256={base_sha}", flush=True)
    torch.manual_seed(args.seed)
    config = AdapterConfig(architecture_version=3, surface_tokens=args.surface_tokens, image_feedback=args.image_feedback,
                           contact_level=args.contact_level, hand_lora_rank=args.hand_lora_rank,
                           residual_mode=args.residual_mode, spatial_mode=args.spatial_mode, gate_mode=args.gate_mode,
                           appearance_mode=args.appearance_mode, appearance_dino=args.appearance_dino,
                           layers=tuple(range(32 // args.injection_layers - 1, 32, 32 // args.injection_layers)))
    signature["adapter_config"] = asdict(config)
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
    prior_options = {} if prior is None else {"hand_prior": prior.to(device)}
    install_hand_adapter(pipe.transformer, adapter.to(device).train(), **prior_options)
    pipe.transformer.enable_gradient_checkpointing()
    adapter_parameters = tuple(parameter for parameter in adapter.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(adapter_parameters, lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sample_rng"])
        noise_rng.set_state(saved["noise_rng"])
    evaluator = (torch.jit.load(str(args.rgb_evaluator), map_location=device).eval().requires_grad_(False)
                 if args.rgb_evaluator else None)
    features = load_perceptual_features(args.rgb_feature_model, device=device) if args.rgb_feature_model else None
    objective_names = tuple(name for name in args.objective_weights if name != "flow")
    gradient_checks = dict(saved.get("gradient_checks", {})) if saved is not None else {}
    observed_coverage = Counter(saved.get("objective_coverage", {})) if saved is not None else Counter()

    def save(path, update):
        state = dict(signature=signature, optimizer=optimizer.state_dict(), sample_rng=rng.get_state(),
                     noise_rng=noise_rng.get_state(), parent_sha=parent_sha, gradient_checks=gradient_checks,
                     objective_coverage=dict(observed_coverage))
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
            auxiliary = (training_output_data(pair, objective_names)
                         if objective_names and sigma.item() <= args.rgb_sigma_max else ((), {}, None))
            masks = token_regions(pair.condition.canvas, pair.regions, "cpu")
            window.append((pair, sigma, auxiliary, masks))
        counts = {name: sum(bool(masks[name].any()) for _, _, _, masks in window) for name in REGION_WEIGHTS}
        rgb_counts = Counter(name for _, _, (active, _, _), _ in window for name in active)
        observed_coverage.update(rgb_counts)
        context_counts = Counter(pair.context_kind for pair, _, _, _ in window if pair.context_kind is not None)
        swap_counts = Counter(pair.person_swap_status for pair, _, _, _ in window)
        total = 0.
        head_terms = Counter()
        for pair, sigma, (active, labels, image_data), _ in window:
            batch = encode_pair(pipe, condition=pair.condition, references=pair.references,
                                target=pair.target, prompt=pair.prompt, source_hashes=pair.source_hashes)
            noise = torch.randn(batch.target.shape, device=device, generator=noise_rng)
            prediction, velocity, clean = predict_flow(pipe.transformer, batch, sigma.to(device), noise,
                                                       prior_strength=args.hand_prior_strength if prior is not None else None)
            loss, _ = region_flow_objective(prediction, velocity, pair.condition.canvas, pair.regions,
                                           denominators=counts)
            loss = loss * args.objective_weights["flow"]
            if active:
                decoded = decode_clean_latent(pipe, clean, pair.condition.canvas)
                scores = output_supervision(evaluator(decoded), labels) if labels else {}
                scores.update(image_output_supervision(decoded, pair.target, image_data, active, features))
                new_heads = {name: value for name, value in scores.items() if name not in gradient_checks}
                if new_heads:
                    measured = check_output_gradients(clean, new_heads, adapter_parameters=adapter_parameters)
                    gradient_checks.update(measured)
                    print(f"objective_gradient={json.dumps(measured, sort_keys=True)}", flush=True)
                for name, value in scores.items():
                    contribution = value / rgb_counts[name]
                    head_terms[name] += contribution.detach().item()
                    loss = loss + args.objective_weights[name] * contribution
            if not torch.isfinite(loss):
                raise RuntimeError("The hand training loss is non-finite")
            loss.backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(adapter_parameters, 1., error_if_nonfinite=True)
        if not norm.item():
            raise RuntimeError("No Adapter gradient on this training update")
        optimizer.step()
        print(f"update={update} loss={total:.6f} grad={norm.item():.6f} rgb_eligible={dict(rgb_counts)} "
              f"rgb_losses={dict(head_terms)} contexts={dict(context_counts)} person_swap={dict(swap_counts)}", flush=True)
        if update < args.updates and update % args.save_every == 0 and all(name in gradient_checks for name in objective_names):
            save(args.out.with_name(f"{args.out.stem}.step-{update:06d}{args.out.suffix}"), update)
    missed = [name for name in objective_names if name not in gradient_checks]
    if missed:
        raise RuntimeError(f"Training completed without a qualified gradient measurement for {', '.join(missed)}")
    save(args.out, args.updates)
    print(f"trainable={sum(p.numel() for p in adapter.parameters())}")


if __name__ == "__main__":
    main()
