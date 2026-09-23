#!/usr/bin/env python3
"""Fixed-pair iPER V6.6 Hand overfit; legacy/off/v66 share stochastic state."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from src.pose_control.v6.checkpoint import (
    build_v66_checkpoint, freeze_for_hand_training, load_v65_checkpoint,
)
from src.pose_control.v6.conditions import AdapterIdentityCondition
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.detail.hand import legacy_to_face_and_hand
from src.pose_control.v6.face.iper_dataset import IPERV65FaceOverfitLoader
from src.pose_control.v6.hand import HandFineCondition, HandReferenceFeatures
from train_iper_v65_face import (
    FP32MasterAdamW, _dtype_from_name, _frame_roles, make_fixed_training_state,
    rasterize_box_mask, resolve_frame_selector,
)
from train_iper_v6_native import get_cached_conditions


def build_hand_inputs(cache: Mapping, source_index: int, target_index: int, legacy_hand) -> tuple[HandFineCondition, HandReferenceFeatures]:
    fine = HandFineCondition.from_legacy(legacy_hand)
    fine.hand_keypoints[0, 0] = cache["hand_keypoints"][target_index]
    fine.mesh_bps[0, 0] = cache["mesh_bps"][target_index]
    fine.mesh_valid[0, 0] = cache["mesh_valid"][target_index]
    patches = torch.zeros(1, 2, 2, 1, 256, 1536, dtype=torch.float32)
    valid = torch.zeros(1, 2, 2, 1, dtype=torch.bool)
    patches[0, 0, :, 0] = cache["dino_patches"][source_index].float()
    valid[0, 0, :, 0] = cache["reference_valid"][source_index]
    return fine.validate(), HandReferenceFeatures(patches, valid).validate()


def frozen_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("hand_preparer.", "control_interface.hand_adapter.", "control_interface.strength_controller.hand_")):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Overfit one iPER pair with independent V6.6 Hand Adapter")
    parser.add_argument("--appearance", required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--v65-checkpoint", required=True)
    parser.add_argument("--sampled-root", default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt")
    parser.add_argument("--assets-root", default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast")
    parser.add_argument("--model-path", default="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
    parser.add_argument("--output-dir", default="experiments/iper_v66_hand_overfit")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-timestep", type=int, default=150)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prompt", default="Change the person pose to match the target pose. Preserve identity, clothing, and background.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_steps <= 0 or args.warmup_steps < 0 or args.checkpoint_every <= 0:
        raise ValueError("step counts must be positive and warmup nonnegative")
    from diffusers import DiffusionPipeline

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets_root) / args.appearance
    cache_path = assets / "v66_hand_features.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    source_index = resolve_frame_selector(cache, args.source_index)
    target_index = resolve_frame_selector(cache, args.target_index)
    source_stem = cache["frame_stems"][source_index]
    target_stem = cache["frame_stems"][target_index]
    roles = _frame_roles(Path(args.sampled_root) / args.appearance)
    pair = IPERV65FaceOverfitLoader(
        sampled_root=args.sampled_root, assets_root=args.assets_root,
        appearance=args.appearance, source_stem=source_stem, source_role=roles[source_stem],
        target_stem=target_stem, target_role=roles[target_stem], resolution=args.resolution,
    ).load()
    _, legacy_hand = legacy_to_face_and_hand(pair.hand_detail_condition)
    hand_condition, hand_features = build_hand_inputs(cache, source_index, target_index, legacy_hand)
    hand_condition = hand_condition.to(device=device, dtype=dtype)
    hand_features = hand_features.to(device=device, dtype=dtype)

    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    if hasattr(pipe, "_load_extras"):
        pipe._load_extras(attn_implementation="sdpa")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe, enable_hand_adapter=True).to(device=device, dtype=dtype)
    old = torch.load(args.v65_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(old, Mapping):
        raise ValueError("V6.5 checkpoint must be a mapping")
    old = dict(old)
    old.pop("optimizer_state_dict", None)  # incompatible by design
    load_v65_checkpoint(adapter, old)
    frozen_before = frozen_digest(adapter)

    batch = pair.batch
    pixels = batch["tgt_image"].to(device=device, dtype=dtype)
    target = pipe.pixels_to_latents(pixels).detach()
    h, w = target.shape[-2:]
    cache_dir = output / "condition_cache"
    cache_dir.mkdir(exist_ok=True)
    reference, sequence, pooled = get_cached_conditions(pipe, batch, args.prompt, str(cache_dir), device, dtype)
    bundle = adapter.condition_injector(
        normal_a=batch["normal"].to(device, dtype),
        pose_heatmap_a=batch["pose_heatmap"].to(device, dtype),
        part_onehot_a=batch["part_onehot"].to(device, dtype),
        smplx_global_a=batch["smplx_global"].to(device, dtype),
        human_mask_a=batch["human_mask"].to(device, dtype),
        task_id=batch["task_id"].to(device=device, dtype=torch.long),
    )
    source_person = torch.zeros(1, 2, reference.shape[1], h, w, device=device, dtype=dtype)
    source_person[:, 0] = reference[:1]
    identity = AdapterIdentityCondition(source_person_latents=source_person, source_indices=torch.tensor([[0, 1]], device=device))
    fixed = make_fixed_training_state(target.shape, seed=args.seed, max_timestep=650, fixed_timestep=args.fixed_timestep, device=device, dtype=dtype)
    noise, timestep = fixed["noise"], fixed["timestep"]
    sigma = (timestep.float() / 1000).to(dtype).reshape(1, 1, 1, 1)
    noisy = (1 - sigma) * target + sigma * noise
    velocity = noise - target
    progress = (1 - timestep.float() / 1000).to(dtype)
    masks = [rasterize_box_mask(hand_condition.target_boxes[:, 0, side].float(), h, w) for side in range(2)]
    hand_mask = torch.stack(masks).amax(0).to(device=device, dtype=dtype)
    weight = 1 + 9 * hand_mask

    def predict(mode):
        prepared = adapter.prepare_conditioning(
            bundle, identity, reference[:1], (h, w),
            hand_condition=hand_condition, hand_features=hand_features,
        )
        control = adapter(
            target_latents=noisy, prepared=prepared, cond_hidden_states=[[reference[0]]],
            encoder_hidden_states=sequence, pooled_projections=pooled,
            timestep=timestep, denoise_progress=progress, geometry_strength=1,
            interaction_strength=0, detail_strength=1, face_strength=0,
            hand_strength=1, hand_mode=mode,
        )
        return pipe.transformer(
            hidden_states=noisy, encoder_hidden_states=sequence, pooled_projections=pooled,
            cond_hidden_states=[[reference[0]]], timestep=timestep,
            block_controlnet_hidden_states=[x.to(dtype=dtype) for x in control.block_controlnet_hidden_states],
            return_dict=False,
        )[0]

    adapter.eval()
    with torch.no_grad():
        off = predict("off").detach()
        legacy = predict("legacy").detach()
    records = []
    phase = None
    optimizer = None
    for step in range(args.max_steps):
        next_phase = "warmup" if step < args.warmup_steps else "full"
        if next_phase != phase:
            selected = freeze_for_hand_training(adapter, warmup=next_phase == "warmup")
            optimizer = FP32MasterAdamW([{"params": selected, "lr": 1e-4 if next_phase == "warmup" else 5e-5}], weight_decay=.01)
            phase = next_phase
        adapter.train()
        optimizer.zero_grad()
        predicted = predict("v66")
        flow = ((predicted.float() - velocity.float()).square() * weight.float()).mean()
        outside = ((predicted.float() - off.float()).square() * (1 - hand_mask.float())).mean()
        loss = flow + .02 * outside
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected, 1.0)
        optimizer.step()
        record = {"step": step + 1, "flow": float(flow.detach()), "outside": float(outside.detach()), "phase": phase}
        records.append(record)
        if step == 0 or (step + 1) % 10 == 0:
            print(record, flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.max_steps:
            adapter.eval()
            with torch.no_grad():
                on = predict("v66")
                comparison = {mode: float((value.float() - velocity.float()).square().mean()) for mode, value in (("off", off), ("legacy", legacy), ("v66", on))}
            torch.save(build_v66_checkpoint(
                adapter, hand_cache_fingerprint=cache["metadata"]["fingerprint"],
                step=step + 1, source_stem=source_stem, target_stem=target_stem,
                optimizer_state_dict=optimizer.state_dict(),
            ), output / "checkpoints" / f"checkpoint_step_{step + 1:06d}.pt")
            (output / "comparison.json").write_text(json.dumps({"fixed_seed": args.seed, "fixed_timestep": int(timestep[0]), "flow_mse": comparison, "history": records}, indent=2), encoding="utf-8")
    if frozen_digest(adapter) != frozen_before:
        raise RuntimeError("frozen DeepGen/Global/Interaction/Face/legacy Hand parameters changed")


if __name__ == "__main__":
    main()
