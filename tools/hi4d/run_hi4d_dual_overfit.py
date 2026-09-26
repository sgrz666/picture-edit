#!/usr/bin/env python3
"""Overfit one real HI4D dual-person source/target pair with V6.3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from diffusers import DiffusionPipeline
from src.pose_control.v6.conditions import AdapterIdentityCondition, TaskType
from src.pose_control.v6.detail import FaceHandDetailCondition
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.controlled_pipeline import ControlledDeepGenPipeline
from train_iper_v6_native import get_cached_conditions


DEFAULT_MANIFEST = (
    ROOT / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_train.jsonl"
)
DEFAULT_MODEL = ROOT / "models/DeepGen-1.0-diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiments/hi4d_dual_overfit_v1")
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--save_every", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Change both people to match the target interaction pose. Preserve both identities, clothing, faces, hands, lighting, and background.")
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def processed_root(row: dict) -> Path:
    return ROOT / "datasets/Hi4D_pilot_v1/processed_512_1500" / row["pair"] / row["target_action"]


def source_root(row: dict) -> Path:
    return ROOT / "datasets/Hi4D_pilot_v1/processed_512_1500" / row["pair"] / row["source_action"]


def png_tensor(path: Path, *, mode: str = "RGB") -> torch.Tensor:
    image = Image.open(path).convert(mode)
    array = np.asarray(image).copy()
    if mode == "RGB":
        return torch.from_numpy(array).float().permute(2, 0, 1) / 255.0
    return torch.from_numpy(array).float().unsqueeze(0) / 255.0


def npy_tensor(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path).copy()).float()


def pose_heatmap(path: Path, size: int = 512, sigma: float = 4.0) -> torch.Tensor:
    points = npy_tensor(path)
    if points.shape != (24, 4):
        raise ValueError(f"expected HI4D pose [24,4], got {tuple(points.shape)}")
    xy = points[:, :2]
    confidence = points[:, 3].clamp(0, 1)
    mid_hip = (xy[1] + xy[2]) * 0.5
    mid_score = torch.minimum(confidence[1], confidence[2])
    xy = torch.cat((xy[:8], mid_hip[None], xy[8:]), dim=0)
    confidence = torch.cat((confidence[:8], mid_score[None], confidence[8:]), dim=0)
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    distance = (x[None] - xy[:, 0, None, None]) ** 2 + (y[None] - xy[:, 1, None, None]) ** 2
    return (confidence[:, None, None] * torch.exp(-distance / (2 * sigma * sigma))).clamp(0, 1)


def person_conditions(root: Path, frame_id: int, person: str, smpl: np.lib.npyio.NpzFile) -> dict[str, torch.Tensor]:
    stem = f"{frame_id:06d}"
    normal = npy_tensor(root / "normal" / f"{person}_npy" / f"{stem}.npy")
    if normal.ndim == 3 and normal.shape[-1] == 3:
        normal = normal.permute(2, 0, 1)
    mask = png_tensor(root / "person_mask" / person / f"{stem}.png", mode="L")
    depth = npy_tensor(root / "depth" / f"{person}_npy" / f"{stem}.npy").unsqueeze(0)
    depth = depth / depth.amax().clamp_min(1e-6)
    pose = pose_heatmap(root / "pose" / f"{person}_npy" / f"{stem}.npy")
    part = torch.zeros(14, 512, 512)
    part[1] = mask[0]
    index = 0 if person == "A" else 1
    betas = torch.from_numpy(smpl["betas"][index]).float()
    orient = torch.from_numpy(smpl["global_orient"][index]).float()
    angle = orient.norm().clamp_min(1e-7)
    axis = orient / angle
    x, y, z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    one = 1 - c
    rotation = torch.stack((c + x*x*one, y*x*one + z*s, z*x*one - y*s,
                            x*y*one - z*s, c + y*y*one, z*y*one + x*s))
    global_params = torch.cat((betas, rotation, torch.from_numpy(smpl["transl"][index]).float(), torch.zeros(7)))
    return {
        "normal": normal * mask,
        "pose": pose,
        "part": part,
        "mask": mask,
        "depth": depth,
        "global": global_params,
    }


def detail_box(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    valid = points[:, 2] > 0.05
    if int(valid.sum()) < 2:
        return torch.zeros(4), False
    xy = points[valid, :2]
    minimum, maximum = xy.amin(0), xy.amax(0)
    padding = (maximum - minimum).clamp_min(1e-3) * 0.2 + 0.01
    box = torch.cat(((minimum - padding).clamp(0, 1), (maximum + padding).clamp(0, 1)))
    return box, bool((box[2:] > box[:2]).all())


def make_detail_condition(detail: list[dict[str, torch.Tensor]]) -> FaceHandDetailCondition:
    face = torch.stack([item["face"] for item in detail])
    hands = torch.stack([item["hands"] for item in detail])
    boxes = []
    valid = []
    for item in detail:
        item_boxes = []
        item_valid = []
        for points in (item["face"], item["hands"][0], item["hands"][1]):
            box, is_valid = detail_box(points)
            item_boxes.append(box)
            item_valid.append(is_valid)
        boxes.append(torch.stack(item_boxes))
        valid.append(torch.tensor(item_valid, dtype=torch.bool))
    boxes = torch.stack(boxes)
    valid = torch.stack(valid)
    return FaceHandDetailCondition(
        face_keypoints=face[None],
        hand_keypoints=hands[None],
        smplx_detail=torch.zeros(1, 2, 103),
        source_boxes=boxes[None],
        target_boxes=boxes[None],
        region_valid=valid[None],
        source_indices=torch.tensor([[0, 1]], dtype=torch.long),
    )


def load_sample(row: dict) -> tuple[dict[str, torch.Tensor], Image.Image, Image.Image, dict, list[dict[str, torch.Tensor]]]:
    target_root = processed_root(row)
    source_path = source_root(row) / "rgb" / f"{row['source_frame_id']:06d}.png"
    target_path = target_root / "rgb" / f"{row['target_frame_id']:06d}.png"
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"processed RGB missing: {source_path} / {target_path}")
    smpl_path = ROOT / "datasets/Hi4D_pilot_v1/raw" / row["pair"] / row["target_action"] / "smpl" / f"{row['target_frame_id']:06d}.npz"
    smpl = np.load(smpl_path)
    conditions = {}
    for person in ("A", "B"):
        conditions[person] = person_conditions(target_root, row["target_frame_id"], person, smpl)
    detail = []
    for person in ("A", "B"):
        pose_path = Path(row[f"target_dwpose_{person}"])
        pose_data = np.load(pose_path)
        points = pose_data["xy_normalized"].astype(np.float32)
        scores = pose_data["scores"].astype(np.float32)
        if points.shape != (134, 2) or scores.shape != (134,):
            raise ValueError(f"expected HI4D DWPose 134 points, got {points.shape} / {scores.shape}")
        detail.append({
            "face": torch.from_numpy(np.concatenate((points[24:92], scores[24:92, None]), axis=1)),
            "hands": torch.from_numpy(np.stack((
                np.concatenate((points[92:113], scores[92:113, None]), axis=1),
                np.concatenate((points[113:134], scores[113:134, None],), axis=1),
            ), axis=0)),
        })
    source = Image.open(source_path).convert("RGB")
    target = Image.open(target_path).convert("RGB")
    return conditions, source, target, {"source_path": str(source_path), "target_path": str(target_path), "smpl_path": str(smpl_path), "detail_points": 134, "detail_note": "HI4D target DWPose is used for both source/target local boxes because the v4.2 source manifest does not store source DWPose."}, detail


def batch_from_sample(conditions: dict, source: Image.Image, target: Image.Image) -> dict[str, object]:
    source_tensor = torch.from_numpy(np.asarray(source).copy()).float().permute(2, 0, 1) / 127.5 - 1
    target_tensor = torch.from_numpy(np.asarray(target).copy()).float().permute(2, 0, 1) / 127.5 - 1
    return {"src_image": source_tensor[None], "tgt_image": target_tensor[None], "appearance": ["hi4d_dual"], "source_stem": ["source"], "target_stem": ["target"]}


def make_bundle(adapter, conditions: dict, device: torch.device, dtype: torch.dtype):
    values = {person: {key: value[None].to(device=device, dtype=dtype) for key, value in item.items()} for person, item in conditions.items()}
    return adapter.condition_injector(
        normal_a=values["A"]["normal"], pose_heatmap_a=values["A"]["pose"], part_onehot_a=values["A"]["part"], smplx_global_a=values["A"]["global"], human_mask_a=values["A"]["mask"], depth_a=values["A"]["depth"],
        normal_b=values["B"]["normal"], pose_heatmap_b=values["B"]["pose"], part_onehot_b=values["B"]["part"], smplx_global_b=values["B"]["global"], human_mask_b=values["B"]["mask"], depth_b=values["B"]["depth"],
        person_b_valid=torch.ones(1, dtype=torch.bool, device=device), task_id=torch.full((1,), int(TaskType.DUAL), dtype=torch.long, device=device),
    ), values


def save_grid(path: Path, source: Image.Image, target: Image.Image, baseline: Image.Image, generated: dict[int, Image.Image]) -> None:
    images = [source, target, baseline] + [generated[step] for step in sorted(generated)]
    canvas = Image.new("RGB", (512 * len(images), 544), (20, 20, 20))
    labels = ["source", "target", "language-only"] + [f"adapter step {step}" for step in sorted(generated)]
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.convert("RGB").resize((512, 512)), (index * 512, 32))
        draw.text((index * 512 + 10, 10), label, fill=(255, 255, 255))
    canvas.save(path)


def save_condition_assets(output_dir: Path, conditions: dict[str, dict[str, torch.Tensor]]) -> None:
    for person, values in conditions.items():
        normal = ((values["normal"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(normal).save(output_dir / f"02_CONDITION_NORMAL_{person}.png")
        depth = (values["depth"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(depth).save(output_dir / f"02_CONDITION_DEPTH_{person}.png")
        mask = (values["mask"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(mask).save(output_dir / f"02_CONDITION_MASK_{person}.png")
        pose = (values["pose"].amax(dim=0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pose).save(output_dir / f"02_CONDITION_POSE_{person}.png")


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError("max_steps and save_every must be positive")
    rows = load_rows(args.manifest)
    row = rows[args.row]
    conditions, source, target, paths, detail_data = load_sample(row)
    report = {"row": args.row, "pair": row["pair"], "action": row["target_action"], "target_contact": row["target_contact"], **paths}
    print(json.dumps(report, indent=2), flush=True)
    if args.validate_only:
        print("HI4D dual sample validation: OK", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.save(args.output_dir / "00_SOURCE_IMAGE.png")
    target.save(args.output_dir / "00_TARGET_IMAGE.png")
    save_condition_assets(args.output_dir, conditions)
    (args.output_dir / "03_LANGUAGE_EDIT_PROMPT.txt").write_text(args.prompt + "\n", encoding="utf-8")
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps({"manifest": str(args.manifest), "row": args.row, "prompt": args.prompt, "seed": args.seed, "max_steps": args.max_steps, "save_every": args.save_every, "inference_steps": args.inference_steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    pipe = DiffusionPipeline.from_pretrained(str(args.model_path), torch_dtype=dtype).to(device)
    pipe.vae.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe._load_extras(attn_implementation="sdpa")
    with torch.inference_mode():
        baseline = pipe(
            prompt=args.prompt,
            image=source,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        ).images[0]
    baseline.save(args.output_dir / "01_original_deepgen_language_only.png")
    UnifiedSMPLXAdapterV6.freeze_deepgen_pipeline_components(pipe)
    adapter = UnifiedSMPLXAdapterV6.from_deepgen_pipeline(pipe).to(device=device, dtype=dtype)
    transformer = pipe.transformer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    batch = batch_from_sample(conditions, source, target)
    cache_dir = str(args.cache_dir or args.output_dir / "condition_cache")
    bundle, values = make_bundle(adapter, conditions, device, dtype)
    detail_condition = make_detail_condition(detail_data)
    source_pixels = batch["src_image"].to(device=device, dtype=dtype)
    target_pixels = batch["tgt_image"].to(device=device, dtype=dtype)
    with torch.no_grad():
        reference_latent, sequence, pooled = get_cached_conditions(
            pipe, batch, args.prompt, cache_dir, device, dtype
        )
        target_latent = pipe.pixels_to_latents(target_pixels)
    identity = AdapterIdentityCondition(source_person_latents=torch.stack((reference_latent[0], reference_latent[0]))[None], source_indices=torch.tensor([[0, 1]], device=device))
    generated: dict[int, Image.Image] = {}
    diagnostics: list[dict] = []

    def inference(step: int) -> None:
        controlled = ControlledDeepGenPipeline(pipeline=pipe, adapter=adapter)
        adapter.eval()
        inference_bundle, _ = make_bundle(adapter, conditions, device, dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            output = controlled(condition_bundle=inference_bundle, identity_condition=identity, source_scene_latents=reference_latent, detail_condition=detail_condition, prompt=args.prompt, image=source, height=args.resolution, width=args.resolution, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale, seed=args.seed, geometry_strength=1.0, interaction_strength=1.0, detail_strength=1.0, face_strength=1.0, hand_strength=1.0).images[0]
        generated[step] = output
        output.save(args.output_dir / f"model_step_{step:04d}.png")
        adapter.train()

    inference(0)
    started = time.time()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(target_latent)
        timestep = torch.rand((1,), device=device) * 1000.0
        sigma = (timestep / 1000.0).reshape(1, 1, 1, 1).to(dtype)
        noisy = (1 - sigma) * target_latent + sigma * noise
        with torch.autocast("cuda", dtype=dtype):
            training_bundle, _ = make_bundle(adapter, conditions, device, dtype)
            prepared = adapter.prepare_conditioning(training_bundle, identity, reference_latent, target_latent.shape[-2:], detail_condition=detail_condition)
            control = adapter(target_latents=noisy, prepared=prepared, cond_hidden_states=[[reference_latent[0]]], encoder_hidden_states=sequence, pooled_projections=pooled, timestep=timestep, denoise_progress=float(1 - timestep.item() / 1000), geometry_strength=1.0, interaction_strength=1.0)
            prediction = transformer(hidden_states=noisy, encoder_hidden_states=sequence, pooled_projections=pooled, cond_hidden_states=[[reference_latent[0]]], timestep=timestep, block_controlnet_hidden_states=control.block_controlnet_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(prediction.float(), (noise - target_latent).float())
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(f"step={step:04d}/{args.max_steps} loss={loss.item():.6f} grad={grad:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save({"step": step, "adapter_state_dict": adapter.state_dict(), "row": row}, args.output_dir / f"adapter_step_{step:04d}.pt")
            inference(step)
            diagnostics.append({"step": step, "loss": float(loss.item()), "gradient_norm": grad})

    save_grid(args.output_dir / "comparison_grid.png", source, target, baseline, generated)
    (args.output_dir / "summary.json").write_text(json.dumps({"completed_steps": args.max_steps, "duration_seconds": time.time() - started, "diagnostics": diagnostics, "row": row}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()