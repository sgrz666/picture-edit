#!/usr/bin/env python3
"""Train the actual Native Geometry/Body, with explicit opt-in CHAMP-style Depth.

Uses the repository's aspect-preserving TikTok canvas loader. --preflight-only
validates real depth records and pair canvases on CPU before loading any model.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRecords

ROOT = Path(__file__).resolve().parents[2]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    p.add_argument("--depth-mode", choices=("off", "relative_png", "camera_z", "auto"), default="off")
    p.add_argument('--depth-version', choices=('v1', 'v2'), default='v2')
    p.add_argument('--upgrade-depth-v2', action='store_true')
    p.add_argument('--body-jump-threshold', type=float, default=.02)
    p.add_argument('--relative-jump-threshold', type=float, default=.05)
    p.add_argument('--train-native-all', action='store_true')
    p.add_argument('--freeze-native-body', action='store_true')
    p.add_argument('--freeze-champ-cnn', action='store_true')
    p.add_argument('--freeze-champ-attention', action='store_true')
    p.add_argument('--freeze-depth-detail', action='store_true')
    p.add_argument('--freeze-depth-fusion', action='store_true')
    p.add_argument("--depth-manifest", type=Path)
    p.add_argument("--depth-strength", type=float, default=1.)
    p.add_argument("--attention-scale", type=int, choices=(8, 16), default=16)
    p.add_argument("--champ-checkpoint", type=Path)
    p.add_argument("--warm-start-native", type=Path)
    p.add_argument("--warm-start-components", choices=("geometry-body", "all"), default="geometry-body")
    p.add_argument("--train-geometry", action="store_true")
    p.add_argument('--geometry-version', choices=('body21', 'legacy_v1'), default='body21')
    p.add_argument('--pose-manifest', type=Path)
    p.add_argument('--pose-rotation', choices=('rotation6d_columns', 'axis_angle'), default='rotation6d_columns')
    p.add_argument('--no-pose-spatial-bias', action='store_true')
    p.add_argument('--freeze-pose', action='store_true')
    p.add_argument('--upgrade-v1-native', action='store_true', help='Explicit v1 -> Body21 weight migration only')
    p.add_argument("--canvas-policy", choices=("native", "aspect_preserving"), default="aspect_preserving")
    p.add_argument("--max-pixels", type=int, default=512*512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="Change the person pose to match the target pose. Preserve identity, clothing, and background.")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--preflight-samples", type=int, default=8)
    return p.parse_args(argv)


def validate_arguments(args):
    import math
    from src.pose_control.qwen21_full.depth_geometry import DepthGeometryConfig
    DepthGeometryConfig(args.body_jump_threshold, args.relative_jump_threshold)
    if args.depth_mode != 'off' and args.depth_version == 'v2' and args.geometry_version != 'body21':
        raise ValueError('Depth V2 requires Body21 paired geometry')
    if args.depth_version == 'v1' and args.depth_mode == 'auto':
        raise ValueError('auto per-record modality is a V2 contract')
    if args.upgrade_depth_v2 and (args.depth_mode == 'off' or args.depth_version != 'v2' or args.warm_start_native is None or args.warm_start_components != 'all'):
        raise ValueError('Depth upgrade needs V2 and explicit all-component Native warm-start')
    if args.champ_checkpoint is not None and args.warm_start_native is not None and args.warm_start_components == 'all':
        raise ValueError('Choose official CHAMP initialization or all-component Native Encoder inheritance, not both')
    if args.train_native_all and any((args.freeze_pose, args.freeze_native_body, args.freeze_champ_cnn,
                                      args.freeze_champ_attention, args.freeze_depth_detail, args.freeze_depth_fusion)):
        raise ValueError('All-Native and individual freeze overrides are mutually exclusive')
    if args.geometry_version == 'body21' and args.pose_manifest is None:
        raise ValueError('Native Body21 default requires --pose-manifest; legacy_v1 is explicit')
    if args.geometry_version == 'legacy_v1' and (args.pose_manifest is not None or args.upgrade_v1_native):
        raise ValueError('legacy_v1 cannot consume v2 Pose caches/migration')
    if args.depth_mode == "off":
        if args.depth_manifest is not None or args.champ_checkpoint is not None:
            raise ValueError("Depth off cannot consume a manifest or CHAMP checkpoint")
    elif args.depth_manifest is None:
        raise ValueError("Enabled Depth requires --depth-manifest")
    if (not args.preflight_only and args.depth_mode != 'off' and args.depth_version == 'v2' and args.champ_checkpoint is None
            and not (args.warm_start_native is not None and args.warm_start_components == 'all')):
        raise ValueError('Depth V2 requires explicit CHAMP weights or a Native Encoder warm-start')
    if args.warm_start_components == "all" and args.warm_start_native is None:
        raise ValueError("--warm-start-components all requires --warm-start-native")
    if args.steps <= 0 or args.preflight_samples <= 0 or args.max_pixels <= 0:
        raise ValueError("steps, preflight-samples and max-pixels must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.depth_strength) or args.depth_strength <= 0:
        raise ValueError("Learning rate and depth strength must be finite and positive")


def preflight_samples(dataset, records, *, count):
    facts = []
    for index in range(min(count, len(dataset))):
        sample = dataset[index]
        if sample["pair_metadata"].get("cross_person") or sample.get("geometry_retargeted", False):
            raise ValueError("Retargeted/cross-person B RGB is not strict paired ground truth")
        depth = None if records is None else records.condition(sample)
        facts.append({"pair_id": sample["pair_metadata"]["pair_id"],
                      "target_token_hw": list(sample["target_token_hw"]),
                      "target_canvas": sample["target_canvas"],
                      "depth": None if depth is None else depth.metadata[0],
                      'pose': sample.get('pose_provenance')})
    if not facts:
        raise ValueError("No training pairs passed preflight")
    return {"checked_pairs": len(facts), "total_pairs": len(dataset),
            "depth_kind": "off" if records is None else records.mode, "pairs": facts}


def _sha(path):
    if path is None:
        return None
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def _batch(sample):
    return {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in sample.items()}


def main(argv=None):
    args = arguments(argv)
    validate_arguments(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; this entry never overwrites an earlier run")
    from src.pose_control.qwen21_full.data_runtime import QwenFullDataset
    from src.data.geometry_pair_dataset_v2 import assert_pair_contract
    if args.geometry_version == 'legacy_v1':
        dataset = QwenFullDataset(args.pairs, resolution="native", dataset_type="tiktok_v3",
                                 align_mode=args.canvas_policy, max_pixels=args.max_pixels)
    else:
        from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
        dataset = NativePoseDataset(args.pairs, args.pose_manifest, canvas_policy=args.canvas_policy, max_pixels=args.max_pixels)
    if args.depth_mode == 'off':
        records = None
    elif args.depth_version == 'v1':
        records = NativeDepthRecords(args.depth_manifest, mode=args.depth_mode)
    else:
        from src.pose_control.qwen21_full.depth_v2_records import NativeDepthV2Records
        from src.pose_control.qwen21_full.depth_geometry import DepthGeometryConfig
        records = NativeDepthV2Records(args.depth_manifest, pose_records=dataset.records, mode=args.depth_mode,
                                      geometry_config=DepthGeometryConfig(args.body_jump_threshold, args.relative_jump_threshold))
    for i in range(min(args.preflight_samples, len(dataset))):
        assert_pair_contract(dataset[i], where=f"Native Depth preflight pair {i}")
    report = preflight_samples(dataset, records, count=args.preflight_samples)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "preflight.json", report)
    print(json.dumps({"checked_pairs": report["checked_pairs"], "depth_kind": report["depth_kind"],
                      "first_grid": report["pairs"][0]["target_token_hw"]}), flush=True)
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; CPU preflight is available")

    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth import NativeDepthConfig, Qwen21NativeDepth
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime, native_velocity_loss
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache, MODEL_REVISION, dataset_image_to_pil, transformer_inputs

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    source = torch.load(args.full_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    from src.pose_control.qwen21_full.body_pose import BodyPoseConfig
    pose_config = None if args.geometry_version == 'legacy_v1' else BodyPoseConfig(
        rotation=args.pose_rotation, spatial_bias=not args.no_pose_spatial_bias)
    geometry = Qwen21NativeGeometry.from_full_checkpoint(source, legacy=args.geometry_version == 'legacy_v1', pose_config=pose_config)
    del source
    from src.pose_control.qwen21_full.native_depth_v2 import Qwen21NativeDepthV2, NativeDepthV2Config
    from src.pose_control.qwen21_full.native_depth_runtime import NativeTrainabilityConfig
    if args.depth_mode == 'off':
        depth = None
    elif args.depth_version == 'v1':
        depth = Qwen21NativeDepth(NativeDepthConfig(attention_scale=args.attention_scale))
    else:
        depth = Qwen21NativeDepthV2(NativeDepthV2Config(attention_scale=args.attention_scale,
                    body_jump_threshold=args.body_jump_threshold, relative_jump_threshold=args.relative_jump_threshold))
    flags = NativeTrainabilityConfig(geometry=args.train_geometry, pose=not args.freeze_pose,
        body=not args.freeze_native_body, champ_cnn=not args.freeze_champ_cnn, champ_attention=not args.freeze_champ_attention,
        depth_detail=not args.freeze_depth_detail, depth_fusion=not args.freeze_depth_fusion)
    system = NativeDepthRuntime(geometry, Qwen21NativeBody(), depth,
                               trainability=NativeTrainabilityConfig.all_native() if args.train_native_all else flags)
    inheritance = {}
    if args.champ_checkpoint is not None:
        inheritance["champ"] = depth.load_champ_state_dict(torch.load(args.champ_checkpoint, map_location="cpu", weights_only=True))
    if args.warm_start_native:
        saved = torch.load(args.warm_start_native, map_location="cpu", weights_only=True)
        inheritance["native"] = system.load_native_weights(saved, include_depth=args.warm_start_components == "all",
                                                          upgrade_pose=args.upgrade_v1_native, upgrade_depth=args.upgrade_depth_v2)
        del saved
    metadata = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "base_revision": MODEL_REVISION,
                "pairs_sha256": _sha(args.pairs), "depth_manifest_sha256": _sha(args.depth_manifest),
                'pose_manifest_sha256': _sha(args.pose_manifest), 'pose_config': None if pose_config is None else pose_config.to_dict(),
                "full_checkpoint_sha256": _sha(args.full_checkpoint),
                "champ_checkpoint_sha256": _sha(args.champ_checkpoint),
                "warm_start_native_sha256": _sha(args.warm_start_native),
                "inheritance": inheritance, "loss": "canvas_coverage_weighted_velocity_mse",
                "geometry_frozen": not system.train_geometry, 'pose_trainable': system.train_pose,
                'depth_version': args.depth_version, 'trainability': asdict(system.trainability),
                "face_hand": "native_reference_only"}
    _write(args.output / "run_manifest.json", metadata)

    # Cache VLM/reference/target features before reserving GPU memory for the DiT.
    device = torch.device("cuda")
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16)
    cache = Qwen21IPERCache(args.output / "cache", resolution="native", prompt=args.prompt,
                            canvas_policy=args.canvas_policy)
    pipe.text_encoder.to(device)
    pipe.vae.to(device)
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["pair_metadata"].get("cross_person") or sample.get("geometry_retargeted", False):
            raise ValueError("Unpaired target RGB cannot become a velocity ground truth")
        if records is not None:
            records.condition(sample)
        cache.source(pipe, dataset_image_to_pil(sample["src_image"]))
        cache.target(pipe, dataset_image_to_pil(sample["tgt_image"]))
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to(device)
    system.to(device=device)
    install_native_body(pipe.transformer, system.body)
    system.train()
    pipe.transformer.enable_gradient_checkpointing()
    parameters = [p for p in system.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable Native parameters; select explicit parameter groups')
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    started = time.perf_counter()
    metrics_path = args.output / "metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as metrics:
        for step in range(args.steps):
            index = int(torch.randint(len(dataset), (1,), generator=torch.Generator().manual_seed(args.seed+step)))
            sample = dataset[index]
            batch = _batch(sample)
            condition = None if records is None else records.condition(sample)
            pair = cache.pair(pipe, batch, device)
            grid = tuple(pair["target_token_hw"])
            if grid != tuple(sample["target_token_hw"]):
                raise ValueError("Cache target grid differs from the pair canvas")
            coverage = F.avg_pool2d(batch["valid_image_mask"].to(device).float(), 16).flatten(1)
            target = pair["target_latent"]
            generator = torch.Generator(device=device).manual_seed(args.seed+step)
            sigma = torch.rand(1, device=device, generator=generator)
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
            noisy = (1-sigma[:, None, None])*target.float() + sigma[:, None, None]*noise.float()
            optimizer.zero_grad(set_to_none=True)
            features = system.encode(batch, target_token_hw=grid, depth=condition, depth_strength=args.depth_strength)
            with bind_native_body(pipe.transformer, features, valid_mask=coverage):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = pipe.transformer(**transformer_inputs(pair, noisy.to(target.dtype), sigma))[0][:, -target.shape[1]:]
                loss = native_velocity_loss(prediction, noise.float()-target.float(), coverage)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm == 0:
                raise FloatingPointError("Native training loss/gradient is nonfinite or zero")
            depth_grad = None if depth is None or depth.projection.weight.grad is None else float(depth.projection.weight.grad.norm())
            depth_conv_grad = None if depth is None or depth.encoder.conv_in.weight.grad is None else float(depth.encoder.conv_in.weight.grad.norm())
            optimizer.step()
            row = {"step": step+1, "pair_index": index, "pair_id": sample["pair_metadata"]["pair_id"],
                   "loss": float(loss.detach()), "grad_norm": float(grad_norm), "depth_head_grad": depth_grad,
                   "depth_conv_grad": depth_conv_grad,
                   "target_token_hw": grid, "elapsed_seconds": time.perf_counter()-started}
            metrics.write(json.dumps(row)+"\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
    payload = system.checkpoint_payload(metadata={**metadata, "step": args.steps})
    payload["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint_path = args.output / "checkpoint_last.pt"
    with checkpoint_path.open("xb") as stream:
        torch.save(payload, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
