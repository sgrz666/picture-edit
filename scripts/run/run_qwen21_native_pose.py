#!/usr/bin/env python3
"""Native Body21 checkpoint inference with target-only geometry and source-only RGB.

--preflight-only checks the full condition/checkpoint path without Qwen or sampling.
This entry never trains, creates an optimizer, or overwrites an output directory.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import torch
from torch.nn import functional as F


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--pose-manifest', type=Path)
    p.add_argument('--geometry-version', choices=('body21', 'legacy_v1'), default='body21')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[2] / 'models/Qwen-Image-2.1')
    p.add_argument('--depth-mode', choices=('off', 'relative_png', 'camera_z', 'auto'), default='off')
    p.add_argument('--depth-version', choices=('v1', 'v2'), help='Default: reconstruct the saved checkpoint version')
    p.add_argument('--depth-manifest', type=Path)
    p.add_argument('--pair-index', type=int, default=0)
    p.add_argument('--max-pixels', type=int, default=512*512)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--reference-resolution', type=int, default=512, help='Explicit Qwen reference-image pixel budget side')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default='Change the person pose to match the target pose. Preserve identity, clothing, and background.')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args(argv)
    if args.geometry_version == 'body21' and args.pose_manifest is None:
        raise ValueError('Body21 requires --pose-manifest')
    if args.geometry_version == 'legacy_v1' and args.pose_manifest is not None:
        raise ValueError('Legacy geometry cannot read v2 Pose cache')
    if (args.depth_mode == 'off') != (args.depth_manifest is None):
        raise ValueError('Depth manifest must be supplied exactly when Depth is enabled')
    if args.max_pixels <= 0 or args.steps <= 0 or args.pair_index < 0 or args.reference_resolution <= 0:
        raise ValueError('Positive budget/steps and nonnegative pair index required')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime, NativeDepthRecords
    from src.pose_control.qwen21_full.native_depth_v2 import Qwen21NativeDepthV2
    from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
    from src.pose_control.qwen21_full.data_runtime import QwenFullDataset
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    system = NativeDepthRuntime.from_native_checkpoint(saved, legacy=args.geometry_version == 'legacy_v1', train_pose=False).eval()
    depth_version = 'v2' if isinstance(system.depth, Qwen21NativeDepthV2) else 'v1'
    if args.depth_version is not None and args.depth_version != depth_version:
        raise ValueError('Requested Depth version differs from checkpoint; migrate explicitly first')
    dataset = (NativePoseDataset(args.pairs, args.pose_manifest, max_pixels=args.max_pixels)
               if args.geometry_version == 'body21' else QwenFullDataset(args.pairs, resolution='native', dataset_type='tiktok_v3',
                    align_mode='aspect_preserving', max_pixels=args.max_pixels))
    sample = dataset[args.pair_index]
    if args.depth_mode == 'off':
        depth = None
    elif depth_version == 'v2':
        from src.pose_control.qwen21_full.depth_v2_records import NativeDepthV2Records
        depth = NativeDepthV2Records(args.depth_manifest, pose_records=dataset.records, mode=args.depth_mode,
                                     geometry_config=system.depth.config.geometry_config()).condition(sample)
    else:
        depth = NativeDepthRecords(args.depth_manifest, mode=args.depth_mode).condition(sample)
    if depth is not None and system.depth is None:
        raise ValueError('This checkpoint has no Depth module')
    batch = {k: v[None] if torch.is_tensor(v) else [v] for k, v in sample.items()}
    grid = tuple(sample['target_token_hw'])
    depth_strength = 0. if args.depth_mode == 'off' else 1.
    with torch.no_grad():
        features = system.encode(batch, target_token_hw=grid, depth=depth, depth_strength=depth_strength)
    report = {'architecture': saved['architecture'], 'pair': sample['pair_metadata'],
              'target_canvas': sample['target_canvas'], 'geometry_shape': list(features.geometry.shape),
              'pose_provenance': sample.get('pose_provenance'), 'depth_mode': args.depth_mode, 'depth_version': depth_version,
              'no_optimizer': True, 'preflight_only': args.preflight_only,
              'reference_resolution': args.reference_resolution, 'use_kv_cache': False}
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.preflight_only:
        from diffusers import QwenImage21Pipeline
        from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
        from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
        pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16, local_files_only=True).to('cuda')
        system.to('cuda')
        with torch.no_grad():
            features = system.encode(batch, target_token_hw=grid, depth=depth, depth_strength=depth_strength)
        install_native_body(pipe.transformer, system.body)
        coverage = F.avg_pool2d(batch['valid_image_mask'].cuda().float(), 16).flatten(1)
        h, w = sample['target_canvas']['canvas_hw']
        with torch.no_grad(), bind_native_body(pipe.transformer, features, valid_mask=coverage):
            # Target RGB never enters the pipeline. Qwen receives A as its sole appearance reference.
            result = pipe(image=[dataset_image_to_pil(sample['src_image'])], prompt=args.prompt,
                          width=w, height=h, num_inference_steps=args.steps, true_cfg_scale=1.,
                          output_resolution=args.reference_resolution, use_kv_cache=False,
                          generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
        if result.size != (w, h):
            raise ValueError('Pipeline output canvas differs from geometry')
        result.save(args.output / 'image.png')
        report.update(seed=args.seed, steps=args.steps)
    with (args.output / 'request.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'architecture': report['architecture'], 'geometry_shape': report['geometry_shape'],
                      'preflight_only': args.preflight_only, 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
