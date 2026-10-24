#!/usr/bin/env python3
"""Native dual V3 inference; actual-prefix identity binding, no target RGB input."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from dataclasses import asdict
import json
import torch


def add_control_arguments(p):
    p.add_argument('--depth-mode', choices=('off', 'scene_only', 'full_geometry'), default='full_geometry')
    p.add_argument('--no-detail', action='store_true')
    p.add_argument('--no-interaction', action='store_true')
    p.add_argument('--no-raster', action='store_true')
    p.add_argument('--no-part-contact', action='store_true')
    p.add_argument('--no-identity', action='store_true')
    for name in ('depth', 'relation', 'interaction', 'identity'):
        p.add_argument('--'+name+'-strength', type=float, default=1.)


def control_options(args):
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    return DualControlOptions(depth_mode=args.depth_mode, detail=not args.no_detail, interaction=not args.no_interaction,
        raster=not args.no_raster, part_contact=not args.no_part_contact, identity=not args.no_identity,
        depth_strength=args.depth_strength, relation_strength=args.relation_strength,
        interaction_strength=args.interaction_strength, identity_strength=args.identity_strength)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--pair-index', type=int, default=0)
    p.add_argument('--max-pixels', type=int, default=512*512)
    p.add_argument('--reference-side', type=int, default=512)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default='Change the poses of the two existing people. Preserve each person identity, clothing, and background.')
    add_control_arguments(p)
    args = p.parse_args()
    options = control_options(args)
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    if args.steps <= 0 or args.pair_index < 0:
        raise ValueError('Positive steps/nonnegative pair index required')
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.dual_data_runtime import NativeDualDataset, collate_native_dual
    from src.pose_control.qwen21_full.native_dual_pipeline import bind_native_dual_pipeline
    from src.pose_control.qwen21_full.native_integration import install_native_body
    from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    runtime = NativeDualRuntime.from_checkpoint(saved).eval().to('cuda')
    dataset = NativeDualDataset(args.manifest, config=runtime.config, max_pixels=args.max_pixels,
                                reference_side=args.reference_side, load_target=False)
    sample = dataset[args.pair_index]
    batch = collate_native_dual([sample])
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.transformer.requires_grad_(False)
    install_native_body(pipe.transformer, runtime.body)
    h, w = sample['target_canvas']['canvas_hw']
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad(), bind_native_dual_pipeline(pipe.transformer, runtime, batch, options=options) as report:
        image = pipe(image=[dataset_image_to_pil(sample['src_image'])], prompt=args.prompt,
            width=w, height=h, output_resolution=args.reference_side, num_inference_steps=args.steps,
            true_cfg_scale=1., use_kv_cache=False,
            generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
    if image.size != (w, h):
        raise ValueError('Generated canvas differs from the controls')
    image.save(args.output/'image.png')
    with (args.output/'request.json').open('x', encoding='utf8') as stream:
        json.dump({'architecture': saved['architecture'], 'pair': sample['pair_metadata'], 'options': asdict(options),
                   'target_canvas': sample['target_canvas'], 'source_canvas': sample['source_canvas'],
                   'identity_provenance': sample['identity_metadata'],
                   'binding': report, 'seed': args.seed, 'steps': args.steps, 'no_optimizer': True}, stream, indent=2)


if __name__ == '__main__':
    main()
