#!/usr/bin/env python3
"""Explicit dual V3 training entry. Existing velocity loss/flat AdamW; no new schedule.

Implementation delivered without running this entry, any model, or an optimizer.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from dataclasses import asdict
import json
import time
import torch
from torch.nn import functional as F

from scripts.run.run_qwen21_native_dual import add_control_arguments, control_options
from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime, DualTrainability, ARCHITECTURE
from src.pose_control.qwen21_full.native_dual_depth import NativeDualConfig
from src.pose_control.qwen21_full.dual_data_runtime import NativeDualDataset, collate_native_dual, identity_from_reference
from src.pose_control.qwen21_full.native_depth_runtime import native_velocity_loss
from src.pose_control.qwen21_full.pose_cache import sha256


def _write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Already migrated V3 checkpoint; no silent warm-start')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[2]/'models/Qwen-Image-2.1')
    p.add_argument('--max-pixels', type=int, default=512*512)
    p.add_argument('--reference-side', type=int, default=512)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default='Change the poses of the two existing people. Preserve each person identity, clothing, and background.')
    p.add_argument('--train-native-all', action='store_true')
    p.add_argument('--freeze-group', action='append', default=[], choices=tuple(DualTrainability.__dataclass_fields__))
    p.add_argument('--preflight-only', action='store_true', help='Read cached data; no Qwen/model forward')
    p.add_argument('--preflight-samples', type=int, default=8)
    add_control_arguments(p)
    args = p.parse_args()
    options = control_options(args)
    import math
    if args.output.exists():
        raise FileExistsError('Use a new experiment directory')
    if args.steps <= 0 or args.preflight_samples <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Positive steps/preflight count/LR required')
    if args.train_native_all and args.freeze_group:
        raise ValueError('Choose all Native or individual freezes')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Migrate to V3 explicitly before training')
    config = NativeDualConfig(**saved['runtime_config']['config'])
    dataset = NativeDualDataset(args.manifest, config=config, max_pixels=args.max_pixels, reference_side=args.reference_side)
    facts = []
    for i in range(min(args.preflight_samples, len(dataset))):
        sample = dataset[i]
        if options.depth_mode == 'full_geometry' and sample['dual_depth'].metadata[0]['mode'] != 'full_geometry':
            raise ValueError('Full geometry cannot read PNG-only samples')
        facts.append({'pair_id': sample['pair_metadata']['pair_id'], 'grid': sample['target_token_hw'],
                      'provenance': sample['dual_depth'].metadata[0]})
    args.output.mkdir(parents=True, exist_ok=False)
    _write(args.output/'preflight.json', {'pairs': facts, 'qwen_or_condition_forward': False})
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; use --preflight-only for data inspection')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    flags = asdict(DualTrainability.all_native() if args.train_native_all else DualTrainability())
    for group in args.freeze_group:
        flags[group] = False
    # Disabled ablation branches are also excluded from optimizer parameters.
    if options.depth_mode == 'off' or options.depth_strength == 0:
        for key in ('champ_cnn', 'champ_attention', 'detail_cnn', 'depth_fusion'):
            flags[key] = False
    elif not options.detail:
        flags['detail_cnn'] = False
    if not options.interaction or options.interaction_strength == 0:
        flags['interaction_fusion'] = flags['relation_raster'] = flags['relation_tokens'] = False
    else:
        flags['relation_raster'] &= options.raster and options.relation_strength > 0
        flags['relation_tokens'] &= options.part_contact and options.relation_strength > 0
    flags['identity_binding'] &= options.identity and options.identity_strength > 0
    system = NativeDualRuntime.from_checkpoint(saved, trainability=DualTrainability(**flags))
    del saved
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache, dataset_image_to_pil, transformer_inputs, MODEL_REVISION
    metadata = {'args': vars(args), 'options': asdict(options), 'trainability': flags,
        'manifest_sha256': sha256(args.manifest), 'checkpoint_sha256': sha256(args.checkpoint),
        'base_revision': MODEL_REVISION, 'optimizer_resumed': False, 'loss': 'canvas_coverage_weighted_velocity_mse'}
    _write(args.output/'run_manifest.json', metadata)
    device = torch.device('cuda')
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16, local_files_only=True)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False).to(device)
    pipe.vae.requires_grad_(False).to(device)
    cache = Qwen21IPERCache(args.output/'cache', resolution='native', prompt=args.prompt, canvas_policy='aspect_preserving')
    for i in range(len(dataset)):
        sample = dataset[i]
        cache.source(pipe, dataset_image_to_pil(sample['src_image']))
        cache.target(pipe, dataset_image_to_pil(sample['tgt_image']))
    pipe.text_encoder.to('cpu')
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    pipe.transformer.to(device)
    system.to(device).train()
    install_native_body(pipe.transformer, system.body)
    pipe.transformer.enable_gradient_checkpointing()
    parameters = [p for p in system.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No active trainable Native groups')
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    started = time.perf_counter()
    with (args.output/'metrics.jsonl').open('x', encoding='utf8') as metrics:
        for step in range(args.steps):
            index = int(torch.randint(len(dataset), (1,), generator=torch.Generator().manual_seed(args.seed+step)))
            batch = collate_native_dual([dataset[index]])
            pair = cache.pair(pipe, batch, device)
            grid = tuple(pair['target_token_hw'])
            if grid != batch['target_token_hw']:
                raise ValueError('Actual Qwen target grid differs from the controls')
            coverage = F.avg_pool2d(batch['valid_image_mask'].to(device).float(), 16).flatten(1)
            identity = identity_from_reference(batch, pair['source_latent'], pair['source_token_hw']) if options.identity else None
            target = pair['target_latent']
            generator = torch.Generator(device=device).manual_seed(args.seed+step)
            sigma = torch.rand(1, device=device, generator=generator)
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
            noisy = (1-sigma[:, None, None])*target.float()+sigma[:, None, None]*noise.float()
            optimizer.zero_grad(set_to_none=True)
            features = system.encode(batch, target_token_hw=grid, identity=identity, options=options)
            with bind_native_body(pipe.transformer, features, valid_mask=coverage, conv_input_gate=True):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = pipe.transformer(**transformer_inputs(pair, noisy.to(target.dtype), sigma))[0][:, -target.shape[1]:]
                loss = native_velocity_loss(prediction, noise.float()-target.float(), coverage)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm == 0:
                raise FloatingPointError('Nonfinite/zero overall Native gradient or loss')
            gradients = {name: [float(p.grad.norm()) for _, p in group if p.grad is not None]
                         for name, group in system.named_parameter_groups().items()}
            optimizer.step()
            row = {'step': step+1, 'pair_index': index, 'loss': float(loss.detach()),
                'grad_norm': float(grad_norm), 'group_gradients': gradients, 'elapsed_seconds': time.perf_counter()-started}
            metrics.write(json.dumps(row)+'\n'); metrics.flush()
            print(json.dumps(row), flush=True)
    # Keep only JSON-compatible values in weights_only-loadable checkpoint metadata.
    metadata = json.loads(json.dumps(metadata, default=str))
    payload = system.checkpoint_payload(metadata={**metadata, 'step': args.steps})
    payload['optimizer_state_dict'] = optimizer.state_dict()
    with (args.output/'checkpoint_last.pt').open('xb') as stream:
        torch.save(payload, stream)


if __name__ == '__main__':
    main()
