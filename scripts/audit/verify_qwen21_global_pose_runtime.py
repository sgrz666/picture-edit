#!/usr/bin/env python3
"""Bounded 7B Body21 + CHAMP Depth forward/backward, no optimizer or weight save.

Only fresh adapter zero heads are perturbed in memory to test gradient reachability.
Real frame geometry, synthetic text/latents; this is not a generation-quality test.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import inspect
import json
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--full-checkpoint', type=Path, required=True)
    p.add_argument('--champ-checkpoint', type=Path, required=True)
    p.add_argument('--pose-manifest', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--max-pixels', type=int, default=512*512)
    p.add_argument('--min-free-gib', type=float, default=40.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from torch.nn import functional as F
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth import Qwen21NativeDepth
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    from src.pose_control.qwen21_full.depth_condition import load_depth_record
    from src.pose_control.qwen21_full.pose_cache import sha256

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    free, total = torch.cuda.mem_get_info()
    if args.min_free_gib <= 0 or free / 1024**3 < args.min_free_gib:
        raise RuntimeError(f'Need {args.min_free_gib} GiB, have {free / 1024**3:.2f}')
    started = time.perf_counter()
    torch.manual_seed(29)
    torch.set_num_threads(4)
    dataset = NativePoseDataset(args.pairs, args.pose_manifest, max_pixels=args.max_pixels)
    sample = dataset[0]
    key = sample['pose_provenance']['record_id']
    raw = dataset.records.raw(key)
    sources = raw['metadata']['sources']
    depth = load_depth_record({'kind': 'relative_png', 'source_id': key, 'polarity': 'near_one',
             'depth_path': sources['depth_vis']['path'], 'valid_path': sources['smpl_mask']['path'],
             'camera_id': key, 'units': 'unknown', 'scale_status': 'unknown'}, sample['target_canvas']).to('cuda')
    grid = tuple(sample['target_token_hw'])
    n = grid[0] * grid[1]
    if grid[0] == grid[1]:
        raise ValueError('This audit requires a rectangular target canvas')
    batch = {k: v[None] if torch.is_tensor(v) else [v] for k, v in sample.items()}
    coverage = F.avg_pool2d(batch['valid_image_mask'].cuda().float(), 16).flatten(1)
    before_stat = args.full_checkpoint.stat()
    payload = torch.load(args.full_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    geometry = Qwen21NativeGeometry.from_full_checkpoint(payload)
    del payload
    depth_module = Qwen21NativeDepth()
    champ = depth_module.load_champ_state_dict(torch.load(args.champ_checkpoint, map_location='cpu', weights_only=True))
    system = NativeDepthRuntime(geometry, Qwen21NativeBody(), depth_module).cuda().eval()
    encoder_calls = []
    hook = geometry.condition_injector.body_pose_encoder.register_forward_hook(lambda *a: encoder_calls.append(1))
    with torch.no_grad():
        initial_features = system.encode(batch, target_token_hw=grid, depth=depth)
    torch.cuda.reset_peak_memory_stats()
    model = QwenImage21Transformer2DModel.from_pretrained(str(args.base_model), subfolder='transformer',
               torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval().cuda()
    parameters = dict(model.named_parameters())
    versions = {k: p._version for k, p in parameters.items()}
    ref_hw = (16, 16)
    ref_n = ref_hw[0] * ref_hw[1]
    text_length = ref_n // 4 + 4
    img_mask = torch.zeros(1, text_length + n // 4, device='cuda', dtype=torch.bool)
    img_mask[:, 2:2 + ref_n // 4] = True
    img_mask[:, text_length:] = True
    values = {'hidden_states': torch.randn(1, ref_n + n, 64, device='cuda', dtype=torch.bfloat16),
              'encoder_hidden_states': torch.randn(1, text_length, 4096, device='cuda', dtype=torch.bfloat16),
              'encoder_hidden_states_mask': torch.ones(1, text_length, device='cuda', dtype=torch.bool),
              'timestep': torch.tensor([.5], device='cuda'), 'img_shapes': [[(1, *ref_hw), (1, *grid)]],
              'img_mask': img_mask, 'return_dict': False}
    with torch.no_grad():
        baseline = model(**values)[0]
    install_native_body(model, system.body)
    with torch.no_grad(), bind_native_body(model, initial_features, valid_mask=coverage):
        zero = model(**values)[0]
    zero_diff = float((zero.float() - baseline.float()).abs().max())
    for head in system.body.output_heads:
        torch.nn.init.normal_(head.weight, std=.002)
    with torch.no_grad(), bind_native_body(model, initial_features, valid_mask=coverage):
        control_baseline = model(**values)[0]
    joint = geometry.reasoner.person_reasoner.joint_attention
    # Exactly one zero projection per new residual path; no multiplicative zero gate.
    torch.nn.init.normal_(joint.projection.weight, std=.01)
    torch.nn.init.normal_(depth_module.projection.weight, std=.01)
    features = system.encode(batch, target_token_hw=grid, depth=depth)
    calls_before_qwen = len(encoder_calls)
    model.enable_gradient_checkpointing()
    with bind_native_body(model, features, valid_mask=coverage):
        result = model(**values)[0]
    prefix_error = float((result[:, :-n].detach().float() - control_baseline[:, :-n].float()).abs().max())
    target_error = float((result[:, -n:].detach().float() - control_baseline[:, -n:].float()).abs().max())
    result[:, -n:].float().square().mean().backward()
    hook.remove()
    encoder = geometry.condition_injector.body_pose_encoder
    def grad(parameter):
        return None if parameter.grad is None else float(parameter.grad.norm())
    grads = {'pose_mlp': grad(encoder.projection[0].weight), 'joint_embedding': grad(encoder.joint_embedding.weight),
             'skeleton_layer0': grad(encoder.skeleton[0].attention.in_proj_weight),
             'skeleton_layer1': grad(encoder.skeleton[1].attention.in_proj_weight),
             'joint_cross_attention': grad(joint.attention.in_proj_weight),
             'joint_projection': grad(joint.projection.weight), 'depth_conv': grad(depth_module.encoder.conv_in.weight),
             'depth_attention': grad(depth_module.encoder.attention.proj_in.weight),
             'depth_projection': grad(depth_module.projection.weight)}
    heads = [grad(h.weight) for h in system.body.output_heads]
    frozen = all(not p.requires_grad and p.grad is None and p._version == versions[k] for k, p in parameters.items())
    old_geometry_frozen = all(not p.requires_grad and p.grad is None for k, p in geometry.named_parameters()
                              if not k.startswith(('condition_injector.body_pose_encoder.', 'reasoner.person_reasoner.joint_attention.')))
    source_unchanged = (before_stat.st_mtime_ns, before_stat.st_size) == (args.full_checkpoint.stat().st_mtime_ns, args.full_checkpoint.stat().st_size)
    sequence_unchanged = result.shape == baseline.shape and len(encoder_calls) == calls_before_qwen == 2
    finite = bool(torch.isfinite(result).all())
    clean = model._native_body_execution.get() is None and model._native_body_binding.get() is None
    passed = (zero_diff == 0 and prefix_error == 0 and target_error > 0 and frozen and old_geometry_frozen
              and source_unchanged and sequence_unchanged and finite and clean
              and all(g is not None and g > 0 and torch.isfinite(torch.tensor(g)) for g in [*grads.values(), *heads]))
    torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    audit = {'schema': 'qwen21_body21_7b_audit_v2', 'passed': bool(passed), 'pair_id': sample['pair_metadata']['pair_id'],
             'target_token_hw': list(grid), 'target_tokens': n, 'reference_tokens': ref_n,
             'prefix_positions_checked': result.shape[1] - n,
             'image_sequence_shape': list(result.shape), 'qwen_added_tokens': 0, 'qwen_blocks': len(model.transformer_blocks),
             'geometry_shape': list(features.geometry.shape), 'zero_body_max_abs': zero_diff,
             'active_prefix_max_abs': prefix_error, 'active_target_max_abs': target_error, 'gradient_norms': grads,
             'body_head_gradient_norms': heads, 'base_frozen_unchanged': frozen, 'old_geometry_frozen': old_geometry_frozen,
             'pose_encoder_calls': len(encoder_calls), 'pose_encoder_calls_during_qwen': len(encoder_calls) - calls_before_qwen,
             'source_checkpoint_stat_unchanged': source_unchanged, 'sequence_unchanged': sequence_unchanged,
             'finite': finite, 'context_clean': clean, 'pose_config': geometry.get_extra_state(),
             'pose_provenance': sample['pose_provenance'], 'depth_metadata': depth.metadata[0], 'champ_inheritance': champ,
             'free_gib_before': free / 1024**3, 'total_gib': total / 1024**3,
             'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3, 'seconds': time.perf_counter() - started,
             'qwen_source_sha256': sha256(inspect.getfile(QwenImage21Transformer2DModel)),
             'source_sha256': {str(path.relative_to(root)): sha256(path) for path in (root / 'src/pose_control/qwen21_full').rglob('*.py')},
             'no_optimizer': True, 'no_weight_save': True,
             'limits': 'Synthetic text/latents, real geometry/depth; fresh heads perturbed only in memory. No image quality conclusion.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(audit, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('source_sha256', 'champ_inheritance', 'pose_provenance', 'depth_metadata', 'pose_config')}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
