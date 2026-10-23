#!/usr/bin/env python3
"""Bounded real-data Body21 contracts and zero-delta parity. No optimizer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
import torch

from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
from src.pose_control.qwen21_full.body_pose import pose_kwargs_from_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-manifest', type=Path, required=True)
    p.add_argument('--pairs', type=Path, nargs='+', required=True)
    p.add_argument('--full-checkpoint', type=Path, required=True)
    p.add_argument('--max-pixels', type=int, default=256*256)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = torch.load(args.full_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    v2 = Qwen21NativeGeometry.from_full_checkpoint(source).eval()
    legacy = Qwen21NativeGeometry.from_full_checkpoint(source, legacy=True).eval()
    del source
    rows = []
    with torch.no_grad():
        for pairs in args.pairs:
            dataset = NativePoseDataset(pairs, args.pose_manifest, max_pixels=args.max_pixels)
            for i in range(len(dataset)):
                sample = dataset[i]
                batch = {k: v[None] for k, v in sample.items() if torch.is_tensor(v)}
                core = {name + '_a': batch[name] for name in ('normal', 'pose_heatmap', 'part_onehot', 'human_mask', 'smplx_global')}
                core['task_id'] = batch['task_id']
                bundle = v2.condition_injector(**core, **pose_kwargs_from_batch(batch, device='cpu')).validate()
                actual = v2(bundle, target_token_hw=sample['target_token_hw'])
                expected = legacy(legacy.condition_injector(**core), target_token_hw=sample['target_token_hw'])
                error = float((actual.geometry - expected.geometry).abs().max())
                if error != 0:
                    raise AssertionError('Zero Pose delta changed the canonical-input legacy result')
                meta = dataset.records.raw(sample['pose_provenance']['record_id'])['metadata']
                rows.append({'record_id': meta['record_id'], 'target_token_hw': list(actual.target_token_hw),
                             'joint_tokens': list(bundle.person_a_joint_tokens.shape),
                             'global_tokens': list(bundle.person_a_global_tokens.shape), 'geometry': list(actual.geometry.shape),
                             'zero_pose_delta_max_abs': error, 'mesh_max_abs_error': meta['mesh_max_abs_error'],
                             'camera': meta['camera']['selection'], 'projectable': int(sample['projection_valid'].sum()),
                             'sigma_px': float(bundle.person_a_joint_sigma[0]),
                             'condition_to_original': meta['condition_to_original'], 'pose_provenance': sample['pose_provenance']})
                print(json.dumps({k: v for k, v in rows[-1].items() if k not in ('condition_to_original', 'pose_provenance')}), flush=True)
    result = {'schema': 'qwen21_body21_data_audit_v2', 'passed': True, 'count': len(rows), 'rows': rows,
              'dimensions': {'body_local_aa': 63, 'rotation6d': [21, 6], 'joint_numeric_input': 11,
                             'joint_tokens': [21, 256], 'global_tokens': [4, 256], 'task_tokens': [1, 256],
                             'joint_attention': [512, 8], 'geometry_width': 512, 'qwen_added_tokens': 0},
              'pose_parameters': sum(p.numel() for p in v2.pose_parameters()),
              'legacy_parameters': sum(p.numel() for p in legacy.parameters()),
              'v2_parameters': sum(p.numel() for p in v2.parameters()),
              'inheritance': v2.inheritance_report, 'no_optimizer': True, 'seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
