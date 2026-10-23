#!/usr/bin/env python3
"""Strict weight-only Full/v1 -> Native Body21 migration, or explicit v1 rebuild.

不创建 optimizer、不恢复训练状态。输出必须是新路径；读取 source 后不改写。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import torch
from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
from src.pose_control.qwen21_full.pose_cache import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('upgrade_pose', 'legacy_rebuild', 'v2_rebuild'), default='upgrade_pose')
    p.add_argument('--seed', type=int, default=42, help='Fresh Pose/Body initialization only; never updates inherited weights')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    if source.get('architecture') == 'qwen21_full_v1':
        if args.mode == 'v2_rebuild':
            raise ValueError('Full checkpoint requires explicit upgrade_pose or legacy_rebuild')
        geometry = Qwen21NativeGeometry.from_full_checkpoint(source, legacy=args.mode == 'legacy_rebuild')
        runtime = NativeDepthRuntime(geometry, Qwen21NativeBody())
        inherited = geometry.inheritance_report
    else:
        runtime = NativeDepthRuntime.from_native_checkpoint(source, legacy=args.mode == 'legacy_rebuild',
                                                            upgrade_pose=args.mode == 'upgrade_pose')
        inherited = {'source_architecture': source['architecture'], 'optimizer_resumed': False}
    result = runtime.checkpoint_payload(metadata={'source_path': str(args.source.resolve()),
             'source_sha256': sha256(args.source), 'mode': args.mode, 'inheritance': inherited,
             'no_optimizer': True, 'no_training': True, 'initialization_seed': args.seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(result, stream)
    print(json.dumps({'architecture': result['architecture'], 'output': str(args.output),
                      'pose_parameters': sum(p.numel() for p in runtime.geometry.pose_parameters()), 'no_optimizer': True}))


if __name__ == '__main__':
    main()
