#!/usr/bin/env python3
"""显式升级至 Depth V2：继承旧 Encoder，丢弃旧 Depth 投影；不恢复 optimizer。

本轮仅实现此工具，不执行。输出必须是新文件，源权重只读。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import torch

from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime, ARCHITECTURE
from src.pose_control.qwen21_full.native_depth_v2 import NativeDepthV2Config, Qwen21NativeDepthV2
from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry, LEGACY_SCHEMA
from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
from src.pose_control.qwen21_full.pose_cache import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--champ-checkpoint', type=Path, help='Required only for Full/Depth-off sources')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    source = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    if source.get('architecture') == ARCHITECTURE:
        raise ValueError('Source is already Depth V2; reconstruct it without migration')
    if source.get('architecture') == 'qwen21_full_v1':
        if args.champ_checkpoint is None:
            raise ValueError('Full source has no Native CHAMP Encoder: supply --champ-checkpoint')
        geometry = Qwen21NativeGeometry.from_full_checkpoint(source)
        base = NativeDepthRuntime(geometry, Qwen21NativeBody())
        frontend_report = geometry.inheritance_report
    else:
        frontend_report = None
        old_geometry = source.get('geometry_config', {}).get('schema') == LEGACY_SCHEMA
        if source.get('depth_config') is not None:
            if args.champ_checkpoint is not None:
                raise ValueError('Choose the Native Encoder source or official CHAMP, not both')
            base = NativeDepthRuntime.from_native_checkpoint(source, upgrade_pose=old_geometry, upgrade_depth=True)
        else:
            if args.champ_checkpoint is None:
                raise ValueError('Depth-off source needs --champ-checkpoint')
            base = NativeDepthRuntime.from_native_checkpoint(source, upgrade_pose=old_geometry)
    if not isinstance(base.depth, Qwen21NativeDepthV2):
        depth = Qwen21NativeDepthV2(NativeDepthV2Config(geometry_dim=base.geometry.reasoner.config.hidden_dim))
        report = depth.load_champ_state_dict(torch.load(args.champ_checkpoint, map_location='cpu', weights_only=True))
        base = NativeDepthRuntime(base.geometry, base.body, depth)
    else:
        report = {**base.inheritance_report, 'source': 'Native V1 Encoder', 'old_depth_projection_discarded': True}
    report['fresh_depth_keys'] = [k for k in base.depth.state_dict() if not k.startswith('encoder.')]
    report['frontend'] = frontend_report
    report['body_initialization'] = 'fresh_zero_heads' if source.get('architecture') == 'qwen21_full_v1' else 'inherited'
    payload = base.checkpoint_payload(metadata={'source_path': str(args.source.resolve()), 'source_sha256': sha256(args.source),
             'champ_sha256': None if args.champ_checkpoint is None else sha256(args.champ_checkpoint),
             'initialization_seed': args.seed, 'inheritance': report,
             'fusion_initialization': 'fresh; final projection zero', 'optimizer_resumed': False,
             'runtime_acceptance': 'not_run'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(payload, stream)
    print(json.dumps({'architecture': payload['architecture'], 'output': str(args.output), 'optimizer_resumed': False}))


if __name__ == '__main__':
    main()
