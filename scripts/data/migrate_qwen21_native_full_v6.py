#!/usr/bin/env python3
"""显式 V5→V6 初始化；不恢复 optimizer，不覆盖源或既有产物。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v5', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pose-config', type=Path)
    a = p.parse_args(argv)
    paths = [a.v5.resolve(), a.output.resolve(), a.report.resolve()]
    if len(set(paths)) != len(paths) or a.output.exists() or a.report.exists():
        raise FileExistsError('V6 checkpoint/report must be distinct new files')
    if not str(a.output.resolve()).startswith('/data/shangguanrz/'):
        raise ValueError('New V6 weights belong under /data/shangguanrz/')
    import torch
    from src.pose_control.qwen21_full.native_v6.contracts import PoseLinkConfig
    from src.pose_control.qwen21_full.native_v6.runtime import NativeFullRuntimeV6
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha, write_json
    config = None if a.pose_config is None else PoseLinkConfig.from_dict(json.loads(a.pose_config.read_text(encoding='utf8')))
    source_sha = file_sha(a.v5)
    saved = torch.load(a.v5, map_location='cpu', mmap=True, weights_only=True)
    runtime = NativeFullRuntimeV6.from_v5_checkpoint(saved, pose_config=config)
    payload = runtime.checkpoint_payload(metadata={'v5_source': {'path': str(a.v5.resolve()), 'sha256': source_sha}})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(payload, f)
    # Rebuild and strict load the actual serialized file, not the in-memory mapping.
    reread = torch.load(a.output, map_location='cpu', mmap=True, weights_only=True)
    checked = NativeFullRuntimeV6.from_checkpoint(reread)
    if file_sha(a.v5) != source_sha:
        raise RuntimeError('Source V5 changed during migration')
    write_json(a.report, {'architecture': 'native_full_v6', 'strict_readback': True,
        'source_sha256': source_sha, 'output_sha256': file_sha(a.output),
        'pose_config': checked.pose_config.to_dict(), 'inheritance': runtime.inheritance_report,
        'optimizer_inherited': False, 'training_started': False})


if __name__ == '__main__':
    main()
