#!/usr/bin/env python3
"""显式 V6→V6.1 初始化；不恢复 optimizer，不覆盖源或既有产物。"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v6', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pose-config', type=Path)
    a = p.parse_args(argv)
    paths = [a.v6.resolve(), a.output.resolve(), a.report.resolve()]
    if len(set(paths)) != len(paths) or a.output.exists() or a.report.exists():
        raise FileExistsError('V6.1 checkpoint/report must be distinct new files')
    if not str(a.output.resolve()).startswith('/data/shangguanrz/'):
        raise ValueError('New V6.1 weights belong under /data/shangguanrz/')
    import torch
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha, write_json
    config = None if a.pose_config is None else PoseLinkConfigV61.from_dict(json.loads(a.pose_config.read_text(encoding='utf8')))
    source_sha = file_sha(a.v6)
    saved = torch.load(a.v6, map_location='cpu', mmap=True, weights_only=True)
    runtime = NativeFullRuntimeV61.from_v6_checkpoint(saved, pose_config=config)
    payload = runtime.checkpoint_payload(metadata={'v6_source': {'path': str(a.v6.resolve()), 'sha256': source_sha}})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(payload, f)
    # Rebuild and strict load the actual serialized file, not the in-memory mapping.
    reread = torch.load(a.output, map_location='cpu', mmap=True, weights_only=True)
    checked = NativeFullRuntimeV61.from_checkpoint(reread)
    if file_sha(a.v6) != source_sha:
        raise RuntimeError('Source V6 changed during migration')
    write_json(a.report, {'architecture': 'native_full_v61', 'strict_readback': True,
        'source_sha256': source_sha, 'output_sha256': file_sha(a.output),
        'pose_config': checked.pose_config.to_dict(), 'inheritance': runtime.inheritance_report,
        'optimizer_inherited': False, 'training_started': False})


if __name__ == '__main__':
    main()
