#!/usr/bin/env python3
"""Explicit V6.1 to V6.2 initialization; no optimizer resume or artifact overwrite."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v61', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pose-config', type=Path)
    a = p.parse_args(argv)
    paths = [a.v61.resolve(), a.output.resolve(), a.report.resolve()]
    if len(set(paths)) != len(paths) or a.output.exists() or a.report.exists():
        raise FileExistsError('V6.2 checkpoint/report must be distinct new files')
    if not str(a.output.resolve()).startswith('/data/shangguanrz/'):
        raise ValueError('New V6.2 weights belong under /data/shangguanrz/')
    import torch
    from src.pose_control.qwen21_full.native_v62.contracts import PoseLinkConfigV62
    from src.pose_control.qwen21_full.native_v62.runtime import NativeFullRuntimeV62
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha, write_json
    config = None if a.pose_config is None else PoseLinkConfigV62.from_dict(json.loads(a.pose_config.read_text(encoding='utf8')))
    source_sha = file_sha(a.v61)
    saved = torch.load(a.v61, map_location='cpu', mmap=True, weights_only=True)
    runtime = NativeFullRuntimeV62.from_v61_checkpoint(saved, pose_config=config)
    payload = runtime.checkpoint_payload(metadata={'v61_source': {'path': str(a.v61.resolve()), 'sha256': source_sha}})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(payload, f)
    # Rebuild and strict load the actual serialized file, not the in-memory mapping.
    reread = torch.load(a.output, map_location='cpu', mmap=True, weights_only=True)
    checked = NativeFullRuntimeV62.from_checkpoint(reread)
    if file_sha(a.v61) != source_sha:
        raise RuntimeError('Source V6.1 changed during migration')
    write_json(a.report, {'architecture': 'native_full_v62', 'strict_readback': True,
        'source_sha256': source_sha, 'output_sha256': file_sha(a.output),
        'pose_config': checked.pose_config.to_dict(), 'inheritance': runtime.inheritance_report,
        'optimizer_inherited': False, 'training_started': False})


if __name__ == '__main__':
    main()
