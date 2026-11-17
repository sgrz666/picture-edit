#!/usr/bin/env python3
"""Explicit V6.1 to V6.2 initialization; no optimizer resume or artifact overwrite."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None,*,source_version='v61',target_version='v62',runtime_type=None,pose_config_type=None):
    description=__doc__ if target_version=='v62' else f'Explicit {source_version} to {target_version} initialization; no optimizer resume or overwrite.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--'+source_version, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pose-config', type=Path)
    a = p.parse_args(argv)
    source=getattr(a,source_version)
    paths = [source.resolve(), a.output.resolve(), a.report.resolve()]
    if len(set(paths)) != len(paths) or a.output.exists() or a.report.exists():
        raise FileExistsError(target_version+' checkpoint/report must be distinct new files')
    if not str(a.output.resolve()).startswith('/data/shangguanrz/'):
        raise ValueError('New '+target_version+' weights belong under /data/shangguanrz/')
    import torch
    from src.pose_control.qwen21_full.native_v62.contracts import PoseLinkConfigV62
    from src.pose_control.qwen21_full.native_v62.runtime import NativeFullRuntimeV62
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha, write_json
    runtime_type=runtime_type or NativeFullRuntimeV62
    pose_config_type=pose_config_type or PoseLinkConfigV62
    config = None if a.pose_config is None else pose_config_type.from_dict(json.loads(a.pose_config.read_text(encoding='utf8')))
    source_sha = file_sha(source)
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=True)
    runtime = getattr(runtime_type,'from_'+source_version+'_checkpoint')(saved, pose_config=config)
    payload = runtime.checkpoint_payload(metadata={source_version+'_source': {'path': str(source.resolve()), 'sha256': source_sha}})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(payload, f)
    # Rebuild and strict load the actual serialized file, not the in-memory mapping.
    reread = torch.load(a.output, map_location='cpu', mmap=True, weights_only=True)
    checked = runtime_type.from_checkpoint(reread)
    if file_sha(source) != source_sha:
        raise RuntimeError('Source V6.1 changed during migration')
    write_json(a.report, {'architecture': 'native_full_'+target_version, 'strict_readback': True,
        'source_sha256': source_sha, 'output_sha256': file_sha(a.output),
        'pose_config': checked.pose_config.to_dict(), 'inheritance': runtime.inheritance_report,
        'optimizer_inherited': False, 'training_started': False})


if __name__ == '__main__':
    main()
