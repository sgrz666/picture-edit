#!/usr/bin/env python3
"""Explicit Depth V2 -> dual V3 migration. New file only; no optimizer resume."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import torch
from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
from src.pose_control.qwen21_full.pose_cache import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    saved = torch.load(args.source, map_location='cpu', mmap=True, weights_only=True)
    runtime = NativeDualRuntime.from_depth_v2(saved)
    payload = runtime.checkpoint_payload(metadata={'source_path': str(args.source.resolve()),
        'source_sha256': sha256(args.source), 'seed': args.seed, 'optimizer_resumed': False,
        'runtime_acceptance': 'not_run'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(payload, stream)
    print(json.dumps({'architecture': payload['architecture'], 'output': str(args.output),
                      'optimizer_resumed': False, 'inheritance': runtime.inheritance_report}))


if __name__ == '__main__':
    main()
