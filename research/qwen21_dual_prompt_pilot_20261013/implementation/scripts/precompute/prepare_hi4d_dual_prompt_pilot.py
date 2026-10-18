"""Prepare fixed real Hi4D pairs and audited source/geometry-only instructions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.hi4d_dual_prompt_pilot import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/shangguanrz/project/pic-edit'))
    parser.add_argument('--output', type=Path, required=True, help='new isolated output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fit-steps', type=int, default=200, help='1..300 mesh fitting steps per frame')
    parser.add_argument('--dino', type=Path, default=Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code/models/dinov2-giant'))
    parser.add_argument('--vae', type=Path, help='default ROOT/models/Qwen-Image-2.1/vae')
    parser.add_argument('--selection-lock', type=Path, help='locked original selection, auto-detected in experiment directory')
    args = parser.parse_args()
    result = prepare_dataset(root=args.root, output=args.output, device=args.device,
        fit_steps=args.fit_steps, dino=args.dino, vae=args.vae or args.root / 'models/Qwen-Image-2.1/vae',
        selection_lock=args.selection_lock)
    print(json.dumps({'dataset': str(result), 'status': 'prepared_and_strict_contract_verified'}), flush=True)


if __name__ == '__main__':
    main()
