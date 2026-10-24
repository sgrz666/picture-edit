#!/usr/bin/env python3
"""Offline complete-SMPL-X import; does not run automatically during loading."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    from src.pose_control.qwen21_full.native_full_assets import import_smplx,write_record
    value=import_smplx(json.loads(args.job.read_text(encoding='utf8')))
    print(json.dumps(write_record(args.output,value,schema='native_smplx_asset_v4')))


if __name__=='__main__': main()
