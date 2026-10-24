#!/usr/bin/env python3
"""Build same-fit Body21/Depth/local geometry/BPS. Requires later real SMPL-X assets.

Run in the independent geometry environment, never install FCL into qwen21.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    from src.pose_control.qwen21_full.native_full_data import assemble_full_cache,BODY_SCHEMA,PAIR_SCHEMA
    from src.pose_control.qwen21_full.native_full_assets import write_record
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    job=json.loads(args.job.read_text(encoding='utf8'))
    sample,local=assemble_full_cache(job,renderer=IndependentMeshRasterizer(args.device))
    args.output.mkdir(parents=True,exist_ok=False)
    body=write_record(args.output/'body.npz',sample,schema=BODY_SCHEMA)
    geometry=write_record(args.output/'local.npz',local,schema='native_local_condition_v4')
    row={'schema':PAIR_SCHEMA,'pair_id':job['pair_id'],'scene_kind':'dual' if local.person_valid[0,1] else 'single',
        'body':body,'local':geometry,'references':job['references'],'source_rgb':job['source_rgb'],'target_rgb':job['target_rgb']}
    with (args.output/'pairs.jsonl').open('x',encoding='utf8') as stream: stream.write(json.dumps(row)+'\n')
    print(json.dumps(row))


if __name__=='__main__': main()
