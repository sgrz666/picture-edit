#!/usr/bin/env python3
"""Explicit V5 sidecar/reference/quality/repair tools; never run automatically."""
from pathlib import Path
import argparse,json,sys
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('promote','references','quality-audit','export-repair','import-repair'):
        s=sub.add_parser(name);s.add_argument('--job',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=p.parse_args();job=json.loads(args.job.read_text(encoding='utf8'))
    if args.output.exists(): raise FileExistsError('Use a new V5 artifact')
    if not str(args.output.resolve()).startswith('/data/shangguanrz/'): raise ValueError('New V5 assets belong under /data/shangguanrz/')
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record,sha256
    if args.command=='promote':
        from src.pose_control.qwen21_full.native_v5.data import promote_manifest
        promote_manifest(job,args.output);return
    if args.command=='references':
        from src.pose_control.qwen21_full.native_full_references import extract_source_references
        refs=extract_source_references(job)
        refs=replace(refs,metadata=tuple({**m,'explicit_multi_reference':bool(job.get('explicit_multi_reference',False))} for m in refs.metadata))
        result=write_record(args.output,refs,schema='native_local_reference_v4')
        print(json.dumps(result));return
    from src.pose_control.qwen21_full.native_v5.quality import audit_hands,export_repair,import_repair
    c=read_record(job['local'],schema='native_local_condition_v4').validate()
    if args.command=='quality-audit':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f: json.dump(audit_hands(c),f,indent=2)
    elif args.command=='export-repair':
        from src.pose_control.qwen21_full.native_v5.data import verified
        verified(job['rgb'])
        print(json.dumps(write_record(args.output,export_repair(c,job['rgb']),schema='native_hand_repair_job_v5')))
    else:
        import torch
        body=read_record(job['body'],schema='native_full_body_record_v4')
        for key in ('pose_provenance','pose_provenance_b'):
            if key in body and body[key]['record_sha256']!=c.metadata[0]['driver_record_sha256']: raise ValueError('Repair Body source differs')
        wrist=torch.stack((body['joint_uv'][[19,20]],body.get('joint_uv_b',torch.zeros_like(body['joint_uv']))[[19,20]]))[None]
        for person,key in enumerate(('projection_valid','projection_valid_b')):
            if key in body: wrist[0,person,~body[key][[19,20]]]=torch.nan
        repaired=read_record(job['repair'],schema='native_hand_repair_result_v5')
        support=read_record(job['trusted_support'],schema='native_hand_repair_support_v5')
        for k in ('driver_rgb_sha256','target_canvas_id','target_person_ids'):
            if support['metadata'].get(k)!=c.metadata[0].get(k): raise ValueError('Trusted repair support provenance differs')
        result=import_repair(c,repaired,body_wrist_uv=wrist,trusted_person_support=support['support'],repair_sha256=job['repair']['sha256'],trusted_support_sha256=job['trusted_support']['sha256'])
        print(json.dumps(write_record(args.output,result,schema='native_local_condition_v4')))


if __name__=='__main__': main()
