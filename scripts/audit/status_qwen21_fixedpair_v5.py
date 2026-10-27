#!/usr/bin/env python3
"""Read-only committed checkpoints, live updates and running pilot processes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import getpass
import hashlib
import json
import re
import subprocess

ROOTS={
    'tiktok':Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1'),
    'hi4d':Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1'),
}


def tail_records(path,limit=4*1024*1024):
    try:
        with path.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-limit));raw=f.read()
    except FileNotFoundError:return []
    result=[]
    for line in raw.splitlines():
        try:result.append(json.loads(line))
        except (ValueError,UnicodeError):continue
    return result


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def status(root,verify=False):
    commits=[]
    for path in (root/'training/checkpoints').glob('*.commit.json'):
        value=json.loads(path.read_text())
        if not Path(value['checkpoint']['path']).is_file():continue
        commits.append((value['global_update'],path.stat().st_mtime_ns,path,value))
    chosen=max(commits,default=None,key=lambda x:x[:2])
    updates=[]
    for path in (root/'training/segments').glob('*/metrics.jsonl'):
        records=[r.get('record',r) for r in tail_records(path)]
        updates += [r for r in records if r.get('optimizer_updated') and 'global_update' in r]
    latest=max(updates,default=None,key=lambda r:r['global_update'])
    events=tail_records(root/'logs/orchestration.jsonl');event=events[-1] if events else None
    if event and event.get('time'):
        event={**event,'time_shanghai':datetime.fromisoformat(event['time']).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}
    checkpoint=None if chosen is None else chosen[3]['checkpoint']
    verified=False
    if verify and checkpoint is not None:
        if digest(Path(checkpoint['path']))!=checkpoint['sha256']:raise ValueError('Committed checkpoint SHA mismatch')
        verified=True
    return {'root':str(root),'target_updates':300,'committed_update':0 if chosen is None else chosen[0],
        'latest_logged_update':0 if latest is None else latest['global_update'],
        'latest_gradient_norm':None if latest is None else latest.get('gradient_norm'),
        'checkpoint':checkpoint,'checkpoint_sha_verified':verified,'latest_event':event,
        'generation_images':sum(1 for _ in (root/'training/output/images').rglob('*.png'))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-latest',action='store_true');a=p.parse_args()
    ps=subprocess.run(['ps','-u',getpass.getuser(),'-o','pid,args'],capture_output=True,text=True,check=True)
    processes=[line.strip() for line in ps.stdout.splitlines() if 'run_qwen21_v5_fixedpair.py' in line and
        re.match(r'^\s*\d+\s+\S*python(?:[0-9.]+)?\s',line)]
    print(json.dumps({'processes':processes,'experiments':{k:status(v,a.verify_latest) for k,v in ROOTS.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
