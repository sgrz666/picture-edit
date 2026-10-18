#!/usr/bin/env python3
"""Render and score one bounded Face repair checkpoint using the frozen inputs."""

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.run_qwen21_face_single_sample import Metrics
from src.pose_control.qwen21_full.data_runtime import read_pairs
from src.pose_control.qwen21_full.execution import json_write

OUT=Path('/home/shangguanrz/nihaochen/qwen21_full/face_repair')
PAIRS=OUT.parent/'face_single_sample/single_pair.jsonl'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,choices=('roi_same_aux','roi_no_aux','roi_aligned_aux'))
    parser.add_argument('--step',type=int,required=True,choices=(0,100,300))
    parser.add_argument('--concurrent-training',action='store_true')
    parser.add_argument('--resolution',type=int,choices=(512,1024),default=512)
    parser.add_argument('--mode',choices=('body','body_face'),default='body_face')
    parser.add_argument('--face-scatter-grid',choices=('native','checkpoint'),default='native')
    args=parser.parse_args()
    checkpoint=OUT/args.run/f'checkpoint_step_{args.step:06d}.pt'
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    directory=OUT/('generations' if args.resolution==512 else 'generations_1024')/args.run/f'step_{args.step:06d}'
    if args.mode=='body':directory=directory/'body'
    elif args.face_scatter_grid=='checkpoint':directory=directory/'checkpoint_grid'
    if not (directory/'metrics.json').exists():
        directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/'eval_iper_qwen21_full.py'),'--pairs',str(PAIRS),
                 '--body-checkpoint',str(checkpoint),'--face-checkpoint',str(checkpoint),
                 '--modes',args.mode,'--output',str(directory),'--resolution',str(args.resolution),
                 '--progress-mode','sigma','--face-strength','1','--no-use-kv-cache']
        command.extend(['--face-scatter-grid',args.face_scatter_grid])
        with (directory/'generation.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(line) for line in (directory/'per_sample.jsonl').read_text().splitlines()]
        if len(rows)!=1:raise AssertionError('Expected the single fixed training pair')
        result=Metrics().evaluate(rows[0],read_pairs(PAIRS)[0])
        result.update(run=args.run,added_step=args.step,concurrent_training=args.concurrent_training,
                      sampling_steps=30,seed=42,true_cfg_scale=1,face_strength=1)
        result['face_scatter_grid']=args.face_scatter_grid
        json_write(directory/'metrics.json',result)
    else:result=json.loads((directory/'metrics.json').read_text())
    print(json.dumps({'run':args.run,'added_step':args.step,'image':result['image'],'metrics':result['metrics']},ensure_ascii=False))

if __name__=='__main__':main()
