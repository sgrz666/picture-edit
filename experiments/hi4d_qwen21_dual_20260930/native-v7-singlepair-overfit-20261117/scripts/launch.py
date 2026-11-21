"""Detach one explicitly named bounded pilot, retaining its PID and exact argv."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import os
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--memory-fraction',type=float,default=.5)
    parser.add_argument('--offload-hidden',action='store_true')
    parser.add_argument('--templates',type=Path)
    args=parser.parse_args()
    if not 0<args.memory_fraction<=.95:parser.error('GPU fraction must leave at least 5% headroom')
    root=args.root.resolve()
    if root.parent!=Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual'):
        raise ValueError('Explicit existing Hi4D experiment root required')
    if not args.name.replace('-','').isalnum():raise ValueError('Plain run name required')
    command=[sys.executable,'-u',str(root/'scripts/run_training.py'),'--root',str(root),
             '--templates',str(args.templates.resolve() if args.templates is not None else root/'configs/native-v1'),'--name',args.name,'--memory-fraction',str(args.memory_fraction),'--generate-baseline']
    if args.offload_hidden:command.append('--offload-hidden')
    environment=dict(os.environ,LD_LIBRARY_PATH='/data/shangguanrz/仪表盘/compat',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    log=root/'logs'/(args.name+'-orchestrator.log')
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd='/home/shangguanrz/project/pic-edit',env=environment,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':process.pid,'argv':command,'started_utc':datetime.now(timezone.utc).isoformat(),
            'log':str(log),'maximum_optimizer_updates':300,'memory_fraction':args.memory_fraction,'offload_hidden':args.offload_hidden,'vae_tiling':False}
    with (root/'logs'/(args.name+'-pid.json')).open('x') as file:json.dump(record,file,ensure_ascii=False,indent=2)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
