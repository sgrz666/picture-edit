#!/usr/bin/env python3
"""Run the two locked formal inference batches, with durable stage status."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.eval.qwen21_condition_combinations import NEW_METHODS, require_protocol_outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root
    status_path=root/'formal_status.json'; logs=root/'logs';logs.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONPATH':str(PROJECT)}
    state={'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'completed':False,'training':False,'stages':[]}
    def save():
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status_path)
    save()
    for resolution in (512,1024):
        stage={'resolution':resolution,'started_utc':datetime.now(timezone.utc).isoformat(),'completed':False}
        state['stages'].append(stage);save()
        cmd=[sys.executable,'scripts/eval/eval_qwen21_native_value.py','--pairs',str(root/'data/pairs.jsonl'),
             '--prompts',str(root/'prompts/locked_prompts.json'),'--methods',*NEW_METHODS,
             '--resolution',str(resolution),'--steps','30','--output',str(root/f'formal_{resolution}'),
             '--index-file',str(root/'generations.jsonl')]
        if resolution==1024:cmd.append('--highres-only')
        stage['command']=cmd;save()
        with (logs/f'formal_{resolution}.log').open('a') as stream:
            result=subprocess.run(cmd,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
        if result.returncode:
            state['failure']=f'Formal {resolution} returned {result.returncode}';save()
            raise SystemExit(result.returncode)
        stage['completed']=True;save()
    rows=[json.loads(x) for x in (root/'generations.jsonl').read_text().splitlines() if x.strip()]
    require_protocol_outputs(root/'protocol.json',rows)
    state.update(completed=True,finished_utc=datetime.now(timezone.utc).isoformat(),merged_outputs=len(rows))
    save();print(json.dumps(state),flush=True)


if __name__=='__main__':main()
