"""Detach one named pilot orchestrator, recording its actual PID and command."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--until',choices=['cache','sft','all'],default='all');p.add_argument('--wait-pid',type=int)
    a=p.parse_args();log=a.root/f'logs/{a.name}.log';pidfile=a.root/f'logs/{a.name}.pid.json'
    if log.exists() or pidfile.exists():raise FileExistsError('Use a new launcher name')
    env=dict(os.environ,PICTUREEDIT_REPO=str(a.repo),PYTHONPATH=str(a.repo),OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    command=['/home/shangguanrz/nihaochen/envs/qwen21/bin/python','-u',str(Path(__file__).with_name('run.py')),
        '--root',str(a.root),'--repo',str(a.repo),'--until',a.until]
    if a.wait_pid:command.extend(('--wait-pid',str(a.wait_pid)))
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:process=subprocess.Popen(command,cwd=a.repo,env=env,stdin=subprocess.DEVNULL,
        stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    result={'pid':process.pid,'command':command,'log':str(log),'started_at_utc':datetime.now(timezone.utc).isoformat()}
    with pidfile.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
