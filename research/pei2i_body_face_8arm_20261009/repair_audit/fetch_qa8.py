"""Copy available QA artifacts without changing any remote experiment files."""
from pathlib import Path
import json, subprocess
remote='/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009'
local=Path(__file__).resolve().parents[1]
for protocol in ('pg','pb','pbg'):
 rel=f'prompts/qa8/{protocol}/train.jsonl'; destination=local/rel
 destination.parent.mkdir(parents=True,exist_ok=True)
 result=subprocess.run(['scp',f'sg:{remote}/{rel}',str(destination)],capture_output=True,text=True)
 if result.returncode:
  print(protocol,'not yet available');continue
 records=[json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
 for row in records:
  paths=[row['raw_path']]+[item['asset_relative_path'] for item in row['inputs']]
  for path in paths:
   dst=local/path
   if not dst.resolve().is_relative_to(local.resolve()): raise ValueError('Invalid artifact path')
   dst.parent.mkdir(parents=True,exist_ok=True)
   subprocess.run(['scp',f'sg:{remote}/{path}',str(dst)],check=True,capture_output=True)
  print(protocol,row['pair_id'],row['tokens'],round(row['seconds'],2),row['compiled_prompt'])
