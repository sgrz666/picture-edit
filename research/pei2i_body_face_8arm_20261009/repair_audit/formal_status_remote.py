"""Read the running experiment without loading models or modifying artifacts."""
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone

run = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
state_path = run / 'repair_audit/full_pipeline_state.json'
state = json.loads(state_path.read_text())
counts = {}
for proto in ('p0', 'pg', 'pb', 'pbg'):
    counts[proto] = {}
    for split in ('train', 'val', 'diagnostic', 'test'):
        file = run / f'prompts/{proto}/{split}.jsonl'
        counts[proto][split] = sum(bool(line.strip()) for line in file.read_text().splitlines()) if file.is_file() else 0
arms = {}
for base in ('qwen', 'head'):
    for proto in ('p0', 'pg', 'pb', 'pbg'):
        arm = f'{base}_{proto}'
        summary = run / f'training/{arm}/summary.json'
        metrics = run / f'training/{arm}/metrics.jsonl'
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.is_file() else []
        arms[arm] = {
            'formal_logged_updates': len(rows),
            'last_logged_step': rows[-1].get('step') if rows else 0,
            'summary': json.loads(summary.read_text()) if summary.is_file() else None,
        }
stages = state.get('stages', [])
pid = stages[-1].get('pid') if stages else None
pids = [str(state['pid'])]
if pid:
    pids.append(str(pid))
    children = pathlib.Path(f'/proc/{pid}/task/{pid}/children')
    if children.exists():
        pids.extend(children.read_text().split())
processes = subprocess.run(['ps', '-p', ','.join(pids), '-o', 'pid,ppid,stat,etime,args'], capture_output=True, text=True).stdout
log = run / stages[-1]['log'] if stages else None
disk = shutil.disk_usage(run)
print(json.dumps({
    'observed_at': datetime.now(timezone.utc).isoformat(),
    'host': subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
    'run_dir': str(run), 'state': state, 'formal_prompt_counts': counts,
    'formal_optimizer_updates': sum(x['formal_logged_updates'] for x in arms.values()),
    'arms': arms, 'processes': processes,
    'current_log_tail': log.read_text(errors='replace')[-2400:] if log and log.is_file() else '',
    'disk_free_gib': round(disk.free / (1024 ** 3), 2),
}, indent=2, ensure_ascii=False))
