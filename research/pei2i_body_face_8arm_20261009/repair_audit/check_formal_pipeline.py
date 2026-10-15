"""Refresh the local snapshot of the actual remote experiment."""
import json
import subprocess
import sys
from pathlib import Path

audit = Path(__file__).resolve().parent
result = subprocess.run(
    [sys.executable, str(audit / 'remote_exec.py'), str(audit / 'formal_status_remote.py')],
    capture_output=True, encoding='utf-8', check=True,
)
payload = json.loads(result.stdout)
(audit / 'formal_pipeline_status.json').write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8',
)
print(json.dumps({
    'observed_at': payload['observed_at'],
    'status': payload['state']['status'],
    'phase': payload['state'].get('phase'),
    'formal_prompt_counts': payload['formal_prompt_counts'],
    'formal_optimizer_updates': payload['formal_optimizer_updates'],
    'processes': payload['processes'],
    'current_log_tail': payload['current_log_tail'],
    'local_snapshot': str(audit / 'formal_pipeline_status.json'),
}, indent=2, ensure_ascii=False))
