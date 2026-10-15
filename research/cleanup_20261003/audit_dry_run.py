"""Run the read-only asset audit and persist its result outside audited journals."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/shangguanrz/project/pic-edit')
LOG = ROOT / 'archive/cleanup_20261003'
OUT = ROOT / 'research/cleanup_20261003'


def inventory():
    return {str(p.relative_to(LOG)): {'mtime_ns': p.stat().st_mtime_ns,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in LOG.rglob('*') if p.is_file() and not p.is_symlink()}


OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'dry_run_result.json').exists():
    raise RuntimeError('Existing result: inspect before rerun')
(OUT / 'dry_run_pid.json').write_text(json.dumps({'pid': os.getpid(), 'started_at': time.time()}))
before = inventory()
started = time.time()
run = subprocess.Popen(['python3', str(LOG / 'tooling/finalize_migration.py')],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
lines = []
for line in run.stdout:
    print(line, end='', flush=True)
    lines.append(line)
code = run.wait()
after = inventory()
result = {'exit_code': code, 'journal_content_and_mtime_unchanged': before == after,
          'stdout': ''.join(lines), 'files_checked': len(before),
          'started_at': started, 'finished_at': time.time()}
(OUT / 'dry_run_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('DRY_RUN_RESULT', json.dumps({k: v for k, v in result.items() if k != 'stdout'}), flush=True)
raise SystemExit(code if before == after else 2)
