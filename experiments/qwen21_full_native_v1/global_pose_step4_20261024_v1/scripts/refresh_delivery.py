"""Deliver only Python source with baseline/previous-delivery SHA checks."""
from pathlib import Path
import subprocess
import hashlib
import json
import tarfile
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[4]
run = Path(__file__).resolve().parents[1]
remote_root = '/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1/source'
tag = 'delivery_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
stage = run / 'source' / tag
stage.mkdir(exist_ok=False)
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=root, text=True).splitlines()
new = ['src/pose_control/qwen21_full/' + n for n in
       ('body_pose.py', 'pose_cache.py', 'pose_data_runtime.py')]
new += ['scripts/data/' + n for n in ('build_qwen21_body_pose_cache.py', 'migrate_qwen21_geometry_v2.py')]
new += ['tests/test_qwen21_global_pose.py', 'tests/test_qwen21_pose_cache.py']
new += ['tests/test_qwen21_full_preservation.py', 'scripts/run/run_qwen21_native_pose.py']
new += [str(p.relative_to(root)).replace('\\', '/') for p in (root / 'scripts/audit').glob('*global_pose*.py')]
paths = sorted(set(new + [p for p in changed if p.endswith('.py')]))
rows = [{'path': p, 'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in paths]
(stage / 'delivery.json').write_text(json.dumps(rows, indent=2))
with tarfile.open(stage / 'delivery.tar.gz', 'w:gz') as tar:
    for p in paths:
        tar.add(root / p, arcname=p)
remote = remote_root + '/' + tag
subprocess.run(['ssh', 'sg', 'mkdir', '-p', remote], check=True)
for path in (stage / 'delivery.tar.gz', stage / 'delivery.json', run / 'source/baseline.json', run / 'scripts/apply_delivery.py'):
    subprocess.run(['scp', str(path), 'sg:' + remote + '/'], check=True)
subprocess.run(['ssh', 'sg', '/usr/bin/python3', remote + '/apply_delivery.py', remote], check=True)
