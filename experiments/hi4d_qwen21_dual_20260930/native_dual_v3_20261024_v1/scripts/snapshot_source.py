"""Stdlib source snapshot only; run on sg before source delivery."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/home/shangguanrz/project/pic-edit')
run = Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native_dual_v3_20261024_v1')
source = run / 'source'
source.mkdir(parents=True, exist_ok=False)
names = subprocess.check_output(['git', 'ls-files', 'src/pose_control/qwen21_full',
    'src/data/hi4d_qwen21_dataset.py', 'src/data/canvas_contract.py',
    'scripts/data', 'scripts/train', 'scripts/run'], cwd=root, text=True).splitlines()
names = [n for n in names if Path(n).suffix in ('.py', '.md', '.json') and (root / n).is_file()]
manifest = {'captured_utc': datetime.now(timezone.utc).isoformat(),
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
    'files': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}}
with tarfile.open(source / 'baseline_source.tar.gz', 'x:gz') as archive:
    for name in names:
        archive.add(root / name, arcname=name)
with (source / 'baseline.json').open('x') as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps({'head': manifest['head'], 'files': len(names), 'source': str(source)}))
