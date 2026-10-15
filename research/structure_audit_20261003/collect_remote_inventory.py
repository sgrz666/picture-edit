"""Read-only inventory. Run on sg with Python 3 through SSH stdin."""
import ast
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/shangguanrz/project/pic-edit')
AUX = Path('/home/shangguanrz/nihaochen')
HOME = ROOT.parents[1]

def run(*args):
    result = subprocess.run(args, capture_output=True, text=True, errors='replace')
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

def walk(root, depth=None):
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', '.cache'})
        level = len(Path(base).relative_to(root).parts)
        if depth is not None and level >= depth:
            dirs[:] = []
        for name in sorted(files):
            yield Path(base) / name

def source_paths():
    paths = set(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py', '.sh', '.md', '.txt', '.ini'})
    for part in ['src', 'tests', 'scripts', 'tools', 'vram_lab', 'third_party', 'docs', 'research']:
        paths.update(p for p in walk(ROOT / part) if p.suffix in {'.py', '.sh', '.md', '.txt', '.ini'} and p.stat().st_size < 500000)
    for part in ['add/tools', 'add/datasets']:
        paths.update(p for p in (ROOT / part).glob('*') if p.is_file() and p.suffix in {'.py', '.sh'})
    for part in ['experiments/16-test', 'model-qwen-edit']:
        paths.update(p for p in (ROOT / part).glob('*') if p.is_file() and p.suffix in {'.py', '.sh', '.md'})
    paths.add(ROOT / 'models/DeepGen-1.0-diffusers/deepgen_pipeline.py')
    for part in ['add/cleanup_records', 'add/migration']:
        paths.update(p for p in walk(ROOT / part, 1) if p.suffix in {'.md', '.sh', '.py'})
    paths.update(p for p in HOME.glob('*') if p.is_file() and p.suffix in {'.py', '.sh'})
    paths.update(p for p in AUX.glob('*') if p.is_file() and p.suffix in {'.py', '.sh'})
    paths.update(p for p in walk(AUX / 'scripts') if p.suffix in {'.py', '.sh'})
    for part in ['iper1024', 'reconstruction_summary_20261001']:
        paths.update(p for p in (AUX / part).glob('*') if p.is_file() and p.suffix in {'.py', '.sh'})
    paths.update(p for p in walk(AUX / 'docs', 1) if p.suffix == '.md')
    return sorted(paths)

sources = []
duplicates = collections.defaultdict(list)
for path in source_paths():
    raw = path.read_bytes()
    body = raw.decode('utf-8', errors='replace')
    item = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'lines': len(body.splitlines()), 'text': body}
    if path.suffix == '.py':
        try:
            tree = ast.parse(body)
            item['symbols'] = [{'name': n.name, 'line': n.lineno, 'kind': type(n).__name__} for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            item['imports'] = [{'module': n.module, 'names': [a.name for a in n.names], 'level': n.level, 'line': n.lineno} if isinstance(n, ast.ImportFrom) else {'names': [a.name for a in n.names], 'line': n.lineno} for n in ast.walk(tree) if isinstance(n, (ast.ImportFrom, ast.Import))]
        except SyntaxError as error:
            item['parse_error'] = str(error)
    sources.append(item)
    if len(raw) > 400 and path.suffix in {'.py', '.sh', '.md'}:
        duplicates[item['sha256']].append(str(path))

artifacts = []
for base, depth in [(ROOT / 'experiments', 4), (AUX, 4), (ROOT / 'add/outputs', 2), (ROOT / 'datasets', 3)]:
    for path in walk(base, depth):
        name = path.name
        if path.suffix in {'.pt', '.safetensors'} and ('checkpoint' in name or 'adapter_step' in name or name == 'adapter.pt' or name.startswith('inherit_body')):
            stat = path.stat()
            artifacts.append({'path': str(path), 'bytes': stat.st_size, 'blocks_bytes': stat.st_blocks * 512, 'inode': stat.st_ino, 'device': stat.st_dev, 'nlink': stat.st_nlink, 'mtime': stat.st_mtime})
        elif path.suffix in {'.json', '.md', '.jsonl'} and path.stat().st_size < 4000000:
            if path.suffix == '.jsonl':
                if name != 'train_metrics.jsonl':
                    continue
                with path.open() as stream:
                    lines = collections.deque(stream, maxlen=2)
                body = ''.join(lines)
            elif path.suffix == '.md' or any(key in name.lower() for key in ['summary', 'report', 'manifest', 'status', 'acceptance', 'review', 'accounting', 'delivery', 'selection', 'completion', 'migration', 'validation', 'inventory', 'contract', 'gate']):
                body = path.read_text(errors='replace')
            else:
                continue
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size, 'text': body})

for relative, depth in [('datasets/TikTokDataset/TikTok_v3/manifests', 2), ('datasets/TikTokDataset/TikTok_v3/qc', 2)]:
    for path in walk(ROOT / relative, depth):
        if path.suffix == '.json' and ('summary' in path.name or path.name == 'stats.json') and path.stat().st_size < 1000000 and not any(a['path'] == str(path) for a in artifacts):
            artifacts.append({'path': str(path), 'bytes': path.stat().st_size, 'text': path.read_text(encoding='utf-8', errors='replace')})

links = collections.defaultdict(lambda: {'count': 0, 'broken': 0, 'inaccessible': 0, 'samples': [], 'broken_samples': [], 'inaccessible_samples': []})
caches = collections.defaultdict(lambda: {'files': 0, 'blocks_bytes': 0})
for base in [ROOT, AUX]:
    for directory, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', 'envs', '.cache'}]
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                group = str(base / Path(directory).relative_to(base).parts[0]) if Path(directory) != base else str(base)
                row = {'path': str(path), 'target': os.readlink(path)}
                links[group]['count'] += 1
                if len(links[group]['samples']) < 12:
                    links[group]['samples'].append(row)
                try:
                    exists = path.exists()
                except PermissionError:
                    links[group]['inaccessible'] += 1
                    if len(links[group]['inaccessible_samples']) < 12:
                        links[group]['inaccessible_samples'].append(row)
                    continue
                if not exists:
                    links[group]['broken'] += 1
                    if len(links[group]['broken_samples']) < 12:
                        links[group]['broken_samples'].append(row)
        if any(part in {'__pycache__', '.pytest_cache'} for part in Path(directory).parts):
            group = str(base)
            for name in files:
                stat = (Path(directory) / name).lstat()
                caches[group]['files'] += 1
                caches[group]['blocks_bytes'] += stat.st_blocks * 512

result = {
    'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'root': str(ROOT), 'auxiliary': str(AUX),
    'git_head': run('git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
    'git_status': run('git', '-C', str(ROOT), '-c', 'core.quotepath=false', 'status', '--porcelain'),
    'git_tracked': run('git', '-C', str(ROOT), 'ls-files'),
    'root_disk_usage': run('du', '-x', '-B1', '--max-depth=1', str(ROOT)),
    'aux_disk_usage': run('du', '-x', '-B1', '--max-depth=2', str(AUX)),
    'data_asset_disk_usage': run('du', '-x', '-B1', '--max-depth=2', '/data/shangguanrz'),
    'data_free_space': run('df', '-B1', '/data'),
    'free_space': run('df', '-B1', str(ROOT)),
    'sources': sources, 'artifacts': artifacts,
    'duplicate_source_groups': [v for v in duplicates.values() if len(v) > 1],
    'symlinks': dict(links), 'disposable_python_caches': dict(caches),
}
print(json.dumps(result, ensure_ascii=False))
