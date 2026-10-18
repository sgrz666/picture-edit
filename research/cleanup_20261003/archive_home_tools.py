"""Archive unchanged, inactive HOME scratch tools without changing their bytes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HOME = Path('/home/shangguanrz')
ROOT = HOME / 'project/pic-edit'
DEST = ROOT / 'archive/home_tools_202609'
MANIFEST = ROOT / 'research/cleanup_20261003/home_classification.json'
JOURNAL = DEST / 'operations.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    temp = JOURNAL.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temp.replace(JOURNAL)


require(HOME.resolve() == HOME and ROOT.resolve() == ROOT, 'Unexpected roots')
require(not JOURNAL.exists(), 'Existing archive operation: inspect instead of rerunning')
items = json.loads(MANIFEST.read_text())
require(len(items) == 66 and len({i['name'] for i in items}) == 66, 'Unexpected helper manifest')
record = {'started_at': time.time(), 'moved': [], 'runner_links': [], 'skipped': []}
DEST.mkdir(parents=True, exist_ok=True)
code = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').rglob('*.py'))
code += list((ROOT / 'tests').rglob('*.py')) + list(ROOT.glob('*.py'))
commands = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
ready = []
for item in items:
    source = HOME / item['name']
    require(source.parent == HOME and source.name == item['name'], 'Unsafe basename')
    require(source.is_file() and not source.is_symlink() and sha(source) == item['sha256'], 'Helper changed: ' + str(source))
    if item['group'] == 'run':
        target = ROOT / 'scripts/run/home' / source.name
        require(not target.exists() and not target.is_symlink(), str(target))
        ready.append((item, source, target, 'runner'))
        continue
    text = source.read_text()
    reason = None
    if '__file__' in text:
        reason = 'location-dependent source'
    elif any(str(source) in line or source.name in line for line in commands):
        reason = 'active command'
    elif any(source.name in p.read_text(errors='replace') for p in code if p.is_file()):
        reason = 'referenced by current project code'
    elif source.suffix == '.py':
        tree = ast.parse(text)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module.split('.')[0])
        if any((HOME / name).is_dir() for name in modules):
            reason = 'imports a HOME package directory'
    if reason:
        record['skipped'].append({'source': str(source), 'reason': reason})
        continue
    category = 'historical_shell' if item['group'] == 'historical_destructive' else 'scratch'
    target = DEST / category / source.name
    require(not target.exists() and not target.is_symlink(), str(target))
    ready.append((item, source, target, 'move'))
save()
for item, source, target, operation in ready:
    target.parent.mkdir(parents=True, exist_ok=True)
    require(sha(source) == item['sha256'], 'Concurrent helper edit')
    if operation == 'runner':
        target.symlink_to(source)
        require(target.resolve() == source, 'Runner alias differs')
        record['runner_links'].append({'source': str(source), 'target': str(target), 'sha256': item['sha256']})
        save()
        continue
    entry = {'source': str(source), 'target': str(target), 'sha256': item['sha256'], 'state': 'prepared'}
    record['moved'].append(entry); save()
    source.rename(target)
    created_inode = None
    try:
        source.symlink_to(target)
        created_inode = source.lstat().st_ino
        require(source.resolve() == target and sha(target) == item['sha256'], 'Archived helper differs')
    except BaseException:
        if source.is_symlink() and created_inode is not None and source.lstat().st_ino == created_inode:
            source.unlink()
        if not source.exists() and not source.is_symlink():
            target.rename(source)
            entry['state'] = 'restored_source'
        else:
            entry['state'] = 'rollback_conflict_original_retained'
        save()
        raise
    entry['state'] = 'completed'; save()
record['finished_at'] = time.time()
save()
print(json.dumps({'moved': len(record['moved']), 'runner_links': len(record['runner_links']), 'skipped': record['skipped']}))
