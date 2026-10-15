"""Small cross-filesystem fixtures exercising actual migration functions."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

SCRIPTS = Path('/home/shangguanrz/project/pic-edit/archive/cleanup_20261003/tooling')


def load_functions(name):
    tree = ast.parse((SCRIPTS / name).read_text())
    safe = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    env = {}
    exec(compile(ast.Module(body=safe, type_ignores=[]), name, 'exec'), env)
    return env


finalizer = load_functions('finalize_migration.py')
organizer = load_functions('organize_remote.py')
results = []
historical = {'ast_files_sha256': {'legacy.py': 'a'}, 'pytest': {'test_files_sha256': {'test_legacy.py': 'a'}}}
expanded = {'ast_files_sha256': {'legacy.py': 'b', 'new.py': 'a'}, 'pytest': {'test_files_sha256': {'test_legacy.py': 'b', 'test_new.py': 'a'}}, 'protected_files_sha256': {'src/core.py': 'b'}}
finalizer['verify_coverage'](historical, expanded, {'protected_before': {'src/core.py': 'a'}})
for kind in ['script', 'test']:
    shrunk = json.loads(json.dumps(expanded))
    mapping = shrunk['ast_files_sha256'] if kind == 'script' else shrunk['pytest']['test_files_sha256']
    del mapping['legacy.py' if kind == 'script' else 'test_legacy.py']
    mapping.update({'new_a.py': 'a', 'new_b.py': 'a'})
    try:
        finalizer['verify_coverage'](historical, shrunk, {'protected_before': {'src/core.py': 'a'}})
    except RuntimeError:
        results.append('historical ' + kind + ' removal rejected despite more replacement files')
    else:
        raise RuntimeError('Historical coverage reduction accepted')
with tempfile.TemporaryDirectory(prefix='cleanup-safety-', dir='/data/shangguanrz') as folder:
    root = Path(folder)
    if root.resolve().parent != Path('/data/shangguanrz'):
        raise RuntimeError('Unexpected fixture directory')
    hold, target = root / 'hold', root / 'target'
    hold.mkdir(); target.mkdir()
    (hold / 'payload').write_bytes(b'ORIGINAL')
    (target / 'payload').write_bytes(b'ORIGINAL')
    expected = {'payload': hashlib.sha256(b'ORIGINAL').hexdigest()}
    finalizer['verify_hashes'](hold, target, expected)
    results.append('identical trees accepted')
    (target / 'payload').write_bytes(b'TAMPERED')
    try:
        finalizer['verify_hashes'](hold, target, expected)
    except RuntimeError:
        if (hold / 'payload').read_bytes() != b'ORIGINAL':
            raise RuntimeError('Original lost after destination mismatch')
        results.append('destination tampering rejected; original retained')
    else:
        raise RuntimeError('Tampering was not rejected')
    metadata = finalizer['snapshot'](hold)
    finalizer['validate_remaining'](metadata, metadata)
    altered = {n: dict(m) for n, m in metadata.items()}
    altered['payload']['mode'] ^= 0o100
    try:
        finalizer['validate_remaining'](altered, metadata)
    except RuntimeError:
        results.append('partial-retirement metadata change rejected')
    else:
        raise RuntimeError('Changed recovery metadata accepted')
    writer = subprocess.Popen(['python3', '-c',
        'import sys,time; f=open(sys.argv[1],"r+b"); print("READY",flush=True); time.sleep(60)',
        str(hold / 'payload')], stdout=subprocess.PIPE, text=True)
    try:
        if writer.stdout.readline().strip() != 'READY':
            raise RuntimeError('Writer fixture failed')
        try:
            finalizer['idle']([hold])
        except RuntimeError:
            results.append('open writer rejected; original retained')
        else:
            raise RuntimeError('Writer was not detected')
    finally:
        writer.terminate(); writer.wait(timeout=5)
    project = root / 'project'; project.mkdir()
    source = project / 'sample.py'
    original = b'from pathlib import Path\nROOT = Path(__file__).resolve().parent\n'
    source.write_bytes(original)
    archive = project / 'archive/cleanup_20261003'; archive.mkdir(parents=True)
    organizer.update(ROOT=project, ARCHIVE=archive, JOURNAL=archive / 'organization.json',
                     record={'moves': []}, preflight_hashes={str(source): hashlib.sha256(original).hexdigest()})
    real_link = Path.symlink_to
    def concurrent_recreation(self, target, target_is_directory=False):
        if self == source:
            self.write_bytes(b'CONCURRENT NEW SOURCE')
        return real_link(self, target, target_is_directory)
    Path.symlink_to = concurrent_recreation
    try:
        try:
            organizer['move_entry'](source, 'audit')
        except FileExistsError:
            kept = archive / 'holds/sample.py'
            if kept.read_bytes() != original or source.read_bytes() != b'CONCURRENT NEW SOURCE':
                raise RuntimeError('Rollback overwrote concurrent data')
            results.append('rollback conflict preserves both original inode and concurrent replacement')
        else:
            raise RuntimeError('Concurrent recreation fixture failed')
    finally:
        Path.symlink_to = real_link
print(json.dumps({'passed': True, 'cases': results}))
