"""Stage one cold-storage migration: copy, SHA-256 verify, link, retain source.

Run on sg via python3, never on the local Windows workspace. The fixed source
allowlist and archive destinations are checked before any destructive action.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import grp
import json
import os
import pwd
from pathlib import Path
import shutil
import subprocess
import time



def require(condition, message):
    if not condition:
        raise RuntimeError(message)

HOME = Path('/home/shangguanrz')
ROOT = HOME / 'project/pic-edit'
ARCHIVE = Path('/data/shangguanrz/pic-edit-archive/cleanup_20261003')
CACHES = Path('/data/shangguanrz/pic-edit-caches')
JOURNAL = ROOT / 'archive/cleanup_20261003/assets.json'
TASKS = [
    (ROOT / 'model-qwen-edit', ARCHIVE / 'models/Qwen-Image-Edit-20B'),
    (ROOT / 'experiments/review_20260908_v2', ARCHIVE / 'experiments/review_20260908_v2'),
    (ROOT / 'experiments/iper_v6_comparative_3000', ARCHIVE / 'experiments/iper_v6_comparative_3000'),
    (HOME / 'nihaochen/qwen21_hi4d_controls', CACHES / 'qwen21_hi4d_controls'),
    *[(ROOT / 'add' / name, ARCHIVE / 'raw/3dpw' / name)
      for name in ('imageFiles.zip', 'sequenceFiles.zip', 'camera_tram.zip')],
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def snapshot(path):
    entries = [path] if path.is_file() else [path] + sorted(path.rglob('*'))
    out = {}
    for p in entries:
        s = p.lstat()
        key = '.' if p == path else str(p.relative_to(path))
        require(not p.is_symlink(), f'Unexpected symlink in migration: {p}')
        require(p.is_file() or p.is_dir(), f'Unsupported file type: {p}')
        out[key] = {'type': 'file' if p.is_file() else 'dir', 'size': s.st_size,
                    'mtime_ns': s.st_mtime_ns, 'mode': s.st_mode & 0o7777,
                    'uid': s.st_uid, 'gid': s.st_gid,
                    'xattrs': {n: os.getxattr(p, n).hex() for n in os.listxattr(p)}}
    return out


def process_users(source):
    """Read /proc command lines and open descriptors; fail closed on matches."""
    hits = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            # Only the source owner can write these non-group/world-writable assets.
            if (p / 'status').stat().st_uid != source.stat().st_uid:
                continue
            command = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if command.strip() == '/usr/lib/systemd/systemd --user':
                continue  # User service manager is non-dumpable, not an asset worker.
            if str(source) in command or source.name in command:
                hits.append({'pid': p.name, 'command': command[:500]})
            for special in ('cwd', 'root'):
                special_target = str((p / special).readlink())
                if special_target == str(source) or special_target.startswith(str(source) + '/'):
                    hits.append({'pid': p.name, special: special_target})
            mapped = (p / 'maps').read_text()
            if str(source) in mapped:
                hits.append({'pid': p.name, 'maps': str(source)})
            for fd in (p / 'fd').iterdir():
                try:
                    target = str(fd.readlink())
                    if target == str(source) or target.startswith(str(source) + '/'):
                        hits.append({'pid': p.name, 'fd': target})
                except (OSError, PermissionError):
                    pass
        except PermissionError as exc:
            hits.append({'pid': p.name, 'unreadable': str(exc)})
        except FileNotFoundError:
            pass  # Process exited during inspection.
    return hits


def persist():
    temp = JOURNAL.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(JOURNAL)


def validate(source, target):
    require((source, target) in TASKS, 'Safety precondition failed')
    require(source.absolute() == source and source.parent.resolve().is_relative_to(HOME), 'Safety precondition failed')
    require(target.absolute() == target, 'Safety precondition failed')
    require(target.is_relative_to(ARCHIVE) or target.is_relative_to(CACHES), 'Safety precondition failed')
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(ancestor.resolve().is_relative_to(Path('/data/shangguanrz')), 'Safety precondition failed')
    require(source.exists() and not source.is_symlink(), source)
    require(not target.exists() and not target.is_symlink(), target)
    require(source.stat().st_dev != os.stat('/data').st_dev, 'Source already on /data')
    require(os.stat('/data').st_dev != os.stat('/').st_dev, '/data must be a separate filesystem')
    require(not process_users(source), f'Asset is in use: {source}')


require(ROOT.resolve() == ROOT and HOME.resolve() == HOME, 'Safety precondition failed')
require(not JOURNAL.exists(), 'Existing migration journal; inspect before resuming.')
require(shutil.which('rsync'), 'Safety precondition failed')
JOURNAL.parent.mkdir(parents=True, exist_ok=True)
record = {'started_at': time.time(), 'tasks': [],
          'disk_before': {n: shutil.disk_usage(n)._asdict() for n in ('/', '/data')}}
persist()
for source, target in TASKS:
    validate(source, target)
    probe = snapshot(source)
    require(not any(meta['mode'] & 0o002 for meta in probe.values()), 'World writable asset')
    for meta in probe.values():
        if meta['mode'] & 0o020:
            members = set(grp.getgrgid(meta['gid']).gr_mem) | {u.pw_name for u in pwd.getpwall() if u.pw_gid == meta['gid']}
            require(members == {pwd.getpwuid(os.getuid()).pw_name}, 'Shared writable group')
require(shutil.disk_usage('/data').free > sum(sum(m['size'] for m in snapshot(s).values()
    if m['type'] == 'file') for s, _ in TASKS) + 1024**3, 'Safety precondition failed')
for source, target in TASKS:
    validate(source, target)
    before = snapshot(source)
    expected = sum(v['size'] for v in before.values() if v['type'] == 'file')
    require(shutil.disk_usage('/data').free > expected + 1024**3, 'Safety precondition failed')
    task = {'source': str(source), 'target': str(target), 'state': 'copying',
            'bytes': expected, 'files': sum(v['type'] == 'file' for v in before.values()),
            'allocated_before': int(subprocess.check_output(['du', '-sx', '--block-size=1', str(source)]).split()[0])}
    record['tasks'].append(task)
    persist()
    target.parent.mkdir(parents=True, exist_ok=True)
    print('COPY', str(source), 'bytes', expected, flush=True)
    if source.is_dir():
        target.mkdir()
        subprocess.run(['rsync', '-aHAXS', '--numeric-ids', '--', str(source) + '/', str(target) + '/'], check=True)
    else:
        subprocess.run(['rsync', '-aHAXS', '--numeric-ids', '--', str(source), str(target)], check=True)
    require(snapshot(source) == before, 'Source changed while copying')
    after = snapshot(target)
    require(set(after) == set(before), 'Destination file list differs')
    for name, meta in before.items():
        got = after[name]
        # Directory allocation size varies between filesystems; ignore only that.
        fields = ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs', 'size') if meta['type'] == 'file' else ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs')
        require(all(meta[k] == got[k] for k in fields), (name, meta, got))
    task['state'] = 'hashing'
    persist()
    hashes = {}
    file_names = [n for n, m in before.items() if m['type'] == 'file']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, name in enumerate(file_names, 1):
            src = source if name == '.' else source / name
            dst = target if name == '.' else target / name
            a = pool.submit(digest, src)
            b = pool.submit(digest, dst)
            left, right = a.result(), b.result()
            require(left == right, f'SHA-256 mismatch: {name}')
            hashes[name] = left
            if i % 100 == 0 or src.stat().st_size > 1024**3 or i == len(file_names):
                print('SHA256', source.name, i, '/', len(file_names), name, flush=True)
    require(snapshot(source) == before, 'Source changed during verification')
    require(snapshot(target) == after, 'Destination changed during verification')
    require(not process_users(source), f'Asset became active: {source}')
    task['sha256'] = hashes
    task['metadata'] = before
    task['state'] = 'verified'
    persist()
    hold = source.with_name(source.name + '.cleanup-hold-20261003')
    task['hold'] = str(hold)
    require(hold.parent.resolve() == source.parent.resolve(), 'Safety precondition failed')
    require(not hold.exists() and not hold.is_symlink(), 'Safety precondition failed')
    source.rename(hold)
    created_inode = None
    try:
        source.symlink_to(target, target_is_directory=hold.is_dir())
        created_inode = source.lstat().st_ino
        require(source.resolve() == target.resolve() and source.exists(), 'Safety precondition failed')
        require(snapshot(hold) == before, 'Safety precondition failed')
    except BaseException:
        if source.is_symlink() and created_inode is not None and source.lstat().st_ino == created_inode:
            source.unlink()
        if not source.exists() and not source.is_symlink():
            hold.rename(source)
            task['state'] = 'restored_source'
        else:
            task['state'] = 'rollback_conflict_original_retained'
        persist()
        raise
    task['state'] = 'linked_with_original_retained'
    persist()
    # Retain source until a separate finalizer rechecks it after the entry tests.
    require(hold.absolute().is_relative_to(HOME) and not hold.is_symlink(), 'Safety precondition failed')
    require(hold.name == source.name + '.cleanup-hold-20261003', 'Safety precondition failed')
    require(snapshot(hold) == before and source.resolve() == target.resolve(), 'Safety precondition failed')
    task['state'] = 'linked_with_original_retained'
    task['finished_at'] = time.time()
    persist()
    print('COMPLETE', source, '->', target, flush=True)
record['disk_after'] = {n: shutil.disk_usage(n)._asdict() for n in ('/', '/data')}
record['finished_at'] = time.time()
persist()
print(json.dumps({'tasks': len(record['tasks']), 'root_freed_bytes':
                  record['disk_after']['/']['free'] - record['disk_before']['/']['free']}), flush=True)
