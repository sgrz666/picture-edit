"""Second-stage source retirement after independent verification and entry tests.

Default is read-only. --apply retires only the fixed, journaled source holds.
All archive destinations and pre-migration source backups remain available.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/shangguanrz/project/pic-edit')
HOME = Path('/home/shangguanrz')
LOG = ROOT / 'archive/cleanup_20261003'
ARCHIVE = Path('/data/shangguanrz/pic-edit-archive/cleanup_20261003')
CACHES = Path('/data/shangguanrz/pic-edit-caches')
ALLOWED = {
    ROOT / 'model-qwen-edit': ARCHIVE / 'models/Qwen-Image-Edit-20B',
    ROOT / 'experiments/review_20260908_v2': ARCHIVE / 'experiments/review_20260908_v2',
    ROOT / 'experiments/iper_v6_comparative_3000': ARCHIVE / 'experiments/iper_v6_comparative_3000',
    HOME / 'nihaochen/qwen21_hi4d_controls': CACHES / 'qwen21_hi4d_controls',
    **{ROOT / 'add' / n: ARCHIVE / 'raw/3dpw' / n
       for n in ('imageFiles.zip', 'sequenceFiles.zip', 'camera_tram.zip')},
}
EXPECTED_ENTRIES = {'scripts/train/train_iper_qwen21_full.py', 'train_iper_qwen21_full.py',
    'scripts/eval/eval_iper_qwen21_full.py', 'scripts/data/prepare_qwen21_full_data.py',
    'scripts/weights/inherit_qwen21_full_weights.py', 'scripts/audit/audit_hi4d_qwen21_prompt_leak.py',
    'scripts/audit/audit_qwen21_weight_compatibility.py', 'scripts/eval/multi_stage_eval_v6.py',
    'multi_stage_eval_v6.py', 'scripts/data/precompute_iper_face_features_v65.py',
    'scripts/audit/smoke_adapter_v6.py'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
        require(not p.is_symlink(), 'Unexpected link: ' + str(p))
        s = p.lstat()
        require(p.is_file() or p.is_dir(), 'Unsupported file type: ' + str(p))
        out['.' if p == path else str(p.relative_to(path))] = {
            'type': 'file' if p.is_file() else 'dir', 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'mode': s.st_mode & 0o7777,
            'uid': s.st_uid, 'gid': s.st_gid,
            'xattrs': {n: os.getxattr(p, n).hex() for n in os.listxattr(p)}}
    return out


def idle(paths):
    """Check all same-owner command lines, descriptors, cwd/root and mappings."""
    prefixes = [str(p) for p in paths]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if (proc / 'status').stat().st_uid != os.getuid():
                continue
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if cmd.strip() == '/usr/lib/systemd/systemd --user':
                continue
            values = [cmd, (proc / 'maps').read_text(), str((proc / 'cwd').readlink()), str((proc / 'root').readlink())]
            for fd in (proc / 'fd').iterdir():
                try:
                    values.append(str(fd.readlink()))
                except FileNotFoundError:
                    pass
            for prefix in prefixes:
                require(not any(prefix in value for value in values), f'Asset in use by PID {proc.name}: {prefix}')
        except FileNotFoundError:
            continue  # Process exited during inspection.
        except PermissionError as exc:
            raise RuntimeError(f'Cannot inspect same-owner process {proc.name}') from exc


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def verify_hashes(hold, target, expected):
    before_hold, before_target = snapshot(hold), snapshot(target)
    require(set(before_hold) == set(before_target), 'File list mismatch')
    file_names = [n for n, m in before_hold.items() if m['type'] == 'file']
    require(set(file_names) == set(expected), 'Unexpected/missing regular file')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, name in enumerate(file_names, 1):
            left_path = hold if name == '.' else hold / name
            right_path = target if name == '.' else target / name
            left = pool.submit(digest, left_path)
            right = pool.submit(digest, right_path)
            require(left.result() == expected[name] and right.result() == expected[name], 'Hash changed: ' + name)
            if i % 300 == 0 or left_path.stat().st_size > 1024**3 or i == len(file_names):
                print('REVERIFY', hold.name, i, '/', len(file_names), flush=True)
    require(snapshot(hold) == before_hold and snapshot(target) == before_target, 'Content metadata changed while hashing')
    return before_hold, before_target


def verify_destination(target, task):
    before = snapshot(target)
    require(set(before) == set(task['metadata']), 'Destination file list changed')
    with ThreadPoolExecutor(max_workers=2) as pool:
        names = list(task['sha256'])
        paths = [target if n == '.' else target / n for n in names]
        for name, got in zip(names, pool.map(digest, paths)):
            require(got == task['sha256'][name], 'Destination hash mismatch: ' + name)
    require(snapshot(target) == before, 'Destination changed during recovery')
    for name, meta in task['metadata'].items():
        fields = ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs', 'size') if meta['type'] == 'file' else ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs')
        require(all(meta[k] == before[name][k] for k in fields), 'Destination metadata mismatch: ' + name)


def validate_remaining(remaining, expected):
    require(set(remaining) <= set(expected), 'Unexpected file in partial hold')
    for name, meta in remaining.items():
        fields = ('type', 'size', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs') if meta['type'] == 'file' else ('type', 'mode', 'uid', 'gid', 'xattrs')
        require(all(meta[k] == expected[name][k] for k in fields), 'Partial hold metadata changed: ' + name)


def verify_coverage(historical, evidence, organization):
    require(set(historical['ast_files_sha256']) <= set(evidence['ast_files_sha256']), 'Historical script coverage shrank')
    require(set(historical['pytest']['test_files_sha256']) <= set(evidence['pytest']['test_files_sha256']), 'Historical regression scope shrank')
    require(set(organization['protected_before']) <= set(evidence['protected_files_sha256']), 'Historical protected file removed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assets = json.loads((LOG / 'assets.json').read_text())
    organization = json.loads((LOG / 'organization.json').read_text())
    small = json.loads((LOG / 'small_operations.json').read_text())
    require(ROOT.resolve() == ROOT and LOG.resolve() == LOG, 'Unexpected project/archive root')
    require({Path(t['source']): Path(t['target']) for t in assets['tasks']} == ALLOWED, 'Asset allowlist differs')
    require(digest(ROOT / '.git/index') == organization['index_before_sha256'], 'Git index changed')
    evidence = json.loads((LOG / 'verification_pre_finalize.json').read_text())
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == organization['head'], 'HEAD changed')
    require(evidence['git_head'] == organization['head'] and evidence['index_sha256'] == organization['index_before_sha256'], 'Evidence Git binding differs')
    require(evidence['started_at'] > assets['finished_at'] and evidence['finished_at'] >= evidence['started_at'], 'Evidence is older than asset stage one')
    require(len(evidence['entries']) == 11 and {t['id'] for t in evidence['entries']} == EXPECTED_ENTRIES,
            'CLI evidence missing or unexpected')
    require(evidence['imports']['id'] == 'compatibility_imports' and evidence['imports']['exit_code'] == 0, 'Import evidence failed')
    require(evidence['ast_count'] >= 131 and len(evidence['ast_files_sha256']) == evidence['ast_count'], 'AST evidence incomplete')
    tests = evidence['pytest']
    verify_coverage(json.loads((LOG / 'verification_pre_finalize_early.json').read_text()), evidence, organization)
    require(tests['command'][:6] == ['/home/shangguanrz/nihaochen/envs/qwen21/bin/python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], 'Unexpected regression command')
    require(tests['passed_count'] >= 177 and tests['exit_code'] == 0 and evidence['passed']
            and all(t['exit_code'] == 0 for t in evidence['entries']), 'Entry/regression checks have not passed')
    current_protected = set(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
                            + list(ROOT.glob('train_*.py')) + list(ROOT.glob('eval_*.py'))
                            + [ROOT / n for n in organization['protected_before']])
    require({str(p.relative_to(ROOT)) for p in current_protected} == set(evidence['protected_files_sha256']), 'Protected file set changed after verification')
    require({str(p.relative_to(ROOT)) for p in (ROOT / 'scripts').rglob('*.py')} == set(evidence['ast_files_sha256']), 'Script file set changed after verification')
    current_tests = set(list((ROOT / 'tests').glob('test_*qwen*.py')) + list((ROOT / 'tests').glob('test_*tiktok*.py')) + list((ROOT / 'tests').glob('test_hi4d*.py')))
    require({str(p.relative_to(ROOT)) for p in current_tests} == set(tests['test_files_sha256']), 'Selected test set changed after verification')
    require(set(evidence['protected_files_sha256']) <= set(evidence['source_bindings']), 'Protected hashes are not covered by runtime bindings')
    for name, expected in {**evidence['source_bindings'], **evidence['protected_files_sha256'], **evidence['ast_files_sha256'], **tests['test_files_sha256']}.items():
        require(digest(ROOT / name) == expected, 'Evidence content binding changed: ' + name)
    for item in organization['moves']:
        relative = str(Path(item['target']).relative_to(ROOT))
        require(digest(Path(item['target'])) == evidence['source_bindings'][relative], 'Moved source changed after fresh tests')
        require(digest(Path(item['backup'])) == item['before_sha256'], 'Rollback backup changed')
    for name, expected in evidence['protected_files_sha256'].items():
        require(digest(ROOT / name) == expected, 'Protected code changed since tests: ' + name)
    final_path = LOG / 'finalization.json'
    if args.apply and final_path.exists():
        output = json.loads(final_path.read_text())
        require('finished_at' not in output, 'Finalization already completed; do not overwrite its evidence')
    else:
        output = {'apply': args.apply, 'started_at': time.time(), 'disk_before': shutil.disk_usage('/')._asdict(), 'retired': [], 'small_intents': []}
    if args.apply:
        atomic_json(final_path, output)
    for task in assets['tasks']:
        source, target, hold = (Path(task[n]) for n in ('source', 'target', 'hold'))
        require(target == ALLOWED[source], 'Unexpected archive target')
        require(hold == source.with_name(source.name + '.cleanup-hold-20261003'), 'Unexpected source hold')
        require(hold.parent.resolve() == source.parent.resolve() and hold.parent.resolve().is_relative_to(HOME), 'Hold parent moved')
        require(target.parent.resolve().is_relative_to(Path('/data/shangguanrz')), 'Target escaped /data')
        require(source.is_symlink() and source.resolve() == target.resolve(), 'Source compatibility link changed')
        if task['state'] == 'completed':
            require(not hold.exists() and not hold.is_symlink(), 'Completed task still has hold')
            continue
        if task['state'] == 'retiring':
            # A prior process may have stopped during rmtree or after unlink.
            # Validate the complete destination before accepting any missing hold.
            verify_destination(target, task)
            if hold.exists():
                require(not hold.is_symlink(), 'Recovery hold is a symlink')
                remaining = snapshot(hold)
                validate_remaining(remaining, task['metadata'])
                for name, meta in remaining.items():
                    if meta['type'] == 'file':
                        require(name in task['sha256'] and digest(hold if name == '.' else hold / name) == task['sha256'][name], 'Partial hold content changed')
                idle([hold, source, target])
                if args.apply:
                    shutil.rmtree(hold) if hold.is_dir() else hold.unlink()
            if args.apply:
                task['state'] = 'completed'
                task['retired_at'] = time.time()
                task['recovered_retirement'] = True
                atomic_json(LOG / 'assets.json', assets)
                if str(hold) not in output['retired']:
                    output['retired'].append(str(hold))
                atomic_json(final_path, output)
            continue
        require(task['state'] == 'linked_with_original_retained' and hold.exists() and not hold.is_symlink(), 'Stage one incomplete')
        require(time.time() - task['finished_at'] >= 30, 'Source hold has not cooled for 30 seconds')
        idle([hold, source, target])
        before_hold, before_target = verify_hashes(hold, target, task['sha256'])
        require(before_hold == task['metadata'], 'Original metadata differs')
        for name, meta in before_hold.items():
            fields = ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs', 'size') if meta['type'] == 'file' else ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs')
            require(all(meta[k] == before_target[name][k] for k in fields), 'Target metadata differs: ' + name)
        # New opens through source already resolve to /data; inspect retained inode
        # descriptors again after hashing. No writer to the retired tree is allowed.
        idle([hold, source, target])
        require(snapshot(hold) == before_hold and snapshot(target) == before_target, 'Tree changed before retirement')
        require(source.is_symlink() and source.resolve() == target.resolve(), 'Source link changed before retirement')
        if args.apply:
            task['state'] = 'retiring'
            atomic_json(LOG / 'assets.json', assets)
            if hold.is_dir():
                shutil.rmtree(hold)
            else:
                hold.unlink()
            task['state'] = 'completed'
            task['retired_at'] = time.time()
            atomic_json(LOG / 'assets.json', assets)
            output['retired'].append(str(hold))
            atomic_json(final_path, output)
            print('RETIRED', hold, flush=True)
    # Retire only duplicate small-file inodes; durable original-byte backups stay.
    small_holds = [(Path(m['hold']), Path(m['backup']), m['before_sha256']) for m in organization['moves']]
    small_holds += [(Path(p), Path(g['archive']), g['sha256']) for g in small['backup_groups'] for p in g['retained_holds']]
    for hold, backup, expected in small_holds:
        require(hold.absolute().is_relative_to(LOG / 'holds') or hold.absolute().is_relative_to(LOG / 'backup_holds'), 'Unexpected small-file hold')
        require(hold.parent.resolve().is_relative_to(LOG), 'Small hold escaped archive')
        if not hold.exists():
            if args.apply and str(hold) in output['small_intents'] and str(hold) not in output['retired']:
                require(digest(backup) == expected, 'Recovery backup changed')
                output['retired'].append(str(hold))
                atomic_json(final_path, output)
            continue
        require(not hold.is_symlink() and hold.is_file(), 'Unexpected small-file type')
        require(digest(hold) == expected and digest(backup) == expected, 'Small hold/backup changed')
        idle([hold])
        if args.apply:
            output['small_intents'].append(str(hold))
            atomic_json(final_path, output)
            hold.unlink()
            output['retired'].append(str(hold))
            atomic_json(final_path, output)
    output['disk_after'] = shutil.disk_usage('/')._asdict()
    output['finished_at'] = time.time()
    if args.apply:
        atomic_json(final_path, output)
    print(json.dumps({'passed': True, 'apply': args.apply, 'asset_count': len(ALLOWED), 'small_hold_count': len(small_holds),
                     'retired': len(output['retired']), 'root_freed_bytes': output['disk_after']['free'] - output['disk_before']['free']}), flush=True)


if __name__ == '__main__':
    main()
