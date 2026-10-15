"""Archive small evidence and identical dated backups, then clear own bytecode."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/home/shangguanrz/project/pic-edit')
ARCHIVE = ROOT / 'archive/cleanup_20261003'
JOURNAL = ARCHIVE / 'small_operations.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    temporary = JOURNAL.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(JOURNAL)


record = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'moves': [], 'backup_groups': [], 'cache_removed': []}
save()
for relative, destination in [('demooutput', 'results/legacy_demo_202609'),
                               ('inference.log', 'logs/legacy/inference_202609.log')]:
    source, target = ROOT / relative, ROOT / destination
    if any(m['source'] == str(source) and m['state'] == 'completed' for m in record['moves']):
        require(source.is_symlink() and source.resolve() == target, 'Existing move differs')
        continue
    require(source.exists() and not source.is_symlink(), str(source))
    require(not target.exists() and not target.is_symlink(), str(target))
    files = [source] if source.is_file() else list(source.rglob('*'))
    require(not any(p.is_symlink() for p in files), 'Unexpected symlink in evidence')
    hashes = {'.' if p == source else str(p.relative_to(source)): sha(p) for p in files if p.is_file()}
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {'source': str(source), 'target': str(target), 'sha256': hashes, 'state': 'prepared'}
    record['moves'].append(item); save()
    source.rename(target)
    try:
        source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=target.is_dir())
        require(source.resolve() == target, 'Compatibility link mismatch')
        for name, digest in hashes.items():
            require(sha(target if name == '.' else target / name) == digest, name)
    except BaseException:
        if source.is_symlink():
            source.unlink()
        target.rename(source)
        raise
    item['state'] = 'completed'; save()

groups = [
    ['add/datasets/dpw_single_pose_edit_dataset_before_use_dwpose_render_20260930_090016.py',
     'add/datasets/dpw_single_pose_edit_dataset_before_dwpose_qc_mapping_20260930_090438.py'],
    ['add/tools/visualize_dpw_single_pose_edit_dataset.py',
     'add/tools/visualize_dpw_single_pose_edit_dataset_backup_20260930_083758.py',
     'add/tools/visualize_dpw_single_pose_edit_dataset_before_black_pose_20260930_084339.py'],
    ['add/tools/visualize_dpw_single_pose_edit_dataset_before_color_pose_20260930_084629.py',
     'add/tools/visualize_dpw_single_pose_edit_dataset_before_dwpose_color_v2_20260930_084922.py',
     'add/tools/visualize_dpw_single_pose_edit_dataset_before_safe_color_pose_20260930_085112.py'],
]
search = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').rglob('*.py'))
search += list((ROOT / 'tests').rglob('*.py')) + list((ROOT / 'add/tools').glob('*.py'))
search += list((ROOT / 'add/datasets').glob('*.py')) + list(ROOT.glob('*.py'))
for i, names in enumerate(groups, 1):
    paths = [ROOT / n for n in names]
    originals = [p if p.exists() else ARCHIVE / 'backup_holds' / p.relative_to(ROOT) for p in paths]
    require(all(p.is_file() and not p.is_symlink() for p in originals), names)
    digests = [sha(p) for p in originals]
    require(len(set(digests)) == 1, 'Backup group content differs')
    target = ARCHIVE / 'add_backups' / ('group_' + str(i)) / paths[0].name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(originals[0], target)
    require(sha(target) == digests[0], 'Archive verification failed')
    group = next((g for g in record['backup_groups'] if g['archive'] == str(target)), None)
    if group is None:
        group = {'original_paths': names, 'archive': str(target), 'sha256': digests[0], 'retained_holds': []}
        record['backup_groups'].append(group); save()
    for source in paths:
        if source.name == 'visualize_dpw_single_pose_edit_dataset.py':
            continue  # The current production helper remains untouched.
        hold = ARCHIVE / 'backup_holds' / source.relative_to(ROOT)
        if hold.exists() and not source.exists():
            require(sha(hold) == group['sha256'], 'Existing hold changed')
            if str(hold) not in group['retained_holds']:
                group['retained_holds'].append(str(hold)); save()
            continue
        refs = [str(p) for p in search if p not in paths and p.is_file() and source.name in p.read_text(encoding='utf-8', errors='replace')]
        require(not refs, ('Referenced backup', str(source), refs))
        require(not hold.exists(), str(hold))
        hold.parent.mkdir(parents=True, exist_ok=True)
        require(sha(source) == digests[0], 'Concurrent backup edit')
        source.rename(hold)
        require(sha(hold) == digests[0], 'Concurrent backup edit retained in hold')
        group['retained_holds'].append(str(hold)); save()

# Clear only self-owned code caches, never environment, third-party or run trees.
for base in [ROOT / '__pycache__', ROOT / '.pytest_cache', ROOT / 'src',
             ROOT / 'tests', ROOT / 'scripts', ROOT / 'vram_lab']:
    candidates = [base] if base.name in ('__pycache__', '.pytest_cache') else list(base.rglob('__pycache__'))
    for cache in candidates:
        if not cache.exists():
            continue
        require(cache.absolute().is_relative_to(ROOT) and not cache.is_symlink(), str(cache))
        require(cache.name in ('__pycache__', '.pytest_cache'), str(cache))
        require(not any(p.is_symlink() for p in cache.rglob('*')), str(cache))
        record['cache_removed'].append({'path': str(cache), 'bytes': sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())})
        save()
        shutil.rmtree(cache)
save()
print(json.dumps({'evidence_moves': len(record['moves']), 'backup_groups': len(record['backup_groups']),
                  'backups_archived': sum(len(g['retained_holds']) for g in record['backup_groups']),
                  'cache_bytes_removed': sum(p['bytes'] for p in record['cache_removed'])}))
