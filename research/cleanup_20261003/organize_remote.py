"""Organize only the explicitly inventoried project entrypoints; run via ssh sg.

No Git index changes, commits, environment edits, or core module edits.
Original source bytes and metadata are retained in archive/cleanup_20261003.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess



def require(condition, message):
    if not condition:
        raise RuntimeError(message)

ROOT = Path('/home/shangguanrz/project/pic-edit')
ARCHIVE = ROOT / 'archive/cleanup_20261003'
JOURNAL = ARCHIVE / 'organization.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path, parent):
    require(path.absolute().is_relative_to(parent), (path, parent))
    require(path.parent.resolve().is_relative_to(parent.resolve()), path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def save():
    temporary = JOURNAL.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(JOURNAL)


require(ROOT.resolve() == ROOT and (ROOT / '.git/index').is_file(), 'Safety precondition failed')
require(not JOURNAL.exists(), 'Existing journal: inspect it before rerunning.')
ARCHIVE.mkdir(parents=True, exist_ok=True)
record = {'root': str(ROOT), 'head': git('rev-parse', 'HEAD').decode().strip(),
          'status_before': git('status', '--porcelain=v1', '-uall').decode(),
          'index_before_sha256': sha(ROOT / '.git/index'), 'moves': [], 'aliases': []}
require(record['head'] == 'b1a00c2a3b2955779af40ab19b5bab71ac19539a', 'Reviewed HEAD changed')
shutil.copy2(ROOT / '.git/index', ARCHIVE / 'git-index.before')
protected = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
protected += list(ROOT.glob('train_*.py')) + list(ROOT.glob('eval_*.py'))
protected += [ROOT / 'scripts/prepare_tiktok_qwen21_training.py',
              ROOT / 'scripts/prepare_tiktok_qwen21_face.py', ROOT / 'test_control_signals.py']
record['protected_before'] = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.exists()}
save()


def move_entry(source, category):
    """Copy/parse before replacing a source with a relative compatibility link."""
    inside(source, ROOT)
    target = ROOT / 'scripts' / category / source.name
    inside(target, ROOT)
    require(not target.exists() and not target.is_symlink(), target)
    require(not source.is_symlink(), source)
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    require(digest == preflight_hashes[str(source)], f'Changed after preflight: {source}')
    text = original.decode('utf-8')
    # Change only executable ROOT expressions, never strings containing source code.
    if source.suffix == '.py':
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    {'ROOT', 'PROJECT', 'PROJECT_ROOT', 'DEFAULT_PROJECT_ROOT'} for t in node.targets):
                segment = ast.get_source_segment(text, node) or ''
                if 'Path(__file__).resolve().parent' in segment:
                    require(node.lineno == node.end_lineno, segment)
                    idx = node.lineno - 1
                    lines[idx] = lines[idx].replace('Path(__file__).resolve().parents[1]',
                                                   'Path(__file__).resolve().parents[2]')
                    if 'Path(__file__).resolve().parent' in lines[idx] and '.parents[' not in lines[idx]:
                        lines[idx] = lines[idx].replace('Path(__file__).resolve().parent',
                                                       'Path(__file__).resolve().parents[2]')
        # Bootstrap import search paths for execution from outside the project cwd.
        insert_at = 0
        for node in tree.body:
            if ((isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                 and isinstance(node.value.value, str)) or
                (isinstance(node, ast.ImportFrom) and node.module == '__future__')):
                insert_at = node.end_lineno
            else:
                if insert_at == 0:
                    insert_at = node.lineno - 1
                break
        lines.insert(insert_at, '\n# Project root for the categorized entrypoint.\n'
                     'from pathlib import Path as _EntryPath\nimport sys as _entry_sys\n'
                     '_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))\n\n')
        text = ''.join(lines)
        ast.parse(text, filename=str(target))
    backup = ARCHIVE / 'before' / source.relative_to(ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    require(sha(backup) == digest, 'Safety precondition failed')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode('utf-8'))
    shutil.copystat(source, target)
    require(sha(source) == digest, f'Concurrent edit: {source}')
    item = {'source': str(source), 'target': str(target), 'backup': str(backup),
            'before_sha256': digest, 'after_sha256': sha(target), 'state': 'copied'}
    record['moves'].append(item)
    save()
    hold = ARCHIVE / 'holds' / source.relative_to(ROOT)
    hold.parent.mkdir(parents=True, exist_ok=True)
    require(not hold.exists() and not hold.is_symlink(), 'Safety precondition failed')
    item['hold'] = str(hold)
    save()
    source.rename(hold)
    created_inode = None
    try:
        source.symlink_to(os.path.relpath(target, source.parent))
        created_inode = source.lstat().st_ino
        require(source.resolve() == target and sha(source) == item['after_sha256'], 'Safety precondition failed')
        require(sha(hold) == digest, f'Concurrent edit retained in {hold}')
    except BaseException:
        if source.is_symlink() and created_inode is not None and source.lstat().st_ino == created_inode:
            source.unlink()
        if not source.exists() and not source.is_symlink():
            hold.rename(source)
            item['state'] = 'restored_source'
        else:
            item['state'] = 'rollback_conflict_original_retained'
        save()
        raise
    item['state'] = 'linked_with_original_retained'
    save()
    print('ENTRY', source.name, '->', target.relative_to(ROOT), flush=True)


def alias(source, category):
    target = ROOT / 'scripts' / category / source.name
    inside(target, ROOT)
    if target == source:
        return
    if target.exists() or target.is_symlink():
        require(target.is_symlink() and target.resolve() == source.resolve(), target)
        return
    require(source.is_file(), 'Safety precondition failed')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent))
    require(target.resolve() == source.resolve(), 'Safety precondition failed')
    record['aliases'].append({'source': str(source), 'target': str(target)})
    save()


root_helpers = {
    'data': ['build_tiktok_v3_candidate_pool.py', 'build_tiktok_v3_condition_inventory.py',
             'build_tiktok_v3_pairs.py', 'build_tiktok_v3_pairs_strict.py',
             'build_tiktok_v3_strict_visual_pool.py', 'build_tiktok_v3_usable_pool.py',
             'fill_tiktok_v3_missing_geometry.py', 'finalize_tiktok_v3_visual_quality_v1.py',
             'run_tiktok_v3_dwpose.py', 'sample_tiktok_v3_dwpose_qc.py',
             'sample_tiktok_v3_geometry_rejects.py', 'split_tiktok_v3_groupwise.py',
             'create_debug_samples.py'],
    'audit': ['audit_hi4d_qwen21_prompt_leak.py', 'audit_tiktok_v3_blur_sample.py',
              'audit_tiktok_v3_final_pairs.py', 'audit_tiktok_v3_pose_redundancy.py',
              'audit_tiktok_v3_strict_final.py', 'audit_tiktok_v3_visual_quality_v2.py',
              'audit_tiktok_v3_visual_quality_v3.py'],
    'reports': ['build_20b_memory_docx.py', 'build_architecture_docx.py',
                'build_brief_architecture_docx.py', 'build_latest_experiment_report_docx.py',
                'plot_vram_requirement_dashboard.py', 'visualize_tiktok_v3_final_global_blur.py'],
    'eval': ['multi_stage_eval_v6.py'],
}
# Existing staged/modified scripts and current TikTok integration are not rewritten.
status = git('status', '--porcelain=v1', '-z', '-uall').decode().split('\0')
dirty = {s[3:] for s in status if s and s[:2] not in ('??', '  ')}
pending = [(ROOT / name, category, False) for category, names in root_helpers.items() for name in names]
for source in sorted((ROOT / 'scripts').glob('*.py')):
    name = source.name
    if name.startswith(('precompute_', 'prepare_', 'split_', 'build_tiktok_')):
        category = 'data'
    elif name.startswith(('audit_', 'check_', 'verify_', 'probe_', 'smoke_', 'inventory_')):
        category = 'audit'
    elif name.startswith(('write_', 'build_')) or name == 'summarize_qwen21_face_groups.py':
        category = 'reports'
    elif name.startswith(('inherit_',)):
        category = 'weights'
    elif name.startswith(('eval_', 'evaluate_')):
        category = 'eval'
    elif name.startswith(('run_', 'finalize_')):
        category = 'run'
    else:
        raise AssertionError('Unclassified script: ' + name)
    pending.append((source, category, str(source.relative_to(ROOT)) in dirty or 'tiktok' in name))

# Full conflict/AST/root-expression preflight precedes the first source mutation.
preflight_hashes = {}
for source, category, preserve in pending:
    require(source.is_file() and not source.is_symlink(), source)
    target = ROOT / 'scripts' / category / source.name
    inside(source, ROOT)
    inside(target, ROOT)
    require(not target.exists() and not target.is_symlink(), target)
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            segment = ast.get_source_segment(source.read_text(encoding='utf-8'), node) or ''
            if '__file__' in segment and isinstance(node, ast.AnnAssign):
                raise RuntimeError('Unexpected annotated ROOT: ' + str(source))
    preflight_hashes[str(source)] = sha(source)
record['preflight_sha256'] = preflight_hashes
save()
for source, category, preserve in pending:
    if preserve:
        alias(source, category)
    else:
        move_entry(source, category)

for source in sorted(ROOT.glob('*.py')):
    if source.is_symlink():
        continue
    if source.name.startswith('train_'):
        alias(source, 'train')
    elif source.name.startswith('eval_'):
        alias(source, 'eval')
    elif source.name.startswith(('run_', 'complete_')):
        alias(source, 'run')
alias(ROOT / 'test_control_signals.py', 'data')
for source in ROOT.glob('*.sh'):
    alias(source, 'run')
record['index_after_sha256'] = sha(ROOT / '.git/index')
require(record['index_after_sha256'] == record['index_before_sha256'], 'Safety precondition failed')
record['protected_after'] = {name: sha(ROOT / name) for name in record['protected_before']}
record['protected_changed'] = [n for n in record['protected_before']
                               if record['protected_before'][n] != record['protected_after'][n]]
require(not record['protected_changed'], record['protected_changed'])
require(git('rev-parse', 'HEAD').decode().strip() == record['head'], 'HEAD changed')
record['status_after'] = git('status', '--porcelain=v1', '-uall').decode()
save()
print(json.dumps({'moves': len(record['moves']), 'aliases': len(record['aliases']),
                  'protected_changed': record['protected_changed'], 'index_unchanged': True}))
