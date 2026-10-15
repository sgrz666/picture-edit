"""Fresh entry/import/AST and regression verification, bound to current content."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/home/shangguanrz/project/pic-edit')
LOG = ROOT / 'archive/cleanup_20261003'
QWEN = '/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
DEEPGEN = '/home/shangguanrz/miniconda3/envs/deepgen/bin/python'
ENTRIES = [
    (QWEN, 'scripts/train/train_iper_qwen21_full.py'), (QWEN, 'train_iper_qwen21_full.py'),
    (QWEN, 'scripts/eval/eval_iper_qwen21_full.py'), (QWEN, 'scripts/data/prepare_qwen21_full_data.py'),
    (QWEN, 'scripts/weights/inherit_qwen21_full_weights.py'),
    (QWEN, 'scripts/audit/audit_hi4d_qwen21_prompt_leak.py'),
    (QWEN, 'scripts/audit/audit_qwen21_weight_compatibility.py'),
    (DEEPGEN, 'scripts/eval/multi_stage_eval_v6.py'), (DEEPGEN, 'multi_stage_eval_v6.py'),
    (DEEPGEN, 'scripts/data/precompute_iper_face_features_v65.py'),
    (DEEPGEN, 'scripts/audit/smoke_adapter_v6.py'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    asset_phase = json.loads((LOG / 'assets.json').read_text())
    if len(asset_phase['tasks']) != 7 or any(t['state'] != 'linked_with_original_retained' for t in asset_phase['tasks']):
        raise RuntimeError('All seven assets must have completed stage one')
    started = time.time()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    index = sha(ROOT / '.git/index')
    organization = json.loads((LOG / 'organization.json').read_text())
    protected_paths = set(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
                          + list(ROOT.glob('train_*.py')) + list(ROOT.glob('eval_*.py'))
                          + [ROOT / n for n in organization['protected_before']])
    protected_files = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    source_paths = set(protected_files) | {str(Path(t['target']).relative_to(ROOT)) for t in organization['moves']}
    bindings = {n: sha(ROOT / n) for n in source_paths}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    entries = []
    for py, name in ENTRIES:
        run = subprocess.run([py, str(ROOT / name), '--help'], cwd='/tmp', env=env, capture_output=True, text=True, timeout=120)
        entries.append({'id': name, 'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
        print('HELP', name, run.returncode, flush=True)
    code = ('import scripts.inherit_qwen21_full_weights as w; import scripts.audit_qwen21_weight_compatibility as a; '
            'import scripts.precompute_iper_face_features_v65 as f; import create_debug_samples as c; '
            'print(w.ROOT,a.ROOT,f.PROJECT_ROOT)')
    run = subprocess.run([QWEN, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    imports = {'id': 'compatibility_imports', 'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
    ast_files = {}
    for p in sorted((ROOT / 'scripts').rglob('*.py')):
        ast.parse(p.read_text(), filename=str(p))
        ast_files[str(p.relative_to(ROOT))] = sha(p)
    tests = sorted(set(list((ROOT / 'tests').glob('test_*qwen*.py')) + list((ROOT / 'tests').glob('test_*tiktok*.py')) + list((ROOT / 'tests').glob('test_hi4d*.py'))))
    command = [QWEN, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *[str(p.relative_to(ROOT)) for p in tests]]
    test_bindings = {str(p.relative_to(ROOT)): sha(p) for p in tests}
    test_run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    matched = re.search(r'(\d+) passed', test_run.stdout)
    pytest = {'exit_code': test_run.returncode, 'passed_count': int(matched.group(1)) if matched else 0,
              'command': command, 'test_files_sha256': test_bindings, 'stdout': test_run.stdout, 'stderr': test_run.stderr}
    if any(sha(ROOT / n) != value for n, value in {**bindings, **test_bindings, **ast_files}.items()):
        raise RuntimeError('Source changed during verification')
    current_protected = set(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
                           + list(ROOT.glob('train_*.py')) + list(ROOT.glob('eval_*.py'))
                           + [ROOT / n for n in organization['protected_before']])
    if current_protected != protected_paths or any(sha(ROOT / n) != s for n, s in protected_files.items()):
        raise RuntimeError('Protected source set changed during verification')
    if sha(ROOT / '.git/index') != index or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head:
        raise RuntimeError('Git state changed during verification')
    passed = (all(r['exit_code'] == 0 for r in entries) and imports['exit_code'] == 0
              and test_run.returncode == 0 and pytest['passed_count'] >= 177)
    result = {'passed': passed, 'git_head': head, 'index_sha256': index, 'started_at': started,
              'finished_at': time.time(), 'asset_phase_finished_at': asset_phase['finished_at'],
              'source_bindings': bindings, 'entries': entries, 'imports': imports,
              'protected_files_sha256': protected_files,
              'ast_files_sha256': ast_files, 'ast_count': len(ast_files), 'pytest': pytest}
    temporary = LOG / 'verification_pre_finalize.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(LOG / 'verification_pre_finalize.json')
    print(json.dumps({'passed': passed, 'cli_count': len(entries), 'ast_count': len(ast_files), 'pytest_passed': pytest['passed_count']}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
