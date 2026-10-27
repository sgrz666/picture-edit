"""Record concurrent workspace drift without approving or reverting any edits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[2]
TROOT = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
HROOT = Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def constructors(data):
    tree = ast.parse(data)
    return {node.name: sha(ast.dump(init, include_attributes=False).encode())
            for node in tree.body if isinstance(node, ast.ClassDef)
            for init in node.body if isinstance(init, (ast.FunctionDef, ast.AsyncFunctionDef)) and init.name == '__init__'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a new immutable drift report')
    if not a.output.resolve().is_relative_to('/data/shangguanrz'):
        raise ValueError('Persistent data volume required')
    ref = TROOT / 'source/delivery_20261026_v1'
    old = json.loads((ref / 'manifest.json').read_text())
    changed = {}
    with tarfile.open(ref / 'source.tar.gz') as tar:
        for name, record in old['files'].items():
            path = REPO / name
            data = path.read_bytes()
            if sha(data) != record['sha256']:
                entry = {'archived_sha256': record['sha256'], 'current_sha256': sha(data),
                         'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                         'protected_neural_file': name in old['protected_neural_modules']}
                if name.endswith('.py'):
                    original = tar.extractfile(name).read()
                    entry['module_constructor_asts_unchanged'] = constructors(original) == constructors(data)
                changed[name] = entry
    fingerprints = {}
    for kind, root in (('tiktok', TROOT), ('hi4d', HROOT)):
        rows = []
        for path in (root / 'training/segments').glob('*/resolved.json'):
            value = json.loads(path.read_text())
            rows.append({'path': str(path), 'config_stage': value['config']['stage']['phase'],
                         'recorded_code': value['identity']['code']})
        fingerprints[kind] = rows
    a.output.mkdir(parents=True)
    # Capture the current state separately. It is deliberately NOT labelled as
    # the exact executed training source or as a passed immutable-core audit.
    files = {str(p.relative_to(REPO)) for p in (REPO / 'src/pose_control/qwen21_full').rglob('*.py')}
    files.update(old['files'])
    files.update(str(p.relative_to(REPO)) for p in (REPO / 'scripts/reports').glob('*fixedpair*') if p.is_file())
    files.add('scripts/audit/audit_qwen21_fixedpair_workspace_drift.py')
    files.add('docs/Qwen21_V5固定样本结果报告工具_20261026.md')
    records = {}
    archive = a.output / 'workspace_after_experiment.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in sorted(files):
            data = (REPO / name).read_bytes()
            if name.endswith('.py'):
                ast.parse(data)
            records[name] = {'sha256': sha(data), 'bytes': len(data)}
            tar.add(REPO / name, arcname=name, recursive=False)
    result = {'schema': 'native_v5_fixedpair_workspace_drift_v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'reference_snapshot': str(ref), 'reference_archive_sha256': old['archive']['sha256'],
              'strict_unchanged_core_audit_passed': not any(r['protected_neural_file'] for r in changed.values()),
              'changed_files': changed, 'recorded_stage_code_fingerprints': fingerprints,
              'current_workspace_archive': {'path': str(archive), 'sha256': sha(archive.read_bytes())},
              'current_workspace_files': records,
              'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'training_source_replaced_or_edits_reverted': False,
              'weights_or_media_archived': False,
              'limitation': 'Disk source changed during the long-lived process. File fingerprints alone do not attest to the exact Python functions already loaded in memory. Preserve the earlier snapshot and do not claim current-worktree identity or bitwise replay.'}
    (a.output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'changed': len(changed), 'changed_protected': sum(r['protected_neural_file'] for r in changed.values()),
                      'report': str(a.output / 'manifest.json'), 'archive': result['current_workspace_archive']}))


if __name__ == '__main__':
    main()
