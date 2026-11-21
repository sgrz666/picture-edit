"""Verify the committed Bilibili 20k lineage, then explicitly initialize Native V7."""
import argparse
import json
from pathlib import Path
import sys

from prepare import read, ref, sha, write

NAME = 'bilibili-single-spatial-train-20261108-ungated-v1'
SOURCE = Path('/data/shangguanrz/pic-edit-experiments/one')/NAME/'stages/S3/checkpoints/update_00020000_f51df6cd2ace4a97b06ab037bd03b939.pt'
EXPECTED = 'd1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3'


def physical(path):
    value = str(path); old = '/data/shangguanrz/pic-edit-experiments/'+NAME+'/'
    if value.startswith(old): value = '/data/shangguanrz/pic-edit-experiments/one/'+NAME+'/'+value[len(old):]
    return Path(value)


def audit(source, root, repo):
    import torch
    sys.path.insert(0, str(repo))
    from src.pose_control.qwen21_full.native_v5.training.config import digest
    from src.pose_control.qwen21_full.native_v5.training.io import verify_prefix
    if sha(source) != EXPECTED: raise ValueError('Selected Bilibili 20k SHA differs')
    lineage = []; path = source; seen = set()
    while path is not None:
        if str(path) in seen: raise ValueError('Checkpoint lineage cycle')
        seen.add(str(path)); marker = read(path.with_suffix('.commit.json')); actual = ref(path)
        if actual['sha256'] != marker['checkpoint']['sha256']: raise ValueError('Committed checkpoint bytes differ')
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if saved['training_schema'] != 'native_v5_training_v1': raise ValueError('Unexpected source branch')
        state = saved['training_state']
        expected = digest({'config': state['config_sha256'], 'identity': state['identity_sha256'],
                           'catalog': state['catalog_sha256'], 'journal': state['journal']})
        if marker['training_state_sha256'] != expected or state['micro_position'] != 0:
            raise ValueError('Source training commit is incomplete')
        journal = {**state['journal'], 'path': str(physical(state['journal']['path']))}
        prefix = verify_prefix(journal)
        phase = state['config']['stage']['phase']
        lineage.append({'checkpoint': actual, 'original_commit': marker, 'phase': phase,
                        'global_update': state['global_update'], 'stage_update': state['stage_update'],
                        'budget': state['config']['stage']['budget_updates'], 'journal_verified': prefix,
                        'parent': state.get('parent'), 'gradient_report': state['gradient_report']})
        parent = state.get('parent'); path = None if parent is None else physical(parent['path'])
        if parent is not None and sha(path) != parent['sha256']: raise ValueError('Parent lineage SHA differs')
        del saved
    end = lineage[0]
    if (end['global_update'], end['phase'], end['stage_update'], end['budget']) != (20000, 'S3', 4000, 4000):
        raise ValueError('Requested 20k S3 endpoint was not completed')
    return write(root/'data/initialization-source.json', {'schema': 'xhs_bilibili20k_source_audit_v1',
        'selected': ref(source), 'passed': True, 'lineage': lineage,
        'relocation': 'old Bilibili root -> one/Bilibili; bytes and original commit remain unchanged',
        'use': 'user-selected explicit initialization, not optimizer resume or a V7-trained 20k claim'})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--audit-only', action='store_true'); a = p.parse_args()
    source_report = a.root/'data/initialization-source.json'
    if not source_report.exists(): audit(SOURCE, a.root, a.repo)
    elif read(source_report)['selected'] != ref(SOURCE): raise ValueError('Pinned initialization changed')
    if a.audit_only: print(json.dumps(read(source_report)['selected']), flush=True); return
    current = SOURCE; migrations = []
    for version, old in (('v6','v5'), ('v61','v6'), ('v62','v61'), ('v7','v62')):
        output = a.root/f'initialization/{version}.pt'; report = a.root/f'report/initialization/{version}.json'
        if not report.exists():
            import importlib
            import torch
            sys.path.insert(0,str(a.repo)); torch.manual_seed(42)
            module=importlib.import_module('scripts.data.migrate_qwen21_native_full_'+version)
            module.main(['--'+old,str(current),'--output',str(output),'--report',str(report)])
        record = read(report)
        if (record['source_sha256'] != sha(current) or record['output_sha256'] != sha(output)
                or not record['strict_readback'] or record['optimizer_inherited'] or record['training_started']):
            raise ValueError('Migration readback/inheritance audit failed: '+version)
        migrations.append({'version': version, 'checkpoint': ref(output), 'report': ref(report)})
        current = output
    write(a.root/'data/initialization.json', {'schema': 'xhs_bilibili20k_to_native_v7_v1', 'source': ref(SOURCE),
          'source_audit': ref(source_report), 'initialization': ref(current), 'migrations': migrations,
          'optimizer_inherited': False, 'migration_seed':42,'source_trained_updates': 20000, 'new_architecture_trained_updates': 0})
    print(json.dumps({'initialization': ref(current), 'source': ref(SOURCE)}), flush=True)


if __name__ == '__main__': main()
