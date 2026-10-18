#!/usr/bin/env python3
"""Read-only validation of the 2026-10-03 project organization journals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise OSError('File changed while hashing: ' + str(path))
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--full-assets', action='store_true', help='Read and hash every archived asset')
    parser.add_argument('--strict-source-baseline', action='store_true', help='Require source/Git bytes to equal the historical cleanup baseline')
    args = parser.parse_args()
    root = args.root.resolve()
    journal_root = root / 'archive/cleanup_20261003'
    organization = json.loads((journal_root / 'organization.json').read_text())
    problems = []
    later_source_changes = []
    def check(condition, message):
        if not condition:
            problems.append(message)
    def checked_digest(path):
        try:
            return digest(path)
        except OSError as exc:
            problems.append(str(exc))
            return None
    home_root = Path('/home/shangguanrz')
    archive_root = Path('/data/shangguanrz/pic-edit-archive/cleanup_20261003')
    expected_assets = {
        (str(root / 'model-qwen-edit'), str(archive_root / 'models/Qwen-Image-Edit-20B')),
        (str(root / 'experiments/review_20260908_v2'), str(archive_root / 'experiments/review_20260908_v2')),
        (str(root / 'experiments/iper_v6_comparative_3000'), str(archive_root / 'experiments/iper_v6_comparative_3000')),
        (str(home_root / 'nihaochen/qwen21_hi4d_controls'), '/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls'),
        *{(str(root / 'add' / name), str(archive_root / 'raw/3dpw' / name))
          for name in ('imageFiles.zip', 'sequenceFiles.zip', 'camera_tram.zip')},
    }
    expected_navigation = {
        str(root / 'reports/qwen21'): str(home_root / 'nihaochen/docs'),
        str(root / 'reports/legacy_demo_202609'): str(root / 'results/legacy_demo_202609'),
        str(root / 'models/Qwen-Image-Edit-20B'): str(root / 'model-qwen-edit'),
        str(root / 'caches/qwen21_hi4d_controls'): '/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls',
    }
    for key, count_expected in (('moves', 62), ('aliases', 31)):
        items = organization[key]
        check(len(items) == count_expected and len({item['source'] for item in items}) == count_expected
              and len({item['target'] for item in items}) == count_expected, 'Organization list incomplete: ' + key)
    head_unchanged = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == organization['head']
    index_unchanged = digest(root / '.git/index') == organization['index_before_sha256']
    if args.strict_source_baseline:
        check(head_unchanged, 'HEAD changed since organization')
        check(index_unchanged, 'Git index changed since organization')
    for item in organization['moves']:
        source, target, backup = (Path(item[n]) for n in ('source', 'target', 'backup'))
        check(source.is_symlink() and source.resolve() == target, 'Entry link differs: ' + str(source))
        check(target.is_file(), 'Moved entry missing: ' + str(target))
        if target.is_file() and checked_digest(target) != item['after_sha256']:
            later_source_changes.append(str(target))
        check(backup.is_file() and checked_digest(backup) == item['before_sha256'], 'Rollback backup differs: ' + str(backup))
    for item in organization['aliases']:
        check(Path(item['target']).is_symlink() and Path(item['target']).resolve() == Path(item['source']).resolve(), 'Classified alias differs: ' + item['target'])
    for name, expected in organization['protected_before'].items():
        check((root / name).is_file(), 'Protected source missing: ' + name)
        if (root / name).is_file() and checked_digest(root / name) != expected:
            later_source_changes.append(name)
    pending = []
    count = 0
    retired_expected = [item['hold'] for item in organization['moves']]
    asset_path = journal_root / 'assets.json'
    if asset_path.exists():
        assets = json.loads(asset_path.read_text())
        check(len(assets['tasks']) == 7 and {(item['source'], item['target']) for item in assets['tasks']} == expected_assets,
              'Fixed seven-asset allowlist differs')
        for item in assets['tasks']:
            retired_expected.append(item['hold'])
            source, target = Path(item['source']), Path(item['target'])
            check(source.is_symlink() and source.resolve() == target.resolve() and target.exists(), 'Asset link differs: ' + str(source))
            if item['state'] != 'completed':
                pending.append({'source': str(source), 'state': item['state']})
            else:
                hold = Path(item['hold'])
                check(not hold.exists() and not hold.is_symlink(), 'Retired original still present: ' + str(hold))
            if args.full_assets and 'sha256' in item:
                actual_entries = [target] if target.is_file() else [target, *target.rglob('*')]
                names_found = {'.' if path == target else str(path.relative_to(target)) for path in actual_entries}
                check(names_found == set(item['metadata']), 'Asset file/directory list differs: ' + str(target))
                for path in actual_entries:
                    name = '.' if path == target else str(path.relative_to(target))
                    check(not path.is_symlink(), 'Unexpected internal asset link: ' + str(path))
                    expected = item['metadata'].get(name)
                    if expected is None:
                        continue
                    info = path.lstat()
                    got = {'type': 'file' if path.is_file() else 'dir', 'size': info.st_size,
                           'mtime_ns': info.st_mtime_ns, 'mode': info.st_mode & 0o7777,
                           'uid': info.st_uid, 'gid': info.st_gid,
                           'xattrs': {key: os.getxattr(path, key).hex() for key in os.listxattr(path)}}
                    fields = ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs', 'size') if expected['type'] == 'file' else ('type', 'mtime_ns', 'mode', 'uid', 'gid', 'xattrs')
                    check(all(got[key] == expected[key] for key in fields), 'Asset metadata differs: ' + str(path))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    names = list(item['sha256'])
                    paths = [target if name == '.' else target / name for name in names]
                    for name, got in zip(names, pool.map(checked_digest, paths)):
                        check(got == item['sha256'][name], 'Asset hash differs: ' + str(target if name == '.' else target / name))
                        count += 1
            elif 'sha256' in item:
                for name in item['sha256']:
                    check((target if name == '.' else target / name).is_file(), 'Missing asset file: ' + name)
    small_path = journal_root / 'small_operations.json'
    if small_path.exists():
        small = json.loads(small_path.read_text())
        check(len(small['moves']) == 2 and len(small['backup_groups']) == 3, 'Small-artifact journal incomplete')
        check(sum(len(group['retained_holds']) for group in small['backup_groups']) == 7, 'Dated backup hold list incomplete')
        for item in small['moves']:
            source, target = Path(item['source']), Path(item['target'])
            check(source.is_symlink() and source.resolve() == target, 'Evidence link differs: ' + str(source))
            for name, expected in item['sha256'].items():
                check(checked_digest(target if name == '.' else target / name) == expected, 'Evidence hash differs: ' + name)
        for group in small['backup_groups']:
            retired_expected.extend(group['retained_holds'])
            check(checked_digest(Path(group['archive'])) == group['sha256'], 'Dated backup archive differs')
    finalization = json.loads((journal_root / 'finalization.json').read_text())
    check(finalization.get('apply') is True and bool(finalization.get('finished_at')), 'Finalization is incomplete')
    retired = finalization.get('retired', [])
    check(len(retired_expected) == len(set(retired_expected)) == len(retired) == len(set(retired)) == 76
          and set(retired) == set(retired_expected), 'Finalization does not cover all 76 original holds')
    for path in retired_expected:
        check(not Path(path).exists() and not Path(path).is_symlink(), 'Retired hold still present: ' + path)
    check(not pending, 'Assets still pending finalization')
    home = json.loads((root / 'archive/home_tools_202609/operations.json').read_text())
    classification = json.loads((root / 'research/cleanup_20261003/home_classification.json').read_text())
    classified = {item['name']: item for item in classification}
    check(len(classification) == len(classified) == 66, 'HOME classification list incomplete')
    check(len(home['moved']) == 62 and len(home['runner_links']) == 4 and not home['skipped']
          and bool(home.get('finished_at')), 'HOME organization journal incomplete')
    home_items = home['moved'] + home['runner_links']
    check(len({item['source'] for item in home_items}) == len({item['target'] for item in home_items}) == 66
          and {Path(item['source']).name for item in home_items} == set(classified), 'HOME source list differs from classification')
    for item in home_items:
        name = Path(item['source']).name
        expected = classified.get(name)
        check(item['source'] == str(home_root / name) and expected is not None
              and item['sha256'] == expected['sha256'], 'HOME source/hash classification differs: ' + name)
        if expected is not None:
            group = expected['group']
            target_expected = (root / 'scripts/run/home' / name if group == 'run' else
                               root / 'archive/home_tools_202609' / ('historical_shell' if group == 'historical_destructive' else group) / name)
            check(item['target'] == str(target_expected), 'Fixed HOME target differs: ' + name)
    check({Path(item['source']).name for item in home['runner_links']} ==
          {name for name, item in classified.items() if item['group'] == 'run'}, 'HOME formal runner set differs')
    for item in home['moved']:
        source, target = Path(item['source']), Path(item['target'])
        check(item['state'] == 'completed', 'HOME move incomplete: ' + str(source))
        check(source.is_symlink() and source.resolve() == target, 'HOME compatibility link differs: ' + str(source))
        check(checked_digest(target) == item['sha256'], 'Archived HOME tool differs: ' + str(target))
    for item in home['runner_links']:
        source, target = Path(item['source']), Path(item['target'])
        check(source.is_file() and not source.is_symlink(), 'Original HOME runner moved: ' + str(source))
        check(target.is_symlink() and target.resolve() == source, 'HOME runner alias differs: ' + str(target))
        if source.is_file() and checked_digest(source) != item['sha256']:
            later_source_changes.append(str(source))
    navigation = json.loads((root / 'research/cleanup_20261003/navigation_operations.json').read_text())
    check(navigation['links'] == expected_navigation, 'Fixed four navigation links differ')
    for source, target in navigation['links'].items():
        check(Path(source).is_symlink() and Path(source).resolve() == Path(target).resolve() and Path(target).exists(), 'Navigation link differs: ' + source)
    registry = json.loads((root / 'configs/asset_registry.json').read_text())
    for section in ('models', 'cold_runs', 'caches'):
        for name, item in registry[section].items():
            for key in ('entry', 'storage', 'compatibility', 'protected_runtime'):
                if key in item:
                    check((root / item[key]).exists(), 'Registered asset missing: ' + section + '/' + name + '/' + key)
            if 'entry' in item and 'storage' in item:
                check((root / item['entry']).resolve() == Path(item['storage']).resolve(), 'Registered asset points elsewhere: ' + name)
    for key in ('raw_3dpw_archive', 'reports_qwen21'):
        check(Path(registry[key]).exists(), 'Registered path missing: ' + key)
    for key in ('protected_runs', 'protected_data'):
        for path in registry[key]:
            check((root / path).exists(), 'Protected asset missing: ' + path)
    for key in ('qwen21_python', 'deepgen_python', 'data_engine_envs'):
        check((root / registry['environments'][key]).exists(), 'Registered environment missing: ' + key)
    if args.strict_source_baseline:
        check(not later_source_changes, 'Source differs from historical baseline')
    print(json.dumps({'passed': not problems, 'moved_entries': len(organization['moves']),
                      'classified_aliases': len(organization['aliases']), 'home_tools_archived': len(home['moved']),
                      'home_runner_aliases': len(home['runner_links']), 'navigation_links': len(navigation['links']),
                      'asset_files_hashed': count,
                      'retired_holds_confirmed': len(retired),
                      'source_changes_since_cleanup': later_source_changes,
                      'historical_head_unchanged': head_unchanged, 'historical_index_unchanged': index_unchanged,
                      'pending_assets': pending, 'problems': problems}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()
