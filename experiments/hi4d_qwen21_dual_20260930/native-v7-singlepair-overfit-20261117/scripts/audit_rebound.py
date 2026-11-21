"""Read-only audit of the existing path relocation and every cached Tensor."""
from pathlib import Path
import argparse
import json
import torch
from rebind_paths import condition_semantics, verified_path


def equal_values(before, after):
    if isinstance(before, torch.Tensor):
        return (isinstance(after, torch.Tensor) and before.dtype == after.dtype
                and before.shape == after.shape and torch.equal(before, after))
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(equal_values(value, after[key]) for key, value in before.items())
    if isinstance(before, (list, tuple)):
        return type(before) is type(after) and len(before) == len(after) and all(equal_values(a, b) for a, b in zip(before, after))
    return before == after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Audit report must be new')
    torch.set_num_threads(2)
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    initial = read(args.root / 'configs/native-v1/S0.json')
    original = read(verified_path(initial['cache_identity']))
    rebound = read(args.root / 'cache/rebound-v1/index.json')
    prepared = read(args.root / 'data-v2/prepared.json')
    rows = []
    for role in ('train', 'observation'):
        old_row = read(verified_path(prepared['manifests'][role]))
        new_row = read(args.root / 'data-rebound-v1' / (role + '.jsonl'))
        old_spec = read(verified_path(old_row['qwen_condition']))
        new_spec = read(verified_path(new_row['qwen_condition']))
        verified_path(old_row['v4_manifest'])
        verified_path(old_spec['prompt']['bank'])
        assert condition_semantics(old_spec) == condition_semantics(new_spec)
        for image in new_spec['images']:
            verified_path(image['file'])
        key = role + '\0' + old_row['pair_id']
        old_ref, new_ref = original['entries'][key]['file'], rebound['entries'][key]['file']
        before = torch.load(verified_path(old_ref), map_location='cpu', mmap=True, weights_only=True)
        after = torch.load(verified_path(new_ref), map_location='cpu', mmap=True, weights_only=True)
        assert before['condition_sha256'] == old_row['qwen_condition']['sha256']
        assert after['condition_sha256'] == new_row['qwen_condition']['sha256']
        before = {key: value for key, value in before.items() if key != 'condition_sha256'}
        after = {key: value for key, value in after.items() if key != 'condition_sha256'}
        if not equal_values(before, after):
            raise ValueError('Frozen cache values changed during relocation: ' + role)
        rows.append({'role': role, 'pair_id': old_row['pair_id'], 'original_cache': old_ref,
                     'rebound_cache': new_ref, 'all_tensor_and_other_values_equal_except_condition_sha': True})
    result = {'passed': True, 'read_only': True, 'original_cache_index': initial['cache_identity'], 'rows': rows}
    with args.report.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
