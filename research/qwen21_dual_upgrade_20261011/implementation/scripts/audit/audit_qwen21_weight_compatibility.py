#!/usr/bin/env python3
"""CPU-only weight provenance, semantic mapping and in-memory module audit.

Never constructs Qwen's base model, an optimizer, or an image pipeline. Never
saves adapter tensors. Source checkpoints and existing entrypoints are read-only.
"""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter
from src.pose_control.qwen21.face_adapter import Qwen21FaceAdapter
from src.pose_control.qwen21_full.data_runtime import QwenFullDataset
from src.pose_control.qwen21_full.execution import new_adapter, load_adapter_checkpoint

RUN = Path('/home/shangguanrz/nihaochen/qwen21_full')
NIHAO = RUN.parent
COMMON = ('condition_injector.', 'reasoner.', 'condition_bridge.')
SOURCES = {
    'body512': ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt',
    'face512': NIHAO / 'qwen21_face/face_train_3k/checkpoint_selected.pt',
    'body1024': NIHAO / 'iper1024/body_selected.pt',
    'face1024': NIHAO / 'iper1024/face_2500/checkpoint_best_loss.pt',
}
ATOL, RTOL = 1e-6, 1e-5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hashes(model):
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def apply_mapping_in_memory(model, source, entries, ignored_source_keys):
    """Atomic preflight. Only explicit destination tensors/rows may be copied."""
    target = model.state_dict()
    names = [entry['source'] for entry in entries]
    destinations = [entry['target'] for entry in entries]
    if len(set(names)) != len(names) or len(set(destinations)) != len(destinations):
        raise ValueError('Duplicate source or destination mapping')
    ignored = set(ignored_source_keys)
    if ignored & set(names) or set(names) | ignored != set(source):
        raise ValueError('Every source key must be explicitly mapped or ignored')
    for entry in entries:
        name, destination = entry['source'], entry['target']
        if name not in source or destination not in target:
            raise ValueError(f'Missing source/destination: {name} -> {destination}')
        a, b = source[name], target[destination]
        if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point():
            raise ValueError(f'Incompatible shape/dtype: {name}')
        if not bool(torch.isfinite(a).all()):
            raise ValueError(f'Nonfinite source: {name}')
        rows = entry.get('rows')
        if rows is not None and (a.ndim != 2 or not rows or len(rows) != len(set(rows))
                                 or min(rows) < 0 or max(rows) >= a.shape[0]):
            raise ValueError(f'Invalid row selection: {name}')
    with torch.no_grad():
        for entry in entries:
            a, b = source[entry['source']], target[entry['target']]
            if 'rows' in entry:
                indices = torch.tensor(entry['rows'], dtype=torch.long)
                b.index_copy_(0, indices, a.index_select(0, indices).to(b))
            else:
                b.copy_(a.to(b))


def compare_tree(a, b, prefix=''):
    records = []
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f'Shape/dtype mismatch: {prefix}')
        floating = a.is_floating_point()
        torch.testing.assert_close(a, b, atol=ATOL if floating else 0, rtol=RTOL if floating else 0)
        records.append({'field': prefix, 'shape': list(a.shape), 'dtype': str(a.dtype),
                        'exact': torch.equal(a, b),
                        'max_abs': float((a - b).abs().max()) if floating and a.numel() else 0.})
    elif is_dataclass(a):
        if [f.name for f in fields(a)] != [f.name for f in fields(b)]:
            raise AssertionError('Dataclass fields differ')
        for field in fields(a):
            records += compare_tree(getattr(a, field.name), getattr(b, field.name), prefix + '.' + field.name)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError('Sequence sizes differ')
        for i, (x, y) in enumerate(zip(a, b)):
            records += compare_tree(x, y, f'{prefix}.{i}')
    elif a != b:
        raise AssertionError(f'Non-tensor field differs: {prefix}')
    return records


def optimizer_evidence(model, saved, body):
    parameters = dict(model.named_parameters())
    groups = ([n for n in parameters if n.startswith(COMMON)], [n for n in parameters if not n.startswith(COMMON)]) if body else (
        [n for n in parameters if n.startswith('zero_heads.')], [n for n in parameters if not n.startswith('zero_heads.')])
    optimizer = saved['optimizer_state_dict']  # Inspection only: never load into an optimizer.
    if len(groups) != len(optimizer['param_groups']):
        raise AssertionError('Optimizer group layout changed')
    result = {}
    for names, group in zip(groups, optimizer['param_groups']):
        if len(names) != len(group['params']):
            raise AssertionError('Cannot align historical optimizer parameters')
        for name, identifier in zip(names, group['params']):
            state = optimizer['state'].get(identifier, {})
            variance = state.get('exp_avg_sq')
            if variance is not None and variance.shape != parameters[name].shape:
                raise AssertionError(f'Optimizer tensor layout changed: {name}')
            active = variance is not None and bool(torch.count_nonzero(variance))
            result[name] = {'optimizer_state_present': bool(state),
                            'optimizer_steps': int(state['step']) if 'step' in state else 0,
                            'nonzero_gradient_history_evidence': active,
                            'status': 'nonzero_gradient_evidence' if active else 'zero_moments_only' if state else 'no_optimizer_state',
                            'last_group_lr': group['lr']}
            if 'embedding.weight' in name and variance is not None and variance.ndim == 2:
                result[name]['active_embedding_rows'] = (variance.abs().sum(1) > 0).nonzero().flatten().tolist()
    return result


def import_fixture(name):
    path = ROOT / 'tests' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('_audit_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def person_kwargs(person, role):
    return {f'{key}_{role}': value for key, value in person.items()}


def cases():
    fixture = import_fixture('test_condition_injector_v1')
    for count in (1, 2):
        a = fixture._person(batch_size=1, image_size=64)
        kw = person_kwargs(a, 'a')
        kw['task_id'] = torch.tensor([0 if count == 1 else 1])
        if count == 2:
            kw.update(person_kwargs(fixture._person(batch_size=1, image_size=64, offset=1), 'b'))
        yield f'synthetic_{count}_person', 512, kw
    a, b = fixture._person(image_size=64), fixture._person(image_size=64, offset=1)
    kw = {**person_kwargs(a, 'a'), **person_kwargs(b, 'b'),
          'person_b_valid': torch.tensor([True, False]), 'task_id': torch.tensor([2, 0]),
          'relative_geometry': torch.randn(2, 11), 'contact_raster': torch.ones(2, 2, 64, 64),
          'contact_relations': fixture._relations()}
    kw['contact_relations'].valid_mask[1] = False
    yield 'synthetic_mixed_contact_invalid_B', 1024, kw
    for resolution in (512, 1024):
        dataset = QwenFullDataset(RUN / 'val_pairs.jsonl', resolution=resolution)
        if len(dataset) != 4:
            raise AssertionError('Expected the fixed four validation pairs')
        for index in range(4):
            sample = dataset[index]
            inputs = {name: sample[name].unsqueeze(0).float() for name in
                      ('normal', 'pose_heatmap', 'part_onehot', 'smplx_global', 'human_mask')}
            kw = person_kwargs(inputs, 'a')
            kw['task_id'] = sample['task_id'].reshape(1).long()
            yield f'validation_{resolution}_{index}', resolution, kw


@torch.no_grad()
def local_checks(original, full, inputs):
    result = []
    for label, resolution, kw in inputs:
        a, b = original.condition_injector(**kw), full.legacy.condition_injector(**kw)
        records = compare_tree(a, b, 'conditions')
        ar, br = original.reasoner(a), full.legacy.reasoner(b)
        records += compare_tree(ar, br, 'reasoner')
        ac, bc = original.condition_bridge(ar), full.legacy.condition_bridge(br)
        records += compare_tree(ac, bc, 'bridge')
        result.append({'case': label, 'rgb_resolution': resolution,
                       'geometry_resolution': list(kw['normal_a'].shape[-2:]),
                       'passed': True, 'compared_fields': records,
                       'high_resolution_features_checked': True,
                       'person_count': a.person_count.tolist()})
        if label in ('validation_512_0', 'validation_1024_0', 'synthetic_mixed_contact_invalid_B'):
            grid = resolution // 16
            batch = a.person_count.numel()
            generator = torch.Generator().manual_seed(42)
            latent = torch.randn(batch, grid * grid, 64, generator=generator)
            persons = torch.randn(batch, 2, grid * grid, 64, generator=generator) if (a.person_count == 2).any() else None
            prepared = full.prepare_conditioning(condition_bundle=b, source_latent=latent,
                                                 target_token_hw=(grid, grid), source_person_latents=persons)
            prepared.validate()
            assert prepared.geometry_condition.shape == (batch, 144, grid * 2, grid * 2)
            result[-1]['full_downstream_contract'] = {
                'geometry_condition': list(prepared.geometry_condition.shape),
                'native_target_token_grid': [grid, grid],
                'synthetic_latents_only': True, 'full_preparation_passed': True,
                'note': 'No equality claim beyond the common bridge: downstream computations differ.'}
        print(json.dumps({'case': label, 'passed': True}), flush=True)
    return result


def module_source_evidence(original, full):
    result = []
    for name in ('condition_injector', 'reasoner', 'condition_bridge'):
        a, b = getattr(original, name), getattr(full.legacy, name)
        pa, pb = Path(inspect.getfile(type(a))), Path(inspect.getfile(type(b)))
        if pa.read_bytes() != pb.read_bytes():
            raise AssertionError(f'Common module implementation differs: {name}')
        if name == 'reasoner' and asdict(a.config) != asdict(b.config):
            raise AssertionError('Reasoner constructor configuration differs')
        result.append({'module': name, 'source_file': str(pa), 'target_file': str(pb),
                       'same_source_bytes': True, 'sha256': file_sha(pa)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'weight_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    full_checkpoint = RUN / 'face_512_100/checkpoint_last.pt'
    tracked_paths = {**SOURCES, 'full_receiver': full_checkpoint,
                     'deepgen_body20k': ROOT / 'experiments/iper_v6_native_20k_v1/checkpoint_last.pt',
                     'deepgen_face1500': ROOT / 'experiments/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_001500.pt'}
    fingerprints = {key: {'path': str(path.absolute()), 'resolved_path': str(path.resolve()),
                          'sha256': file_sha(path), 'bytes': path.stat().st_size} for key, path in tracked_paths.items()}
    write_json(args.output / 'source_fingerprints.json', fingerprints)
    full = new_adapter('cpu').eval().requires_grad_(False)
    load_adapter_checkpoint(full, full_checkpoint)
    original = Qwen21BodyAdapter().eval().requires_grad_(False)
    face = Qwen21FaceAdapter().eval().requires_grad_(False)
    interface_evidence = module_source_evidence(original, full)
    inputs = list(cases())
    target_baseline = {n: v.clone() for n, v in full.state_dict().items() if n.startswith('legacy.' + COMMON[0])
                       or n.startswith('legacy.' + COMMON[1]) or n.startswith('legacy.' + COMMON[2])}
    initial_hashes = state_hashes(full)
    embedding_names = {n + '.weight' for n, m in original.named_modules() if isinstance(m, torch.nn.Embedding)}
    summary, parameter_rows, whitelist = {}, [], {}
    deepgen = read_checkpoint(tracked_paths['deepgen_body20k'])['adapter_state_dict']
    body512_state = read_checkpoint(SOURCES['body512'])['adapter_state_dict']
    face512_state = read_checkpoint(SOURCES['face512'])['face_state_dict']
    for label, path in SOURCES.items():
        body = label.startswith('body')
        saved = read_checkpoint(path)
        state = saved['adapter_state_dict' if body else 'face_state_dict']
        model = original if body else face
        model.load_state_dict(state, strict=True)
        evidence = optimizer_evidence(model, saved, body)
        compatible, recommended, rows = [], [], []
        baseline = deepgen if label == 'body512' else body512_state if label == 'body1024' else face512_state if label == 'face1024' else {}
        for name, value in state.items():
            if not bool(torch.isfinite(value).all()):
                raise AssertionError(f'Nonfinite checkpoint value: {label}/{name}')
            compatible_key = body and name.startswith(COMMON)
            target = 'legacy.' + name if compatible_key else None
            if compatible_key and (target not in full.state_dict() or full.state_dict()[target].shape != value.shape):
                raise AssertionError(f'Unexpected common-module shape mismatch: {name}')
            row = {'source_id': label, 'source': name, 'target': target, 'shape': list(value.shape),
                   'dtype': str(value.dtype), 'finite': True, 'training_evidence': evidence.get(name, {}),
                   'structurally_compatible': compatible_key,
                   'classification': 'direct_common_module' if compatible_key else 'requires_separate_adaptation',
                   'same_as_baseline': torch.equal(value.float(), baseline[name].float()) if name in baseline and baseline[name].shape == value.shape else None,
                   'baseline': 'DeepGen Body20k' if label == 'body512' else 'Qwen 512 same branch' if label.endswith('1024') else 'no_equivalent_legacy_initialization'}
            if compatible_key:
                entry = {'source': name, 'target': target}
                compatible.append(entry)
                if evidence[name]['nonzero_gradient_history_evidence']:
                    entry = dict(entry)
                    if name in embedding_names:
                        active = evidence[name].get('active_embedding_rows', [])
                        if active and len(active) != value.shape[0]:
                            entry['rows'] = active
                    recommended.append(entry)
            rows.append(row)
        checks = []
        protection = None
        if body:
            apply_mapping_in_memory(full, state, compatible, set(state) - {e['source'] for e in compatible})
            checks = local_checks(original, full, inputs)
            # Restore first; the recommended subset must preserve every other parameter.
            with torch.no_grad():
                for name, value in target_baseline.items(): full.state_dict()[name].copy_(value)
            if state_hashes(full) != initial_hashes:
                raise AssertionError('Structural test mutated noncandidate state')
            apply_mapping_in_memory(full, state, recommended, set(state) - {e['source'] for e in recommended})
            after = state_hashes(full)
            allowed = {e['target'] for e in recommended}
            outsiders = [name for name in after if name not in allowed and after[name] != initial_hashes[name]]
            preserved_rows = []
            for entry in recommended:
                actual, expected = full.state_dict()[entry['target']], state[entry['source']].float()
                selected = entry.get('rows')
                if selected is None:
                    assert torch.equal(actual, expected)
                else:
                    assert torch.equal(actual[selected], expected[selected])
                    rest = [i for i in range(actual.shape[0]) if i not in selected]
                    assert torch.equal(actual[rest], target_baseline[entry['target']][rest])
                    preserved_rows.append({'target': entry['target'], 'copied_rows': selected, 'preserved_rows': rest})
            if outsiders:
                raise AssertionError(f'Unexpected parameter mutation: {outsiders}')
            # Single-person outputs should match the source even with unused rows retained.
            selective = local_checks(original, full, [item for item in inputs if item[0].startswith('validation_')])
            protection = {'passed': True, 'changed_outside_whitelist': outsiders,
                          'partially_copied_embeddings': preserved_rows,
                          'selective_single_person_checks': selective,
                          'note': 'Shared encoders affect dual inputs too; preserving private dual weights does not preserve all dual outputs.'}
            with torch.no_grad():
                for name, value in target_baseline.items(): full.state_dict()[name].copy_(value)
        parameter_rows += rows
        whitelist[label] = {'source_sha256': fingerprints[label]['sha256'], 'receiver_sha256': fingerprints['full_receiver']['sha256'],
                            'entries': recommended, 'declared_ignored_source_keys': sorted(set(state) - {e['source'] for e in recommended}),
                            'structural_complete_module_entries': compatible,
                            'policy': 'Recommended set copies only compatible parameters with nonzero gradient evidence; untouched embedding rows are preserved.',
                            'enabled': False, 'no_checkpoint_written': True}
        summary[label] = {'path': str(path), 'step': saved['step'], 'resolution': saved['args'].get('resolution', 512),
                          'resolution_explicit_in_checkpoint': 'resolution' in saved['args'],
                          'args': saved['args'], 'tensor_count': len(state), 'compatible_tensors': len(compatible),
                          'recommended_tensors': len(recommended),
                          'training_evidence': dict(Counter(v['status'] for v in evidence.values())),
                          'changed_from_comparable_baseline': sum(r['same_as_baseline'] is False for r in rows),
                          'local_checks': checks, 'whitelist_protection': protection}
        write_json(args.output / f'{label}_audit.json', summary[label])
        print(json.dumps({'source': label, 'compatible': len(compatible), 'recommended': len(recommended)}), flush=True)
    if state_hashes(full) != initial_hashes:
        raise AssertionError('Receiver was not restored after in-memory tests')
    for key, path in tracked_paths.items():
        if file_sha(path) != fingerprints[key]['sha256']:
            raise AssertionError(f'Existing checkpoint was modified: {path}')
    write_json(args.output / 'parameter_inventory.json', parameter_rows)
    write_json(args.output / 'load_whitelists.json', whitelist)
    write_json(args.output / 'module_source_evidence.json', interface_evidence)
    result = {'passed': True, 'sources': summary, 'cpu_fp32_atol': ATOL, 'cpu_fp32_rtol': RTOL,
              'checkpoint_files_unchanged': True, 'receiver_restored': True,
              'training_performed': False, 'images_generated': False, 'final_test_set_used': False,
              'optimizer_states_inspected_not_loaded': True,
              'validation_pairs_sha256': file_sha(RUN / 'val_pairs.jsonl'),
              'seconds': time.perf_counter() - started,
              'limitations': ['Compatibility does not establish image quality or fix Face artifacts.',
                             'Nonzero Adam second moment is evidence of historical gradients, not useful learning.',
                             'Moment-free parameters are not proven untrained over their entire history.',
                             '1024 RGB uses the same existing 512 geometry maps.',
                             'No 512-versus-1024 quality ranking is made.']}
    write_json(args.output / 'summary.json', result)
    print('Audit completed without training, generation or checkpoint writes', flush=True)


if __name__ == '__main__':
    main()
