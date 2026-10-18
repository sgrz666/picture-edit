"""CPU regression guards for the final untouched-test inference runner."""
import importlib
from dataclasses import dataclass
import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np
from PIL import Image


def evaluator():
    return importlib.import_module('src.evaluation.qwen21_dual_prompt_pilot')


def test_background_canvas_uses_native_metadata_not_materialized_512_size():
    from src.data.hi4d_smplx_dataset import letterbox_affine
    from src.evaluation.qwen21_dual_v8_metrics import background_error
    assets = dict(preprocessing=dict(native_size=[940, 1280], resolution=512,
        affine=letterbox_affine(940, 1280).tolist()))
    valid = evaluator().source_canvas_mask(assets)
    assert valid.shape == (512, 512) and valid.dtype == bool
    assert int(valid.sum()) == 376 * 512
    assert valid[:, 68:444].all()
    assert not valid[:, :68].any() and not valid[:, 444:].any()
    source = np.zeros((512, 512, 3), dtype=np.uint8)
    generated = source.copy()
    generated[~valid] = 255
    masks = np.zeros((2, 512, 512), dtype=np.float32)
    score = background_error(generated, source, masks, masks, valid_canvas=valid)
    assert score['mae_rgb_0_1'] == 0.0
    assert score['pixel_denominator'] == 376 * 512


def test_background_canvas_rejects_missing_or_inconsistent_preprocessing():
    with pytest.raises(ValueError, match='preprocessing'):
        evaluator().source_canvas_mask({})
    with pytest.raises(ValueError, match='affine'):
        evaluator().source_canvas_mask(dict(preprocessing=dict(native_size=[940, 1280],
            resolution=512, affine=np.eye(3).tolist())))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def dataset(tmp_path):
    from src.data.hi4d_smplx_dataset import GENERATION_KEYS
    from src.pose_control.qwen21_full.dual.schema import generation_versions
    split_path = write(tmp_path / 'split.json', dict(train_pairs=['pair12', 'pair14'],
        val_pairs=['pair22'], test_pairs=['pair00']))
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    samples, splits = [], {}
    for split, pairs in [('train', ['pair12', 'pair12', 'pair14', 'pair14']),
                         ('val', ['pair22']), ('test', ['pair00'])]:
        rows = []
        ids = []
        for index, pair in enumerate(pairs):
            sid = f'{split}_{index}'
            ids.append(sid)
            source_action = {'pair12': 'highfive12', 'pair14': 'talk14', 'pair22': 'game22', 'pair00': 'taichi00'}[pair]
            target_action = 'highfive12' if pair == 'pair12' else pair.replace('pair', 'hug')
            source_frame = {'pair12': 28, 'pair14': 32, 'pair22': 52, 'pair00': 41}[pair]
            target_frame = ([33, 82][index] if pair == 'pair12' else [45, 62][index - 2]) if split == 'train' else (46 if split == 'val' else 108)
            row = dict(source_rgb=str(tmp_path / 'assets/sources' / pair / 'cam4' / source_action / f'{source_frame:06}' / 'rgb.png'),
                source_masks=[str(tmp_path / pair / f'{a}.png') for a in 'AB'],
                source_features=str(tmp_path / pair / 'features.pt'),
                source_features_sha256='a' * 64, target_geometry=str(tmp_path / sid / f'{target_action}_{target_frame:06}_cam4_geometry.npz'),
                person_uids=[pair + '_A', pair + '_B'], prompt='short ' + sid, seed=42,
                resolution=512, versions=generation_versions(), target_geometry_known=True)
            assert set(row) == GENERATION_KEYS
            rows.append(row)
            samples.append(dict(sample_id=sid, split=split, pair=pair,
                source_action=source_action, target_action=target_action, source_frame=source_frame,
                target_frame=target_frame, camera=4, person_uids=row['person_uids']))
        short = write(tmp_path / split / 'short_generation.json', rows)
        structured = write(tmp_path / split / 'structured_generation.json',
            [dict(row, prompt='structured ' + sid) for row, sid in zip(rows, ids)])
        supervision = tmp_path / split / 'supervision.json'
        # The test supervision remains absent: preflight cannot open it.
        if split != 'test':
            write(supervision, [])
        splits[split] = dict(short_generation=str(short), structured_generation=str(structured),
            supervision=str(supervision), sample_ids=ids,
            short_generation_sha256=file_sha256(short), structured_generation_sha256=file_sha256(structured))
    locked = []
    for split in ['train', 'val', 'test']:
        selected = [row for row in samples if row['split'] == split]
        original_path = tmp_path / 'original' / f'{split}.jsonl'
        original_path.parent.mkdir(parents=True, exist_ok=True)
        originals = [dict(split=split, pair=row['pair'], source_action=row['source_action'],
            target_action=row['target_action'], source_frame_id=row['source_frame'],
            target_frame_id=row['target_frame'], camera_id=row['camera']) for row in selected]
        lines = [json.dumps(row) for row in originals]
        original_path.write_text('\n'.join(lines) + '\n')
        for index, (row, original, line) in enumerate(zip(selected, originals, lines)):
            row_sha = hashlib.sha256(line.encode()).hexdigest()
            row.update(original_manifest=str(original_path), original_manifest_line=index + 1,
                original_manifest_sha256=file_sha256(original_path), original_row_sha256=row_sha)
            locked.append(dict(split=split, original_manifest=str(original_path), one_based_line=index + 1,
                manifest_sha256=file_sha256(original_path), row_sha256=row_sha, row=original))
    lock_path = write(tmp_path / 'selection_lock.json', dict(selection_locked_before_training=True,
        split_manifest_sha256=file_sha256(split_path), rows=locked))
    payload = dict(schema_version='hi4d_dual_prompt_pilot_v1',
        split_manifest=dict(path=str(split_path), sha256=file_sha256(split_path)),
        selected_samples=samples, splits=splits, selection_lock=str(lock_path),
        selection_lock_sha256=file_sha256(lock_path))
    write(tmp_path / 'dataset.json', payload)
    return payload


def training(tmp_path, bundle):
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    root = tmp_path / 'training'
    branches = {}
    for variant in ['short', 'structured']:
        path = root / variant / 'checkpoint_best.pt'
        path.parent.mkdir(parents=True)
        metadata = dict(architecture_version='dual_v8', variant=variant,
            prompt_pilot_schema_version='qwen21_dual_prompt_pilot_v1',
            dataset_sha256=bundle['dataset_sha256'], split_manifest=bundle['split_manifest'],
            train_pair_ids=bundle['pair_ids']['train'], val_pair_ids=bundle['pair_ids']['val'],
            train_sample_ids=bundle['sample_ids']['train'], val_sample_ids=bundle['sample_ids']['val'],
            validation_selection='true_val_split_only', backbone_sha256='b' * 64,
            model_id='/models/Qwen-Image-2.1', best_step=250, best_validation=.2,
            step=250, validation_loss=.2)
        metadata['manifest_sha256'] = {split: {key: file_sha256(bundle['manifests'][split][kind])
            for key, kind in [('short_generation', 'short'), ('structured_generation', 'structured'),
                              ('supervision', 'supervision')]} for split in ['train', 'val']}
        metadata['run_contract'] = dict(data_content_sha256={str(bundle['dataset_path']): bundle['dataset_sha256']})
        from src.pose_control.qwen21_full.dual.prompt_pilot import prompt_pilot_code_sha256
        metadata['code_sha256'] = prompt_pilot_code_sha256(Path(__file__).resolve().parents[1])
        metadata['run_contract']['code_sha256'] = metadata['code_sha256']
        torch.save(dict(step=250, metadata=metadata, adapter_state_dict={'weight': torch.ones(2)}), path)
        record = dict(status='success', variant=variant, completed_steps=300, requested_steps=300,
            best_step=250, best_validation=.2, validation_records=[dict(step=step, loss=loss)
                for step, loss in zip(range(50, 301, 50), [.5, .45, .4, .35, .2, .3])],
            best_checkpoint=dict(path=str(path), sha256=file_sha256(path), step=250),
            dataset_sha256=bundle['dataset_sha256'], split_manifest=bundle['split_manifest'],
            initial_adapter_sha256='c' * 64, backbone_sha256_before='b' * 64,
            backbone_sha256_after='b' * 64, backbone_unchanged=True)
        write(root / variant / 'training_complete.json', record)
        branches[variant] = record
    write(root / 'training_complete.json', dict(status='success', completed_steps=300,
        requested_steps=300, variants=branches, dataset_sha256=bundle['dataset_sha256'],
        split_manifest=bundle['split_manifest'], initialization_sha256_equal=True,
        initial_adapter_sha256='c' * 64, backbone_sha256_before='b' * 64,
        backbone_sha256_after='b' * 64, backbone_unchanged=True))
    return root


@pytest.mark.parametrize('operation', ['tensor_backward', 'autograd_backward', 'autograd_grad', 'optimizer_step'])
def test_global_guard_rejects_real_backprop_even_with_grad_disabled_and_restores(operation):
    module = evaluator()
    tensor = torch.tensor(2., requires_grad=True)
    optimizer = torch.optim.AdamW([tensor])
    original = (torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad, torch.optim.AdamW.step)
    actions = dict(tensor_backward=lambda: tensor.backward(),
        autograd_backward=lambda: torch.autograd.backward(tensor),
        autograd_grad=lambda: torch.autograd.grad(tensor, tensor), optimizer_step=lambda: optimizer.step())
    guard = module.InferenceOnlyGuard()
    with pytest.raises(RuntimeError, match='inference-only'):
        with guard:
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            actions[operation]()
    assert guard.attempts[operation] == 1
    assert original == (torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad, torch.optim.AdamW.step)
    assert guard.restored
    assert torch.is_grad_enabled()


def test_guard_rejects_optimizer_construction():
    module = evaluator()
    with module.InferenceOnlyGuard() as guard:
        with pytest.raises(RuntimeError, match='inference-only'):
            torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)
    assert guard.attempts['optimizer_init'] == 1


def test_test_membership_and_split_sha_are_verified_before_supervision(tmp_path):
    payload = dataset(tmp_path)
    bundle = evaluator().validate_test_dataset(tmp_path)
    assert bundle['pair_ids']['test'] == ['pair00']
    assert bundle['sample_ids']['test'] == ['test_0']
    payload['selected_samples'][-1]['pair'] = 'pair12'
    write(tmp_path / 'dataset.json', payload)
    with pytest.raises(ValueError, match='membership|overlap'):
        evaluator().validate_test_dataset(tmp_path)


def test_test_must_have_exactly_one_preselected_row(tmp_path):
    payload = dataset(tmp_path)
    payload['splits']['test']['sample_ids'] += ['test_extra']
    write(tmp_path / 'dataset.json', payload)
    with pytest.raises(ValueError, match='exactly one|count'):
        evaluator().validate_test_dataset(tmp_path)


def test_alternate_test_frame_in_allowed_pair_is_rejected(tmp_path):
    payload = dataset(tmp_path)
    payload['selected_samples'][-1]['target_frame'] = 109
    write(tmp_path / 'dataset.json', payload)
    with pytest.raises(ValueError, match='locked test|selection lock'):
        evaluator().validate_test_dataset(tmp_path)


def test_selection_lock_content_is_authenticated_without_opening_gt(tmp_path):
    payload = dataset(tmp_path)
    lock = Path(payload['selection_lock'])
    value = json.loads(lock.read_text())
    value['rows'][-1]['row']['target_frame_id'] = 109
    write(lock, value)
    with pytest.raises(ValueError, match='selection lock.*SHA'):
        evaluator().validate_test_dataset(tmp_path)


def test_stale_runtime_core_hash_in_checkpoint_is_rejected(tmp_path):
    dataset(tmp_path)
    module = evaluator()
    bundle = module.validate_test_dataset(tmp_path)
    root = training(tmp_path, bundle)
    path = root / 'short/checkpoint_best.pt'
    value = torch.load(path, weights_only=True)
    value['metadata']['code_sha256']['src/pose_control/qwen21_full/dual/runtime.py'] = '0' * 64
    torch.save(value, path)
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    branch = json.loads((root / 'short/training_complete.json').read_text())
    branch['best_checkpoint']['sha256'] = file_sha256(path)
    write(root / 'short/training_complete.json', branch)
    complete = json.loads((root / 'training_complete.json').read_text())
    complete['variants']['short'] = branch
    write(root / 'training_complete.json', complete)
    with pytest.raises(ValueError, match='training.*source|runtime.*SHA'):
        module.validate_training_contracts(root, bundle)


def test_native_structured_prompt_has_geometry_text_without_dense_control():
    plan = evaluator().planned_runs([42, 43, 44], 'short', 'structured')
    native_short = next(row for row in plan if row['key'] == 'native_short_seed42')
    native_structured = next(row for row in plan if row['key'] == 'native_structured_seed42')
    dual_structured = next(row for row in plan if row['key'] == 'dual_structured_seed42')
    assert not native_short['explicit_target_geometry_control']
    assert not native_short['prompt_geometry_descriptions']
    assert not native_structured['explicit_target_geometry_control']
    assert native_structured['prompt_geometry_descriptions']
    assert native_structured['prompt_geometry_provenance'] == 'reliable_external_target_geometry'
    assert dual_structured['explicit_target_geometry_control']
    assert 'target_geometry_known' not in native_structured


def test_existing_materializer_split_path_and_sha_shape_is_supported(tmp_path):
    payload = dataset(tmp_path)
    binding = payload['split_manifest']
    payload['split_manifest'] = binding['path']
    payload['split_manifest_sha256'] = binding['sha256']
    write(tmp_path / 'dataset.json', payload)
    assert evaluator().validate_test_dataset(tmp_path)['split_manifest'] == binding


def test_complete_records_bind_best_checkpoint_and_validation_selection(tmp_path):
    dataset(tmp_path)
    bundle = evaluator().validate_test_dataset(tmp_path)
    root = training(tmp_path, bundle)
    contracts = evaluator().validate_training_contracts(root, bundle)
    assert contracts['short']['step'] == 250
    best = root / 'short' / 'checkpoint_best.pt'
    with best.open('ab') as stream:
        stream.write(b'stale')
    with pytest.raises(ValueError, match='SHA|sha256'):
        evaluator().validate_training_contracts(root, bundle)


def test_authenticates_fixed_test_prompt_content_before_generation(tmp_path):
    payload = dataset(tmp_path)
    path = Path(payload['splits']['test']['short_generation'])
    value = json.loads(path.read_text())
    value[0]['prompt'] = 'select a different instruction after results'
    write(path, value)
    with pytest.raises(ValueError, match='generation.*SHA|manifest.*SHA'):
        evaluator().validate_test_dataset(tmp_path)


def test_checkpoint_rejects_changed_train_manifest_even_with_same_dataset_json(tmp_path):
    payload = dataset(tmp_path)
    module = evaluator()
    bundle = module.validate_test_dataset(tmp_path)
    root = training(tmp_path, bundle)
    write(Path(payload['splits']['train']['short_generation']), [])
    with pytest.raises(ValueError, match='manifest.*SHA'):
        module.validate_training_contracts(root, bundle)


def test_checkpoint_cannot_claim_a_train_pair_as_test_holdout(tmp_path):
    dataset(tmp_path)
    bundle = evaluator().validate_test_dataset(tmp_path)
    root = training(tmp_path, bundle)
    path = root / 'short' / 'checkpoint_best.pt'
    value = torch.load(path, weights_only=True)
    value['metadata']['train_pair_ids'].append('pair00')
    torch.save(value, path)
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    branch = json.loads((root / 'short/training_complete.json').read_text())
    branch['best_checkpoint']['sha256'] = file_sha256(path)
    write(root / 'short/training_complete.json', branch)
    complete = json.loads((root / 'training_complete.json').read_text())
    complete['variants']['short'] = branch
    write(root / 'training_complete.json', complete)
    with pytest.raises(ValueError, match='pair|holdout'):
        evaluator().validate_training_contracts(root, bundle)


def test_best_payload_validation_must_match_recorded_minimum(tmp_path):
    dataset(tmp_path)
    module = evaluator()
    bundle = module.validate_test_dataset(tmp_path)
    root = training(tmp_path, bundle)
    path = root / 'short/checkpoint_best.pt'
    value = torch.load(path, weights_only=True)
    value['metadata']['best_validation'] = .9
    torch.save(value, path)
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    branch = json.loads((root / 'short/training_complete.json').read_text())
    branch['best_checkpoint']['sha256'] = file_sha256(path)
    write(root / 'short/training_complete.json', branch)
    complete = json.loads((root / 'training_complete.json').read_text())
    complete['variants']['short'] = branch
    write(root / 'training_complete.json', complete)
    with pytest.raises(ValueError, match='payload.*validation|validation.*payload'):
        module.validate_training_contracts(root, bundle)


def test_real_cpu_weights_and_gradients_unchanged_and_target_opened_last(tmp_path):
    module = evaluator()
    native = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    adapters = {kind: torch.nn.Linear(2, 2) for kind in ['short', 'structured']}
    pipe = SimpleNamespace(transformer=native)
    @dataclass(frozen=True)
    class Generation:
        prompts: tuple
        seeds: tuple
    values = {kind: Generation(prompts=(kind,), seeds=(42,)) for kind in adapters}
    events = []
    def generate(pipe, generation, *, adapter, steps, cfg, use_kv_cache):
        events.append('generate')
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert generation.seeds[0] in [42, 43, 44]
        result = pipe.transformer(torch.ones(1, 2))
        if adapter is not None:
            result = adapter(result)
        assert not result.requires_grad
        return Image.new('RGB', (512, 512), 'gray')
    def target_loader():
        assert events == ['generate'] * 12
        events.append('target')
        return object()
    def score(records, target):
        assert events[-1] == 'target'
        return dict(outputs=len(records))
    plan = module.planned_runs([42, 43, 44], 'short', 'structured')
    outputs, audit, scores = module.generate_then_evaluate(generate, pipe, values, adapters,
        plan, tmp_path / 'new', {}, target_loader=target_loader, score_fn=score)
    assert scores['outputs'] == 12
    assert audit['weights_unchanged'] and audit['all_parameter_grads_none']
    assert audit['optimizer_instances_before'] == audit['optimizer_instances_after'] == 0
    assert len(audit['generation_samples']) == 12
    assert len(audit['native_forward_samples']) == 12
    assert all(not row['grad_enabled'] and row['inference_mode'] for row in audit['generation_samples'])
    assert len(outputs) == 12 and all(path['output_sha256'] for path in outputs)
    assert 'target' not in inspect.signature(module.run_generation_phase).parameters
    assert 'evaluation_target' not in inspect.signature(module.run_generation_phase).parameters


def test_failed_generation_writes_failure_and_no_success_or_target_read(tmp_path):
    module = evaluator()
    native = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    adapters = {kind: torch.nn.Linear(2, 2) for kind in ['short', 'structured']}
    output = tmp_path / 'new'
    @dataclass(frozen=True)
    class Generation:
        prompts: tuple = ('short',)
        seeds: tuple = (42,)
    def generate(*args, **kwargs):
        raise RuntimeError('expected generation failure')
    with pytest.raises(RuntimeError, match='expected generation failure'):
        module.generate_then_evaluate(generate, SimpleNamespace(transformer=native),
            {'short': Generation(), 'structured': Generation()}, adapters,
            module.planned_runs([42, 43, 44], 'short', 'structured'), output, {},
            target_loader=lambda: pytest.fail('target read on generation failure'), score_fn=lambda *args: None)
    assert json.loads((output / 'evaluation_failed.json').read_text())['status'] == 'failed'
    assert not (output / 'evaluation_complete.json').exists()


def test_generation_destination_never_overwrites_existing_outputs(tmp_path):
    module = evaluator()
    path = tmp_path / 'existing'
    path.mkdir()
    with pytest.raises(ValueError, match='new|exists'):
        module.run_generation_phase(None, None, {}, {}, [], path, {})


def test_actual_camera_observations_use_affine_and_reject_stale_content(tmp_path):
    module = evaluator()
    from src.evaluation.qwen21_dual_v8_metrics import file_sha256
    paths = []
    for actor in range(2):
        path = tmp_path / f'observation{actor}.npz'
        np.savez(path, xy_native=np.full((134, 2), [100., 200.]),
            scores=np.ones(134), width=np.array(1000), height=np.array(500))
        paths.append(str(path))
    geometry = tmp_path / 'geometry.npz'
    write(geometry.with_suffix('.json'), dict(pose_observation_paths=paths,
        pose_observation_sha256=[file_sha256(path) for path in paths],
        observation_width=1000, observation_height=500, K512=np.eye(3).tolist()))
    xy, scores, metadata = module.read_external_pose(geometry)
    assert xy.shape == (2, 134, 2) and scores.shape == (2, 134)
    assert xy[0, 0] == pytest.approx([51.2, 230.4])
    assert metadata['provenance'] == 'actual_camera_dwpose_observations'
    with Path(paths[0]).open('ab') as stream:
        stream.write(b'stale')
    with pytest.raises(ValueError, match='SHA'):
        module.read_external_pose(geometry)
