"""Strict preflight and inference-only execution for the held-out prompt pilot.

No target RGB or supervision object is accepted by the generation phase.
GPU/model imports stay in the CLI; this module also runs with real CPU models.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import torch
import numpy as np

from .qwen21_dual_v8_metrics import file_sha256


VARIANTS = ('short', 'structured')
LOCKED_TEST = dict(pair='pair00', source_action='taichi00', source_frame=41,
    target_action='hug00', target_frame=108, camera=4)


def source_canvas_mask(source_assets):
    """Exclude padding using original-size provenance, not the saved 512 PNG."""
    from src.data.hi4d_smplx_dataset import letterbox_affine
    preprocessing = source_assets.get('preprocessing') if isinstance(source_assets, dict) else None
    if not isinstance(preprocessing, dict):
        raise ValueError('source preprocessing metadata is required for background scoring')
    size = preprocessing.get('native_size')
    resolution = preprocessing.get('resolution')
    if (not isinstance(size, (tuple, list)) or len(size) != 2 or
            any(type(value) is not int or value <= 0 for value in size) or
            type(resolution) is not int or resolution != 512):
        raise ValueError('source preprocessing native dimensions/resolution are invalid')
    affine = np.asarray(preprocessing.get('affine'), dtype=np.float64)
    expected = letterbox_affine(*size)
    if affine.shape != (3, 3) or not np.isfinite(affine).all() or not np.allclose(affine, expected, rtol=0, atol=1e-5):
        raise ValueError('source preprocessing affine does not match original dimensions')
    left, top = int(round(affine[0, 2])), int(round(affine[1, 2]))
    width, height = int(round(size[0] * affine[0, 0])), int(round(size[1] * affine[1, 1]))
    if min(left, top) < 0 or min(width, height) < 1 or left + width > resolution or top + height > resolution:
        raise ValueError('source preprocessing affine content lies outside the canvas')
    valid = np.zeros((resolution, resolution), dtype=bool)
    valid[top:top + height, left:left + width] = True
    return valid


def configuration_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve(base, path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _selection_identity(row, *, original=False):
    return dict(pair=row.get('pair'), source_action=row.get('source_action'),
        source_frame=row.get('source_frame_id' if original else 'source_frame'),
        target_action=row.get('target_action'),
        target_frame=row.get('target_frame_id' if original else 'target_frame'),
        camera=row.get('camera_id' if original else 'camera'))


def validate_selection_lock(data, data_dir, split_sha256):
    """Authenticate the pretraining selection and original JSONL rows, not GT."""
    if not isinstance(data.get('selection_lock'), str) or not data['selection_lock']:
        raise ValueError('final test requires the original pretraining selection lock')
    path = _resolve(data_dir, data['selection_lock'])
    if file_sha256(path) != data.get('selection_lock_sha256'):
        raise ValueError('selection lock content SHA mismatch')
    lock = _read(path)
    if lock.get('selection_locked_before_training') is not True or lock.get('split_manifest_sha256') != split_sha256:
        raise ValueError('selection lock does not bind the authoritative split before training')
    locked_rows = lock.get('rows', [])
    by_line = {(row['split'], row['one_based_line']): row for row in locked_rows}
    samples = data['selected_samples']
    if len(by_line) != len(locked_rows) or set(by_line) != {(row['split'], row['original_manifest_line']) for row in samples}:
        raise ValueError('selection lock rows differ from preselected dataset rows')
    originals = {}
    for selected in samples:
        locked = by_line[(selected['split'], selected['original_manifest_line'])]
        original_path = _resolve(data_dir, selected['original_manifest'])
        if original_path != _resolve(data_dir, locked['original_manifest']):
            raise ValueError('selection lock original manifest path differs')
        if original_path not in originals:
            originals[original_path] = (file_sha256(original_path), original_path.read_text(encoding='utf-8').splitlines())
        manifest_sha, lines = originals[original_path]
        line_index = selected['original_manifest_line'] - 1
        if not 0 <= line_index < len(lines):
            raise ValueError('selection lock original manifest line is invalid')
        line = lines[line_index]
        original = json.loads(line)
        row_sha = hashlib.sha256(line.encode()).hexdigest()
        if (manifest_sha != selected['original_manifest_sha256'] or manifest_sha != locked['manifest_sha256']
                or row_sha != selected['original_row_sha256'] or row_sha != locked['row_sha256']
                or original != locked.get('row')):
            raise ValueError('selection lock original manifest/row SHA binding differs')
        if original.get('split') != selected['split'] or _selection_identity(original, original=True) != _selection_identity(selected):
            raise ValueError('selection lock source/target identity differs from selected sample')
    return dict(path=str(path), sha256=file_sha256(path), selected_rows=len(samples),
        selection_locked_before_training=True, locked_test=LOCKED_TEST)


def validate_test_dataset(data_dir):
    """Verify preselected test membership without opening any supervision file."""
    from src.data.hi4d_smplx_dataset import validate_generation_manifest
    data_dir = Path(data_dir).resolve()
    dataset_path = data_dir / 'dataset.json'
    data = _read(dataset_path)
    if data.get('schema_version') != 'hi4d_dual_prompt_pilot_v1':
        raise ValueError('requires current hi4d_dual_prompt_pilot_v1 dataset')
    split_binding = data.get('split_manifest', {})
    if isinstance(split_binding, str):
        split_binding = dict(path=split_binding, sha256=data.get('split_manifest_sha256'))
    split_path = _resolve(data_dir, split_binding['path'])
    if file_sha256(split_path) != split_binding.get('sha256'):
        raise ValueError('authoritative split manifest SHA mismatch')
    authoritative = _read(split_path)
    membership = {name: set(authoritative[name + '_pairs']) for name in ('train', 'val', 'test')}
    if any(membership[a] & membership[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('authoritative split pair overlap')
    samples = data.get('selected_samples')
    if not isinstance(samples, list) or len(samples) != 6:
        raise ValueError('pilot requires four train, one val and exactly one test sample')
    by_id = {row['sample_id']: row for row in samples}
    if len(by_id) != len(samples):
        raise ValueError('duplicate selected sample IDs')
    pair_ids, sample_ids, manifests, rows = {}, {}, {}, {}
    for split, count in [('train', 4), ('val', 1), ('test', 1)]:
        entry = data['splits'][split]
        ids = entry.get('sample_ids')
        if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count:
            raise ValueError(f'{split} sample count must be {count}; test requires exactly one')
        selected = [by_id.get(sid) for sid in ids]
        if any(row is None or row.get('split') != split or row.get('pair') not in membership[split] for row in selected):
            raise ValueError(f'{split} sample pair membership does not match authoritative split')
        pair_ids[split] = sorted({row['pair'] for row in selected})
        sample_ids[split] = ids
        manifests[split], rows[split] = {}, {}
        for kind in VARIANTS:
            path = _resolve(data_dir, entry[kind + '_generation'])
            if file_sha256(path) != entry.get(kind + '_generation_sha256'):
                raise ValueError(f'{split}/{kind} generation manifest SHA mismatch')
            current = _read(path)
            if not isinstance(current, list) or len(current) != count:
                raise ValueError(f'{split}/{kind} strict generation row count mismatch')
            for row in current:
                validate_generation_manifest(row)
            manifests[split][kind] = path
            rows[split][kind] = current
        for first, second, selected_row in zip(rows[split]['short'], rows[split]['structured'], selected):
            if {k: v for k, v in first.items() if k != 'prompt'} != {k: v for k, v in second.items() if k != 'prompt'}:
                raise ValueError('short/structured generation must share the exact source and geometry')
            if first['person_uids'] != selected_row['person_uids']:
                raise ValueError('selected source UID bindings differ from generation')
        # Store only the path now. File content and GT assets are opened after generation.
        manifests[split]['supervision'] = _resolve(data_dir, entry['supervision'])
    if set(by_id) != {sid for ids in sample_ids.values() for sid in ids}:
        raise ValueError('selected sample IDs differ from actual split lists')
    if pair_ids['test'][0] in pair_ids['train'] + pair_ids['val']:
        raise ValueError('test pair overlaps training or validation')
    test_sample = by_id[sample_ids['test'][0]]
    if _selection_identity(test_sample) != LOCKED_TEST:
        raise ValueError('final evaluation requires the exact locked test pair00 taichi00/41 to hug00/108 cam4')
    selection = validate_selection_lock(data, data_dir, file_sha256(split_path))
    from src.data.hi4d_smplx_dataset import hi4d_asset_identity
    generation = rows['test']['short'][0]
    source_identity = hi4d_asset_identity(generation['source_rgb'], target=False)
    if source_identity != dict(pair='pair00', action='taichi00', camera=4, frame=41):
        raise ValueError('locked test source asset identity differs from taichi00/41 cam4')
    if Path(generation['target_geometry']).name != 'hug00_000108_cam4_geometry.npz':
        raise ValueError('locked test geometry asset identity differs from hug00/108 cam4')
    return dict(dataset_path=dataset_path, dataset_sha256=file_sha256(dataset_path),
        split_manifest=dict(path=str(split_path), sha256=file_sha256(split_path)),
        pair_ids=pair_ids, sample_ids=sample_ids, manifests=manifests, rows=rows,
        test_sample=test_sample, selection_lock=selection, metadata=data)


def validate_training_code(metadata, project=None):
    """Require the exact training closure, including controller/core/conditions."""
    from src.pose_control.qwen21_full.dual.prompt_pilot import prompt_pilot_code_sha256
    project = Path(__file__).resolve().parents[2] if project is None else Path(project).resolve()
    expected = metadata.get('code_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('checkpoint lacks the complete training runtime source SHA contract')
    current = prompt_pilot_code_sha256(project)
    if expected != current:
        changed = sorted(name for name in set(expected) | set(current) if expected.get(name) != current.get(name))
        raise ValueError(f'training runtime source SHA mismatch: {changed}')
    if metadata.get('run_contract', {}).get('code_sha256') != expected:
        raise ValueError('training runtime source SHA differs between checkpoint and run contract')
    return current


def validate_training_contracts(training_root, bundle):
    """Authenticate both finished runs and the validation-selected best payloads."""
    root = Path(training_root).resolve()
    complete_path = root / 'training_complete.json'
    complete = _read(complete_path)
    common = dict(dataset_sha256=bundle['dataset_sha256'], split_manifest=bundle['split_manifest'])
    def finished(record):
        if record.get('status') != 'success' or record.get('completed_steps') != 300 or record.get('requested_steps') != 300:
            raise ValueError('final test requires real success completion of both 300-step trainings')
        if any(record.get(k) != v for k, v in common.items()):
            raise ValueError('training completion dataset/split binding is stale')
        if record.get('backbone_unchanged') is not True or record.get('backbone_sha256_before') != record.get('backbone_sha256_after'):
            raise ValueError('training native backbone integrity was not verified')
    finished(complete)
    if complete.get('initialization_sha256_equal') is not True:
        raise ValueError('both variants must have verified identical initialization')
    result = {}
    for variant in VARIANTS:
        branch_path = root / variant / 'training_complete.json'
        branch = _read(branch_path)
        finished(branch)
        if branch.get('variant') != variant or complete.get('variants', {}).get(variant) != branch:
            raise ValueError('root/variant completion records disagree')
        if branch.get('initial_adapter_sha256') != complete.get('initial_adapter_sha256'):
            raise ValueError('variant initialization SHA differs')
        best = branch.get('best_checkpoint', {})
        path = _resolve(root / variant, best.get('path', ''))
        if path != root / variant / 'checkpoint_best.pt' or file_sha256(path) != best.get('sha256'):
            raise ValueError('best checkpoint SHA/path differs from completed training')
        validations = branch.get('validation_records', [])
        if not validations or any(type(row.get('step')) is not int or row['step'] not in range(50, 301, 50)
                or not isinstance(row.get('loss'), (int, float)) or not math.isfinite(row['loss']) for row in validations):
            raise ValueError('best selection requires actual finite validation records')
        if [row['step'] for row in validations] != list(range(50, 301, 50)):
            raise ValueError('validation must contain all six true-val evaluations')
        selected = min(validations, key=lambda row: (row['loss'], row['step']))
        if branch.get('best_step') != selected['step'] or branch.get('best_validation') != selected['loss'] or best.get('step') != selected['step']:
            raise ValueError('checkpoint best selection does not match minimum validation loss')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        metadata = payload.get('metadata', {})
        verified_training_code = validate_training_code(metadata)
        expected = dict(common, variant=variant, architecture_version='dual_v8',
            validation_selection='true_val_split_only', train_pair_ids=bundle['pair_ids']['train'],
            val_pair_ids=bundle['pair_ids']['val'], train_sample_ids=bundle['sample_ids']['train'],
            val_sample_ids=bundle['sample_ids']['val'])
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('checkpoint variant/dataset/pair holdout or validation contract mismatch')
        current_manifests = {split: {key: file_sha256(bundle['manifests'][split][kind])
            for key, kind in [('short_generation', 'short'), ('structured_generation', 'structured'),
                              ('supervision', 'supervision')]} for split in ('train', 'val')}
        if metadata.get('manifest_sha256') != current_manifests:
            raise ValueError('checkpoint training/validation manifest SHA mismatch')
        contents = metadata.get('run_contract', {}).get('data_content_sha256')
        if not isinstance(contents, dict) or contents.get(str(bundle['dataset_path'])) != bundle['dataset_sha256']:
            raise ValueError('checkpoint training data content contract is missing')
        test_supervision = bundle['manifests']['test']['supervision']
        for asset, digest in contents.items():
            if Path(asset).resolve() == test_supervision:
                raise ValueError('training data contract improperly includes test supervision')
            if file_sha256(asset) != digest:
                raise ValueError('checkpoint training data content SHA mismatch')
        if metadata.get('prompt_pilot_schema_version') != 'qwen21_dual_prompt_pilot_v1':
            raise ValueError('checkpoint is not a current prompt-pilot training')
        if payload.get('step') != selected['step'] or metadata.get('backbone_sha256') != branch['backbone_sha256_after']:
            raise ValueError('best payload step/native hash does not match completed training')
        if (metadata.get('best_step') != selected['step'] or metadata.get('best_validation') != selected['loss']
                or metadata.get('validation_loss') != selected['loss']):
            raise ValueError('best payload validation does not match the recorded minimum')
        if not payload.get('adapter_state_dict'):
            raise ValueError('best payload lacks actual adapter tensors')
        result[variant] = dict(path=path, sha256=best['sha256'], step=selected['step'],
            metadata=metadata, completion_sha256=file_sha256(branch_path),
            root_completion_sha256=file_sha256(complete_path), best_validation=selected['loss'],
            verified_training_code_sha256=verified_training_code)
        del payload
    if result['short']['metadata']['backbone_sha256'] != result['structured']['metadata']['backbone_sha256']:
        raise ValueError('variant native backbones differ')
    return result


def read_external_pose(geometry_path):
    """Load reliable camera observations; no target photographs are read."""
    from src.data.hi4d_smplx_dataset import letterbox_affine
    sidecar_path = Path(geometry_path).with_suffix('.json')
    sidecar = _read(sidecar_path)
    paths, shas = sidecar.get('pose_observation_paths'), sidecar.get('pose_observation_sha256')
    if not isinstance(paths, list) or len(paths) != 2 or not isinstance(shas, list) or len(shas) != 2:
        raise ValueError('geometry sidecar lacks two actual camera pose observations')
    points, confidence, sizes = [], [], []
    for path, expected in zip(paths, shas):
        if file_sha256(path) != expected:
            raise ValueError('actual target camera pose observation SHA mismatch')
        with np.load(path, allow_pickle=False) as values:
            xy, scores = values['xy_native'].copy(), values['scores'].copy()
            width, height = int(values['width']), int(values['height'])
        if xy.shape != (134, 2) or scores.shape != (134,) or not np.isfinite(scores).all():
            raise ValueError('actual pose observations require official reordered DWPose134')
        affine = letterbox_affine(width, height)
        points.append(xy * affine[0, 0] + affine[:2, 2])
        confidence.append(scores)
        sizes.append([width, height])
    if sizes[0] != sizes[1]:
        raise ValueError('actor pose observations must share the actual camera canvas')
    return np.asarray(points), np.asarray(confidence), dict(provenance='actual_camera_dwpose_observations',
        paths=paths, sha256=shas, native_sizes=sizes, geometry_sidecar_sha256=file_sha256(sidecar_path),
        confidence_threshold=.3, fitted_missing_joint_priors_used=False)


def optimizer_instance_count():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return sum(isinstance(value, torch.optim.Optimizer) for value in gc.get_objects())


class InferenceOnlyGuard:
    """Reject every reverse-mode and optimizer entry point, then restore them.

    Optimizer subclasses are patched individually because AdamW/SGD override
    Optimizer.step. The guard also rejects constructing a new optimizer.
    """
    def __init__(self):
        self.attempts = {name: 0 for name in ('tensor_backward', 'autograd_backward',
            'autograd_grad', 'optimizer_step', 'optimizer_init')}
        self._patches = []
        self.restored = False
        self.samples = []

    def _patch(self, owner, attribute, operation):
        original = getattr(owner, attribute)
        owned = attribute in owner.__dict__
        def rejected(*args, **kwargs):
            self.attempts[operation] += 1
            raise RuntimeError(f'inference-only evaluation forbids {operation}')
        self._patches.append((owner, attribute, original, owned))
        setattr(owner, attribute, rejected)

    def __enter__(self):
        self.mode = torch.inference_mode()
        self.mode.__enter__()
        try:
            self._patch(torch.Tensor, 'backward', 'tensor_backward')
            self._patch(torch.autograd, 'backward', 'autograd_backward')
            self._patch(torch.autograd, 'grad', 'autograd_grad')
            classes, queue = set(), [torch.optim.Optimizer]
            while queue:
                cls = queue.pop()
                if cls not in classes:
                    classes.add(cls)
                    queue.extend(cls.__subclasses__())
            for cls in sorted(classes, key=lambda cls: cls.__name__):
                if 'step' in cls.__dict__:
                    self._patch(cls, 'step', 'optimizer_step')
            self._patch(torch.optim.Optimizer, '__init__', 'optimizer_init')
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise
        return self

    def sample(self, key):
        row = dict(key=key, grad_enabled=torch.is_grad_enabled(),
            inference_mode=torch.is_inference_mode_enabled())
        self.samples.append(row)
        if row['grad_enabled'] or not row['inference_mode']:
            raise RuntimeError('actual generation escaped inference-only mode')
        return row

    def __exit__(self, exc_type, exc, traceback):
        for owner, name, original, owned in reversed(self._patches):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._patches.clear()
        self.restored = True
        return self.mode.__exit__(exc_type, exc, traceback)


def model_state_sha256(module):
    # Match the existing runtime hash while including actual state/buffers.
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def model_snapshot(models):
    return {name: dict(sha256=model_state_sha256(model),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        requires_grad_count=sum(parameter.requires_grad for parameter in model.parameters()),
        grad_count=sum(parameter.grad is not None for parameter in model.parameters()),
        training=model.training) for name, model in models.items()}


def planned_runs(seeds, short_prompt, structured_prompt):
    if list(seeds) != [42, 43, 44]:
        raise ValueError('final pilot requires the fixed preselected seeds 42 43 44')
    return [dict(key=f'{method}_{kind}_seed{seed}', method=method, prompt_kind=kind,
        prompt=prompt, seed=seed, explicit_target_geometry_control=method == 'dual',
        prompt_geometry_descriptions=kind == 'structured',
        prompt_geometry_provenance='reliable_external_target_geometry' if kind == 'structured' else 'action_label_only')
        for method in ('native', 'dual') for kind, prompt in [('short', short_prompt),
            ('structured', structured_prompt)] for seed in seeds]


def run_generation_phase(generate_fn, pipe, generations, adapters, plan, output, provenance,
                         *, steps=40, cfg=1., use_kv_cache=True):
    """Generate and audit all 12 images; this API has no evaluator GT input."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('evaluation requires a new output directory; destination already exists')
    if steps != 40 or cfg != 1. or len(plan) != 12 or len({row['key'] for row in plan}) != 12:
        raise ValueError('final comparison requires all 12 fixed 40-step CFG1 generations')
    output.mkdir(parents=True)
    guard = InferenceOnlyGuard()
    forward_samples, forward_hooks = [], []
    current_key = None
    models, outputs = {}, []
    before, after = {}, {}
    before_optimizers = optimizer_instance_count()
    try:
        if before_optimizers:
            raise RuntimeError('final test process must have no optimizer instances')
        for name in ('transformer', 'text_encoder', 'vae'):
            model = getattr(pipe, name, None)
            if isinstance(model, torch.nn.Module):
                models['native_' + name] = model
        for kind in VARIANTS:
            adapters[kind].eval().requires_grad_(False)
            models['adapter_' + kind] = adapters[kind]
        before = model_snapshot(models)
        if any(row['grad_count'] or row['requires_grad_count'] or row['training'] for row in before.values()):
            raise RuntimeError('native and adapter models must be frozen, eval, and have no parameter gradients')
        def observe_forward(module, inputs):
            sample = dict(key=current_key, grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled())
            forward_samples.append(sample)
            if sample['grad_enabled'] or not sample['inference_mode']:
                raise RuntimeError('actual native forward escaped inference-only mode')
        forward_hooks.append(pipe.transformer.register_forward_pre_hook(observe_forward))
        with guard:
            for spec in plan:
                current_key = spec['key']
                value = generations[spec['prompt_kind']]
                # Typed immutable production conditions; simple CPU doubles may
                # carry no seed-dependent state, but receive identical config.
                if hasattr(value, '__dataclass_fields__'):
                    value = replace(value, prompts=(spec['prompt'],), seeds=(spec['seed'],))
                else:
                    raise ValueError('generation requires an immutable typed source/geometry input')
                adapter = adapters[spec['prompt_kind']] if spec['method'] == 'dual' else None
                checkpoint = provenance.get('checkpoints', {}).get(spec['prompt_kind']) if adapter is not None else None
                configuration = dict(provenance, **spec, checkpoint=checkpoint,
                    steps=steps, cfg=cfg, use_kv_cache=use_kv_cache)
                started = time.time()
                mode = guard.sample(spec['key'])
                image = generate_fn(pipe, value, adapter=adapter, steps=steps, cfg=cfg, use_kv_cache=use_kv_cache)
                if image.size != (512, 512):
                    raise ValueError('generation must return actual 512 square images')
                path = output / (spec['key'] + '.png')
                image.convert('RGB').save(path)
                record = dict(configuration=configuration, configuration_sha256=configuration_sha256(configuration),
                    output=str(path), output_sha256=file_sha256(path),
                    explicit_target_geometry_control=spec['explicit_target_geometry_control'],
                    prompt_geometry_descriptions=spec['prompt_geometry_descriptions'],
                    prompt_geometry_provenance=spec['prompt_geometry_provenance'],
                    elapsed_seconds=time.time() - started, actual_gradient_state=mode,
                    generation_api_target_rgb_argument=False,
                    diagnostics=getattr(pipe, 'last_dual_v8_diagnostics', None))
                write_json(path.with_suffix('.json'), record)
                outputs.append(record)
                print(json.dumps(dict(generated=spec['key'], output_sha256=record['output_sha256'])), flush=True)
        after = model_snapshot(models)
        after_optimizers = optimizer_instance_count()
        audit = dict(before=before, after=after,
            weights_unchanged=all(before[name]['sha256'] == after[name]['sha256'] for name in models),
            all_parameter_grads_none=all(row['grad_count'] == 0 for row in after.values()),
            all_models_frozen_eval=all(not row['requires_grad_count'] and not row['training'] for row in after.values()),
            optimizer_instances_before=before_optimizers, optimizer_instances_after=after_optimizers,
            blocked_attempts=guard.attempts, patched_methods_restored=guard.restored,
            generation_samples=guard.samples, native_forward_samples=forward_samples,
            target_supervision_opened_during_generation=False)
        write_json(output / 'inference_guard.json', audit)
        if not audit['weights_unchanged'] or not audit['all_parameter_grads_none'] or not audit['all_models_frozen_eval'] or any(guard.attempts.values()) or after_optimizers:
            raise RuntimeError('final inference audit failed; weights/gradients/optimizer guard violation')
        return outputs, audit
    except BaseException as exc:
        write_json(output / 'evaluation_failed.json', dict(status='failed', phase='generation',
            error=f'{type(exc).__name__}: {exc}', generated_outputs=len(outputs),
            blocked_attempts=guard.attempts, patched_methods_restored=guard.restored,
            before=before, after=after, target_supervision_opened=False))
        raise
    finally:
        for hook in forward_hooks:
            hook.remove()


def generate_then_evaluate(generate_fn, pipe, generations, adapters, plan, output, provenance,
                           *, target_loader, score_fn, **settings):
    """Keep target loading behind successful completion of the frozen phase."""
    records, audit = run_generation_phase(generate_fn, pipe, generations, adapters,
        plan, output, provenance, **settings)
    target = target_loader()
    return records, audit, score_fn(records, target)
