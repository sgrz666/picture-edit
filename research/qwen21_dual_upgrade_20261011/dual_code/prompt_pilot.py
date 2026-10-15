"""Matched multi-sample prompt training with a closed train/validation boundary.

The data loader deliberately never resolves, stats or opens test manifests.
Native generation and the dual_v8 flow objective remain in their existing modules.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch

from src.data.hi4d_smplx_dataset import (
    GENERATION_KEYS, SUPERVISION_KEYS, Hi4DSMPLXDataset, TrainingTarget,
    file_sha256, hi4d_asset_identity, read_mask, read_rgb, validate_generation_manifest,
    validate_supervision_manifest)
from .checkpoint import warm_start_from_body
from .controller import DualV8Adapter
from .experiment import pilot_code_sha256, tensor_cache_sha256


DATA_SCHEMA_VERSION = 'hi4d_dual_prompt_pilot_v1'
TRAINING_SCHEMA_VERSION = 'qwen21_dual_prompt_training_complete_v1'
SMOKE_TRAINING_SCHEMA_VERSION = 'qwen21_dual_prompt_smoke_complete_v1'
VARIANTS = ('short', 'structured')
VALIDATION_SEEDS = (420001, 420002, 420003)
PROVENANCE_KEYS = {'split', 'pair', 'sample_id', 'action', 'source_frame', 'target_frame', 'camera'}
# Reviewed authorization map: pair, source action/frame, target action/frame,
# camera. The test row is intentionally absent from this training boundary.
APPROVED_TRAINING_ROWS = {
    'train': (('pair14','talk14',32,'hug14',45,4), ('pair14','talk14',32,'hug14',62,4),
              ('pair12','highfive12',28,'highfive12',33,4), ('pair12','highfive12',28,'highfive12',82,4)),
    'val': (('pair22','game22',52,'hug22',46,4),),
}


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def state_sha256(module):
    """Canonical tensor content hash, independent of torch archive metadata."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp',
                                     mode='w', encoding='utf-8', delete=False) as stream:
        pending = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def require_fresh_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('prompt pilot requires a fresh empty output directory; incomplete runs cannot be resumed')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError('manifest paths must be nonempty strings')
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def _rows(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ValueError('prompt pilot manifests require nonempty row lists')
    return value


def _canonical_row(row, *, supervision, base):
    allowed = SUPERVISION_KEYS if supervision else GENERATION_KEYS
    unknown = set(row)-allowed-PROVENANCE_KEYS
    if unknown:
        raise ValueError(f'unsupported prompt pilot row fields: {sorted(unknown)}')
    canonical = copy.deepcopy({key: value for key, value in row.items() if key in allowed})
    for name in ('target_rgb', 'source_rgb', 'source_features', 'target_geometry'):
        if name in canonical:
            canonical[name] = str(_path(canonical[name], base))
    for name in ('source_masks', 'target_masks'):
        if name in canonical:
            canonical[name] = [str(_path(value, base)) for value in canonical[name]]
    return canonical


@dataclass(frozen=True)
class TrainingSplit:
    name: str
    sample_ids: tuple[str, ...]
    rows: dict
    supervision_rows: tuple[dict, ...]
    manifest_paths: dict
    pair_ids: tuple[str, ...]

    def generation(self, index, variant):
        # Reuse the strict historical tensor loader without rewriting input files.
        dataset = Hi4DSMPLXDataset.__new__(Hi4DSMPLXDataset)
        dataset.path = self.manifest_paths[f'{variant}_generation']
        dataset.rows = self.rows[variant]
        dataset.supervision_manifest = None
        return dataset.generation(index)

    def supervision(self, index):
        metadata = validate_supervision_manifest(self.supervision_rows[index], self.rows['short'][index])
        metadata = dict(metadata, supervision_manifest_sha256=file_sha256(self.manifest_paths['supervision']))
        masks = torch.stack([read_mask(path) for path in metadata['target_masks']])[None]
        return TrainingTarget(read_rgb(metadata['target_rgb'])[None], masks, metadata)


@dataclass(frozen=True)
class PromptPilotData:
    dataset_path: Path
    dataset_sha256: str
    split_manifest: dict
    authoritative_pair_ids: dict
    splits: dict[str, TrainingSplit]
    content_sha256: dict[str, str]
    binding_sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self,'binding_sha256',self.current_binding_sha256())

    def current_binding_sha256(self):
        return json_sha256(dict(dataset_path=str(self.dataset_path),dataset_sha256=self.dataset_sha256,
            split_manifest=self.split_manifest,authoritative_pair_ids=self.authoritative_pair_ids,
            content_sha256=self.content_sha256,splits={name:dict(sample_ids=split.sample_ids,
                rows=split.rows,supervision_rows=split.supervision_rows,pair_ids=split.pair_ids,
                manifest_paths={key:str(path) for key,path in split.manifest_paths.items()})
                for name,split in self.splits.items()}))

    def assert_unchanged(self):
        if self.current_binding_sha256()!=self.binding_sha256:
            raise ValueError('prompt pilot in-memory data binding changed')
        for path, expected in self.content_sha256.items():
            if file_sha256(path) != expected:
                raise ValueError(f'prompt pilot input content changed: {path}')


def load_training_data(data_dir):
    """Validate true train/val rows while leaving every test asset untouched."""
    dataset_path = (Path(data_dir)/'dataset.json').resolve()
    manifest = json.loads(dataset_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != DATA_SCHEMA_VERSION:
        raise ValueError(f'prompt pilot requires {DATA_SCHEMA_VERSION}')
    binding = manifest.get('split_manifest')
    if isinstance(binding, str):
        binding = dict(path=binding, sha256=manifest.get('split_manifest_sha256'))
    if not isinstance(binding, dict) or not {'path', 'sha256'} <= set(binding):
        raise ValueError('split manifest requires its original path and SHA256')
    original = _path(binding['path'], dataset_path.parent)
    split_sha = file_sha256(original)
    if split_sha != binding['sha256']:
        raise ValueError('split manifest SHA256 does not match its original content')
    original_split = json.loads(original.read_text(encoding='utf-8'))
    pairs = {}
    for name in ('train', 'val', 'test'):
        value = original_split.get(f'{name}_pairs', original_split.get(name))
        if not isinstance(value, list) or any(not isinstance(pair, str) or not pair.startswith('pair') for pair in value):
            raise ValueError(f'original split requires explicit {name} pair membership')
        if len(set(value)) != len(value):
            raise ValueError(f'original split {name} pair membership has duplicates')
        pairs[name] = tuple(value)
    if any(set(pairs[a]) & set(pairs[b]) for a, b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('original split pair membership is not disjoint')
    content = {str(dataset_path): file_sha256(dataset_path), str(original): split_sha}
    splits = {}
    all_samples = set()
    split_uids = {}
    for name in ('train', 'val'):
        entry = manifest.get('splits', {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f'missing {name} split manifests')
        ids = entry.get('sample_ids')
        if not isinstance(ids, list) or any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{name} sample IDs must be nonempty and unique')
        if all_samples & set(ids):
            raise ValueError('train and val sample IDs must be disjoint')
        if name == 'train' and len(ids) < 2:
            raise ValueError('multi-sample train split requires at least two samples')
        if name == 'train' and len(ids) != 4:
            raise ValueError('approved production pilot requires exactly four train samples')
        if name == 'val' and len(ids) != 1:
            raise ValueError('approved prompt pilot requires exactly one true val sample')
        all_samples.update(ids)
        paths = {key: _path(entry.get(key), dataset_path.parent)
                 for key in ('short_generation', 'structured_generation', 'supervision')}
        raw = {key: _rows(path) for key, path in paths.items()}
        if any(len(rows) != len(ids) for rows in raw.values()):
            raise ValueError(f'{name} manifest row counts do not match sample IDs')
        rows = {variant: tuple(_canonical_row(row, supervision=False, base=paths[f'{variant}_generation'].parent)
                                for row in raw[f'{variant}_generation']) for variant in VARIANTS}
        truths = tuple(_canonical_row(row, supervision=True, base=paths['supervision'].parent)
                       for row in raw['supervision'])
        selected_pairs, uids, observations = [], set(), set()
        for index, sid in enumerate(ids):
            short, structured, truth = rows['short'][index], rows['structured'][index], truths[index]
            for generation in (short, structured):
                validate_generation_manifest(generation)
            if {key:value for key,value in short.items() if key!='prompt'} != {key:value for key,value in structured.items() if key!='prompt'}:
                raise ValueError(f'{name} prompt variants must have identical source, geometry, actor slots and sample order')
            if not all(isinstance(row['prompt'], str) and row['prompt'].strip() for row in (short, structured)) or short['prompt'] == structured['prompt']:
                raise ValueError(f'{name} short and structured variants require different prompts')
            if truth.get('pair') not in pairs[name]:
                raise ValueError(f'{name} row is outside its original {name} pair membership')
            actual_source = hi4d_asset_identity(short['source_rgb'],target=False)
            authorized_row = (truth['pair'],actual_source['action'],actual_source['frame'],
                              truth['action'],truth['target_frame'],truth['camera'])
            if authorized_row not in APPROVED_TRAINING_ROWS[name]:
                raise ValueError(f'{name} source/target tuple is outside the approved pilot selection: {authorized_row}')
            for key in ('short_generation', 'structured_generation', 'supervision'):
                provenance = raw[key][index]
                for field, expected in dict(split=name, sample_id=sid, pair=truth['pair'],
                                            action=truth['action'], source_frame=truth['source_frame'],
                                            target_frame=truth['target_frame'], camera=truth['camera']).items():
                    if field in provenance and provenance[field] != expected:
                        raise ValueError(f'{name} row provenance {field} disagrees with its sample binding')
            validate_supervision_manifest(truth, short)
            if file_sha256(short['source_features']) != short['source_features_sha256']:
                raise ValueError('source feature SHA256 does not match generation manifest')
            observation = (truth['pair'],truth['action'],truth['source_frame'],truth['target_frame'],truth['camera'])
            if observation in observations:
                raise ValueError(f'{name} contains duplicate source/target observations')
            observations.add(observation)
            selected_pairs.append(truth['pair'])
            uids.update(short['person_uids'])
            assets = [short['source_rgb'], short['source_features'], short['target_geometry'],
                      str(Path(short['target_geometry']).with_suffix('.json')), truth['target_rgb'],
                      *short['source_masks'], *truth['target_masks']]
            for path in assets:
                content[path] = file_sha256(path)
        for path in paths.values():
            content[str(path)] = file_sha256(path)
        split_uids[name] = uids
        splits[name] = TrainingSplit(name, tuple(ids), rows, truths, paths, tuple(sorted(set(selected_pairs))))
    if set(splits['train'].pair_ids) & set(splits['val'].pair_ids) or split_uids['train'] & split_uids['val']:
        raise ValueError('selected train/val pair identities and manifest person_uids must be disjoint')
    return PromptPilotData(dataset_path, content[str(dataset_path)], dict(path=str(original), sha256=split_sha),
                           pairs, splits, content)


@dataclass(frozen=True)
class Microbatch:
    step: int
    micro: int
    sample_index: int
    sample_id: str
    noise_seed: int


def shared_schedule(sample_ids, *, steps, seed):
    ids = tuple(sample_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or not all(isinstance(sid, str) and sid for sid in ids):
        raise ValueError('shared schedule requires at least two unique sample IDs')
    if type(steps) is not int or not 1 <= steps <= 300 or type(seed) is not int or not 0 <= seed <= 2**63-1200:
        raise ValueError('shared schedule requires 1..300 steps and a valid nonnegative seed')
    return tuple(Microbatch(step=index//4+1, micro=index%4, sample_index=index%len(ids),
                           sample_id=ids[index%len(ids)], noise_seed=seed+index)
                 for index in range(steps*4))


def schedule_sha256(schedule):
    return json_sha256([asdict(item) for item in schedule])


def source_prompt_cache_key(row):
    """A text-image condition key uses source-only content and the exact prompt."""
    return json_sha256(dict(prompt=row['prompt'],source_rgb_path=row['source_rgb'],
        source_rgb_sha256=file_sha256(row['source_rgb']),source_mask_sha256=[file_sha256(path) for path in row['source_masks']],
        source_features_sha256=row['source_features_sha256'],person_uids=row['person_uids'],versions=row['versions']))


def _cache_value(value, *, clone=False):
    if torch.is_tensor(value):
        return value.detach().cpu().clone() if clone else value
    if is_dataclass(value):
        return {item.name:_cache_value(getattr(value,item.name),clone=clone) for item in fields(value)}
    if isinstance(value, dict):
        return {key:_cache_value(child,clone=clone) for key,child in value.items()}
    if isinstance(value, (tuple,list)):
        return tuple(_cache_value(child,clone=clone) for child in value)
    return value


@dataclass(frozen=True)
class CachedSample:
    sample_id: str
    generation: object
    source: dict
    clean: torch.Tensor
    content_sha256: str = field(init=False)

    def __post_init__(self):
        # New cache entries returned by the native encoder can be inference
        # tensors. Clone outside inference_mode before any training backward.
        object.__setattr__(self,'source',_cache_value(self.source,clone=True))
        object.__setattr__(self,'clean',_cache_value(self.clean,clone=True))
        object.__setattr__(self,'content_sha256',self.current_sha256())

    def current_sha256(self):
        return tensor_cache_sha256(dict(sample_id=self.sample_id,generation=_cache_value(self.generation),
                                       source=self.source,clean=self.clean))

    def assert_unchanged(self):
        if self.current_sha256()!=self.content_sha256:
            raise ValueError(f'prompt pilot cache content changed: {self.sample_id}')


def prompt_pilot_code_sha256(project):
    project = Path(project)
    closure = pilot_code_sha256(project)
    for name in ('src/pose_control/qwen21_full/dual/prompt_pilot.py',
                 'scripts/train/train_qwen21_dual_prompt_pilot.py',
                 'src/data/hi4d_dual_prompt_pilot.py',
                 'scripts/precompute/prepare_hi4d_dual_prompt_pilot.py'):
        path = project/name
        if path.exists():
            closure[name] = file_sha256(path)
    return closure


def fresh_adapter(seed, warm_start, *, device='cpu', adapter_kwargs=None):
    # Native loading and earlier runs cannot consume the initialization RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = DualV8Adapter(**({} if adapter_kwargs is None else adapter_kwargs))
    report = warm_start_from_body(adapter, warm_start, require_frozen_coverage=True, expected_source_step=19500)
    adapter.freeze_warm_geometry()
    initial = state_sha256(adapter)
    return adapter.to(device=device, dtype=torch.float32).train(), report, initial


@contextmanager
def _validation_guard(optimizer):
    counts = {name:0 for name in ('tensor_backward','autograd_backward','autograd_grad','optimizer_step')}
    saved = []
    def block(name):
        def forbidden(*args, **kwargs):
            counts[name] += 1
            raise RuntimeError(f'validation {name} is forbidden')
        return forbidden
    targets = [(torch.Tensor,'backward','tensor_backward'), (torch.autograd,'backward','autograd_backward'),
               (torch.autograd,'grad','autograd_grad'), (type(optimizer),'step','optimizer_step')]
    try:
        for obj, attr, label in targets:
            saved.append((obj,attr,getattr(obj,attr)))
            setattr(obj,attr,block(label))
        with torch.inference_mode():
            yield counts
    finally:
        for obj, attr, previous in reversed(saved):
            setattr(obj,attr,previous)


def validation_pass(adapter, backbone, optimizer, evaluate, *, sample_count):
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError('validation requires actual val samples')
    optimizer.zero_grad(set_to_none=True)
    if any(param.grad is not None for param in adapter.parameters()):
        raise AssertionError('validation requires cleared adapter gradients')
    if any(param.requires_grad or param.grad is not None for param in backbone.parameters()):
        raise AssertionError('validation native backbone must be frozen without gradients')
    before = state_sha256(adapter)
    backbone_versions = [(name,id(value),value._version) for name,value in backbone.named_parameters()]
    was_training = adapter.training
    adapter.eval()
    losses = []
    try:
        with _validation_guard(optimizer) as attempts:
            for index in range(sample_count):
                for seed in VALIDATION_SEEDS:
                    value = evaluate(index, seed)
                    loss = float(value.detach()) if torch.is_tensor(value) else float(value)
                    if not math.isfinite(loss):
                        raise FloatingPointError('validation loss is nonfinite')
                    losses.append(loss)
        after = state_sha256(adapter)
        unchanged = backbone_versions == [(name,id(value),value._version) for name,value in backbone.named_parameters()]
        grads_none = all(param.grad is None for model in (adapter,backbone) for param in model.parameters())
        if before != after or not unchanged or not grads_none:
            raise AssertionError('validation changed weights or parameter gradients')
        evidence = dict(adapter_sha256_before=before,adapter_sha256_after=after,adapter_unchanged=True,
            backbone_parameter_versions_unchanged=unchanged,autograd_enabled=False,inference_mode=True,
            gradients_none=grads_none,optimizer_steps=0,forbidden_attempts=dict(attempts),
            seeds=list(VALIDATION_SEEDS),sample_count=sample_count,losses=losses,selection='true_val_split_only')
        return sum(losses)/len(losses), evidence
    finally:
        adapter.train(was_training)


def run_variant_updates(adapter, backbone, optimizer, schedule, train_forward, validation_forward, *,
                        sample_count, save_every, on_save, metrics_path):
    """Four real loss/backward microbatches per update; select only true val losses."""
    schedule = tuple(schedule)
    if not schedule or len(schedule)%4 or not 1 <= len(schedule)//4 <= 300 or type(save_every) is not int or save_every < 1:
        raise ValueError('invalid bounded prompt training schedule or save interval')
    if any((item.step,item.micro)!=(index//4+1,index%4) for index,item in enumerate(schedule)):
        raise ValueError('training schedule must contain four ordered microbatches per step')
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        raise ValueError('training metrics require a fresh output path')
    parameters = [param for param in adapter.parameters() if param.requires_grad]
    best, best_step = float('inf'), None
    validation_records = []
    began = time.monotonic()
    with metrics_path.open('x', encoding='utf-8', buffering=1) as metrics:
        for offset in range(0,len(schedule),4):
            step = schedule[offset].step
            optimizer.zero_grad(set_to_none=True)
            losses, micro_stats = [], []
            for item in schedule[offset:offset+4]:
                loss, stats = train_forward(item, 4)
                scalar = float(loss.detach())
                if not math.isfinite(scalar):
                    raise FloatingPointError(f'nonfinite training loss at step {step}')
                losses.append(scalar)
                micro_stats.append(dict(sample_id=item.sample_id,noise_seed=item.noise_seed,**stats))
                del loss
            norms = {}
            for name, param in adapter.named_parameters():
                if param.grad is not None:
                    if not bool(torch.isfinite(param.grad).all()):
                        raise FloatingPointError(f'nonfinite training gradient: {name}')
                    norms[name] = float(param.grad.detach().float().norm())
            if not norms or not any(value > 0 for value in norms.values()):
                raise AssertionError('adapter received no nonzero training gradients')
            if any(param.grad is not None or param.requires_grad for param in backbone.parameters()):
                raise AssertionError('native backbone received gradients or became trainable')
            total_norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.))
            if not math.isfinite(total_norm):
                raise FloatingPointError('training gradient norm is nonfinite')
            optimizer.step()
            record = dict(step=step,loss=sum(losses)/4,microbatches=micro_stats,
                          gradient_norm=total_norm,finite_gradients=True,elapsed_seconds=time.monotonic()-began)
            if step%save_every==0 or offset+4==len(schedule):
                score,evidence = validation_pass(adapter,backbone,optimizer,validation_forward,sample_count=sample_count)
                is_best = score < best
                if is_best:
                    best,best_step = score,step
                validation_records.append(dict(step=step,loss=score))
                record.update(validation_loss=score,validation=evidence,best_step=best_step,best_validation=best)
                on_save(step,score,is_best,evidence)
            metrics.write(json.dumps(record,allow_nan=False)+'\n')
            print('TRAIN',step,record['loss'],'elapsed',round(record['elapsed_seconds'],1),flush=True)
    return dict(completed_steps=len(schedule)//4,best_step=best_step,best_validation=best,
                validation_records=validation_records,elapsed_seconds=time.monotonic()-began)
