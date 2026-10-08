"""Training utilities for the complete V6 controller connected to Qwen21.

The 47 copied architecture files remain unchanged. This module only manages
native Qwen I/O, optimizer selection, and execution of the preserved branches.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import FACE_ONLY_PREFIXES
from .data_runtime import FaceConditionStore, QwenFullDataset, build_condition_bundle, pair_metadata
from .face.adapter import scatter_face_roi_residuals
from .qwen_adapter import Qwen21FullAdapter, Qwen21FullControlContext
from .qwen_runtime import MODEL_REVISION, PROMPT, Qwen21IPERCache, dataset_image_to_pil, transformer_inputs

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full')
BODY_CHECKPOINT = ROOT / 'experiments/iper_v6_native_20k_v1/checkpoint_last.pt'
FACE_CHECKPOINT = ROOT / 'experiments/iper_v65_face_overfit_024_6/checkpoints/checkpoint_step_001500.pt'
MODEL = ROOT / 'models/Qwen-Image-2.1'
ARCHITECTURE = 'qwen21_full_v1'


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


def batch_one(dataset, index):
    sample = dataset[index]
    return {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
            for key, value in sample.items()}


def new_adapter(device='cpu'):
    adapter = Qwen21FullAdapter.from_checkpoints(
        ROOT / 'models/DeepGen-1.0-diffusers/transformer/config.json',
        BODY_CHECKPOINT, FACE_CHECKPOINT, enable_hand=True)
    return adapter.to(device=device, dtype=torch.float32)


def load_adapter_checkpoint(adapter, path):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected a qwen21_full checkpoint; simplified Qwen weights are incompatible')
    adapter.load_state_dict(saved['adapter_state_dict'], strict=True)
    return saved


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(path, saved):
    """Record the actual initialized weights and their original run manifest."""
    path = Path(path).resolve()
    manifest_path = path.parent / 'run_manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = {'path': str(manifest_path), 'sha256': file_sha256(manifest_path),
                    'content': json.loads(manifest_path.read_text())}
    return {'checkpoint_path': str(path), 'checkpoint_sha256': file_sha256(path),
            'checkpoint_step': saved.get('step'), 'checkpoint_branch': saved.get('branch'),
            'checkpoint_architecture': saved.get('architecture'),
            'checkpoint_model_revision': saved.get('model_revision'),
            'source_run_manifest': manifest}


def validate_resume_configuration(saved, args, previous_manifest=None):
    """Reject a continuation whose data, loss, or random sequence has changed.

    Missing newer options keep the defaults used before they were introduced.
    An init checkpoint deliberately does not use this continuation contract.
    """
    if saved['resolution'] != args.resolution or saved['branch'] != args.branch:
        raise ValueError('Resume requires the same resolution and branch')
    stored = saved['args']
    for key in ('seed', 'prompt', 'model', 'warmup_steps', 'train_pairs', 'val_pairs'):
        if str(stored[key]) != str(getattr(args, key)):
            raise ValueError(f'Resume configuration differs: {key}')
    defaults = {
        'dataset_type': 'auto', 'align_mode': 'resize',
        'tiktok_raw_root': ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset',
        'tiktok_assets_root': ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native',
        'face_flow_loss': 'global', 'face_aux_every': 4, 'face_aux_alignment': 'box',
        'face_evaluator_range': 'legacy', 'lr_scale': 1.0,
        'features': RUN_ROOT / 'face_cache', 'face_supervision': RUN_ROOT / 'face_supervision.json',
        'cache': RUN_ROOT / 'native_cache',
        'diagnostic_probes': False, 'diagnostic_pair_indices': [0],
        'require_active_gradients': False,
        'single_pass': False,
    }
    for key, default in defaults.items():
        previous = stored.get(key, default)
        current = getattr(args, key, default)
        if str(previous) != str(current):
            raise ValueError(f'Resume configuration differs: {key}; use init-checkpoint for a new experiment')
    previous_manifest = previous_manifest or {}
    for key in ('train_pairs', 'val_pairs'):
        expected = saved.get(f'{key}_sha256', previous_manifest.get(f'{key}_sha256'))
        if expected is None or expected != file_sha256(getattr(args, key)):
            raise ValueError(f'Resume dataset fingerprint differs or is missing: {key}')


def _metrics_snapshot(metrics_path, saved):
    """Validate every logged update before permitting a checkpoint rollback."""
    step = saved.get('step')
    order = saved.get('sample_order')
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError('Resume checkpoint step must be a nonnegative integer')
    if not isinstance(order, (list, tuple)) or len(order) != step:
        raise ValueError('Resume checkpoint sample_order length must equal checkpoint step')
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in order):
        raise ValueError('Resume checkpoint sample_order must contain nonnegative integer indices')
    path = Path(metrics_path)
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    rows = []
    for expected_step, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f'Resume metrics has an invalid JSON row at line {expected_step}') from error
        if not isinstance(row, dict):
            raise ValueError(f'Resume metrics row {expected_step} must be an object')
        actual_step = row.get('step')
        if (not isinstance(actual_step, int) or isinstance(actual_step, bool)
                or actual_step != expected_step):
            raise ValueError(f'Resume metrics steps must be exactly 1..N without missing or duplicate steps: '
                             f'line {expected_step} has step {actual_step!r}')
        index = row.get('pair_index')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f'Resume metrics row {expected_step} has an invalid pair_index')
        if expected_step <= step and index != order[expected_step - 1]:
            raise ValueError(f'Resume metrics checkpoint sample_order prefix differs at step {expected_step}')
        rows.append(row)
    if len(rows) < step:
        raise ValueError(f'Resume metrics is missing committed steps: expected 1..{step}, found 1..{len(rows)}')
    prefix = b''.join(lines[:step])
    # The trainer appends complete JSONL records. A truncated final record must
    # never be silently joined to the first update after a successful resume.
    if prefix and not prefix.endswith(b'\n'):
        raise ValueError('Resume metrics committed prefix must end with a complete JSONL newline')
    return raw, rows, prefix, b''.join(lines[step:])


def _atomic_replace_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.{path.name}.{uuid.uuid4().hex}.tmp'
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_reconciliation(path, ledger):
    _atomic_replace_bytes(path, (json.dumps(ledger, ensure_ascii=False, indent=2) + '\n').encode())


def _reconciliation_ledger(path):
    path = Path(path)
    if not path.exists():
        return {'schema_version': 1, 'events': []}
    ledger = json.loads(path.read_text())
    if ledger.get('schema_version') != 1 or not isinstance(ledger.get('events'), list):
        raise ValueError('Invalid resume reconciliation ledger')
    if sum(event.get('status') == 'pending' for event in ledger['events']) > 1:
        raise ValueError('Resume reconciliation has multiple unfinished events')
    for event in ledger['events']:
        if event.get('status') not in ('pending', 'completed'):
            raise ValueError('Invalid resume reconciliation event status')
        archive = Path(event['archive_path'])
        if not archive.is_file() or file_sha256(archive) != event['archive_sha256']:
            raise ValueError(f'Resume discarded-update archive is missing or changed: {archive}')
    return ledger


def _ledger_totals(ledger):
    completed = [event for event in ledger['events'] if event['status'] == 'completed']
    unknown_seconds = any(event['discarded_attempt_seconds'] is None for event in completed)
    return {'discarded_update_attempts': sum(event['discarded_update_attempts'] for event in completed),
            'discarded_attempt_seconds': (None if unknown_seconds else
                                         sum(event['discarded_attempt_seconds'] for event in completed)),
            'discarded_archive_paths': [event['archive_path'] for event in completed]}


def _logged_seconds(row):
    value = row.get('elapsed_seconds') if row else 0.0
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        return None
    return float(value)


def reconcile_resume_metrics(metrics_path, saved, *, checkpoint_path=None, checkpoint_sha256=None):
    """Keep the committed trajectory and durably archive any unsaved attempts.

    A pending ledger entry precedes the atomic log replacement. Retrying after
    interruption accepts either the original log or its retained prefix and
    completes the same event, without archiving or counting the tail twice.
    No Python, NumPy, or Torch random state is consumed by this operation.
    """
    metrics_path = Path(metrics_path).resolve()
    raw, rows, prefix, tail = _metrics_snapshot(metrics_path, saved)
    ledger_path = metrics_path.parent / 'resume_reconciliation.json'
    ledger = _reconciliation_ledger(ledger_path)
    step = saved['step']
    original_sha = hashlib.sha256(raw).hexdigest()
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    pending = next((event for event in ledger['events'] if event['status'] == 'pending'), None)
    if pending is not None:
        if (pending['checkpoint_step'] != step or pending['metrics_path'] != str(metrics_path)
                or pending['checkpoint_sha256'] != checkpoint_sha256):
            raise ValueError('Unfinished resume reconciliation belongs to a different checkpoint')
        if original_sha not in (pending['original_metrics_sha256'], pending['retained_metrics_sha256']):
            raise ValueError('Metrics changed during unfinished resume reconciliation')
        if prefix_sha != pending['retained_metrics_sha256']:
            raise ValueError('Committed prefix changed during unfinished resume reconciliation')
        if tail and hashlib.sha256(tail).hexdigest() != pending['archive_sha256']:
            raise ValueError('Discarded tail changed during unfinished resume reconciliation')
    elif tail:
        event_id = uuid.uuid4().hex
        archive_path = metrics_path.parent / f'{metrics_path.stem}.discarded_after_step_{step:06d}.{event_id}.jsonl'
        with archive_path.open('xb') as stream:
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
        original_seconds = _logged_seconds(rows[-1])
        retained_seconds = _logged_seconds(rows[step - 1] if step else None)
        discarded_seconds = (original_seconds - retained_seconds
                             if original_seconds is not None and retained_seconds is not None
                             and original_seconds >= retained_seconds else None)
        pair_uses = {}
        for row in rows[step:]:
            key = str(row['pair_index'])
            pair_uses[key] = pair_uses.get(key, 0) + 1
        pending = {
            'event_id': event_id, 'status': 'pending',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'metrics_path': str(metrics_path),
            'checkpoint_path': str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
            'checkpoint_sha256': checkpoint_sha256, 'checkpoint_step': step,
            'original_last_step': len(rows), 'retained_last_step': step,
            'original_row_count': len(rows), 'retained_row_count': step,
            'discarded_first_step': step + 1, 'discarded_last_step': len(rows),
            'discarded_update_attempts': len(rows) - step,
            'discarded_training_pair_uses': pair_uses,
            'discarded_unique_training_pairs': len(pair_uses),
            'original_metrics_sha256': original_sha, 'retained_metrics_sha256': prefix_sha,
            'archive_path': str(archive_path), 'archive_sha256': hashlib.sha256(tail).hexdigest(),
            'original_elapsed_seconds': original_seconds, 'retained_elapsed_seconds': retained_seconds,
            'discarded_attempt_seconds': discarded_seconds,
        }
        ledger['events'].append(pending)
        _write_reconciliation(ledger_path, ledger)
    if pending is not None:
        if tail:
            if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != original_sha:
                raise ValueError('Metrics changed while preparing resume reconciliation')
            _atomic_replace_bytes(metrics_path, prefix)
        pending['status'] = 'completed'
    ledger.update(_ledger_totals(ledger))
    ledger['last_resume'] = {
        'validated_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_step': step,
        'original_last_step': len(rows), 'retained_last_step': step,
        'rolled_back_update_attempts': len(rows) - step,
        'retained_metrics_sha256': prefix_sha, 'checkpoint_sha256': checkpoint_sha256,
        'recovered_pending_event': bool(pending and pending['original_last_step'] != len(rows)),
    }
    _write_reconciliation(ledger_path, ledger)
    return ledger


def training_attempt_accounting(metrics_path, *, step, sample_order, seconds):
    """Count the final model trajectory separately from archived update attempts."""
    _, rows, _, tail = _metrics_snapshot(metrics_path, {'step': step, 'sample_order': sample_order})
    if tail:
        raise ValueError('Final training metrics contains uncommitted update attempts')
    ledger = _reconciliation_ledger(Path(metrics_path).parent / 'resume_reconciliation.json')
    if any(event['status'] == 'pending' for event in ledger['events']):
        raise ValueError('Final training summary has an unfinished resume reconciliation')
    totals = _ledger_totals(ledger)
    discarded_seconds = totals['discarded_attempt_seconds']
    return {
        'optimizer_steps': step, 'training_pair_uses': len(rows),
        'unique_training_pairs': len(set(sample_order)),
        'repeated_training_pair_uses': len(rows) - len(set(sample_order)),
        **totals,
        'total_update_attempts': len(rows) + totals['discarded_update_attempts'],
        'training_seconds_including_discarded_attempts': (None if discarded_seconds is None else
                                                       float(seconds) + discarded_seconds),
        'attempt_time_scope': 'Logged training time; excludes initialization and unlogged work after the last metric',
        'nonzero_gradient_steps': sum(row.get('grad_norm', 0) > 0 for row in rows),
        'face_active_steps': sum(row.get('control_diagnostics', {}).get('face_active_rows', 0) > 0 for row in rows),
    }


def validate_probe_indices(dataset, pair_indices):
    indices = list(pair_indices)
    if not indices:
        raise ValueError('diagnostic-pair-indices must contain at least one training pair')
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError('diagnostic-pair-indices must be integer training pair indices')
    if len(set(indices)) != len(indices):
        raise ValueError('diagnostic-pair-indices must not contain duplicates')
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError('diagnostic-pair-indices contains an out-of-range training pair')
    return indices


def precompute(pipe, cache, datasets, *, should_stop=None):
    seen = {'source': set(), 'target': set()}
    for dataset in datasets:
        for row in dataset.samples:
            if should_stop is not None and should_stop():
                raise TimeoutError('Safe stop requested during native cache preparation')
            meta = pair_metadata(row, dataset.sampled_root)
            for side in ('source', 'target'):
                path = meta[f'{side}_rgb']
                # Precompute exactly the RGB transform used by __getitem__,
                # once per image/role, without constructing body heatmaps.
                if path in seen[side]:
                    continue
                image = dataset_image_to_pil(dataset.rgb_transform(dataset._load_image(path)))
                getattr(cache, side)(pipe, image)
                seen[side].add(path)
        print(f'Native Qwen cache ready: {len(dataset)} pairs at {cache.resolution}', flush=True)


def prepare(adapter, batch, pair, metadata, store, *, with_face):
    device = next(adapter.parameters()).device
    grid = int(pair['resolution']) // 16
    # Original typed condition records require one dtype across geometry/tokens.
    # Preserve those contracts by preparing in FP32; the dynamic core uses AMP.
    with torch.autocast(device.type, enabled=False):
        bundle = build_condition_bundle(adapter.legacy, batch, device=device, dtype=torch.float32)
        kwargs = {}
        if with_face:
            meta = [row if 'source_appearance' in row and 'target_appearance' in row else pair_metadata(row) for row in metadata]
            from .face_coordinates import batch_alignment_records
            face_condition, references = store.for_pairs(meta, device=device, dtype=torch.float32,
                                                        alignments=batch_alignment_records(batch, len(meta)))
            kwargs = dict(face_condition=face_condition, face_references=references)
        return adapter.prepare_conditioning(
            condition_bundle=bundle, source_latent=pair['source_latent'],
            target_token_hw=(grid, grid), **kwargs)


BODY_PREFIXES = (
    'condition_injector.spatial_encoder.', 'condition_injector.global_encoder.',
    'condition_injector.task_encoder.', 'reasoner.person_reasoner.', 'reasoner.role_embedding.',
    'reasoner.single_geometry_block.', 'reasoner.single_high_projection.', 'reasoner.single_high_block.',
    'condition_bridge.geometry_projection.', 'condition_bridge.geometry_high_downsample.',
    'geometry_token_projection.', 'appearance_token_encoder.', 'person_token_binder.',
    'control_interface.control_core.pos_embed.', 'control_interface.control_core.time_text_embed.',
    'control_interface.control_core.context_embedder.', 'control_interface.control_core.shared_block.',
    'control_interface.control_core.geometry_branch.',
    'control_interface.strength_controller.geometry_log_group_scale',
)


def parameter_groups(adapter, branch, *, lr_scale=1.0):
    if not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError('lr_scale must be positive and finite')
    groups = {'heads_and_interface': [], 'encoders': [], 'control_core': []}
    names = {key: [] for key in groups}
    for name, parameter in adapter.named_parameters():
        if name.startswith(f'output_bridges.{"geometry" if branch == "body" else "face"}.'):
            group = 'heads_and_interface'
        elif branch == 'body' and name.startswith(('latent_bridge.', 'text_bridge.', 'pooled_bridge.')):
            group = 'heads_and_interface'
        elif name.startswith('legacy.'):
            short = name[len('legacy.'):]
            allowed = BODY_PREFIXES if branch == 'body' else FACE_ONLY_PREFIXES
            if not short.startswith(allowed):
                continue
            if '.zero_heads.' in short or '_log_group_scale' in short:
                group = 'heads_and_interface'
            elif short.startswith(('control_interface.control_core.', 'control_interface.face_adapter.')):
                group = 'control_core'
            else:
                group = 'encoders'
        else:
            continue
        groups[group].append(parameter)
        names[group].append(name)
    rates = {'heads_and_interface': 1e-4, 'encoders': 5e-5, 'control_core': 1e-5}
    result = [{'params': groups[key], 'lr': rates[key] * lr_scale, 'name': key} for key in groups if groups[key]]
    if not result:
        raise ValueError('No trainable parameters selected')
    return result, names


def set_training_phase(adapter, branch, *, warmup):
    adapter.requires_grad_(False)
    _, names = parameter_groups(adapter, branch)
    allowed = {name for group in names.values() for name in group}
    head_prefix = f'output_bridges.{"geometry" if branch == "body" else "face"}.'
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad_(name.startswith(head_prefix) if warmup else name in allowed)
    return {name: sum(p.numel() for n, p in adapter.named_parameters() if p.requires_grad and n in group)
            for name, group in names.items()}


def frozen_digest(adapter, branch):
    _, groups = parameter_groups(adapter, branch)
    allowed = {name for group in groups.values() for name in group}
    digest = hashlib.sha256()
    for name, tensor in adapter.state_dict().items():
        if name in allowed:
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_frozen_gradients(adapter, **frozen_models):
    unexpected = [f'adapter.{name}' for name, parameter in adapter.named_parameters()
                  if not parameter.requires_grad and parameter.grad is not None]
    for scope, model in frozen_models.items():
        unexpected.extend(f'{scope}.{name}' for name, parameter in model.named_parameters()
                          if parameter.grad is not None)
    if unexpected:
        raise AssertionError(f'Frozen parameters received gradients: {unexpected[:8]}')


def assert_fp32_training_state(adapter, optimizer):
    unexpected = [name for name, parameter in adapter.named_parameters()
                  if parameter.dtype != torch.float32]
    if unexpected:
        raise AssertionError(f'Adapter parameters must remain FP32: {unexpected[:8]}')
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in state and state[key].dtype != torch.float32:
                raise AssertionError(f'Adam state must remain FP32: {key}')


def face_mask(prepared, grid):
    face = prepared.face
    if face is None:
        raise ValueError('Face loss requires full face conditioning')
    b = face.batch_size
    ones = torch.ones((b, 2, 16, 16, 1), device=face.target_boxes.device, dtype=face.target_boxes.dtype)
    mask = scatter_face_roi_residuals(ones, face.face_masks, face.target_boxes, face.face_valid, (grid, grid))
    return mask.detach().clamp(0, 1)


def face_loss_mask(prepared, grid):
    face = prepared.face
    if face is None:
        raise ValueError('Face loss requires full face conditioning')
    # Preserve train_iper_v65_face.rasterize_box_mask for the outside loss.
    # The Adapter itself still scatters its original feathered face residuals.
    boxes = face.target_boxes[:, 0].float()
    yy = (torch.arange(grid, device=boxes.device, dtype=boxes.dtype) + .5) / grid
    xx = (torch.arange(grid, device=boxes.device, dtype=boxes.dtype) + .5) / grid
    mask = ((xx[None, None, :] >= boxes[:, 0, None, None])
            & (xx[None, None, :] <= boxes[:, 2, None, None])
            & (yy[None, :, None] >= boxes[:, 1, None, None])
            & (yy[None, :, None] <= boxes[:, 3, None, None]))
    return mask.to(boxes.dtype).reshape(face.batch_size, grid * grid, 1).detach()


def crop_batch(images, boxes, side=112):
    # Match the endpoint grid and align_corners convention of the V6.5 trainer.
    axis = torch.linspace(0., 1., side, device=images.device, dtype=images.dtype)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    x0, y0, x1, y1 = [boxes[:, index, None, None] for index in range(4)]
    coords = torch.stack(((x0 + x[None] * (x1 - x0)) * 2 - 1,
                          (y0 + y[None] * (y1 - y0)) * 2 - 1), dim=-1)
    return F.grid_sample(images, coords, mode='bilinear', padding_mode='zeros', align_corners=True)


def decode_tokens(pipe, tokens, resolution):
    latents = pipe._unpack_latents(tokens, resolution, resolution, pipe.vae_scale_factor).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    pixels = (pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0].float() * .5 + .5).clamp(0, 1)
    return pixels[:, :3] * pixels[:, 3:4] + (1 - pixels[:, 3:4]) if pixels.shape[1] == 4 else pixels[:, :3]


def feature(model, image):
    value = model(image.float())
    return (value[-1] if isinstance(value, (tuple, list)) else value).flatten(1)


def face_velocity_loss(squared, mask, mode='global'):
    """Use a separately normalized face objective without discarding diagnostics.

    Historical checkpoints used the whole-image mean. A small face must not
    become a proportionally weaker objective solely because the canvas is large.
    """
    if mode not in ('global', 'roi'):
        raise ValueError('Face flow loss must be global or roi')
    if squared.ndim != 3 or mask.shape != (*squared.shape[:2], 1):
        raise ValueError('Expected squared [B,N,D] and face mask [B,N,1]')
    if mode == 'roi' and not bool((mask.sum(dim=1) > 0).all()):
        raise ValueError('ROI Face training requires a nonempty face in every sample')
    global_mean = squared.mean()
    face_mean = (squared * mask).sum() / (mask.sum() * squared.shape[-1]).clamp_min(1)
    outside_mean = (squared * (1 - mask)).sum() / ((1 - mask).sum() * squared.shape[-1]).clamp_min(1)
    return (face_mean if mode == 'roi' else global_mean), global_mean, face_mean, outside_mean


def run_loss(pipe, adapter, cache, dataset, store, index, branch, *, seed,
             auxiliary=False, evaluators=None, supervision=None, backward=True, fixed_sigma=None,
             check_auxiliary_gradients=False, require_active_face=False,
             normalized_face_evaluators=False, face_flow_loss='global', face_aux_alignment='box'):
    if face_aux_alignment not in ('box', 'five_point'):
        raise ValueError('Face auxiliary alignment must be box or five_point')
    if branch == 'face' and getattr(dataset, 'align_mode', 'resize') == 'letterbox' and face_flow_loss != 'roi':
        raise ValueError('Letterbox Face training requires the separately normalized face-region objective')
    device = next(adapter.parameters()).device
    batch = batch_one(dataset, index)
    metadata = [dataset.samples[index]]
    pair = cache.pair(pipe, batch, device)
    target = pair['target_latent']
    grid = cache.resolution // 16
    tokens = grid * grid
    generator = torch.Generator(device=device).manual_seed(seed)
    sigma = torch.rand((1,), device=device, dtype=target.dtype, generator=generator)
    if branch == 'face':
        # The inherited Face schedule is inactive at sigma >= .65. Sample
        # its active interval so each bounded Face step can update parameters;
        # auxiliary decoded-image losses use the lower-noise texture interval.
        sigma = sigma * (.35 if auxiliary else .65)
    if fixed_sigma is not None:
        sigma.fill_(fixed_sigma)
    sigma_resamples = 0
    if branch == 'face' and require_active_face and fixed_sigma is None:
        # BF16 cosine rounding can make the first sliver of the nominal active
        # interval exactly zero. Reject only those zero-schedule draws, preserving
        # the inherited schedule and the declared noise-range upper bound.
        schedule = adapter.legacy.control_interface.strength_controller.face_schedule
        dtype = torch.bfloat16 if torch.is_autocast_enabled('cuda') else torch.float32
        while not bool((schedule.multiplier(1 - sigma.to(dtype)) > 0).all()):
            sigma = torch.rand((1,), device=device, dtype=target.dtype, generator=generator) * (.35 if auxiliary else .65)
            sigma_resamples += 1
            if sigma_resamples > 100:
                raise RuntimeError('Unable to draw an active Face timestep')
    noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    velocity = noise - target
    prepared = prepare(adapter, batch, pair, metadata, store, with_face=branch == 'face')
    inputs = transformer_inputs(pair, noisy, sigma)
    outside_reference = None
    if branch == 'face':
        # The baseline touches trainable Face projections with zero inputs.
        # Do not cache no-grad autocast weights for the following train pass.
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
            with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                         face_strength=0, hand_strength=0, hand_mode='off'):
                outside_reference = pipe.transformer(**inputs)[0][:, -tokens:]
    with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                 face_strength=1 if branch == 'face' else 0,
                                 hand_strength=0, hand_mode='off') as context:
        prediction = pipe.transformer(**inputs)[0][:, -tokens:]
        squared = (prediction.float() - velocity.float()).square()
        identity = perceptual = outside = squared.new_zeros(())
        if branch == 'body':
            human = F.interpolate(batch['human_mask'].to(device=device, dtype=torch.float32),
                                  size=(grid, grid), mode='bilinear', align_corners=False)
            from src.data.letterbox import valid_body_velocity_loss
            valid = batch.get('valid_image_mask')
            if valid is not None:
                valid = F.interpolate(valid.to(device=device,dtype=torch.float32), size=(grid,grid),mode='area').flatten(2).transpose(1,2)
            flow = valid_body_velocity_loss(squared, human.flatten(2).transpose(1,2), valid)
        else:
            from .face_coordinates import valid_image_tokens
            mask = face_loss_mask(prepared, grid).to(device)
            valid_tokens = valid_image_tokens(batch, grid, device)
            if valid_tokens is not None:
                mask = mask * valid_tokens
            flow, global_flow, face_flow, nonface_flow = face_velocity_loss(squared, mask, face_flow_loss)
            outside_weight = (1 - mask) if valid_tokens is None else (1 - mask) * valid_tokens
            outside = ((prediction.float() - outside_reference.float()).square() * outside_weight).sum() / (
                outside_weight.sum() * prediction.shape[-1]).clamp_min(1)
            if auxiliary:
                if evaluators is None:
                    raise ValueError('Auxiliary face losses require identity and perceptual evaluators')
                if supervision is None:
                    raise ValueError('Face auxiliary losses require Target visibility eligibility')
                valid = prepared.face.face_valid[:, 0] & supervision.valid_mask(metadata, cache.resolution, device,
                                                                                 align_mode=getattr(dataset, 'align_mode', 'resize'))
                if valid.any():
                    x0 = noisy - sigma[:, None, None] * prediction
                    boxes = prepared.face.target_boxes[:, 0].float()
                    decoded = decode_tokens(pipe, x0, cache.resolution)
                    target_pixels = (batch['tgt_image'].to(device).float() * .5 + .5).clamp(0, 1)
                    if face_aux_alignment == 'five_point':
                        from .face_loss_alignment import build_arcface_alignment_grid, align_face_rgb
                        points = []
                        for item, keep in zip(metadata, valid.detach().cpu().tolist()):
                            if not keep:
                                continue
                            record = supervision.record(item)
                            eligibility = record['by_resolution'][str(cache.resolution)]
                            selected = eligibility['selected_detection']
                            keypoints = record['detections'][selected]['landmarks5_original_px']
                            if keypoints is None:
                                raise ValueError('Valid aligned face supervision is missing its five points')
                            output_points = record['detections'][selected].get('landmarks5_output_px', {}).get(str(cache.resolution))
                            value = torch.tensor(output_points if output_points is not None else keypoints, dtype=torch.float32)
                            if output_points is None:
                                if getattr(dataset, 'align_mode', 'resize') != 'resize':
                                    raise ValueError('Letterbox supervision is missing transformed five points')
                                value *= value.new_tensor([cache.resolution / record['original_width'],
                                                           cache.resolution / record['original_height']])
                            points.append(value)
                        grid = build_arcface_alignment_grid(torch.stack(points), decoded.shape[-2:],
                                                            device=device)
                        generated = align_face_rgb(decoded[valid], grid=grid)
                        reference = align_face_rgb(target_pixels[valid], grid=grid)
                    else:
                        generated = crop_batch(decoded, boxes)[valid]
                        reference = crop_batch(target_pixels, boxes)[valid]
                    if normalized_face_evaluators:
                        # The pinned ArcFace TorchScript consumes normalized RGB;
                        # the pinned VGG wrapper maps [-1,1] back to [0,1] itself.
                        generated = generated * 2 - 1
                        reference = reference * 2 - 1
                    with torch.no_grad(), torch.autocast('cuda', enabled=False):
                        target_id = feature(evaluators[0], reference)
                        target_perc = feature(evaluators[1], reference)
                    with torch.autocast('cuda', enabled=False):
                        identity = (1 - F.cosine_similarity(feature(evaluators[0], generated), target_id)).mean()
                        perceptual = (feature(evaluators[1], generated) - target_perc).square().mean()
        auxiliary_gradient_norms = {}
        if check_auxiliary_gradients:
            if not auxiliary or not backward:
                raise ValueError('Auxiliary gradient checks require a backward auxiliary Face step')
            parameter = adapter.output_bridges['face'].weight
            for name, loss in (('identity', identity), ('perceptual', perceptual)):
                gradients = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True) if loss.requires_grad else (None,)
                auxiliary_gradient_norms[name] = 0.0 if gradients[0] is None else float(gradients[0].detach().float().norm())
        total = flow + .10 * identity + .05 * perceptual + .02 * outside
        if backward:
            total.backward()
    diagnostics = getattr(context, 'output', None)
    meta = pair_metadata(metadata[0])
    report = {'loss': float(total.detach()), 'flow': float(flow.detach()), 'identity': float(identity.detach()),
              'perceptual': float(perceptual.detach()), 'outside': float(outside.detach()), 'sigma': float(sigma.item()),
              'pair_index': index, 'pair_id': meta['pair_id'], 'appearance': meta['source_appearance'],
              'source_appearance': meta['source_appearance'], 'target_appearance': meta['target_appearance'],
              'source': meta['source_stem'], 'target': meta['target_stem'], 'noise_seed': seed}
    if diagnostics is not None:
        report['control_diagnostics'] = diagnostics.diagnostics
    if auxiliary:
        report['auxiliary_eligible_faces'] = int(valid.sum())
    if check_auxiliary_gradients:
        report['auxiliary_gradient_norms'] = auxiliary_gradient_norms
    if require_active_face:
        report['zero_schedule_sigma_resamples'] = sigma_resamples
    if branch == 'face':
        report['face_flow_loss_mode'] = face_flow_loss
        report['whole_image_flow_mse'] = float(global_flow.detach())
        report['face_region_flow_mse'] = float(face_flow.detach())
        report['nonface_region_flow_mse'] = float(nonface_flow.detach())
        report['face_region_token_fraction'] = float(mask.mean())
    if auxiliary:
        report['face_evaluator_input_range'] = '[-1,1]' if normalized_face_evaluators else '[0,1] legacy'
        report['face_aux_alignment'] = face_aux_alignment
    return report


def capture_rng():
    return {'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state()}


def save_checkpoint(path, adapter, optimizer, *, step, branch, args, sample_order):
    payload = {'architecture': ARCHITECTURE, 'model_revision': MODEL_REVISION,
               'adapter_state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'step': step, 'branch': branch, 'resolution': args.resolution,
               'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'initialization': adapter.loading_report,
               'initialization_provenance': getattr(adapter, 'initialization_provenance',
                                                    {'mode': 'legacy_checkpoints',
                                                     'legacy_loading_report': adapter.loading_report,
                                                     'optimizer_state_restored': False}),
               'sample_order': sample_order,
               'train_pairs_sha256': hashlib.sha256(Path(args.train_pairs).read_bytes()).hexdigest(),
               'val_pairs_sha256': hashlib.sha256(Path(args.val_pairs).read_bytes()).hexdigest(),
               **capture_rng()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(saved):
    torch.set_rng_state(saved['torch_rng_state'])
    if saved['cuda_rng_state']:
        torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    random.setstate(saved['python_rng_state'])
    np.random.set_state(saved['numpy_rng_state'])


@contextmanager
def preserved_rng():
    saved = capture_rng()
    try:
        yield
    finally:
        restore_rng(saved)


def probe(pipe, adapter, cache, dataset, store, branch, *, pair_indices=(0,), **loss_options):
    indices = validate_probe_indices(dataset, pair_indices)
    device = next(adapter.parameters()).device
    rows = []
    with preserved_rng(), torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for offset, index in enumerate(indices):
            seed = 24042 + offset
            row = run_loss(pipe, adapter, cache, dataset, store, index, branch, seed=seed,
                           backward=False, fixed_sigma=.2 if branch == 'face' else .5, **loss_options)
            rows.append({**row, 'noise_seed': seed})
    if len(rows) == 1:
        return rows[0]
    return {'pair_indices': indices, 'rows': rows, 'fixed_noise_seeds': [row['noise_seed'] for row in rows],
            'loss': float(np.mean([row['loss'] for row in rows])),
            'flow': float(np.mean([row['flow'] for row in rows]))}


def diagnostic_probe(pipe, adapter, cache, dataset, store, branch, *, step,
                     pair_indices=(0,), **loss_options):
    """Eight fixed sigma/noise draws for every explicitly selected training pair."""
    indices = validate_probe_indices(dataset, pair_indices)
    sigmas = (.05, .10, .15, .20, .30, .40, .50, .60) if branch == 'face' else (.05, .15, .25, .35, .50, .65, .80, .95)
    device = next(adapter.parameters()).device
    rows = []
    with preserved_rng(), torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for pair_offset, pair_index in enumerate(indices):
            for sigma_offset, sigma in enumerate(sigmas):
                seed = 41042 + pair_offset * len(sigmas) + sigma_offset
                row = run_loss(pipe, adapter, cache, dataset, store, pair_index, branch, seed=seed,
                               backward=False, fixed_sigma=sigma, **loss_options)
                rows.append({**row, 'noise_seed': seed, 'fixed_sigma': sigma})
    return {'step': step, 'pair_indices': indices, 'sigmas': list(sigmas),
            'fixed_noise_seeds': [row['noise_seed'] for row in rows], 'rows': rows,
            'mean_flow': float(np.mean([row['flow'] for row in rows]))}
