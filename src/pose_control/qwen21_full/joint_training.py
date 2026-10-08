"""Joint Body/Face optimization without changing the inherited architecture."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def freeze_evaluator(model):
    # RecursiveScriptModule explicitly forbids Module.requires_grad_.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def assert_exact_state(stored, current, path='state'):
    import numpy as np
    if isinstance(stored, torch.Tensor):
        equal = torch.equal(stored.detach().cpu(), current.detach().cpu())
    elif isinstance(stored, np.ndarray):
        equal = np.array_equal(stored, current)
    elif isinstance(stored, dict):
        if stored.keys() != current.keys():
            raise AssertionError(f'Restored keys differ: {path}')
        for key in stored:
            assert_exact_state(stored[key], current[key], f'{path}.{key}')
        return
    elif isinstance(stored, (tuple, list)):
        if len(stored) != len(current):
            raise AssertionError(f'Restored sequence differs: {path}')
        for index, (a, b) in enumerate(zip(stored, current)):
            assert_exact_state(a, b, f'{path}[{index}]')
        return
    else:
        equal = stored == current
    if not equal:
        raise AssertionError(f'Restored value differs: {path}')


def validate_group_gradients(norms, *, branches=('body', 'face')):
    for branch in branches:
        groups = {name: value for name, value in norms.items() if name.startswith(branch + '_')}
        if not groups:
            raise AssertionError(f'No {branch} optimizer groups')
        for name, value in groups.items():
            if not math.isfinite(value) or value <= 0:
                raise AssertionError(f'Gradient interrupted/nonfinite: {name}={value}')


def validate_preflight(report, signature, initialization_sha256):
    if report.get('passed') is not True:
        raise ValueError('A passed gradient preflight is required')
    validate_joint_signature(report['joint_signature'], signature)
    if report.get('initialization_sha256') != initialization_sha256:
        raise ValueError('Gradient preflight initialization checkpoint differs')


def validate_locked_split(train_path, val_path):
    train_path, val_path = Path(train_path).resolve(), Path(val_path).resolve()
    roles = {'common256_pairs.jsonl': 'common', 'branch256_pairs.jsonl': 'branch'}
    if train_path.name not in roles:
        raise ValueError('Training manifest role must be common256 or branch256')
    lock_path = train_path.parent / 'lock.json'
    lock = json.loads(lock_path.read_text())
    manifests, hashes = {}, {}
    for name, count in [('common256_pairs.jsonl', 256), ('branch256_pairs.jsonl', 256),
                        ('train512_pairs.jsonl', 512), ('val24_pairs.jsonl', 24)]:
        path = train_path.parent / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != lock['sha256'][name]:
            raise ValueError(f'Locked manifest changed: {name}')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != count or len({row['pair_id'] for row in rows}) != count:
            raise ValueError(f'Wrong count/duplicate pairs: {name}')
        manifests[name] = rows
    if val_path != train_path.parent / 'val24_pairs.jsonl':
        raise ValueError('Validation must use the locked held-out val24 manifest')
    common, branch, full, val = [manifests[name] for name in manifests]
    if common + branch != full:
        raise ValueError('Common/branch order differs from locked full512 split')
    if len({(row['sequence_id'], row['target_stem']) for row in full}) != 512:
        raise ValueError('Repeated target frames in full512 split')
    if {row['source_video_group'] for row in full} & {row['source_video_group'] for row in val}:
        raise ValueError('Train/validation original-video overlap')
    return {'role': roles[train_path.name], 'lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            'full512_sha256': hashes['train512_pairs.jsonl'], 'common_sha256': hashes['common256_pairs.jsonl'],
            'branch_sha256': hashes['branch256_pairs.jsonl']}


def joint_parameter_groups(adapter, *, body_scale=.25, face_scale=.5):
    from .execution import parameter_groups
    result, names, seen = [], {}, set()
    for branch, scale in [('body', body_scale), ('face', face_scale)]:
        groups, branch_names = parameter_groups(adapter, branch, lr_scale=scale)
        for group in groups:
            key = f'{branch}_{group["name"]}'
            retained = [(name, parameter) for name, parameter in zip(branch_names[group['name']], group['params'])
                        if id(parameter) not in seen]
            if not retained:
                continue
            seen.update(id(parameter) for _, parameter in retained)
            names[key] = [name for name, _ in retained]
            result.append({**group, 'name': key, 'params': [parameter for _, parameter in retained],
                           'base_lr': group['lr']})
    return result, names


def enable_joint_training(adapter, names):
    allowed = {name for group in names.values() for name in group}
    for name, parameter in adapter.named_parameters():
        parameter.requires_grad_(name in allowed)
    return allowed


def frozen_joint_digest(adapter, names):
    allowed = {name for group in names.values() for name in group}
    digest = hashlib.sha256()
    for name, tensor in adapter.state_dict().items():
        if name not in allowed:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_joint_signature(stored, current):
    if stored.keys() != current.keys():
        raise ValueError('Joint resume signature fields differ')
    for key in stored:
        if stored[key] != current[key]:
            raise ValueError(f'Joint resume configuration differs: {key}')


def joint_velocity_loss(squared, human, valid, face_mask, *, face_active, face_weight=.5):
    from src.data.letterbox import valid_body_velocity_loss
    body = valid_body_velocity_loss(squared, human, valid)
    face = squared.new_zeros(())
    if face_active:
        weight = face_mask if valid is None else face_mask * valid
        denominator = weight.sum() * squared.shape[-1]
        if not bool(denominator > 0):
            raise ValueError('Active Face has an empty valid ROI')
        face = (squared * weight).sum() / denominator
    return body + face_weight * face, {'body': body, 'face': face}


def joint_group_gradients(adapter, names):
    parameters = dict(adapter.named_parameters())
    return {key: sum(float(parameters[name].grad.detach().float().square().sum())
                     for name in group if parameters[name].grad is not None) ** .5
            for key, group in names.items()}


def run_joint_loss(pipe, adapter, cache, dataset, store, index, *, seed, auxiliary=False,
                   evaluators=None, supervision=None, backward=True, fixed_sigma=None,
                   face_weight=.5, check_auxiliary_gradients=False):
    from .execution import (batch_one, crop_batch, decode_tokens, face_loss_mask, feature,
                            pair_metadata, prepare, transformer_inputs)
    from .face_coordinates import valid_image_tokens
    from .qwen_adapter import Qwen21FullControlContext
    device = next(adapter.parameters()).device
    batch = batch_one(dataset, index)
    metadata = [dataset.samples[index]]
    pair = cache.pair(pipe, batch, device)
    target = pair['target_latent']
    grid = cache.resolution // 16
    generator = torch.Generator(device=device).manual_seed(seed)
    sigma = torch.rand((1,), device=device, dtype=target.dtype, generator=generator)
    if fixed_sigma is not None:
        sigma.fill_(fixed_sigma)
    noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
    noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
    velocity = noise - target
    prepared = prepare(adapter, batch, pair, metadata, store, with_face=True)
    if prepared.face is None or not bool(prepared.face.face_valid[:, 0].all()):
        raise ValueError(f'Joint training requires a valid source identity and target Face geometry: {index}')
    schedule = adapter.legacy.control_interface.strength_controller.face_schedule
    schedule_dtype = torch.bfloat16 if torch.is_autocast_enabled(device.type) else target.dtype
    face_active = bool((schedule.multiplier(1 - sigma.to(schedule_dtype)) != 0).all())
    context = Qwen21FullControlContext(
        pipe.transformer, adapter, prepared,
        geometry_strength=1, interaction_strength=0, detail_strength=1,
        face_strength=1, hand_strength=0, hand_mode='off',
    )
    # detail_strength is also a Face gate in the inherited interface. No shared
    # Detail payload is provided, so that branch remains absent.
    if prepared.detail is not None or prepared.hand is not None:
        raise AssertionError('Joint Body/Face pilot must not prepare Hand/shared Detail')
    with context:
        prediction = pipe.transformer(**transformer_inputs(pair, noisy, sigma))[0][:, -(grid * grid):]
        squared = (prediction.float() - velocity.float()).square()
        human = F.interpolate(batch['human_mask'].to(device=device, dtype=torch.float32),
                              size=(grid, grid), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        valid_tokens = valid_image_tokens(batch, grid, device)
        total, parts = joint_velocity_loss(squared, human, valid_tokens, face_loss_mask(prepared, grid),
                                           face_active=face_active, face_weight=face_weight)
        identity = perceptual = squared.new_zeros(())
        eligible = 0
        auxiliary_used = auxiliary and face_active and float(sigma.item()) <= .35
        if auxiliary_used:
            if evaluators is None or supervision is None:
                raise ValueError('Joint auxiliary losses require frozen evaluators and supervision gate')
            valid = prepared.face.face_valid[:, 0] & supervision.valid_mask(
                metadata, cache.resolution, device, align_mode=getattr(dataset, 'align_mode', 'resize'))
            eligible = int(valid.sum())
            if eligible:
                x0 = noisy - sigma[:, None, None] * prediction
                decoded = decode_tokens(pipe, x0, cache.resolution)
                target_pixels = (batch['tgt_image'].to(device).float() * .5 + .5).clamp(0, 1)
                boxes = prepared.face.target_boxes[:, 0].float()
                generated = crop_batch(decoded, boxes)[valid] * 2 - 1
                reference = crop_batch(target_pixels, boxes)[valid] * 2 - 1
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    target_id = feature(evaluators[0], reference)
                    target_perceptual = feature(evaluators[1], reference)
                with torch.autocast(device.type, enabled=False):
                    identity = (1 - F.cosine_similarity(feature(evaluators[0], generated), target_id)).mean()
                    perceptual = (feature(evaluators[1], generated) - target_perceptual).square().mean()
                total = total + .10 * identity + .05 * perceptual
        auxiliary_gradients = {}
        if check_auxiliary_gradients and eligible:
            parameter = adapter.output_bridges['face'].weight
            for name, loss in [('identity', identity), ('perceptual', perceptual)]:
                gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
                auxiliary_gradients[name] = 0. if gradient is None else float(gradient.detach().float().norm())
        if backward:
            total.backward()
    meta = pair_metadata(metadata[0])
    report = {
        'loss': float(total.detach()), 'body_flow': float(parts['body'].detach()),
        'face_flow': float(parts['face'].detach()), 'identity': float(identity.detach()),
        'perceptual': float(perceptual.detach()), 'sigma': float(sigma.item()),
        'face_active': face_active, 'auxiliary_requested': auxiliary,
        'auxiliary_used': auxiliary_used, 'auxiliary_eligible_faces': eligible,
        'pair_index': index, 'pair_id': meta['pair_id'], 'noise_seed': seed,
        'source_stem': meta['source_stem'], 'target_stem': meta['target_stem'],
        'source_appearance': meta['source_appearance'], 'target_appearance': meta['target_appearance'],
    }
    if check_auxiliary_gradients:
        report['auxiliary_gradient_norms'] = auxiliary_gradients
    return report
