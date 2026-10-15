"""Training utilities for the complete V6 controller connected to Qwen21.

The 47 copied architecture files remain unchanged. This module only manages
native Qwen I/O, optimizer selection, and execution of the preserved branches.
"""
from __future__ import annotations

import hashlib
import json
import random
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


def precompute(pipe, cache, datasets):
    for dataset in datasets:
        for index in range(len(dataset)):
            sample = dataset[index]
            cache.source(pipe, dataset_image_to_pil(sample['src_image']))
            cache.target(pipe, dataset_image_to_pil(sample['tgt_image']))
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
            meta = [row if 'source_stem' in row else pair_metadata(row) for row in metadata]
            face_condition, references = store.for_pairs(meta, device=device, dtype=torch.float32)
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


def parameter_groups(adapter, branch):
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
    result = [{'params': groups[key], 'lr': rates[key], 'name': key} for key in groups if groups[key]]
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
            weight = 1 + 2 * human.flatten(2).transpose(1, 2)
            flow = (squared * weight).sum() / (weight.sum() * squared.shape[-1])
        else:
            mask = face_loss_mask(prepared, grid).to(device)
            flow, global_flow, face_flow, nonface_flow = face_velocity_loss(squared, mask, face_flow_loss)
            outside = ((prediction.float() - outside_reference.float()).square() * (1 - mask)).sum() / (
                (1 - mask).sum() * prediction.shape[-1]).clamp_min(1)
            if auxiliary:
                if evaluators is None:
                    raise ValueError('Auxiliary face losses require identity and perceptual evaluators')
                if supervision is None:
                    raise ValueError('Face auxiliary losses require Target visibility eligibility')
                valid = prepared.face.face_valid[:, 0] & supervision.valid_mask(metadata, cache.resolution, device)
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
                            value = torch.tensor(keypoints, dtype=torch.float32)
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
    report = {'loss': float(total.detach()), 'flow': float(flow.detach()), 'identity': float(identity.detach()),
              'perceptual': float(perceptual.detach()), 'outside': float(outside.detach()), 'sigma': float(sigma.item()),
              'pair_index': index, 'appearance': metadata[0].get('appearance'),
              'source': metadata[0]['source']['stem'], 'target': metadata[0]['target']['stem']}
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


def save_checkpoint(path, adapter, optimizer, *, step, branch, args, sample_order):
    payload = {'architecture': ARCHITECTURE, 'model_revision': MODEL_REVISION,
               'adapter_state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'step': step, 'branch': branch, 'resolution': args.resolution,
               'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'initialization': adapter.loading_report, 'sample_order': sample_order,
               'train_pairs_sha256': hashlib.sha256(Path(args.train_pairs).read_bytes()).hexdigest(),
               'val_pairs_sha256': hashlib.sha256(Path(args.val_pairs).read_bytes()).hexdigest(),
               'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
               'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(saved):
    torch.set_rng_state(saved['torch_rng_state'])
    torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    random.setstate(saved['python_rng_state'])
    np.random.set_state(saved['numpy_rng_state'])


def probe(pipe, adapter, cache, dataset, store, branch, **loss_options):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return run_loss(pipe, adapter, cache, dataset, store, 0, branch, seed=24042,
                        backward=False, fixed_sigma=.2 if branch == 'face' else .5, **loss_options)
