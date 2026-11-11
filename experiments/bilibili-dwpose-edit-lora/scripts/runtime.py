# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Adapted from Diffusers train_dreambooth_lora_qwenimage21_img2img.py (Apache-2.0).
# Upstream snapshot and license are preserved in source/diffusers (remote: LoRA/source/diffusers).
"""Frozen disk cache and target-only multi-image Edit LoRA training."""
from __future__ import annotations

import gc
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
import diffusers
from accelerate.utils import set_seed
from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                       QwenImage21Pipeline, QwenImage21Transformer2DModel)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from core import (cache_key, cache_manifest_names, digest, joint_layout, load_training_rows,
                  read_json, read_jsonl, sha256, training_permutation, verify_row_assets, write_json)
from resources import ResourceMonitor


def release():
    gc.collect()
    torch.cuda.empty_cache()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def setup(config):
    torch.set_num_threads(4)
    set_seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('A real CUDA GPU is required')
    required = {'hidden_states', 'encoder_hidden_states', 'img_shapes', 'img_mask',
                'encoder_hidden_states_mask', 'timestep'}
    available = set(inspect.signature(QwenImage21Transformer2DModel.forward).parameters)
    if not required <= available:
        raise RuntimeError(f'Installed transformer API mismatch: {required - available}')
    if 'image_pad_mask' not in inspect.signature(QwenImage21Pipeline.encode_prompt).parameters:
        raise RuntimeError('Installed pipeline does not expose image_pad_mask')
    return {name: importlib.metadata.version(name) for name in
            ['diffusers', 'torch', 'peft', 'accelerate', 'transformers', 'safetensors']}


def model_identity(config):
    root = Path(config['model_root']).resolve()
    files = sorted([*root.rglob('*.json'), *root.rglob('*.safetensors')])
    records = {}
    for path in files:
        stat = path.stat()
        records[str(path.relative_to(root))] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                               'sha256': sha256(path)}
    import diffusers.pipelines.qwenimage21.pipeline_qwenimage21 as pipeline_module
    import diffusers.models.transformers.transformer_qwenimage21 as transformer_module
    return {'root': str(root), 'files': records, 'diffusers': diffusers.__version__,
            'pipeline_sha256': sha256(pipeline_module.__file__),
            'transformer_sha256': sha256(transformer_module.__file__),
            'preprocess': 'original_dwpose134_validmask_letterbox_v1', 'canvas_wh': config['canvas_wh'],
            'rgba': 'opaque_alpha_255', 'latent_channels': 64}


def validate_model_identity(identity, config):
    if str(Path(config['model_root']).resolve()) != identity['root']:
        raise ValueError('Cache model root changed')
    if identity['diffusers'] != diffusers.__version__:
        raise ValueError('Cache Diffusers version changed')
    for name, expected in identity['files'].items():
        stat = (Path(identity['root']) / name).stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Cached model weights/config changed: {name}')
    import diffusers.pipelines.qwenimage21.pipeline_qwenimage21 as p
    import diffusers.models.transformers.transformer_qwenimage21 as t
    if sha256(p.__file__) != identity['pipeline_sha256'] or sha256(t.__file__) != identity['transformer_sha256']:
        raise ValueError('Installed Qwen pipeline/transformer changed since cache creation')


def cache_data(config, data, name, reuse_cache=None):
    versions = setup(config)
    data = Path(data)
    out = Path(config['workspace_root']) / 'cache' / name
    out.mkdir(parents=True, exist_ok=False)
    with ResourceMonitor(config, out / 'resources.json') as monitor:
        manifests = cache_manifest_names(config)
        rows = [row for manifest in manifests for row in read_jsonl(data / manifest)]
        if len({r['pair_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate pair IDs in cache manifests')
        reuse_index = read_json(Path(reuse_cache) / 'index.json') if reuse_cache else None
        if reuse_index:
            identity = reuse_index['identity']
            validate_model_identity(identity, config)
            if reuse_index['versions'] != versions or identity['canvas_wh'] != config['canvas_wh']:
                raise ValueError('Cannot reuse encodings from a different runtime or canvas')
            print('Reusing verified frozen model identity; checking cached sample hashes', flush=True)
        else:
            print('Fingerprinting model weights/configs', flush=True)
            identity = model_identity(config)
        if any(r['prompt'] != config['prompt'] or r['canvas_wh'] != config['canvas_wh'] for r in rows):
            raise ValueError('Data Prompt/canvas differs from cache configuration')
        print(f'Cache manifests={manifests}, samples={len(rows)}', flush=True)
        write_json(out / 'model_identity.json', identity)
        all_paths = {path: expected for row in rows for path, expected in
                     zip([*row['condition_images'], row['target_image']],
                         [*row['condition_sha256'], row['target_sha256']])}
        for path, expected in all_paths.items():
            if sha256(path) != expected:
                raise ValueError(f'Prepared input changed: {path}')
        # Encode VAE first, then release it before loading Qwen3-VL.
        vae = AutoencoderKLQwenImage21.from_pretrained(config['model_root'], subfolder='vae',
                                                     torch_dtype=torch.bfloat16).to('cuda')
        vae.requires_grad_(False).eval()
        if vae.config.z_dim != 64:
            raise ValueError('This experiment requires native 64-channel Qwen 2.1 latents')
        mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
        scale = torch.tensor(vae.config.latents_std, device='cuda', dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
        latent_root = out / 'vae'
        latent_root.mkdir()
        for i, (path, key) in enumerate(sorted(all_paths.items())):
            monitor.check()
            destination = latent_root / f'{key}.pt'
            if destination.exists():
                continue  # Identical prepared pixels share one immutable posterior.
            with Image.open(path) as image:
                array = np.array(image.convert('RGBA'), copy=True)
            pixels = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].to('cuda', torch.bfloat16) / 127.5 - 1
            with torch.no_grad():
                posterior = vae.encode(pixels).latent_dist
                payload = {'mean': cpu((posterior.mean - mean) / scale),
                           'std': cpu(posterior.std / scale), 'image_sha256': key}
            torch.save(payload, destination)
            if i % 10 == 0:
                print(f'VAE {i+1}/{len(all_paths)}', flush=True)
        del vae, mean, scale, pixels, posterior, payload
        release()
        pipeline = QwenImage21Pipeline.from_pretrained(config['model_root'], vae=None, transformer=None,
                                                      scheduler=None, torch_dtype=torch.bfloat16).to('cuda')
        pipeline.text_encoder.requires_grad_(False).eval()
        entries, reused = {}, 0
        for i, row in enumerate(rows):
            monitor.check()
            key = cache_key(row, identity)
            previous = reuse_index['entries'].get(row['pair_id']) if reuse_index else None
            if previous and previous['key'] == key:
                source = Path(reuse_cache) / previous['file']
                if sha256(source) != previous['sha256']:
                    raise ValueError(f'Reused cache content changed: {source}')
                shutil.copyfile(source, out / f'{key}.pt')
                entries[row['pair_id']] = {'key': key, 'file': f'{key}.pt', 'sha256': previous['sha256']}
                reused += 1
                continue
            images = [Image.open(p).convert('RGBA') for p in row['condition_images']]
            with torch.no_grad():
                embeds, attention_mask, image_pad_mask = pipeline.encode_prompt(prompt=row['prompt'], image=images)
            latents = [torch.load(latent_root / f'{h}.pt', weights_only=True) for h in row['condition_sha256']]
            target = torch.load(latent_root / f'{row["target_sha256"]}.pt', weights_only=True)
            layout = joint_layout([tuple(z['mean'].shape[-2:]) for z in latents],
                                  tuple(target['mean'].shape[-2:]), image_pad_mask[0].sum().item())
            item = {'schema': 'dual_edit_cache_sample_v1', 'key': key, 'pair_id': row['pair_id'],
                    'condition_latents': [z['mean'] for z in latents],
                    'target_mean': target['mean'], 'target_std': target['std'],
                    'prompt_embeds': cpu(embeds), 'prompt_embeds_mask': cpu(attention_mask),
                    'image_pad_mask': cpu(image_pad_mask), 'layout': layout,
                    'condition_images': row['condition_images'], 'prompt': row['prompt']}
            torch.save(item, out / f'{key}.pt')
            entries[row['pair_id']] = {'key': key, 'file': f'{key}.pt', 'sha256': sha256(out / f'{key}.pt')}
            if len(rows) < 100 or i % 25 == 0 or i+1 == len(rows):
                print(f'VLM {i+1}/{len(rows)} {row["split"]} {row["pair_id"][:12]} slots={int(image_pad_mask.sum())}', flush=True)
            del embeds, attention_mask, image_pad_mask, item, latents, target, images
        pipeline.to('cpu')
        del pipeline
        release()
        write_json(out / 'index.json', {'schema': 'dual_edit_disk_cache_v1', 'data': str(data.resolve()),
                   'data_index_sha256': sha256(data / 'index.json'), 'config_sha256': digest(config),
                   'versions': versions, 'identity': identity, 'entries': entries,
                   'reused_samples': reused, 'reuse_cache': str(reuse_cache) if reuse_cache else None,
                   'manifest_sha256': {n: sha256(data / n) for n in sorted(set(manifests +
                                      ['pilot_train.jsonl', 'pilot_val.jsonl', 'diagnostics.jsonl']))}})
    return out


def open_cache(config, data, cache):
    data, cache = Path(data), Path(cache)
    index = read_json(cache / 'index.json')
    current_versions = {name: importlib.metadata.version(name) for name in index['versions']}
    if current_versions != index['versions']:
        raise ValueError(f'Cache runtime versions changed: {current_versions}')
    if index['config_sha256'] != digest(config) or index['data_index_sha256'] != sha256(data / 'index.json'):
        raise ValueError('Cache/config/data identity mismatch')
    for name, expected in index['manifest_sha256'].items():
        if sha256(data / name) != expected:
            raise ValueError(f'Pilot manifest changed: {name}')
    validate_model_identity(index['identity'], config)
    return index


def load_cached(row, index, cache, device='cuda'):
    verify_row_assets(row)
    entry = index['entries'][row['pair_id']]
    if entry['key'] != cache_key(row, index['identity']):
        raise ValueError('Sample cache key mismatch')
    path = Path(cache) / entry['file']
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Cached tensor content changed: {path}')
    value = torch.load(path, map_location='cpu', weights_only=True)
    def move(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, list):
            return [move(x) for x in v]
        return v
    return {k: move(v) for k, v in value.items()}


def forward_loss(model, item, scheduler, density_value=None):
    # Same native normalization, flattening, timestep and target-only velocity loss as upstream.
    clean = item['target_mean'] + item['target_std'] * torch.randn_like(item['target_mean'])
    noise = torch.randn_like(clean)
    u = (compute_density_for_timestep_sampling(weighting_scheme='none', batch_size=1)
         if density_value is None else torch.tensor([density_value]))
    position = int((u * scheduler.config.num_train_timesteps).long().item())
    timestep = scheduler.timesteps[position].reshape(1).to('cuda')
    sigma = scheduler.sigmas[position].to('cuda', dtype=clean.dtype).reshape(1, 1, 1, 1, 1)
    noisy = (1 - sigma) * clean + sigma * noise
    def pack(latent):
        return QwenImage21Pipeline._pack_latents(latent, 1, latent.shape[1], *latent.shape[-2:])
    target_packed = pack(noisy)
    condition_packed = [pack(z) for z in item['condition_latents']]
    layout = joint_layout([tuple(z.shape[-2:]) for z in item['condition_latents']], tuple(clean.shape[-2:]),
                          item['image_pad_mask'][0].sum().item())
    img_mask = torch.cat([item['image_pad_mask'], item['image_pad_mask'].new_ones(1, layout['target_slots'])], dim=1)
    prediction = model(hidden_states=torch.cat([*condition_packed, target_packed], dim=1),
                       encoder_hidden_states=item['prompt_embeds'],
                       encoder_hidden_states_mask=item['prompt_embeds_mask'],
                       timestep=timestep / 1000, img_shapes=[layout['img_shapes']],
                       img_mask=img_mask, return_dict=False)[0]
    prediction = prediction[:, -layout['target_tokens']:]
    target = pack(noise - clean)
    weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigma).reshape(1, 1, 1)
    loss = (weighting.float() * (prediction.float() - target.float()).square()).mean()
    return loss, {'sigma': float(sigma.item()), 'timestep': float(timestep.item()),
                  'condition_tokens': layout['condition_tokens'], 'target_tokens': layout['target_tokens'],
                  'image_slots': int(img_mask.sum().item())}


def adapter_state(model):
    return {k: cpu(v).clone() for k, v in get_peft_model_state_dict(model).items()}


def validate(model, config, data, cache, index, scheduler, out, step, monitor):
    rows = read_jsonl(Path(data) / config['validation_manifest'])
    if not rows or any(row['split'] != 'val' for row in rows):
        raise ValueError('Validation must use only val samples')
    was_training = model.training
    values = []
    try:
        model.eval()
        # Keep validation from perturbing the following training noise/timestep stream.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for i, row in enumerate(rows):
                item = load_cached(row, index, cache)
                for density in config['validation_density_values']:
                    monitor.check()
                    torch.manual_seed(config['validation_seed'] + i)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss, details = forward_loss(model, item, scheduler, density_value=density)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite validation loss')
                    values.append({'pair_id': row['pair_id'], 'density': density,
                                   'loss': float(loss), **details})
                del item
    finally:
        model.train(was_training)
    result = {'step': step, 'samples': len(rows), 'seed': config['validation_seed'],
              'mean_loss': float(np.mean([v['loss'] for v in values])), 'records': values,
              'training_rng_preserved': True}
    write_json(out / f'validation_{step:06d}.json', result)
    print(f'Validation step={step} samples={len(rows)} mean_loss={result["mean_loss"]:.6f}', flush=True)


def save_checkpoint(model, optimizer, lr_scheduler, config, cache, out, step, initial):
    checkpoint = out / f'checkpoint-{step:06d}'
    checkpoint.mkdir(exist_ok=False)
    state = adapter_state(model)
    QwenImage21Pipeline.save_lora_weights(str(checkpoint), transformer_lora_layers=state)
    delta_sq = sum(float((state[k].float() - initial[k].float()).square().sum()) for k in state)
    payload = {'step': step, 'lora': state, 'optimizer': cpu(optimizer.state_dict()),
               'scheduler': lr_scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
               'config_sha256': digest(config), 'cache_index_sha256': sha256(Path(cache) / 'index.json')}
    with (checkpoint / 'training_state.pt').open('xb') as stream:
        torch.save(payload, stream)
    write_json(checkpoint / 'checkpoint.json', {'step': step, 'config_sha256': digest(config),
               'cache_index_sha256': payload['cache_index_sha256'], 'lora_delta_l2': delta_sq ** 0.5,
               'lora_sha256': sha256(checkpoint / 'pytorch_lora_weights.safetensors'),
               'training_state_sha256': sha256(checkpoint / 'training_state.pt')})
    return checkpoint


def train(config, data, cache, name, until_step=None, resume=None):
    versions = setup(config)
    index = open_cache(config, data, cache)
    rows = load_training_rows(config, data)
    if any(row['pair_id'] not in index['entries'] for row in rows):
        raise ValueError('The selected train dataset is not fully cached')
    limit = config['steps'] if until_step is None else until_step
    if not 1 <= limit <= config['steps']:
        raise ValueError('until-step must be within the approved update budget')
    out = Path(config['workspace_root']) / 'output' / name
    if resume is None:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'config.json', config)
    elif Path(resume).resolve().parent != out.resolve() or read_json(out / 'config.json') != config:
        raise ValueError('Resume must use the same run directory and exact config')
    starting_step = read_json(Path(resume) / 'checkpoint.json')['step'] if resume else 0
    if starting_step >= limit:
        raise ValueError('Resume update limit must exceed the committed checkpoint step')
    with ResourceMonitor(config, out / f'resources_{starting_step:06d}_{limit:06d}.json') as monitor:
        model = QwenImage21Transformer2DModel.from_pretrained(config['model_root'], subfolder='transformer',
                                                            torch_dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['alpha'],
                                    lora_dropout=config['dropout'], init_lora_weights='gaussian',
                                    target_modules=config['lora_modules']))
        model.to('cuda')
        model.enable_gradient_checkpointing()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named or any('lora_' not in n for n, _ in named):
            raise RuntimeError('Only LoRA weights may be trainable')
        for _, p in named:
            p.data = p.data.float()
        parameters = [p for _, p in named]
        optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'],
                                     betas=(0.9, 0.999), weight_decay=config['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / max(config['warmup_steps'], 1), 1.0))
        initial_path = out / 'initial_lora.pt'
        if resume:
            record = read_json(Path(resume) / 'checkpoint.json')
            state_path = Path(resume) / 'training_state.pt'
            if sha256(state_path) != record['training_state_sha256']:
                raise ValueError('Resume checkpoint checksum mismatch')
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            if state['config_sha256'] != digest(config) or state['cache_index_sha256'] != sha256(Path(cache) / 'index.json'):
                raise ValueError('Resume config/cache mismatch')
            set_peft_model_state_dict(model, state['lora'])
            restored = adapter_state(model)
            if set(restored) != set(state['lora']) or any(not torch.equal(restored[k], state['lora'][k]) for k in restored):
                raise RuntimeError('LoRA weights did not restore exactly')
            optimizer.load_state_dict(state['optimizer'])
            lr_scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            random.setstate(state['python_rng'])
            initial = torch.load(initial_path, weights_only=True)
            write_json(out / f'resume_{starting_step:06d}.json',
                       {'step': starting_step, 'lora_exact': True, 'optimizer_states': len(optimizer.state),
                        'scheduler_step': lr_scheduler.last_epoch, 'rng_restored': True})
            del state, restored
        else:
            initial = adapter_state(model)
            with initial_path.open('xb') as stream:
                torch.save(initial, stream)
            write_json(out / 'trainable.json', {'versions': versions,
                       'trainable_parameters': sum(p.numel() for p in parameters),
                       'trainable_tensors': len(named), 'names': [n for n, _ in named],
                       'base_frozen': all(not p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n),
                       'optimizer_only_lora': {id(p) for group in optimizer.param_groups for p in group['params']} ==
                                              {id(p) for p in parameters},
                       'train_pairs': len(rows), 'cache_index_sha256': sha256(Path(cache) / 'index.json')})
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config['model_root'], subfolder='scheduler')
        model.train()
        last_checkpoint = None
        losses = []
        current_epoch, order = None, None
        log_path = out / f'updates_{starting_step:06d}_{limit:06d}.jsonl'
        import json
        with log_path.open('x', encoding='utf-8', buffering=1) as log:
            for step in range(starting_step, limit):
                monitor.check()
                epoch, offset = divmod(step, len(rows))
                if epoch != current_epoch:
                    order = training_permutation(len(rows), config['seed'], epoch)
                    current_epoch = epoch
                row = rows[order[offset]]
                item = load_cached(row, index, cache)
                start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, details = forward_loss(model, item, scheduler)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at step {step+1}')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'], error_if_nonfinite=True)
                if not float(gradient_norm) > 0:
                    raise RuntimeError('All LoRA gradients are zero')
                if step == starting_step and any(p.grad is not None for n, p in model.named_parameters() if 'lora_' not in n):
                    raise RuntimeError('A frozen base weight received a gradient')
                learning_rate = optimizer.param_groups[0]['lr']
                optimizer.step()
                lr_scheduler.step()
                torch.cuda.synchronize()
                value = float(loss.detach())
                losses.append(value)
                record = {'step': step+1, 'pair_id': row['pair_id'], 'loss': value,
                          'gradient_norm': float(gradient_norm), 'lr': learning_rate,
                          'step_seconds': time.monotonic()-start,
                          'cuda_allocated_mib': torch.cuda.memory_allocated()/2**20, **details}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if step == starting_step or (step+1) % 10 == 0:
                    print(json.dumps(record, allow_nan=False), flush=True)
                del loss, item
                validation_every = config.get('validation_every', 0)
                if validation_every and ((step+1) % validation_every == 0 or step+1 == limit):
                    validate(model, config, data, cache, index, scheduler, out, step+1, monitor)
                if (step+1) % config['save_every'] == 0 or step+1 == limit:
                    last_checkpoint = save_checkpoint(model, optimizer, lr_scheduler, config, cache, out, step+1, initial)
                    print(f'Checkpoint committed: {last_checkpoint}', flush=True)
                monitor.check()
        final_state = adapter_state(model)
        changed = [key for key in final_state if not torch.equal(final_state[key], initial[key])]
        if not changed:
            raise RuntimeError('No LoRA parameter changed')
        write_json(out / f'train_result_{starting_step:06d}_{limit:06d}.json',
                   {'start_step': starting_step, 'end_step': limit, 'updates': len(losses),
                    'loss_first': losses[0], 'loss_last': losses[-1],
                    'loss_first_20_mean': float(np.mean(losses[:20])),
                    'loss_last_20_mean': float(np.mean(losses[-20:])),
                    'changed_lora_tensors': len(changed), 'checkpoint': str(last_checkpoint),
                    'base_frozen': True, 'optimizer_only_lora': True})
    return last_checkpoint
