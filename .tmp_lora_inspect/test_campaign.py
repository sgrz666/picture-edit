#!/usr/bin/env python3
"""One-shot full-test evaluation of the full-run LoRA (checkpoint-009520).

Stages: prereg -> gate -> audit -> cache_test -> infer -> evaluate -> report.
The test split never enters the training cache: core.cache_manifest_names forbids
it, so this driver builds a separate test-only cache (schema *_test_v1) that the
training loader cannot open (different config digest and schema).

Serial discipline: the gate waits until the ungated-v1 line's chain/eval PIDs are
gone, no other qwen21 CUDA process of this user exists, and >= gate_free_gib is
free. CPU-only work (contract audit, copyA/shuffle controls) runs during the wait.

Every pair gets a five-column panel. No config may be changed after seeing test
results; the pre-registration file is written before the gate and is immutable.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from core import (cache_key, digest, pose_stats, read_json, read_jsonl, sha256,  # noqa: E402
                  verify_row_assets, write_json)

POSE_FIELDS = ['pck_002_diagonal', 'pck_005_diagonal', 'pck_010_diagonal', 'nme_diagonal']
SIGNS = {'pck_002_diagonal': 1, 'pck_005_diagonal': 1, 'pck_010_diagonal': 1,
         'nme_diagonal': -1, 'identity_a_cosine': 1}


def log(events_path, event, **kw):
    line = json.dumps({'time': time.time(), 'at': time.strftime('%F %T'), 'event': event, **kw},
                      ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with open(events_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def write_json_replace(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
    os.replace(tmp, path)


def pid_alive(pid):
    return Path('/proc/%d' % pid).exists()


def cmdline(pid):
    try:
        return Path('/proc/%d/cmdline' % pid).read_bytes().replace(b'\0', b' ').decode('utf8', 'replace')
    except OSError:
        return ''


def smi(config, query):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = config['nvml_compat'] + ':' + env.get('LD_LIBRARY_PATH', '')
    return subprocess.check_output(['nvidia-smi', '--query-gpu=' + query,
                                    '--format=csv,noheader,nounits'], env=env, text=True, timeout=20)


def smi_compute_apps(config):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = config['nvml_compat'] + ':' + env.get('LD_LIBRARY_PATH', '')
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], env=env, text=True, timeout=20)
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit():
            apps.append((int(parts[0]), float(parts[1])))
    return apps


def test_rows(config, data):
    rows = read_jsonl(Path(data) / config['test_manifest'])
    if len(rows) != config['expected_test_pairs']:
        raise ValueError('test rows=%d expected=%d' % (len(rows), config['expected_test_pairs']))
    if any(r['split'] != 'test' for r in rows):
        raise ValueError('test manifest contains non-test rows')
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate test pair ids')
    return rows


# ---------------------------------------------------------------- prereg
def prereg(config, data, tag, report_dir, events):
    path = Path(report_dir) / ('test_preregistration_%s.json' % tag)
    if path.exists():
        log(events, 'prereg_exists', path=str(path))
        return read_json(path)
    rows = test_rows(config, data)
    adapter = Path(config['adapter_checkpoint'])
    value = {
        'tag': tag,
        'written_before_any_test_result': True,
        'split': 'test', 'pairs': len(rows),
        'video_groups': sorted({r['group_id'] for r in rows}),
        'adapter': str(adapter),
        'adapter_lora_sha256': sha256(adapter / 'pytorch_lora_weights.safetensors'),
        'test_manifest_sha256': sha256(Path(data) / config['test_manifest']),
        'protocol': {'seed': config['seed'], 'inference_steps': config['inference_steps'],
                     'true_cfg_scale': config['true_cfg_scale'], 'canvas_wh': config['canvas_wh'],
                     'prompt_sha256': digest(config['prompt']),
                     'identical_to_val_protocol': True},
        'primary_metric': 'pck_002_diagonal',
        'secondary_metrics': ['pck_005_diagonal', 'pck_010_diagonal', 'nme_diagonal', 'identity_a_cosine'],
        'estimator': 'mean paired per-pair delta (lora - baseline), sign-adjusted so positive = better',
        'inference': 'cluster bootstrap over video groups, resample groups with replacement, '
                     'pair-level mean inside each replicate, percentile 95% CI',
        'bootstrap_reps': config['bootstrap_reps'], 'bootstrap_seed': config['bootstrap_seed'],
        'decision_rule': 'improvement iff CI lower bound > 0; regression iff CI upper bound < 0; '
                         'otherwise inconclusive. Applied per metric independently.',
        'exclusions': 'pairs where either method has pose_detected=false are excluded from paired pose '
                      'metrics and counted; pairs with a missing face embedding are excluded from '
                      'FaceID and counted. No imputation.',
        'controls': 'copyA (detect pose on letterboxed A, score vs own B) and shuffleA (A of another '
                    'pair vs this B, fixed derangement) establish floor and headroom; not part of the '
                    'decision rule.',
        'one_shot': 'After this run no hyperparameter, prompt, canvas or adapter change may be justified '
                    'by these test numbers. Any further iteration needs a new holdout or an explicit '
                    'user decision.',
        'resource_bounds': {'min_free_gpu_mib': config['min_free_gpu_mib'],
                            'min_available_ram_mib': config['min_available_ram_mib']},
        'expected_outputs': {'images': 2 * len(rows), 'panels': len(rows)},
    }
    write_json(path, value)
    log(events, 'prereg_written', path=str(path), pairs=len(rows))
    return value


# ---------------------------------------------------------------- gate
def gate(config, events):
    mode = config.get('gate_mode', 'serial')
    block = list(config['gate_block_pids'])
    deadline = time.time() + config['gate_max_wait_seconds']
    last_report = 0.0
    if mode == 'vram_only':
        log(events, 'gate_mode_vram_only', authorization=config.get('gate_authorization', ''),
            note='PID/foreign-process checks disabled by explicit user authorization; '
                 'only free-VRAM threshold enforced')
    while True:
        alive = [p for p in block if pid_alive(p)]
        apps = smi_compute_apps(config)
        foreign = [(p, m) for p, m in apps
                   if p != os.getpid() and 'qwen21/bin/python' in cmdline(p)]
        free_gib = float(smi(config, 'memory.free').splitlines()[0]) / 1024.0
        blocked = (alive or foreign) if mode == 'serial' else []
        if not blocked and free_gib >= config['gate_free_gib']:
            log(events, 'gate_passed', mode=mode, free_gib=round(free_gib, 2),
                blocked_pids=alive, foreign_qwen21=[p for p, _ in foreign])
            return
        if time.time() > deadline:
            log(events, 'gate_timeout', mode=mode, free_gib=round(free_gib, 2),
                blocked_pids=alive, foreign_qwen21=[p for p, _ in foreign])
            raise RuntimeError('gate wait exceeded %ds' % config['gate_max_wait_seconds'])
        if time.time() - last_report > 600:
            last_report = time.time()
            log(events, 'gate_wait', mode=mode, free_gib=round(free_gib, 2), blocked_pids=alive,
                foreign_qwen21=[p for p, _ in foreign])
        time.sleep(60)


# ---------------------------------------------------------------- audit
def audit(config, data, events):
    rows = test_rows(config, data)
    problems = []
    src = {}
    for split in ('train', 'val', 'test'):
        src[split] = {r['pair_id'] for r in read_jsonl(Path(config['dataset_root']) / split / 'pairs.jsonl')}
    ids = {r['pair_id'] for r in rows}
    if ids != src['test']:
        problems.append('test manifest != source test split')
    if ids & src['train'] or ids & src['val']:
        problems.append('test overlaps train/val')
    pilot_val = {r['pair_id'] for r in read_jsonl(Path(data) / 'pilot_val.jsonl')}
    if ids & pilot_val:
        problems.append('test overlaps pilot_val')
    n_img = 0
    for r in rows:
        if r['prompt'] != config['prompt']:
            problems.append('prompt drift %s' % r['pair_id'])
        if r['canvas_wh'] != config['canvas_wh']:
            problems.append('canvas drift %s' % r['pair_id'])
        for path, expected in zip([*r['condition_images'], r['target_image']],
                                  [*r['condition_sha256'], r['target_sha256']]):
            n_img += 1
            if not Path(path).is_file():
                problems.append('missing image %s' % path)
            elif sha256(path) != expected:
                problems.append('image sha drift %s' % path)
        meta = Path(r['target_pose_metadata'])
        if not meta.is_file():
            problems.append('missing pose metadata %s' % meta)
        else:
            m = read_json(meta)
            if m['size'] != config['canvas_wh']:
                problems.append('pose canvas drift %s' % r['pair_id'])
    adapter = Path(config['adapter_checkpoint'])
    ck = read_json(adapter / 'checkpoint.json')
    if sha256(adapter / 'pytorch_lora_weights.safetensors') != ck['lora_sha256']:
        problems.append('adapter sha != checkpoint.json lora_sha256')
    if ck['step'] != 9520:
        problems.append('adapter step != 9520')
    result = {'pairs': len(rows), 'images_checked': n_img,
              'video_groups': sorted({r['group_id'] for r in rows}),
              'adapter_sha256': ck['lora_sha256'],
              'data_index_sha256': sha256(Path(data) / 'index.json'),
              'problems': problems, 'passed': not problems}
    log(events, 'audit_done', passed=result['passed'], pairs=len(rows),
        images_checked=n_img, problems=problems[:10])
    if problems:
        raise RuntimeError('contract audit failed: %s' % problems[:5])
    return result


# ---------------------------------------------------------------- controls (CPU)
def controls(config, data, campaign_dir, events):
    sys.path.insert(0, config['project_root'])
    from src.evaluation.joint_pilot_metrics import PoseMetric
    rows = test_rows(config, data)
    store = Path(campaign_dir) / 'control_detections.npz'
    pose = PoseMetric(Path(config['project_root']) / 'models')
    det = {}
    if store.exists():
        z = np.load(store, allow_pickle=False)
        for k in z.files:
            if k.endswith('_xy'):
                det[k[:-3]] = (z[k], z[k[:-3] + '_sc'])
        log(events, 'controls_reused', detections=len(det))
    todo = [r for r in rows if r['pair_id'] not in det]
    for i, r in enumerate(todo):
        with Image.open(r['condition_images'][0]) as im:
            if list(im.size) != list(config['canvas_wh']):
                raise ValueError('A condition canvas != config canvas: %s' % r['pair_id'])
            d = pose.detect(im)
        if d is None:
            det[r['pair_id']] = (np.zeros((17, 2)), np.zeros(17))
        else:
            det[r['pair_id']] = (np.asarray(d[0]), np.asarray(d[1]))
        if (i + 1) % 200 == 0:
            log(events, 'controls_progress', done=i + 1, total=len(todo))
    if todo:
        payload = {}
        for k, (xy, sc) in det.items():
            payload[k + '_xy'] = xy
            payload[k + '_sc'] = sc
        np.savez(store, **payload)
    refs = {}
    for r in rows:
        m = read_json(r['target_pose_metadata'])
        refs[r['pair_id']] = (np.asarray(m['keypoints_xy']), np.asarray(m['valid'], bool), m['size'])
    names = [r['pair_id'] for r in rows]
    rng = random.Random(config['control_seed'])
    for _ in range(100000):
        perm = names[:]
        rng.shuffle(perm)
        if all(perm[i] != names[i] for i in range(len(names))):
            break
    else:
        raise RuntimeError('no derangement')
    partner = dict(zip(names, perm))
    copy_a, shuffle_a = {}, {}
    for pid in names:
        xy, sc = det[pid]
        detected = None if sc.sum() == 0 else (xy, sc)
        ref_xy, ref_valid, size = refs[pid]
        copy_a[pid] = pose_stats(detected, ref_xy, ref_valid, size)
        p = partner[pid]
        pxy, psc = det[p]
        pdet = None if psc.sum() == 0 else (pxy, psc)
        shuffle_a[pid] = pose_stats(pdet, ref_xy, ref_valid, size)
    out = {'seed': config['control_seed'], 'detections': len(det),
           'a_detection_failures': sum(1 for v in det.values() if v[1].sum() == 0),
           'copyA': copy_a, 'shuffleA': shuffle_a, 'partner': partner}
    write_json_replace(Path(campaign_dir) / 'controls.json', out)
    log(events, 'controls_done', detections=len(det),
        copyA_pck002=float(np.mean([v['pck_002_diagonal'] for v in copy_a.values()])),
        shuffleA_pck002=float(np.mean([v['pck_002_diagonal'] for v in shuffle_a.values()])))
    return out


# ---------------------------------------------------------------- test-only cache
def cache_test(config, data, name, events):
    import runtime
    from resources import ResourceMonitor
    versions = runtime.setup(config)
    data = Path(data)
    out = Path(config['workspace_root']) / 'cache' / name
    out.mkdir(parents=True, exist_ok=False)
    rows = test_rows(config, data)
    with ResourceMonitor(config, out / 'resources.json') as monitor:
        identity = runtime.model_identity(config)
        if any(r['prompt'] != config['prompt'] or r['canvas_wh'] != config['canvas_wh'] for r in rows):
            raise ValueError('test prompt/canvas differs from config')
        all_paths = {p: e for r in rows for p, e in
                     zip([*r['condition_images'], r['target_image']],
                         [*r['condition_sha256'], r['target_sha256']])}
        for path, expected in all_paths.items():
            if sha256(path) != expected:
                raise ValueError('prepared input changed: %s' % path)
        log(events, 'cache_test_start', samples=len(rows), unique_images=len(all_paths))
        import torch
        from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
        vae = AutoencoderKLQwenImage21.from_pretrained(config['model_root'], subfolder='vae',
                                                      torch_dtype=torch.bfloat16).to('cuda')
        vae.requires_grad_(False).eval()
        mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
        scale = torch.tensor(vae.config.latents_std, device='cuda', dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
        latent_root = out / 'vae'
        latent_root.mkdir()
        for i, (path, key) in enumerate(sorted(all_paths.items())):
            monitor.check()
            dest = latent_root / ('%s.pt' % key)
            if dest.exists():
                continue
            with Image.open(path) as image:
                array = np.array(image.convert('RGBA'), copy=True)
            pixels = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].to('cuda', torch.bfloat16) / 127.5 - 1
            with torch.no_grad():
                posterior = vae.encode(pixels).latent_dist
                payload = {'mean': runtime.cpu((posterior.mean - mean) / scale),
                           'std': runtime.cpu(posterior.std / scale), 'image_sha256': key}
            torch.save(payload, dest)
            if (i + 1) % 200 == 0:
                log(events, 'cache_test_vae', done=i + 1, total=len(all_paths))
        del vae, mean, scale, pixels, posterior, payload
        runtime.release()
        pipeline = QwenImage21Pipeline.from_pretrained(config['model_root'], vae=None, transformer=None,
                                                       scheduler=None, torch_dtype=torch.bfloat16).to('cuda')
        pipeline.text_encoder.requires_grad_(False).eval()
        entries = {}
        for i, row in enumerate(rows):
            monitor.check()
            key = cache_key(row, identity)
            images = [Image.open(p).convert('RGBA') for p in row['condition_images']]
            with torch.no_grad():
                embeds, attention_mask, image_pad_mask = pipeline.encode_prompt(prompt=row['prompt'], image=images)
            latents = [torch.load(latent_root / ('%s.pt' % h), weights_only=True) for h in row['condition_sha256']]
            target = torch.load(latent_root / ('%s.pt' % row['target_sha256']), weights_only=True)
            layout = runtime.joint_layout([tuple(z['mean'].shape[-2:]) for z in latents],
                                          tuple(target['mean'].shape[-2:]), image_pad_mask[0].sum().item())
            item = {'schema': 'dual_edit_cache_sample_v1', 'key': key, 'pair_id': row['pair_id'],
                    'condition_latents': [z['mean'] for z in latents],
                    'target_mean': target['mean'], 'target_std': target['std'],
                    'prompt_embeds': runtime.cpu(embeds), 'prompt_embeds_mask': runtime.cpu(attention_mask),
                    'image_pad_mask': runtime.cpu(image_pad_mask), 'layout': layout,
                    'condition_images': row['condition_images'], 'prompt': row['prompt']}
            torch.save(item, out / ('%s.pt' % key))
            entries[row['pair_id']] = {'key': key, 'file': '%s.pt' % key,
                                       'sha256': sha256(out / ('%s.pt' % key))}
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                log(events, 'cache_test_vlm', done=i + 1, total=len(rows))
            del embeds, attention_mask, image_pad_mask, item, latents, target, images
        pipeline.to('cpu')
        del pipeline
        runtime.release()
        write_json(out / 'index.json', {
            'schema': 'dual_edit_disk_cache_test_v1', 'data': str(data.resolve()),
            'data_index_sha256': sha256(data / 'index.json'), 'config_sha256': digest(config),
            'versions': versions, 'identity': identity, 'entries': entries,
            'reused_samples': 0, 'reuse_cache': None,
            'manifest_sha256': {config['test_manifest']: sha256(data / config['test_manifest'])}})
    log(events, 'cache_test_done', path=str(out), entries=len(entries))
    return out


def open_test_cache(config, data, cache):
    import importlib.metadata
    import runtime
    data, cache = Path(data), Path(cache)
    index = read_json(cache / 'index.json')
    if index['schema'] != 'dual_edit_disk_cache_test_v1':
        raise ValueError('not a test-only cache; training caches must never be used for test')
    current = {n: importlib.metadata.version(n) for n in index['versions']}
    if current != index['versions']:
        raise ValueError('cache runtime versions changed')
    if index['config_sha256'] != digest(config) or index['data_index_sha256'] != sha256(data / 'index.json'):
        raise ValueError('test cache/config/data identity mismatch')
    for name, expected in index['manifest_sha256'].items():
        if sha256(data / name) != expected:
            raise ValueError('test manifest changed: %s' % name)
    runtime.validate_model_identity(index['identity'], config)
    return index


# ---------------------------------------------------------------- inference
def infer_test(config, data, cache, name, events):
    import torch
    from diffusers import QwenImage21Pipeline
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    import runtime
    from inference import CachedValidationPipeline
    from resources import ResourceMonitor
    runtime.setup(config)
    index = open_test_cache(config, data, cache)
    rows = test_rows(config, data)
    cases = [('test_%04d' % i, r, r, 'normal') for i, r in enumerate(rows)]
    out = Path(config['workspace_root']) / 'output' / name
    resume = out.exists()
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    if resume:
        for line in (out / 'outputs.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r['method'], r['case'])] = r
        for k, r in done.items():
            if sha256(r['image']) != r['image_sha256']:
                raise RuntimeError('resumed output sha mismatch: %s' % r['image'])
        log(events, 'infer_resume', already_done=len(done))
    cfg_path = out / 'inference_config.json'
    if not cfg_path.exists():
        write_json(cfg_path, {'config': config, 'config_sha256': digest(config),
                              'adapter': config['adapter_checkpoint'], 'data': str(data),
                              'cache_index_sha256': sha256(Path(cache) / 'index.json'),
                              'cases': len(cases)})
    adapter = Path(config['adapter_checkpoint'])
    total = 2 * len(cases)
    counter = len(done)
    start_all = time.monotonic()
    res_path = out / 'resources.json'
    if res_path.exists():
        res_path = out / ('resources_resume_%d.json' % int(time.time()))
    with ResourceMonitor(config, res_path) as monitor:
        pipeline = CachedValidationPipeline.from_pretrained(config['model_root'], text_encoder=None,
                                                            torch_dtype=torch.bfloat16).to('cuda')
        pipeline.transformer.requires_grad_(False).eval()
        pipeline.vae.requires_grad_(False).eval()
        pipeline.set_progress_bar_config(disable=True)
        with open(out / 'outputs.jsonl', 'a', encoding='utf-8') as stream:
            for method in ('baseline', 'lora'):
                if method == 'lora':
                    pipeline.load_lora_weights(str(adapter), adapter_name='pilot')
                    saved = load_file(str(adapter / 'pytorch_lora_weights.safetensors'))
                    restored = get_peft_model_state_dict(pipeline.transformer, adapter_name='pilot')
                    exact, difference = True, 0.0
                    for key, value in restored.items():
                        original = saved['transformer.' + key]
                        exact &= torch.equal(original.to(value.dtype), value.detach().cpu())
                        difference = max(difference, float((original.float() - value.detach().cpu().float()).abs().max()))
                    write_json_replace(out / 'native_lora_reload.json',
                                       {'exact_after_dtype_cast': bool(exact),
                                        'max_absolute_weight_difference': difference,
                                        'tensors': len(restored), 'saved_tensors': len(saved),
                                        'adapter_sha256': sha256(adapter / 'pytorch_lora_weights.safetensors')})
                    if not exact or len(restored) != len(saved):
                        raise RuntimeError('native LoRA reload verification failed')
                    del saved, restored
                    log(events, 'lora_loaded', exact=bool(exact), max_diff=difference)
                for case, cache_row, original_row, intervention in cases:
                    if (method, case) in done:
                        continue
                    monitor.check()
                    item = runtime.load_cached(cache_row, index, cache)
                    pipeline.cached_image_pad_mask = item['image_pad_mask']
                    images = [Image.open(p).convert('RGBA') for p in cache_row['condition_images']]

                    def callback(pipe, step, timestep, values):
                        monitor.check()
                        return values
                    start = time.monotonic()
                    with torch.no_grad():
                        generated = pipeline(image=images, prompt_embeds=item['prompt_embeds'],
                                             prompt_embeds_mask=item['prompt_embeds_mask'],
                                             width=config['canvas_wh'][0], height=config['canvas_wh'][1],
                                             output_resolution=512, num_inference_steps=config['inference_steps'],
                                             true_cfg_scale=config['true_cfg_scale'],
                                             generator=torch.Generator(device='cuda').manual_seed(config['seed']),
                                             use_kv_cache=True, callback_on_step_end=callback).images[0]
                    target = out / ('%s_%s.png' % (method, case))
                    generated.convert('RGB').save(target)
                    row = {'method': method, 'case': case, 'intervention': intervention,
                           'image': str(target), 'image_sha256': sha256(target),
                           'cache_row': cache_row, 'original_row': original_row,
                           'seconds': time.monotonic() - start, 'seed': config['seed'],
                           'steps': config['inference_steps'], 'true_cfg_scale': config['true_cfg_scale']}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    write_json_replace(out / ('%s_%s.json' % (method, case)), row)
                    browse = Path(config['browse_root']) / method
                    browse.mkdir(parents=True, exist_ok=True)
                    bdest = browse / ('%s.png' % case)
                    if bdest.exists():
                        if sha256(bdest) != row['image_sha256']:
                            raise RuntimeError('browse copy sha mismatch: %s' % bdest)
                    else:
                        shutil.copyfile(target, bdest)
                        if sha256(bdest) != row['image_sha256']:
                            raise RuntimeError('browse copy sha mismatch: %s' % bdest)
                    counter += 1
                    if counter % 25 == 0 or counter == total:
                        elapsed = time.monotonic() - start_all
                        rate = elapsed / max(counter - len(done), 1)
                        log(events, 'infer_progress', method=method, done=counter, total=total,
                            eta_seconds=round(rate * (total - counter)))
                    del item, images, generated
    log(events, 'infer_done', images=counter, expected=total)
    if counter != total:
        raise RuntimeError('inference incomplete: %d/%d' % (counter, total))
    return out


# ---------------------------------------------------------------- evaluation
def cluster_ci(deltas_by_group, reps, seed):
    groups = sorted(g for g, v in deltas_by_group.items() if len(v))
    if not groups:
        return float('nan'), float('nan')
    arrays = {g: np.asarray(deltas_by_group[g], dtype=float) for g in groups}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        pooled = np.concatenate([arrays[groups[j]] for j in pick])
        means[i] = pooled.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_test(config, inference_dir, name, campaign_dir, events):
    from inference import panels
    sys.path.insert(0, config['project_root'])
    from src.evaluation.joint_pilot_metrics import PoseMetric, IdentityMetric
    from resources import ResourceMonitor
    inference_dir = Path(inference_dir)
    out = Path(config['workspace_root']) / 'report' / name
    out.mkdir(parents=True, exist_ok=False)
    pose = PoseMetric(Path(config['project_root']) / 'models')
    identity = IdentityMetric(Path(config['project_root']) / 'models')
    embeddings = {}

    def embedding(path):
        if path not in embeddings:
            with Image.open(path) as image:
                embeddings[path] = identity.embedding(image)
        return embeddings[path]

    rows = read_jsonl(inference_dir / 'outputs.jsonl')
    results = []
    with ResourceMonitor(config, out / 'resources.json') as monitor:
        for i, row in enumerate(rows):
            monitor.check()
            with Image.open(row['image']) as image:
                detected = pose.detect(image)
            reference = read_json(row['cache_row']['target_pose_metadata'])
            metrics = pose_stats(detected, reference['keypoints_xy'], reference['valid'], reference['size'])
            src_emb = embedding(row['original_row']['source_rgb_path'])
            gen_emb = embedding(row['image'])
            metrics.update({'identity_a_cosine': identity.cosine(src_emb, gen_emb),
                            'source_face_detected': src_emb is not None,
                            'generated_face_detected': gen_emb is not None})
            results.append({**{k: row[k] for k in ('method', 'case', 'intervention', 'image', 'seconds')},
                            'split': row['original_row']['split'], 'pair_id': row['original_row']['pair_id'],
                            'group_id': row['original_row']['group_id'],
                            'desired_pose_pair_id': row['cache_row']['pair_id'], **metrics})
            if (i + 1) % 200 == 0 or i + 1 == len(rows):
                log(events, 'evaluate_progress', done=i + 1, total=len(rows))
        write_json_replace(out / 'per_image.jsonl', results)
        grouped = defaultdict(list)
        for r in results:
            grouped[(r['method'], r['split'], r['intervention'])].append(r)
        summary = []
        for (method, split, intervention), values in grouped.items():
            item = {'method': method, 'split': split, 'intervention': intervention, 'images': len(values),
                    'pose_detection_failures': sum(not v['pose_detected'] for v in values),
                    'source_face_failures': sum(not v['source_face_detected'] for v in values),
                    'generated_face_failures': sum(not v['generated_face_detected'] for v in values)}
            for key in POSE_FIELDS + ['identity_a_cosine']:
                valid = [v[key] for v in values if v[key] is not None]
                item[key] = float(np.mean(valid)) if valid else None
                item[key + '_n'] = len(valid)
            summary.append(item)
        ctrl = read_json(Path(campaign_dir) / 'controls.json')
        by_case = {}
        for r in results:
            by_case.setdefault(r['case'], {})[r['method']] = r
        paired, excluded_pose, excluded_face = [], 0, 0
        for case, methods in sorted(by_case.items()):
            if 'baseline' not in methods or 'lora' not in methods:
                continue
            b, l = methods['baseline'], methods['lora']
            entry = {'case': case, 'pair_id': b['pair_id'], 'group_id': b['group_id']}
            ok_pose = b['pose_detected'] and l['pose_detected']
            ok_face = b['identity_a_cosine'] is not None and l['identity_a_cosine'] is not None
            if not ok_pose:
                excluded_pose += 1
            if not ok_face:
                excluded_face += 1
            for key in POSE_FIELDS:
                entry[key] = (l[key] - b[key]) if ok_pose and b[key] is not None and l[key] is not None else None
            entry['identity_a_cosine'] = (l['identity_a_cosine'] - b['identity_a_cosine']) if ok_face else None
            entry['baseline_pck005_at_ceiling'] = b['pck_005_diagonal'] == 1.0
            paired.append(entry)
        decisions = {}
        for key, sign in SIGNS.items():
            vals = [p[key] * sign for p in paired if p[key] is not None]
            by_group = defaultdict(list)
            for p in paired:
                if p[key] is not None:
                    by_group[p['group_id']].append(p[key] * sign)
            lo, hi = cluster_ci(by_group, config['bootstrap_reps'], config['bootstrap_seed'])
            mean = float(np.mean(vals))
            verdict = 'improvement' if lo > 0 else ('regression' if hi < 0 else 'inconclusive')
            decisions[key] = {'n': len(vals), 'mean_delta_sign_adjusted': mean,
                              'ci95': [lo, hi], 'verdict': verdict,
                              'per_group_mean': {g: float(np.mean(v)) for g, v in sorted(by_group.items())}}
        ceiling = float(np.mean([p['baseline_pck005_at_ceiling'] for p in paired]))
        ctrl_summary = {}
        for arm in ('copyA', 'shuffleA'):
            ctrl_summary[arm] = {k: float(np.mean([v[k] for v in ctrl[arm].values() if v[k] is not None]))
                                 for k in POSE_FIELDS}
        metrics = {'inference': str(inference_dir), 'summary': summary, 'paired_n': len(paired),
                   'per_image_records': len(results),
                   'excluded_pose_pairs': excluded_pose, 'excluded_face_pairs': excluded_face,
                   'baseline_pck005_ceiling_fraction': ceiling,
                   'decisions': decisions, 'controls': ctrl_summary,
                   'pose_protocol': 'COCO17 joints mapped from DWPose134; image-diagonal normalization; '
                                    'detected score>0.3; target-valid denominator includes missed joints',
                   'identity_protocol': 'existing CPU SCRFD/glintr100, largest detected face; source is native A RGB; '
                                        'failures recorded, not imputed',
                   'test_used': True, 'one_shot': True,
                   'preregistration': str(Path(config['workspace_root']) / 'report' /
                                          ('test_preregistration_%s.json' % config['_tag']))}
        write_json_replace(out / 'metrics.json', metrics)
        with open(out / 'paired.jsonl', 'w', encoding='utf-8', newline='\n') as f:
            for p in paired:
                f.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + '\n')
        panels(config, rows, out)
        n_panels = len(list(out.glob('*_test_comparison.png')))
        log(events, 'evaluate_done', per_image=len(results), panels=n_panels,
            paired=len(paired), excluded_pose=excluded_pose, excluded_face=excluded_face)
        if n_panels != len(by_case):
            raise RuntimeError('panel count %d != cases %d' % (n_panels, len(by_case)))
        write_report(config, out, metrics, ctrl, events)
        browse = Path(config['browse_root'])
        (browse / 'panels').mkdir(parents=True, exist_ok=True)
        panels(config, rows, browse / 'panels')
        n_browse = len(list((browse / 'panels').glob('*_test_comparison.png')))
        if n_browse != len(by_case):
            raise RuntimeError('browse panel count %d != cases %d' % (n_browse, len(by_case)))
        for fn in ('metrics.json', 'REPORT.md', 'per_image.jsonl', 'paired.jsonl'):
            shutil.copyfile(out / fn, browse / fn)
        records = []
        for r in rows:
            p = Path(r['image'])
            records.append({'dest': 'pictures/test/%s/%s.png' % (r['method'], r['case']),
                            'source': str(p), 'kind': 'generated', 'method': r['method'],
                            'bytes': p.stat().st_size, 'sha256': r['image_sha256'], 'verified': True})
        for p in sorted((browse / 'panels').glob('*.png')):
            records.append({'dest': 'pictures/test/panels/' + p.name, 'source': str(out / p.name),
                            'kind': 'panel', 'method': None, 'bytes': p.stat().st_size,
                            'sha256': sha256(p), 'verified': True})
        for fn in ('metrics.json', 'REPORT.md', 'per_image.jsonl', 'paired.jsonl'):
            p = browse / fn
            records.append({'dest': 'pictures/test/' + fn, 'source': str(out / fn), 'kind': 'report',
                            'method': None, 'bytes': p.stat().st_size, 'sha256': sha256(p),
                            'verified': True})
        write_json_replace(browse / 'manifest.json', {'run': 'test_20261112_v1', 'files': records})
        log(events, 'browse_archived', path=str(browse), records=len(records))
    return out


def write_report(config, out, metrics, ctrl, events):
    dec = metrics['decisions']
    lines = ['# 全量 test 一次性评测（%s）' % config['_tag'], '',
             '1146 对 test、5 个视频组，baseline 与 LoRA(checkpoint-009520) 同 seed/同输入各出图一次，'
             '逐对五列 panel 见本目录 `test_*_test_comparison.png`。判据在出图前预注册：'
             '`../test_preregistration_%s.json`。跑完不回头调参。' % config['_tag'], '',
             '## 判定（cluster bootstrap over 5 video groups, %d reps, 95%% CI）' % config['bootstrap_reps'], '',
             '| 指标 | n | 平均差(符号调整后, 正=LoRA好) | 95% CI | 判定 |', '|---|---|---|---|---|']
    for key in SIGNS:
        d = dec[key]
        lines.append('| %s | %d | %+.5f | [%+.5f, %+.5f] | %s |'
                     % (key, d['n'], d['mean_delta_sign_adjusted'], d['ci95'][0], d['ci95'][1], d['verdict']))
    lines += ['', '排除：pose 任一侧检测失败 %d 对；人脸 embedding 缺失 %d 对。'
              % (metrics['excluded_pose_pairs'], metrics['excluded_face_pairs']),
              'baseline PCK@.05 满分（天花板）比例 %.3f。' % metrics['baseline_pck005_ceiling_fraction'], '',
              '## 控制组（同协议，CPU）', '',
              '| 臂 | PCK@.02 | PCK@.05 | PCK@.10 | NME |', '|---|---|---|---|---|']
    for arm in ('shuffleA', 'copyA'):
        c = metrics['controls'][arm]
        lines.append('| %s | %.5f | %.5f | %.5f | %.5f |'
                     % (arm, c['pck_002_diagonal'], c['pck_005_diagonal'], c['pck_010_diagonal'], c['nme_diagonal']))
    for row in metrics['summary']:
        if row['method'] in ('baseline', 'lora') and row['intervention'] == 'normal':
            lines.append('| %s | %.5f | %.5f | %.5f | %.5f |'
                         % (row['method'], row['pck_002_diagonal'], row['pck_005_diagonal'],
                            row['pck_010_diagonal'], row['nme_diagonal']))
    lines += ['', '## 分视频组平均差（符号调整后）', '']
    for key in ('pck_002_diagonal', 'identity_a_cosine'):
        lines.append('- %s: %s' % (key, json.dumps(dec[key]['per_group_mean'], ensure_ascii=False)))
    lines += ['', '## 资源与完整性', '',
              '推理与评测的 resources.json 在各自目录；violation 见其中字段。',
              'panel 数 %d，per_image 记录 %d。' % (len(list(out.glob('*_test_comparison.png'))),
                                                  metrics['per_image_records']),
              'test 编码独立于训练缓存（schema dual_edit_disk_cache_test_v1），训练 loader 无法打开。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    log(events, 'report_written', path=str(out / 'REPORT.md'))


# ---------------------------------------------------------------- campaign
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=['audit', 'controls', 'campaign'], default='campaign')
    args = parser.parse_args()
    config = read_json(args.config)
    config['_tag'] = args.tag
    data = Path(config['workspace_root']) / 'data' / 'data_20261111_v1'
    campaign = Path(config['workspace_root']) / 'output' / ('test_campaign_%s' % args.tag)
    campaign.mkdir(parents=True, exist_ok=True)
    events = campaign / 'events.jsonl'
    report_dir = Path(config['workspace_root']) / 'report'
    try:
        if args.stage == 'campaign':
            prereg(config, data, args.tag, report_dir, events)
        if args.stage in ('audit', 'campaign'):
            result = audit(config, data, events)
            write_json_replace(campaign / 'audit.json', result)
        if args.stage == 'audit':
            return
        if args.stage in ('controls', 'campaign'):
            controls(config, data, campaign, events)
        if args.stage == 'controls':
            return
        gate(config, events)
        cache = cache_test(config, data, 'cache_test_%s' % args.tag, events)
        inference = infer_test(config, data, cache, 'test_comparison_%s' % args.tag, events)
        evaluate_test(config, inference, 'test_report_%s' % args.tag, campaign, events)
        log(events, 'campaign_completed', report=str(report_dir / ('test_report_%s' % args.tag)))
    except Exception as error:
        (campaign / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        log(events, 'campaign_failed', error=str(error))
        raise


if __name__ == '__main__':
    main()