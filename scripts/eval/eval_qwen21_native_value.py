#!/usr/bin/env python3
"""Fixed-input native Qwen/Body/Face comparison; never trains or changes weights.

Prompts are supplied as {pair_id: {method: full_prompt}}. For independent prompt
calibration, use {variant: {pair_id: {method: full_prompt}}} and --prompt-variant.
Every source/target path and identity is explicit: sequence equality is never
used to infer person identity, and B's RGB is never an Adapter reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import importlib.metadata
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.letterbox import Letterbox
from scripts.eval.qwen21_condition_combinations import NEW_METHODS, COMBINATION_REFERENCES

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('/home/shangguanrz/nihaochen/qwen21_full/native_value_audit_20261008')
FULL_RUN = Path('/home/shangguanrz/nihaochen/qwen21_full')
METHODS = ('native_ab', 'native_ab_crop', 'native_skeleton', 'native_geometry',
           'simple_body', 'full_body', 'full_body_face', 'full_body_native_face', 'simple_body1024')
LEGACY_METHODS = METHODS
METHODS = LEGACY_METHODS + NEW_METHODS
BODY_PATH = FULL_RUN / 'tiktok_body_fullpass_20261004/training/checkpoint_step_006000.pt'
FACE_PATH = FULL_RUN / 'tiktok_face_v1_20261004/training/checkpoint_step_003000.pt'
SIMPLE_PATH = ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt'
SIMPLE1024_PATH = Path('/home/shangguanrz/nihaochen/iper1024/body_selected.pt')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({row['pair_id'] for row in rows}) != len(rows):
        raise ValueError('Pair manifest must be nonempty with unique pair_id values')
    for row in rows:
        if row.get('dataset') not in ('iper', 'tiktok'):
            raise ValueError('Dataset must be explicitly iper or tiktok')
        for field in ('source_rgb', 'target_rgb', 'source_appearance', 'target_appearance',
                      'source_stem', 'target_stem'):
            if not row.get(field):
                raise ValueError(f'Missing {field} for {row["pair_id"]}')
        if row['source_appearance'] == row['target_appearance']:
            raise ValueError('This audit requires explicitly different-person A/B cases')
        if '/' in row['pair_id'] or '..' in row['pair_id']:
            raise ValueError('pair_id must be safe as a filename')
    return rows


def read_prompts(path, rows, methods, variant=None):
    data = json.loads(Path(path).read_text())
    if variant is not None:
        data = data[variant]
    resolved = {}
    for row in rows:
        pair = data[row['pair_id']]
        resolved[row['pair_id']] = {}
        for method in methods:
            alias = 'simple_body' if method == 'simple_body1024' else method
            value = pair.get(method, pair.get(alias))
            if not isinstance(value, str) or len(value.strip()) < 120:
                raise ValueError(f'Explicit detailed prompt missing/too short: {row["pair_id"]}/{method}')
            resolved[row['pair_id']][method] = value.strip()
    return resolved


def read_rgb(path):
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba).convert('RGB')


def rgb_tensor(image):
    array = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


def source_face_crop(row, output):
    """Use A's reliable original-image detector box, never B's geometry box."""
    path = row.get('source_face_crop')
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return read_rgb(path), {'path': str(path), 'rule': 'manifest_locked_crop',
                                'source': str(row['source_rgb'])}
    box = row.get('source_face_bbox_pixels')
    if box is None:
        return None, {'available': False, 'reason': 'no reliable original A detection/crop'}
    if row.get('source_face_usable', True) is not True:
        return None, {'available': False, 'reason': 'A detector record explicitly unusable'}
    rgb = read_rgb(row['source_rgb'])
    width, height = rgb.size
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('A original detection box is invalid')
    side = max(x1 - x0, y1 - y0) * 1.8
    left = max(0, min(width - side, (x0 + x1) / 2 - side / 2))
    top = max(0, min(height - side, (y0 + y1) / 2 - side / 2))
    crop_box = [int(left), int(top), int(left + side), int(top + side)]
    image = rgb.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)
    path = Path(output) / 'inputs' / row['pair_id'] / 'A_face.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image, {'available': True, 'path': str(path), 'source': str(row['source_rgb']),
                   'detection_bbox_pixels': list(box), 'crop_bbox_pixels': crop_box,
                   'expansion': 1.8, 'output_size': [512, 512], 'resize': 'LANCZOS'}


def save_inputs(row, resolution, output):
    directory = Path(output) / 'inputs' / row['pair_id']
    directory.mkdir(parents=True, exist_ok=True)
    a_original, b_original = read_rgb(row['source_rgb']), read_rgb(row['target_rgb'])
    a = Letterbox.from_size(a_original.size, resolution)
    b = Letterbox.from_size(b_original.size, resolution)
    images = {'A': a.image(a_original, fill=(128, 128, 128)),
              'B': b.image(b_original, fill=(128, 128, 128))}
    paths = {}
    for name in ('A', 'B'):
        paths[name] = str(directory / f'{name}.png')
        images[name].save(paths[name])
    for name, field in (('B_skeleton', 'target_dwpose'), ('B_normal', 'target_normal'),
                        ('B_depth', 'target_depth_vis')):
        if row.get(field):
            mode = 'L' if name == 'B_depth' else 'RGB'
            with Image.open(row[field]) as image:
                images[name] = b.image(image.convert(mode), fill=0).convert('RGB')
            paths[name] = str(directory / f'{name}.png')
            images[name].save(paths[name])
    crop, crop_meta = source_face_crop(row, output)
    if crop is not None:
        images['A_face'] = crop
        paths['A_face'] = crop_meta['path']
    return images, paths, {'source': a.metadata(), 'target': b.metadata()}, crop_meta


def references_for_method(method, images, paths, body_row=None):
    keys = {'native_ab': ('A', 'B'), 'native_ab_crop': ('A', 'B', 'A_face'),
            'native_skeleton': ('A', 'B_skeleton'),
            'native_geometry': ('A', 'B_normal', 'B_depth'),
            'simple_body': ('A',), 'simple_body1024': ('A',),
            'full_body': ('A',), 'full_body_face': ('A',)}
    keys.update(COMBINATION_REFERENCES)
    if method == 'full_body_native_face':
        if body_row is None:
            raise ValueError('Native face edit requires the exact matching full_body generation')
        return [read_rgb(body_row['image']), images['A_face']], [
            {'index': 1, 'role': 'generated_C_from_full_body', 'path': body_row['image']},
            {'index': 2, 'role': 'A_face', 'path': paths['A_face']}]
    missing = [name for name in keys[method] if name not in images]
    if missing:
        raise ValueError(f'Missing locked reference inputs for {method}: {missing}')
    return ([images[name] for name in keys[method]],
            [{'index': index + 1, 'role': name, 'path': paths[name]} for index, name in enumerate(keys[method])])


def validate_completed_record(saved, args, prompt, prepared):
    """A resume must preserve all experimental inputs, not just its filename."""
    expected = {'resolution': args.resolution, 'steps': args.steps, 'seed': args.seed,
                'true_cfg_scale': 1.0, 'use_kv_cache': False,
                'prompt_variant': args.prompt_variant}
    for name, value in expected.items():
        if saved.get(name) != value:
            raise ValueError(f'Resume configuration differs: {name}')
    if saved.get('prompt') != prompt or saved.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError('Resume prompt differs')
    images, paths, _, _ = prepared
    if saved['method'] != 'full_body_native_face':
        _, references = references_for_method(saved['method'], images, paths)
        if [(r['index'], r['role'], r['path']) for r in saved['references']] != [(r['index'], r['role'], r['path']) for r in references]:
            raise ValueError('Resume reference order/roles/path differ')
    for record in saved['references']:
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Resume reference bytes differ')
    if saved.get('lora_loaded') is not False:
        raise ValueError('Resume unexpectedly used LoRA')


class AuditDataset:
    """Read B geometry with existing readers; attach independent actual A metadata."""
    def __init__(self, rows, resolution, workdir):
        from src.data.iper_dataset import IPERPoseDataset
        from src.data.tiktok_dataset import TikTokV3PoseDataset
        self.rows, self.resolution, self.lookup, self.readers = rows, resolution, {}, {}
        directory = Path(workdir) / 'geometry_proxy_manifests'
        directory.mkdir(parents=True, exist_ok=True)
        for dataset in ('iper', 'tiktok'):
            selected = [(index, row) for index, row in enumerate(rows) if row['dataset'] == dataset]
            if not selected:
                continue
            proxies = []
            for local_index, (index, row) in enumerate(selected):
                self.lookup[index] = dataset, local_index
                original = dict(row['target_original_row'])
                if dataset == 'iper':
                    if original['target']['stem'] != row['target_stem']:
                        raise ValueError('B proxy target stem mismatch')
                    original['appearance'] = row['target_appearance']
                    original['source'] = original['target']
                else:
                    original.update(sequence_id=row['target_appearance'],
                                    source_rgb=row['target_rgb'], target_rgb=row['target_rgb'],
                                    source_stem=row['target_stem'], target_stem=row['target_stem'])
                    for field in ('target_person_mask', 'target_dwpose', 'target_normal', 'target_depth_vis'):
                        if row.get(field):
                            original[field] = row[field]
                proxies.append(original)
            path = directory / f'{dataset}_{resolution}.jsonl'
            content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in proxies)
            if path.exists() and path.read_text() != content:
                raise ValueError('Geometry proxy manifest changed; use a new output directory')
            path.write_text(content)
            if dataset == 'iper':
                self.readers[dataset] = IPERPoseDataset(str(path), str(ROOT / 'datasets/iPER/iper_sampled_6src64tgt'),
                                                       str(ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast'),
                                                       resolution=resolution, augment=False)
            else:
                self.readers[dataset] = TikTokV3PoseDataset(path, resolution=resolution,
                                                           align_mode='letterbox', strict_conditions=True)

    def sample(self, index, images, alignment):
        row = self.rows[index]
        dataset, local_index = self.lookup[index]
        if dataset == 'iper':
            for side in ('source', 'target'):
                record = alignment[side]
                if record['width'] != record['height']:
                    raise ValueError('Existing iPER geometry requires square original FoV; refusing silent stretching')
        sample = self.readers[dataset][local_index]
        sample['src_image'], sample['tgt_image'] = rgb_tensor(images['A']), rgb_tensor(images['B'])
        sample['alignment_meta'] = json.dumps(alignment, sort_keys=True)
        sample['appearance'], sample['source_stem'] = row['source_appearance'], row['source_stem']
        sample['target_stem'] = row['target_stem']
        sample['pair_metadata'] = {**row, 'cross_person': True}
        return sample


class AuditFaceStore:
    """Construct the unchanged full Face store with explicit per-appearance cache paths."""
    def __new__(cls, rows):
        from src.pose_control.qwen21_full.data_runtime import FaceConditionStore
        locations = {}
        for row in rows:
            for side in ('source', 'target'):
                appearance = str(row[f'{side}_appearance'])
                path = str(Path(row[f'{side}_face_cache']).resolve())
                if appearance in locations and locations[appearance] != path:
                    raise ValueError('One appearance resolves to multiple face caches')
                locations[appearance] = path

        class ExplicitStore(FaceConditionStore):
            def __init__(self):
                self._caches = {}

            def _load(self, appearance):
                if appearance not in self._caches:
                    from scripts.precompute_iper_face_features_v65 import load_v65_face_cache
                    cache = load_v65_face_cache(locations[appearance])
                    if ('geometry_valid' not in cache or cache['geometry_valid'].shape != cache['valid'].shape
                            or cache['geometry_valid'].tolist() != cache['metadata'].get('geometry_valid')):
                        raise ValueError('Face cache geometry validity is missing or unaudited')
                    self._caches[appearance] = cache
                return self._caches[appearance]
        return ExplicitStore()


class AdapterPool:
    def __init__(self, args):
        self.args, self.current, self.adapter, self.payload = args, None, None, None

    def activate(self, method):
        source = {'simple_body': self.args.simple_checkpoint, 'simple_body1024': self.args.simple1024_checkpoint,
                  'full_body': self.args.body_checkpoint, 'full_body_face': self.args.face_checkpoint}.get(method)
        if source is None:
            self.release()
            return None, None
        key = str(source)
        if self.current != key:
            self.release()
            if method.startswith('simple_'):
                from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter
                self.adapter = Qwen21BodyAdapter()
                self.payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
                self.adapter.load_state_dict(self.payload['adapter_state_dict'], strict=True)
            else:
                from src.pose_control.qwen21_full.execution import new_adapter, load_adapter_checkpoint
                self.adapter = new_adapter('cpu')
                self.payload = load_adapter_checkpoint(self.adapter, source)
            self.adapter.to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)
            self.current = key
        return self.adapter, self.payload

    def release(self):
        self.adapter = self.payload = self.current = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SamplingWitness:
    """Observe actual sampling without changing hidden states or predictions.

    Bracket each control hook to verify native prefix values remain bitwise
    unchanged. Native routes install no control context; only this observer.
    Face locality is checked at token-cell resolution, including cells which
    intersect a box boundary rather than pretending masks are pixel-perfect.
    """
    def __init__(self, transformer, context, steps, grid, prepared=None):
        self.transformer, self.context, self.steps, self.grid = transformer, context, steps, grid
        self.prepared, self.handles, self.prefixes = prepared, [], {}
        self.forward_calls = self.prefix_checks = self.face_checks = 0
        self.prefix_token_counts, self.max_face_outside = set(), 0.0
        self.target_tokens = grid * grid

    def _before_model(self, module, args, kwargs):
        values = dict(inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments)
        if 'kwargs' in values:
            values.update(values.pop('kwargs'))
        shapes = values['img_shapes']
        if any(tuple(row[-1]) != (1, self.grid, self.grid) for row in shapes):
            raise AssertionError('Native target token layout differs from requested resolution')
        prefix = values['hidden_states'].shape[1] - self.target_tokens
        if prefix < 0:
            raise AssertionError('Native target token segment is missing')
        self.prefix_token_counts.add(prefix)
        self.forward_calls += 1

    def _capture_prefix(self, index):
        def capture(module, args, output):
            self.prefixes[index] = output[:, :-self.target_tokens].detach().clone()
        return capture

    def _check_prefix(self, index):
        def check(module, args, output):
            before = self.prefixes.pop(index)
            if not torch.equal(before, output[:, :-self.target_tokens]):
                raise AssertionError(f'Control directly changed native prefix at block {index}')
            self.prefix_checks += 1
            # Check once per denoising step after the complete raw Face output
            # exists. Final scheduling zeros do not hide earlier leakage.
            if index != 0 or self.prepared is None or self.prepared.face is None:
                return
            raw_face = self.context.output.raw_residuals.face
            if raw_face is None:
                return
            face = self.prepared.face
            boxes = face.target_boxes.float()
            lower = torch.floor(boxes[..., :2] * self.grid)
            upper = torch.ceil(boxes[..., 2:] * self.grid)
            y = torch.arange(self.grid, device=boxes.device)[None, None, :, None]
            x = torch.arange(self.grid, device=boxes.device)[None, None, None, :]
            support = ((x >= lower[..., 0, None, None]) & (x < upper[..., 0, None, None])
                       & (y >= lower[..., 1, None, None]) & (y < upper[..., 1, None, None])
                       & face.face_valid[..., None, None]).any(1).flatten(1)
            for delta in raw_face:
                outside = delta[~support]
                amount = float(outside.float().abs().max()) if outside.numel() else 0.0
                self.max_face_outside = max(self.max_face_outside, amount)
                if amount != 0:
                    raise AssertionError('Face residual leaked beyond target face token-cell region')
            self.face_checks += 1
        return check

    def _check_prediction(self, module, args, output):
        value = output[0] if isinstance(output, (tuple, list)) else getattr(output, 'sample', output)
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite Qwen velocity prediction')

    def __enter__(self):
        self.handles.append(self.transformer.register_forward_pre_hook(self._before_model, with_kwargs=True))
        if self.context is not None:
            for index, block in enumerate(self.transformer.transformer_blocks):
                self.handles.append(block.register_forward_hook(self._capture_prefix(index)))
            self.context.__enter__()
            for index, block in enumerate(self.transformer.transformer_blocks):
                self.handles.append(block.register_forward_hook(self._check_prefix(index)))
        self.handles.append(self.transformer.register_forward_hook(self._check_prediction))
        return self

    def __exit__(self, kind, error, traceback):
        if self.context is not None:
            self.context.__exit__(kind, error, traceback)
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if kind is None and self.forward_calls != self.steps:
            raise AssertionError('Incomplete Qwen sampling witness count')
        return False

    def record(self):
        return {'native_transformer_calls': self.forward_calls,
                'control_prefix_bitwise_checks': self.prefix_checks,
                'native_prefix_image_token_counts': sorted(self.prefix_token_counts),
                'target_token_count': self.target_tokens, 'target_token_grid': [self.grid, self.grid],
                'face_region_checks': self.face_checks,
                'face_outside_target_token_cell_region_max_abs': self.max_face_outside,
                'region_definition': 'all token cells intersecting the target face box; includes boundary cells',
                'control_context_installed': self.context is not None}


def encode_source_latent(pipe, source, resolution, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.convert('RGB').tobytes()).hexdigest()
    path = directory / f'source_vae_letterbox_v1_{resolution}_{digest}.pt'
    if path.exists():
        latent = torch.load(path, map_location='cpu', weights_only=True)
    else:
        image = pipe.image_processor.preprocess(source.convert('RGBA'), width=resolution, height=resolution)
        image = image.unsqueeze(2).to(device='cuda', dtype=pipe.vae.dtype)
        latent = pipe._encode_vae_image(image, torch.Generator(device='cuda').manual_seed(0))
        latent = latent.flatten(3).squeeze(2).transpose(1, 2).detach().cpu().contiguous()
        torch.save(latent, path)
    if latent.shape != (1, (resolution // 16) ** 2, 64) or not torch.isfinite(latent).all():
        raise ValueError('Source Qwen VAE cache shape/finite check failed')
    return latent.to('cuda')


def verify_frozen_body_checkpoint_pair(body_path, face_path):
    """Face-on and Face-off must start from exactly the same frozen Body."""
    from src.pose_control.qwen21_full.checkpoint import FACE_ONLY_PREFIXES
    body = torch.load(body_path, map_location='cpu', weights_only=False, mmap=True)
    face = torch.load(face_path, map_location='cpu', weights_only=False, mmap=True)
    for payload in (body, face):
        if payload.get('architecture') != 'qwen21_full_v1':
            raise ValueError('Full Body/Face comparison requires verified full checkpoints')
    a, b = body['adapter_state_dict'], face['adapter_state_dict']
    if set(a) != set(b):
        raise AssertionError('Full Body and Face checkpoint keys differ')
    allowed = ('output_bridges.face.',) + tuple('legacy.' + prefix for prefix in FACE_ONLY_PREFIXES)
    changed_face = checked_frozen = 0
    for name in a:
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
            raise AssertionError(f'Body/Face state shape or dtype changed: {name}')
        equal = torch.equal(a[name], b[name])
        if name.startswith(allowed):
            changed_face += int(not equal)
        else:
            checked_frozen += 1
            if not equal:
                raise AssertionError(f'Face checkpoint changed a frozen Body/shared/other parameter: {name}')
    return {'body_sha256': sha256(body_path), 'face_sha256': sha256(face_path),
            'frozen_tensors_bitwise_equal': checked_frozen,
            'changed_face_tensors': changed_face,
            'allowed_face_prefixes': list(allowed),
            'body_training_resolution': body.get('resolution'), 'face_training_resolution': face.get('resolution'),
            'scope': 'All non-Face state tensors match exactly; Face-only disabled/invalid computation uses this same Body'}


def checkpoint_resolution(payload, checkpoint_sha):
    value = payload.get('resolution', payload.get('args', {}).get('resolution'))
    if value in (512, 1024):
        return value, 'explicit_checkpoint_or_training_args'
    historical = FULL_RUN / 'weight_audit'
    fingerprints = json.loads((historical / 'source_fingerprints.json').read_text())
    summary = json.loads((historical / 'summary.json').read_text())
    for name, source in fingerprints.items():
        if source['sha256'] == checkpoint_sha and name in summary['sources']:
            value = summary['sources'][name]['resolution']
            if value in (512, 1024):
                return value, f'verified_historical_weight_audit:{name}'
    raise ValueError('Checkpoint training resolution cannot be verified')


def append_record(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        key = ('pair_id', 'method', 'resolution', 'steps', 'prompt_variant')
        for existing in previous:
            if all(existing.get(name) == row.get(name) for name in key):
                if existing != row:
                    raise ValueError(f'Conflicting generation index record: {row["pair_id"]}/{row["method"]}')
                return
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def build_config(args, rows, prompts):
    checkpoints = {method: source for method, source in (
        ('simple_body', args.simple_checkpoint), ('simple_body1024', args.simple1024_checkpoint),
        ('full_body', args.body_checkpoint), ('full_body_face', args.face_checkpoint)) if method in args.methods}
    if 'full_body_native_face' in args.methods:
        checkpoints['full_body'] = args.body_checkpoint
    input_fields = ('source_rgb', 'target_rgb', 'target_dwpose', 'target_normal', 'target_depth_vis',
                    'source_face_crop', 'source_face_cache', 'target_face_cache')
    fingerprints = {}
    for row in rows:
        for field in input_fields:
            path = row.get(field)
            if path and str(path) not in fingerprints:
                fingerprints[str(path)] = sha256(path)
    return {'architecture': 'native_value_audit_v1', 'model': str(args.model), 'resolution': args.resolution,
            'model_revision': '790c92633540aa0cb11d9abf19eb46d861714758',
            'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                         'diffusers': importlib.metadata.version('diffusers'),
                         'transformers': importlib.metadata.version('transformers'),
                         'torch_cuda': torch.version.cuda},
            'steps': args.steps, 'seed': args.seed, 'true_cfg_scale': 1.0, 'use_kv_cache': False,
            'align_mode': 'letterbox_v1', 'progress_mode': 'sigma', 'body_strength': 1.0, 'face_strength': 1.0,
            'face_scatter_grid': args.face_scatter_grid, 'methods': args.methods,
            'prompt_variant': args.prompt_variant, 'manifest_path': str(args.pairs), 'manifest_sha256': sha256(args.pairs),
            'prompts_path': str(args.prompts), 'prompts_sha256': sha256(args.prompts),
            'pair_ids': [row['pair_id'] for row in rows], 'resolved_prompts': prompts,
            'checkpoint_sha256': {method: sha256(path) for method, path in checkpoints.items()},
            'checkpoint_paths': {method: str(path) for method, path in checkpoints.items()},
            'input_files_sha256': fingerprints, 'training': False, 'lora_loaded': False,
            'cross_person_B_is_reconstruction_target': False}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--prompt-variant')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index-file', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(LEGACY_METHODS[:-1]))
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--steps', type=int, choices=(30, 40), default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--highres-only', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--body-checkpoint', type=Path, default=BODY_PATH)
    parser.add_argument('--face-checkpoint', type=Path, default=FACE_PATH)
    parser.add_argument('--simple-checkpoint', type=Path, default=SIMPLE_PATH)
    parser.add_argument('--simple1024-checkpoint', type=Path, default=SIMPLE1024_PATH)
    parser.add_argument('--face-scatter-grid', choices=('checkpoint', 'native'), default='checkpoint')
    parser.add_argument('--validate-only', action='store_true', help='Prepare/audit inputs and prompts without loading the GPU model')
    return parser.parse_args()


def main():
    args = arguments()
    if any(method in NEW_METHODS for method in args.methods) and args.steps != 30:
        raise ValueError('Supplementary combinations are fixed at 30 steps')
    if 'simple_body1024' in args.methods and args.resolution != 1024:
        raise ValueError('simple_body1024 is only the explicitly trained 1024 secondary control')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('methods must be unique')
    if 'full_body_native_face' in args.methods and 'full_body' not in args.methods:
        args.methods.insert(args.methods.index('full_body_native_face'), 'full_body')
    rows = read_rows(args.pairs)
    if args.highres_only:
        rows = [row for row in rows if row.get('highres_check') is True]
    if args.max_samples:
        rows = rows[:args.max_samples]
    if not rows:
        raise ValueError('Empty selected sample set')
    prompts = read_prompts(args.prompts, rows, args.methods, args.prompt_variant)
    args.output.mkdir(parents=True, exist_ok=True)
    config = build_config(args, rows, prompts)
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run configuration/input/prompt differs; use a separate output directory')
    json_write(config_path, config)
    needs_adapter = any(method.startswith(('simple_', 'full_')) and method != 'full_body_native_face'
                        for method in args.methods)
    dataset = AuditDataset(rows, args.resolution, args.output) if needs_adapter else None
    store = AuditFaceStore(rows) if 'full_body_face' in args.methods else None
    prepared_inputs = [save_inputs(row, args.resolution, args.output) for row in rows]
    if 'full_body_face' in args.methods:
        proof = verify_frozen_body_checkpoint_pair(args.body_checkpoint, args.face_checkpoint)
        json_write(args.output / 'body_face_frozen_weight_check.json', proof)
    if args.validate_only:
        validations = []
        for index, row in enumerate(rows):
            images, paths, alignment, crop = prepared_inputs[index]
            for method in args.methods:
                if method != 'full_body_native_face':
                    references_for_method(method, images, paths)
            result = {'pair_id': row['pair_id'], 'alignment': alignment, 'crop': crop}
            if dataset is not None:
                sample = dataset.sample(index, images, alignment)
                for name in ('src_image', 'normal', 'depth', 'part_onehot', 'pose_heatmap', 'smplx_global'):
                    if not torch.isfinite(sample[name]).all():
                        raise FloatingPointError(f'Nonfinite input {name}')
                result['condition_shapes'] = {key: list(value.shape) for key, value in sample.items() if torch.is_tensor(value)}
                if store is not None:
                    condition, references = store.for_pairs([row], device='cpu', dtype=torch.float32,
                                                            alignments=[alignment])
                    result['face_valid'] = bool(condition.face_valid.any())
                    result['source_arcface_nonzero'] = bool(references.arcface.abs().sum() > 0)
            validations.append(result)
        json_write(args.output / 'input_validation.json', validations)
        print(json.dumps({'status': 'input_validation_passed', 'n': len(rows), 'gpu_loaded': False}), flush=True)
        return
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21.body_adapter import Qwen21ControlContext
    from src.pose_control.qwen21_full.execution import prepare, seed_all
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
    seed_all(args.seed)
    torch.set_num_threads(4)
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe.transformer, 'peft_config', {}):
        raise AssertionError('Native audit must not load any LoRA')
    pool = AdapterPool(args)
    records = args.output / 'per_sample.jsonl'
    existing = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    completed = {(row['pair_id'], row['method']): row for row in existing}
    for method in args.methods:
        # A previous context/prepared payload keeps its Adapter tensors alive;
        # remove those references before releasing or switching checkpoints.
        context = prepared = witness = adapter = payload = None
        adapter, payload = pool.activate(method)
        for index, pair in enumerate(rows):
            key = pair['pair_id'], method
            if key in completed:
                saved = completed[key]
                if not Path(saved['image']).is_file() or sha256(saved['image']) != saved['image_sha256']:
                    raise ValueError('Completed output is missing or changed')
                validate_completed_record(saved, args, prompts[pair['pair_id']][method], prepared_inputs[index])
                if args.index_file:
                    append_record(args.index_file, saved)
                continue
            images, paths, alignment, crop_meta = prepared_inputs[index]
            body_row = completed.get((pair['pair_id'], 'full_body'))
            reference_images, reference_records = references_for_method(method, images, paths, body_row)
            prompt = prompts[pair['pair_id']][method]
            if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
                raise AssertionError('Leaked Adapter hook before sampling')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            context, prepared = None, None
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                if method.startswith('simple_'):
                    sample = dataset.sample(index, images, alignment)
                    batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                             for name, value in sample.items()}
                    # The reused typed geometry records require a single FP32
                    # dtype, as in the historical simplified evaluation path.
                    with torch.autocast('cuda', enabled=False):
                        geometry = adapter.encode_geometry(batch, (args.resolution // 16,) * 2)
                    context = Qwen21ControlContext(pipe.transformer, adapter, geometry)
                elif method in ('full_body', 'full_body_face'):
                    sample = dataset.sample(index, images, alignment)
                    batch = {name: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                             for name, value in sample.items()}
                    latent = encode_source_latent(pipe, images['A'], args.resolution, args.output / 'native_cache')
                    prepared = prepare(adapter, batch, {'source_latent': latent, 'resolution': args.resolution},
                                       [pair], store, with_face=method == 'full_body_face')
                    reference_grid = None
                    if method == 'full_body_face' and args.face_scatter_grid == 'checkpoint':
                        trained = payload.get('resolution')
                        if trained not in (512, 1024):
                            raise ValueError('Face checkpoint has no verified resolution')
                        reference_grid = (trained // 16, trained // 16)
                    context = Qwen21FullControlContext(
                        pipe.transformer, adapter, prepared, face_strength=1.0 if method == 'full_body_face' else 0,
                        hand_strength=0, hand_mode='off', progress_mode='sigma', num_inference_steps=args.steps,
                        face_reference_token_hw=reference_grid)
                witness = SamplingWitness(pipe.transformer, context, args.steps,
                                          args.resolution // 16, prepared)
                with witness:
                    result = pipe(prompt=prompt, image=reference_images,
                                  width=args.resolution, height=args.resolution, output_resolution=args.resolution,
                                  num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=False,
                                  generator=torch.Generator(device='cuda').manual_seed(args.seed))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            peak = torch.cuda.max_memory_allocated() / 1024**3
            rgba = result.images[0].convert('RGBA')
            generated = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
            if generated.size != (args.resolution, args.resolution):
                raise ValueError('Qwen output resolution differs from the fixed experiment')
            image_dir = args.output / 'images'
            image_dir.mkdir(exist_ok=True)
            image_path = image_dir / f'{pair["pair_id"]}_{method}.png'
            generated.save(image_path)
            calls = {} if context is None else context.call_counts
            if context is not None and any(count != args.steps for count in calls.values()):
                raise AssertionError('Incomplete control layer sampling')
            if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
                raise AssertionError('Adapter hooks were not removed after sampling')
            for reference in reference_records:
                reference['sha256'] = sha256(reference['path'])
            is_two_stage = method == 'full_body_native_face'
            row = {'pair_id': pair['pair_id'], 'pair_index': index, 'dataset': pair['dataset'], 'method': method,
                   'resolution': args.resolution, 'steps': args.steps, 'seed': args.seed, 'true_cfg_scale': 1.0,
                   'use_kv_cache': False, 'prompt_variant': args.prompt_variant,
                   'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                   'image': str(image_path), 'image_sha256': sha256(image_path),
                   'source_rgb': paths['A'], 'target_rgb': paths['B'],
                   'source_original_rgb': pair['source_rgb'], 'target_original_rgb': pair['target_rgb'],
                   'references': reference_records, 'A_face_crop': crop_meta, 'alignment': alignment,
                   'metadata': pair, 'seconds': elapsed + (body_row['seconds'] if is_two_stage else 0),
                   'stage1_seconds': body_row['seconds'] if is_two_stage else None,
                   'stage2_seconds': elapsed if is_two_stage else None,
                   'peak_cuda_gib': max(peak, body_row['peak_cuda_gib']) if is_two_stage else peak,
                   'stage2_peak_cuda_gib': peak if is_two_stage else None,
                   'control_forwards': (0 if context is None else getattr(context, 'forward_count', args.steps)),
                   'block_calls': calls, 'lora_loaded': False,
                   'correctness_observation': witness.record(),
                   'target_rgb_in_native_references': any(ref['role'] == 'B' for ref in reference_records),
                   'face_valid': bool(prepared.face.face_valid.any()) if prepared is not None and prepared.face is not None else None,
                   'face_scatter_grid': args.face_scatter_grid if method == 'full_body_face' else None,
                   'checkpoint': config['checkpoint_paths'].get(method),
                   'checkpoint_sha256': config['checkpoint_sha256'].get(method),
                   'checkpoint_training_resolution': payload.get('resolution') if payload is not None else None,
                   'sampling_schedule': getattr(context, 'step_diagnostics', []),
                   'last_control_diagnostics': context.output.diagnostics if context is not None and hasattr(context, 'output') else None}
            if is_two_stage:
                row['first_stage_generation'] = {'image': body_row['image'], 'image_sha256': body_row['image_sha256'],
                                                 'checkpoint_sha256': body_row['checkpoint_sha256'],
                                                 'prompt_sha256': body_row['prompt_sha256']}
            if payload is not None:
                trained_resolution, evidence = checkpoint_resolution(payload, row['checkpoint_sha256'])
                row['checkpoint_training_resolution'] = trained_resolution
                row['checkpoint_training_resolution_evidence'] = evidence
            append_record(records, row)
            if args.index_file:
                append_record(args.index_file, row)
            completed[key] = row
            print(json.dumps({name: row[name] for name in ('pair_id', 'method', 'seconds', 'peak_cuda_gib', 'face_valid')}), flush=True)
    pool.release()
    summary = {method: {'n': sum((pair['pair_id'], method) in completed for pair in rows),
                         'seconds_mean': float(np.mean([completed[(pair['pair_id'], method)]['seconds'] for pair in rows])),
                         'peak_cuda_gib': max(completed[(pair['pair_id'], method)]['peak_cuda_gib'] for pair in rows)}
               for method in args.methods}
    json_write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
