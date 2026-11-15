# Validation mask bridge adapted from the pinned Diffusers Edit LoRA example, Apache-2.0.
"""Native pipeline inference, pose intervention, and CPU pose/identity evaluation."""
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import QwenImage21Pipeline
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

from core import digest, pose_stats, read_json, read_jsonl, sha256, write_json, write_jsonl
from resources import ResourceMonitor
from runtime import load_cached, open_cache, setup


class CachedValidationPipeline(QwenImage21Pipeline):
    cached_image_pad_mask = None

    def encode_prompt(self, *args, **kwargs):
        if kwargs.get('image_pad_mask') is None and self.cached_image_pad_mask is not None:
            kwargs['image_pad_mask'] = self.cached_image_pad_mask
        return super().encode_prompt(*args, **kwargs)


def inference_cases(data, limit):
    training = read_jsonl(Path(data) / 'pilot_train.jsonl')
    validation = read_jsonl(Path(data) / 'pilot_val.jsonl')
    selected, seen = [], set()
    for row in training:
        if row['group_id'] not in seen:
            selected.append(row)
            seen.add(row['group_id'])
    normal = [*validation, *selected]
    if limit is not None:
        return [(f'normal_{i:02d}', row, row, 'normal') for i, row in enumerate(normal[:limit])]
    result = [(f'normal_{i:02d}', row, row, 'normal') for i, row in enumerate(normal)]
    original, alternate = validation[:2]
    if original['condition_images'][0] != alternate['condition_images'][0]:
        raise ValueError('Swap diagnostic needs the same A with two different B poses')
    # Alternate's whole cache is valid: source A and generic prompt are byte-identical.
    result.append(('swap_pose', alternate, original, 'swap_pose'))
    blank = read_jsonl(Path(data) / 'diagnostics.jsonl')[0]
    result.append(('blank_pose', blank, original, 'blank_pose'))
    return result


def infer(config, data, cache, name, adapter=None, limit=None):
    setup(config)
    index = open_cache(config, data, cache)
    out = Path(config['workspace_root']) / 'output' / name
    out.mkdir(parents=True, exist_ok=False)
    cases = inference_cases(data, limit)
    write_json(out / 'inference_config.json', {'config': config, 'config_sha256': digest(config),
               'adapter': str(adapter) if adapter else None, 'limit': limit,
               'data': str(data), 'cache_index_sha256': sha256(Path(cache) / 'index.json')})
    with ResourceMonitor(config, out / 'resources.json') as monitor:
        pipeline = CachedValidationPipeline.from_pretrained(config['model_root'], text_encoder=None,
                                                            torch_dtype=torch.bfloat16).to('cuda')
        pipeline.transformer.requires_grad_(False).eval()
        pipeline.vae.requires_grad_(False).eval()
        pipeline.set_progress_bar_config(disable=True)
        rows = []
        for method in (['baseline', 'lora'] if adapter else ['baseline']):
            if method == 'lora':
                pipeline.load_lora_weights(str(adapter), adapter_name='pilot')
                saved = load_file(str(Path(adapter) / 'pytorch_lora_weights.safetensors'))
                restored = get_peft_model_state_dict(pipeline.transformer, adapter_name='pilot')
                exact, difference = True, 0.0
                for key, value in restored.items():
                    original = saved['transformer.' + key]
                    exact &= torch.equal(original.to(value.dtype), value.detach().cpu())
                    difference = max(difference, float((original.float() - value.detach().cpu().float()).abs().max()))
                write_json(out / 'native_lora_reload.json', {'exact_after_dtype_cast': bool(exact),
                           'max_absolute_weight_difference': difference, 'tensors': len(restored),
                           'saved_tensors': len(saved), 'adapter_sha256': sha256(Path(adapter) / 'pytorch_lora_weights.safetensors')})
                if not exact or len(restored) != len(saved):
                    raise RuntimeError('Native pipeline LoRA reload verification failed')
                del saved, restored
            for case, cache_row, original_row, intervention in cases:
                monitor.check()
                item = load_cached(cache_row, index, cache)
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
                target = out / f'{method}_{case}.png'
                generated.convert('RGB').save(target)
                row = {'method': method, 'case': case, 'intervention': intervention,
                       'image': str(target), 'image_sha256': sha256(target),
                       'cache_row': cache_row, 'original_row': original_row,
                       'seconds': time.monotonic()-start, 'seed': config['seed'],
                       'steps': config['inference_steps'], 'true_cfg_scale': config['true_cfg_scale']}
                rows.append(row)
                write_json(out / f'{method}_{case}.json', row)
                print(f'{method} {case}: {row["seconds"]:.2f}s', flush=True)
                del item, images, generated
        write_jsonl(out / 'outputs.jsonl', rows)
    return out


def evaluate(config, inference, name):
    inference = Path(inference)
    out = Path(config['workspace_root']) / 'report' / name
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, config['project_root'])
    from src.evaluation.joint_pilot_metrics import PoseMetric, IdentityMetric
    pose = PoseMetric(Path(config['project_root']) / 'models')
    identity = IdentityMetric(Path(config['project_root']) / 'models')
    embeddings = {}
    def embedding(path):
        if path not in embeddings:
            with Image.open(path) as image:
                embeddings[path] = identity.embedding(image)
        return embeddings[path]
    results = []
    with ResourceMonitor(config, out / 'resources.json'):
        rows = read_jsonl(inference / 'outputs.jsonl')
        for row in rows:
            with Image.open(row['image']) as image:
                detected = pose.detect(image)
            reference = read_json(row['cache_row']['target_pose_metadata'])
            metrics = pose_stats(detected, reference['keypoints_xy'], reference['valid'], reference['size'])
            source_embedding = embedding(row['original_row']['source_rgb_path'])
            generated_embedding = embedding(row['image'])
            metrics.update({'identity_a_cosine': identity.cosine(source_embedding, generated_embedding),
                            'source_face_detected': source_embedding is not None,
                            'generated_face_detected': generated_embedding is not None})
            # Blank input has no requested joints: its pose comparison refers to original B for diagnosis only.
            result = {**{k: row[k] for k in ('method', 'case', 'intervention', 'image', 'seconds')},
                      'split': row['original_row']['split'], 'pair_id': row['original_row']['pair_id'],
                      'desired_pose_pair_id': row['cache_row']['pair_id'], **metrics}
            results.append(result)
            print(f'evaluate {row["method"]} {row["case"]}: PCK@.05={metrics["pck_005_diagonal"]}, ID={metrics["identity_a_cosine"]}', flush=True)
        grouped = defaultdict(list)
        for result in results:
            grouped[(result['method'], result['split'], result['intervention'])].append(result)
        summary = []
        for (method, split, intervention), values in grouped.items():
            item = {'method': method, 'split': split, 'intervention': intervention, 'images': len(values),
                    'pose_detection_failures': sum(not v['pose_detected'] for v in values),
                    'source_face_failures': sum(not v['source_face_detected'] for v in values),
                    'generated_face_failures': sum(not v['generated_face_detected'] for v in values)}
            for key in ['pck_002_diagonal', 'pck_005_diagonal', 'pck_010_diagonal', 'nme_diagonal', 'identity_a_cosine']:
                valid = [v[key] for v in values if v[key] is not None]
                item[key] = float(np.mean(valid)) if valid else None
                item[key + '_n'] = len(valid)
            summary.append(item)
        write_jsonl(out / 'per_image.jsonl', results)
        write_json(out / 'metrics.json', {'inference': str(inference), 'summary': summary,
                   'pose_protocol': 'COCO17 joints mapped from DWPose134; image-diagonal normalization; detected score>0.3; target-valid denominator includes missed joints',
                   'identity_protocol': 'existing CPU SCRFD/glintr100, largest detected face; source is native A RGB; failures recorded, not imputed',
                   'blank_pose_protocol': 'PCK against original B is diagnostic, not a matched conditioning quality score',
                   'test_used': False})
        panels(config, rows, out)
    return out


def panels(config, rows, out):
    normal = defaultdict(dict)
    for row in rows:
        if row['intervention'] == 'normal':
            normal[row['case']][row['method']] = row
    size = tuple(config['canvas_wh'])
    overview = []
    for case, methods in normal.items():
        if 'baseline' not in methods or 'lora' not in methods:
            continue
        row = methods['baseline']['original_row']
        paths = [*row['condition_images'], row['target_image'], methods['baseline']['image'], methods['lora']['image']]
        panel = Image.new('RGB', (size[0] * 5, size[1] + 32), 'white')
        draw = ImageDraw.Draw(panel)
        for i, (path, label) in enumerate(zip(paths, ['A', 'B DWPose', 'B RGB (GT)', 'Base Qwen', 'Edit LoRA'])):
            with Image.open(path) as image:
                panel.paste(image.convert('RGB'), (i * size[0], 32))
            draw.text((i * size[0] + 5, 8), label, fill='black')
        panel.save(out / f'{case}_{row["split"]}_comparison.png')
        if row['split'] == 'val' and len(overview) < 4:
            overview.append(panel.resize((1280, round(panel.height * 1280 / panel.width))))
    if overview:
        combined = Image.new('RGB', (1280, sum(i.height for i in overview)), 'white')
        offset = 0
        for image in overview:
            combined.paste(image, (0, offset))
            offset += image.height
        combined.save(out / 'validation_overview.png')
