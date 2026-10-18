#!/usr/bin/env python3
"""Balanced, matched-prompt BFS experiment on sg; no training or production edits."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_compare_30_20261004')
VARIANTS = ('body', 'nolora', 'bfs', 'alternative')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_face/face_val_pairs.jsonl'))
    p.add_argument('--output', type=Path, default=RUN_ROOT)
    p.add_argument('--sampled-root', type=Path, default=ROOT / 'datasets/iPER/iper_sampled_6src64tgt')
    p.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast')
    p.add_argument('--body-checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    p.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    p.add_argument('--bfs-dir', type=Path, default=Path('/home/shangguanrz/models/bfs'))
    p.add_argument('--per-identity', type=int, default=3)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage2-resolution', type=int, default=768)
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()


def balanced_pairs(rows, per_identity):
    if per_identity < 1:
        raise ValueError('per-identity must be positive')
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['appearance']].append((index, row))
    selected = []
    for identity in sorted(groups):
        group = sorted(groups[identity], key=lambda item: (item[1]['target']['frame_index0'], item[0]))
        count = min(per_identity, len(group))
        indices = [len(group) // 2] if count == 1 else [round(i * (len(group) - 1) / (count - 1)) for i in range(count)]
        selected.extend(group[i] for i in indices)
    return selected


def normalized_detection(detected, width, height):
    """The legacy comparator expects pixels on a 512 x 512 canvas."""
    if detected is None:
        return None
    points, scores = detected
    points = points.copy()
    points[:, 0] *= 512.0 / width
    points[:, 1] *= 512.0 / height
    return points, scores


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def self_test():
    import numpy as np
    rows = [{'appearance': person, 'target': {'frame_index0': n}} for person in ('b', 'a') for n in (40, 0, 20, 10, 30)]
    chosen = balanced_pairs(rows, 3)
    assert [r['target']['frame_index0'] for _, r in chosen] == [0, 20, 40, 0, 20, 40]
    assert len({i for i, _ in chosen}) == 6
    points = np.array([[384., 256.], [768., 512.]], dtype=np.float32)
    scores = np.ones(2)
    actual, _ = normalized_detection((points, scores), 768, 512)
    np.testing.assert_allclose(actual, [[256., 256.], [512., 512.]])
    np.testing.assert_allclose(points, [[384., 256.], [768., 512.]])
    assert normalized_detection(None, 768, 768) is None
    print('PASS: balanced frame selection; rectangular coordinate scaling; inputs preserved', flush=True)


def face_detection(identity, image):
    import cv2
    import numpy as np
    boxes, _ = identity.det.detect(cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR))
    if len(boxes) == 0:
        return None, 0
    index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    return [float(x) for x in boxes[index, :4]], len(boxes)


def verify_loaded_lora(pipe, path, name):
    import torch
    from bfs_explore import convert_bfs_state_dict
    state = convert_bfs_state_dict(path, gate_first=True)
    pipe.load_lora_weights(state, adapter_name=name)
    pipe.set_adapters([name], adapter_weights=[1.0])
    targets, scales = set(), set()
    dtype_cast_tensors = 0
    for key, expected in state.items():
        module_name, suffix = key.removeprefix('transformer.').split('.lora_')
        part = suffix.split('.')[0]
        module = pipe.transformer.get_submodule(module_name)
        live = getattr(module, 'lora_' + part)[name].weight.detach().cpu()
        if live.shape != expected.shape or not torch.isfinite(live).all():
            raise RuntimeError(f'Invalid loaded tensor: {key}')
        if not torch.equal(live, expected.to(live.dtype)):
            if not torch.equal(live, expected.to(pipe.transformer.dtype).to(live.dtype)):
                raise RuntimeError(f'Loaded tensor differs from file: {key}')
            dtype_cast_tensors += 1
        targets.add(module_name)
        scales.add(float(module.scaling[name]))
    if scales != {1.0}:
        raise RuntimeError(f'Unexpected LoRA scales: {scales}')
    return {'tensors_checked': len(state), 'modules_checked': len(targets), 'all_values_match_after_dtype_conversion': True,
            'tensors_cast_through_base_dtype': dtype_cast_tensors,
            'gate_first': True, 'strength': 1.0, 'module_scales': sorted(scales)}


def generate(pipe, prompt, images, size, args):
    import torch
    with torch.inference_mode():
        result = pipe(prompt=prompt, image=images, width=size, height=size, output_resolution=size,
                      num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=True,
                      generator=torch.Generator(device='cuda').manual_seed(args.seed))
    return result.images[0]


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    if args.stage2_resolution < 512 or args.stage2_resolution % 32:
        raise ValueError('stage2-resolution must be >=512 and divisible by 32')
    import numpy as np
    import torch
    from PIL import Image
    from bfs_explore import BFS_HEAD_PROMPT, face_crop, load_trusted_state
    from eval_iper_four_arm import IdentityMetric, PoseMetric
    from src.data.tiktok_dataset import load_pose_dataset
    from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
    from src.pose_control.qwen21.runtime import PROMPT, dataset_image_to_pil

    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    chosen = balanced_pairs(rows, args.per_identity)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'images').mkdir(exist_ok=True)
    selected_path = args.output / 'selected_pairs.jsonl'
    selected_text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for _, row in chosen)
    weights = {'bfs': args.bfs_dir / 'bfs_head_v1.1_qwen_2.1.safetensors',
               'alternative': args.bfs_dir / 'bfs_head_v1.1_alternative_qwen_2.1.safetensors'}
    data_paths = set()
    for _, row in chosen:
        data_paths.add(args.sampled_root / row['appearance'] / row['source']['rgb_1024'])
        data_paths.add(args.assets_root / row['appearance'] / 'v6_conditions.pt')
        data_paths.update(args.assets_root / row['appearance'] / folder / row['target']['role'] / (row['target']['stem'] + '.png')
                          for folder in ('normal_512', 'mask_512', 'part_512'))
    config = {'pairs': str(args.pairs), 'pairs_sha256': sha256(args.pairs), 'selection': [i for i, _ in chosen],
              'selected_pairs_sha256': hashlib.sha256(selected_text.encode()).hexdigest(),
              'appearance_counts': dict(collections.Counter(row['appearance'] for _, row in chosen)),
              'actor_counts': dict(collections.Counter(row['appearance'].split('_')[0] for _, row in chosen)),
              'sampled_root': str(args.sampled_root.resolve()), 'assets_root': str(args.assets_root.resolve()),
              'input_fingerprints': {str(path): sha256(path) if path.is_file() else None for path in sorted(data_paths)},
              'selection_rule': 'per identity, chronological first/middle/last target frames; no outcome filtering',
              'stage1': {'body_checkpoint': str(args.body_checkpoint), 'checkpoint_sha256': sha256(args.body_checkpoint),
                         'resolution': 512, 'face_adapter': False, 'prompt': PROMPT},
              'stage2': {'resolution': args.stage2_resolution, 'prompt': BFS_HEAD_PROMPT, 'body_adapter': False,
                         'face_adapter': False, 'inputs': ['whole generated C at 512', 'source A face crop at 512'],
                         'weights': {v: {'path': str(p), 'sha256': sha256(p)} for v, p in weights.items()}, 'strength': 1.0},
              'seed': args.seed, 'steps': args.steps, 'true_cfg_scale': 1.0, 'use_kv_cache': True,
              'model': str(args.model), 'model_config_sha256': sha256(args.model / 'transformer/config.json'),
              'runner_sha256': sha256(__file__), 'legacy_helper_sha256': sha256(ROOT / 'bfs_explore.py')}
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing run config differs; choose a new output directory')
    write_json(config_path, config)
    selected_path.write_text(selected_text, encoding='utf-8')
    print(f'SELECTION n={len(chosen)} actors={len(config["actor_counts"])} appearances={len(config["appearance_counts"])}', flush=True)
    dataset = load_pose_dataset(selected_path, dataset_type='iper', sampled_root=args.sampled_root,
                                assets_root=args.assets_root, resolution=512, align_mode='square_crop', augment=False)
    if len(dataset) != len(chosen):
        raise RuntimeError('Dataset filtered or reordered selected samples')
    for index, (_, row) in enumerate(chosen):
        meta = dataset.samples[index]
        if (meta['appearance'], meta['source']['stem'], meta['target']['stem']) != (row['appearance'], row['source']['stem'], row['target']['stem']):
            raise RuntimeError('Dataset order differs from fixed selection')
    identity = IdentityMetric()
    records = []
    record_path = args.output / 'records.json'
    prior = {r['stem']: r for r in json.loads(record_path.read_text())} if record_path.exists() else {}
    for i, (_, row) in enumerate(chosen):
        stem = f'{i:03d}_{row["appearance"]}_{row["source"]["stem"]}_{row["target"]["stem"]}'
        source_path = args.sampled_root / row['appearance'] / row['source']['rgb_1024']
        source = Image.open(source_path).convert('RGB')
        bbox, faces = face_detection(identity, source)
        record = prior.get(stem, {'stem': stem, 'appearance': row['appearance'], 'original_pair_index': chosen[i][0],
                                  'source_path': str(source_path), 'target_path': str(args.sampled_root / row['appearance'] / row['target']['rgb_1024']),
                                  'target_stem': row['target']['stem'], 'seconds': {}})
        record.update({'source_bbox': bbox, 'source_faces': faces, 'source_face_valid': bbox is not None and faces == 1,
                       'source_face_short_side': min(bbox[2]-bbox[0], bbox[3]-bbox[1]) if bbox else None})
        if record['source_face_valid']:
            face_crop(source_path, bbox).save(args.output / 'images' / f'{stem}_reference.png')
        else:
            record['stage2_skip_reason'] = 'source face missing or ambiguous'
        records.append(record)
    write_json(record_path, records)

    if not args.score_only:
        from diffusers import QwenImage21Pipeline
        import diffusers, peft, transformers
        write_json(args.output / 'environment.json', {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                   'peft': peft.__version__, 'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0)})
        pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer.eval().requires_grad_(False)
        body = Qwen21BodyAdapter().to('cuda', dtype=torch.float32).eval().requires_grad_(False)
        body.load_state_dict(load_trusted_state(args.body_checkpoint)['adapter_state_dict'], strict=True)
        load_checks = {}
        for variant, path in weights.items():
            load_checks[variant] = verify_loaded_lora(pipe, path, 'bfs_compare')
            pipe.unload_lora_weights()
        write_json(args.output / 'lora_load_checks.json', load_checks)
        print(f'PREFLIGHT {load_checks}', flush=True)
        for i, record in enumerate(records):
            path = args.output / 'images' / f'{record["stem"]}_body_512.png'
            if not path.exists():
                sample = dataset[i]
                batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in sample.items()}
                with torch.inference_mode():
                    geometry = body.encode_geometry(batch, (32, 32))
                start = time.perf_counter()
                with Qwen21ControlContext(pipe.transformer, body, geometry):
                    image = generate(pipe, PROMPT, [dataset_image_to_pil(sample['src_image'])], 512, args)
                image.save(path)
                record['seconds']['body'] = time.perf_counter() - start
            Image.open(path).convert('RGB').resize((args.stage2_resolution,) * 2, Image.Resampling.LANCZOS).save(
                args.output / 'images' / f'{record["stem"]}_body.png')
            write_json(record_path, records)
            print(f'GEN body {i+1}/{len(records)} {record["stem"]}', flush=True)
        if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
            raise RuntimeError('Body control leaked into stage 2')
        for variant in ('nolora', 'bfs', 'alternative'):
            if variant != 'nolora':
                load_checks[variant] = verify_loaded_lora(pipe, weights[variant], 'bfs_compare')
                write_json(args.output / 'lora_load_checks.json', load_checks)
                print(f'LOAD {variant} {load_checks[variant]}', flush=True)
            for i, record in enumerate(records):
                if not record['source_face_valid']:
                    continue
                path = args.output / 'images' / f'{record["stem"]}_{variant}.png'
                if not path.exists():
                    images = [Image.open(args.output / 'images' / f'{record["stem"]}_body_512.png').convert('RGB'),
                              Image.open(args.output / 'images' / f'{record["stem"]}_reference.png').convert('RGB')]
                    start = time.perf_counter()
                    generate(pipe, BFS_HEAD_PROMPT, images, args.stage2_resolution, args).save(path)
                    record['seconds'][variant] = time.perf_counter() - start
                    write_json(record_path, records)
                print(f'GEN {variant} {i+1}/{len(records)} {record["stem"]}', flush=True)
            if variant != 'nolora':
                pipe.unload_lora_weights()
        del pipe, body
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    pose = PoseMetric()
    conditions = {}
    for i, record in enumerate(records):
        appearance = record['appearance']
        if appearance not in conditions:
            conditions[appearance] = load_trusted_state(args.assets_root / appearance / 'v6_conditions.pt')
        bundle = conditions[appearance]
        target_points = bundle['kps_25_coords'][bundle['stem_to_idx'][record['target_stem']]].numpy()
        source_embedding = identity.embedding(Image.open(record['source_path']).convert('RGB'))
        body_image = Image.open(args.output / 'images' / f'{record["stem"]}_body.png').convert('RGB')
        head_bbox, _ = face_detection(identity, body_image)
        record['body_head_bbox_for_eval'] = head_bbox
        outside = np.ones((body_image.height, body_image.width), dtype=bool)
        if head_bbox:
            x0, y0, x1, y1 = head_bbox
            cx, cy = (x0+x1)/2, (y0+y1)/2
            side = max(x1-x0, y1-y0) * 2.4
            outside[max(0, int(cy-side/2)):min(body_image.height, math.ceil(cy+side/2)),
                    max(0, int(cx-side/2)):min(body_image.width, math.ceil(cx+side/2))] = False
        record['metrics'] = {}
        for variant in VARIANTS:
            path = args.output / 'images' / f'{record["stem"]}_{variant}.png'
            if not path.exists():
                record['metrics'][variant] = {'status': 'missing', 'identity_A': None, 'face_detected': False}
                continue
            image = Image.open(path).convert('RGB')
            if image.size != body_image.size:
                raise RuntimeError(f'Unexpected scoring resolution: {path}')
            embedding = identity.embedding(image)
            metrics = {'status': 'complete', 'identity_A': identity.cosine(source_embedding, embedding),
                       'face_detected': embedding is not None, 'source_identity_valid': source_embedding is not None}
            metrics.update(pose.compare(normalized_detection(pose.detect(image), image.width, image.height), target_points))
            metrics['outside_head_mae_255'] = float(np.abs(np.asarray(image, dtype=np.float32)-np.asarray(body_image, dtype=np.float32))[outside].mean()) if head_bbox and outside.any() else None
            record['metrics'][variant] = metrics
        write_json(record_path, records)
        print(f'SCORE {i+1}/{len(records)} {record["stem"]}', flush=True)
    summarize(args.output, records)
    print('DONE', flush=True)


def summarize(output, records):
    import numpy as np
    summary = {}
    for variant in VARIANTS:
        metrics = [r['metrics'][variant] for r in records]
        summary[variant] = {'n_total': len(records), 'n_complete': sum(m['status'] == 'complete' for m in metrics),
                            'n_face_detected': sum(m['face_detected'] for m in metrics)}
        for key in ('identity_A', 'pose_pck_002', 'outside_head_mae_255'):
            values = [m[key] for m in metrics if m.get(key) is not None]
            summary[variant][key + '_mean'] = float(np.mean(values)) if values else None
            summary[variant][key + '_n'] = len(values)
    paired, by_identity = {}, {}
    for variant in ('bfs', 'alternative'):
        for baseline in ('nolora', 'body'):
            key = variant + '_minus_' + baseline
            groups = collections.defaultdict(list)
            for r in records:
                a, b = r['metrics'][variant].get('identity_A'), r['metrics'][baseline].get('identity_A')
                if a is not None and b is not None:
                    groups[r['appearance'].split('_')[0]].append(a-b)
            values = [x for xs in groups.values() for x in xs]
            means = np.array([np.mean(xs) for xs in groups.values()])
            rng = np.random.default_rng(42)
            bootstrap = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1) if len(means) else np.array([])
            paired[key] = {'n_pairs': len(values), 'n_identities': len(means), 'mean_delta': float(np.mean(values)) if values else None,
                           'wins': sum(x > 0 for x in values), 'losses': sum(x < 0 for x in values),
                           'equal_identity_mean_delta': float(means.mean()) if len(means) else None,
                           'identity_cluster_bootstrap_95_ci': [float(x) for x in np.quantile(bootstrap, [.025, .975])] if len(means) > 1 else None}
    for person in sorted({r['appearance'] for r in records}):
        by_identity[person] = {}
        for variant in VARIANTS:
            values = [r['metrics'][variant]['identity_A'] for r in records if r['appearance'] == person and r['metrics'][variant].get('identity_A') is not None]
            by_identity[person][variant] = {'n': len(values), 'mean': float(np.mean(values)) if values else None}
    write_json(output / 'results.json', {'summary': summary, 'paired_identity': paired, 'by_identity': by_identity,
               'limitations': ['single seed; 30 pairs from ten validation appearances', 'simplified Body first stage; Full Face not compared',
                               'body PCK is not a head-pose/expression metric', 'whole-image second stage; outside-head edits are possible',
                               'body comparator is a deterministic 512-to-768 upscale, not an extra diffusion pass']})
    flat = [{'stem': r['stem'], 'appearance': r['appearance'], 'variant': v, **r['metrics'][v]} for r in records for v in VARIANTS]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (output / 'per_pair.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
