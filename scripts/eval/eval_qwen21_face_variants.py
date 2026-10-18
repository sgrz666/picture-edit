#!/usr/bin/env python3
"""Measure saved Qwen21 Face variants on the same pairs and valid face subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

from eval_iper_four_arm import IdentityMetric, PoseMetric, image_metrics, open_rgb


ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')
DEFAULT_MODES = ('body', 'body_crop_same', 'body_crop_role', 'body_face', 'body_face_crop', 'body_face_gated')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--inventory', type=Path, default=FACE_ROOT / 'face_val_inventory.jsonl')
    parser.add_argument('--features', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--face-train-pairs', type=Path, default=FACE_ROOT / 'face_train_pairs.jsonl')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--image-roots', nargs='+', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=list(DEFAULT_MODES))
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'variant_eval')
    return parser.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_image(roots, stem, mode, fallback):
    for root in roots:
        aliases = (mode, 'off') if mode == 'native' else ((mode, 'on') if mode == 'body' else (mode,))
        for alias in aliases:
            image = root / 'images' / f'{stem}_{alias}.png'
            if image.exists():
                return image
    if fallback and mode in ('body_crop_same', 'body_crop_role', 'body_face_crop'):
        return find_image(roots, stem, 'body_face' if mode == 'body_face_crop' else 'body', False)
    raise FileNotFoundError(f'{stem}_{mode}.png')


def runtime_rows(roots):
    indexed = {}
    for root in roots:
        body_rows = root / 'per_sample.json'
        if body_rows.exists():
            for row in json.loads(body_rows.read_text()):
                for mode, key in (('native', 'off'), ('body', 'on')):
                    if row.get('seconds', {}).get(key) is not None:
                        indexed[row['image_stem'], mode] = {
                            'seconds': row['seconds'][key],
                            'peak_cuda_gib': row['peak_cuda_gib'][key],
                        }
        path = root / 'per_sample.jsonl'
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for mode, report in (row.get('results') or row.get('modes') or {}).items():
                indexed[row['stem'], mode] = report
    return indexed


def summary_mean(rows, mode, key, *, usable=False, jointly=False, strict=False):
    values = []
    for row in rows:
        if usable and not row['source_face_usable']:
            continue
        if strict and not row['strict_reference_usable']:
            continue
        value = row['variants'][mode].get(key)
        if value is None or not np.isfinite(value):
            continue
        if jointly and any(row['variants'][other].get(key) is None or
                           (strict and not row['variants'][other]['generated_face_reliable'])
                           for other in row['variants']):
            continue
        if strict and not row['variants'][mode]['generated_face_reliable']:
            continue
        values.append(float(value))
    return {'mean': float(np.mean(values)) if values else None, 'n': len(values)}


def main():
    args = arguments()
    pairs, inventory = read_jsonl(args.pairs), read_jsonl(args.inventory)
    train_person_ids = {row['appearance'].split('_')[0] for row in read_jsonl(args.face_train_pairs)}
    if args.max_samples:
        pairs, inventory = pairs[:args.max_samples], inventory[:args.max_samples]
    if len(pairs) != len(inventory):
        raise ValueError('pair and inventory counts disagree')
    pose = PoseMetric()
    identity = IdentityMetric()
    sampled = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'
    assets = ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast'
    conditions = {}
    feature_apps = {}
    timings = runtime_rows(args.image_roots)
    results = []
    for index, (pair, record) in enumerate(zip(pairs, inventory)):
        app = pair['appearance']
        source_meta, target_meta = pair['source'], pair['target']
        stem = f"{index:03d}_{app}_{source_meta['stem']}_{target_meta['stem']}"
        if (record['appearance'], record['source_stem'], record['target_stem']) != (app, source_meta['stem'], target_meta['stem']):
            raise AssertionError(f'inventory mismatch at {index}')
        source = open_rgb(sampled / app / source_meta['rgb_1024'])
        target = open_rgb(sampled / app / target_meta['rgb_1024'])
        source_embedding = identity.embedding(source)
        target_embedding = identity.embedding(target)
        source_mask = np.asarray(Image.open(assets / app / 'mask_512' / source_meta['role'] / f"{source_meta['stem']}.png").convert('L')) / 255
        target_mask = np.asarray(Image.open(assets / app / 'mask_512' / target_meta['role'] / f"{target_meta['stem']}.png").convert('L')) / 255
        if app not in conditions:
            conditions[app] = torch.load(assets / app / 'v6_conditions.pt', map_location='cpu', weights_only=False)
        if app not in feature_apps:
            feature_apps[app] = torch.load(args.features / f'{app}.pt', map_location='cpu', weights_only=False)
        feature = feature_apps[app]
        source_feature = feature['source'][source_meta['stem']]
        target_visibility = feature['target_visibility'][target_meta['stem']]['visible']
        target_box = feature['target_geometry'][target_meta['stem']]['box']
        target_points = conditions[app]['kps_25_coords'][conditions[app]['stem_to_idx'][target_meta['stem']]].numpy()
        row = {'pair_idx': index, 'stem': stem, 'appearance': app, 'source_face_usable': record['face_usable'],
               'person_seen_in_face_training': app.split('_')[0] in train_person_ids,
               'source_face_side_original': (source_feature['detection']['side']
                                            if source_feature['detection'] else None),
               'target_face_visible': bool(target_visibility),
               'strict_reference_usable': bool(source_feature['valid'] and target_visibility),
               'source_target_identity_cosine': identity.cosine(source_embedding, target_embedding), 'variants': {}}
        for mode in args.modes:
            path = find_image(args.image_roots, stem, mode, not record['face_usable'])
            generated = open_rgb(path)
            measurements = image_metrics(generated, target, source, target_mask, source_mask)
            measurements.update(pose.compare(pose.detect(generated), target_points))
            generated_embedding = identity.embedding(generated)
            boxes, _ = identity.det.detect(cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2BGR))
            if len(boxes):
                face_index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
                face_box = boxes[face_index]
                face_side = float(min(face_box[2] - face_box[0], face_box[3] - face_box[1]))
                face_score = float(face_box[4])
                cx = float((face_box[0] + face_box[2]) / 1024)
                cy = float((face_box[1] + face_box[3]) / 1024)
                x0, y0, x1, y1 = target_box
                face_in_target_head = x0 - .08 <= cx <= x1 + .08 and y0 - .08 <= cy <= y1 + .08
            else:
                face_box, face_side, face_score, face_in_target_head = None, 0., 0., False
            measurements['generated_face_box'] = [float(v) for v in face_box[:4]] if face_box is not None else None
            measurements['generated_face_side'] = face_side
            measurements['generated_face_score'] = face_score
            measurements['generated_face_in_target_head'] = bool(face_in_target_head)
            measurements['generated_face_reliable'] = bool(face_side >= 16 and face_score >= .5 and face_in_target_head)
            measurements['identity_cosine'] = identity.cosine(source_embedding, generated_embedding)
            measurements['identity_target_cosine'] = identity.cosine(target_embedding, generated_embedding)
            measurements['image_path'] = str(path)
            timing = timings.get((stem, mode), {})
            measurements['inference_seconds'] = timing.get('seconds')
            measurements['peak_cuda_gib'] = timing.get('peak_cuda_gib')
            row['variants'][mode] = measurements
        results.append(row)
        print(f'{index + 1}/{len(pairs)} {stem}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'per_sample.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = ('identity_cosine', 'identity_target_cosine', 'pose_pck_002', 'pose_nme_diagonal',
               'background_mae_source', 'human_mae', 'psnr', 'ssim',
               'inference_seconds', 'peak_cuda_gib')
    summary = {'pairs': len(results), 'modes': list(args.modes), 'metrics': {}}
    for metric in metrics:
        summary['metrics'][metric] = {
            mode: {'all': summary_mean(results, mode, metric),
                   'source_face_usable': summary_mean(results, mode, metric, usable=True),
                   'jointly_scored': summary_mean(results, mode, metric, usable=True, jointly=True),
                   'strict_joint': summary_mean(results, mode, metric, strict=True, jointly=True)}
            for mode in args.modes
        }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
