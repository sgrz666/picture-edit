#!/usr/bin/env python3
"""Prepare source-only ArcFace/DINO caches for full Qwen TikTok training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_tiktok_qwen21_training import read_rows, write_rows, check_split_disjoint
from scripts.precompute_iper_face_features_v65 import (
    LocalFaceFeatureExtractor, _extractor_metadata, atomic_save_cache, compute_cache_fingerprint,
    compute_face_box, safe_expanded_crop, load_v65_face_cache,
)
from scripts.precompute_tiktok_face_features_v65 import project_tiktok_face72_to_rgb_normalized


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assemble_cache(stems, geometry, source_features, metadata):
    count = len(stems)
    metadata = dict(metadata, geometry_valid=[bool(item['geometry_valid']) for item in geometry])
    cache = {'metadata': metadata, 'frame_stems': stems,
             'stem_to_idx': {stem: i for i, stem in enumerate(stems)},
             'arcface': torch.zeros(count, 512, dtype=torch.float16),
             'dino_patches': torch.zeros(count, 256, 1536, dtype=torch.float16),
             'valid': torch.zeros(count, dtype=torch.bool),
             'geometry_valid': torch.tensor(metadata['geometry_valid'], dtype=torch.bool)}
    for name in ('face_landmarks', 'jaw_pose', 'expression', 'face_boxes'):
        cache[name] = torch.stack([item[name] for item in geometry]).float()
    for i, stem in enumerate(stems):
        if stem in source_features:
            cache['arcface'][i], cache['dino_patches'][i] = (value.half() for value in source_features[stem])
            cache['valid'][i] = True
    metadata['cache_fingerprint'] = compute_cache_fingerprint(cache)
    return cache


def extract_source(extractor, rgb, geometry, *, min_score, min_side):
    """Recognition/texture only from source RGB; the target image is never supplied."""
    detector = extractor._arcface.app.det_model
    boxes, five_points = detector.detect(np.ascontiguousarray(rgb[..., ::-1]), max_num=0)
    if not len(boxes):
        return None, {'reason': 'no_source_face'}
    index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    box = boxes[index]
    side = float(min(box[2] - box[0], box[3] - box[1]))
    record = {'score': float(box[4]), 'short_side_native_px': side}
    if not geometry['geometry_valid'] or box[4] < min_score or side < min_side:
        return None, dict(record, reason='source_face_ineligible')
    tight = safe_expanded_crop(rgb, geometry['face_boxes'], scale=1.0)
    expanded = safe_expanded_crop(rgb, geometry['face_boxes'], scale=1.35)
    try:
        arc, dino = extractor.extract(tight, expanded)
        record['alignment'] = 'legacy_tight_crop_detection'
    except RuntimeError as error:
        if 'InsightFace found no face in the tight crop' not in str(error):
            raise
        from insightface.utils import face_align
        aligned = face_align.norm_crop(np.ascontiguousarray(rgb[..., ::-1]), five_points[index], image_size=112)
        values = extractor._arcface.app.models['recognition'].get_feat(aligned)
        arc = F.normalize(torch.as_tensor(values, dtype=torch.float32).reshape(-1), dim=0)
        with torch.inference_mode():
            inputs = extractor._processor(images=Image.fromarray(expanded), return_tensors='pt')
            tokens = extractor._dino(**{k: v.to(extractor._device) for k, v in inputs.items()}).last_hidden_state
            if tuple(tokens.shape[1:]) != (257, 1536):
                raise ValueError(f'Unexpected DINO sequence shape: {tuple(tokens.shape)}')
            dino = tokens[0, 1:].float().cpu()
        record['alignment'] = 'original_source_five_point_112'
    if arc.shape != (512,) or dino.shape != (256, 1536):
        raise ValueError('Full Face requires ArcFace[512] and DINO[256,1536]')
    if not torch.isfinite(arc).all() or not torch.isfinite(dino).all():
        raise ValueError('Nonfinite source face features')
    return (arc.cpu(), dino.cpu()), dict(record, reason='accepted')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', type=Path, required=True)
    parser.add_argument('--val-pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--min-source-side', type=float, default=36.)
    parser.add_argument('--min-source-score', type=float, default=.5)
    parser.add_argument('--min-target-output-side', type=float, default=32.)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    splits = {'train': read_rows(args.train_pairs), 'val': read_rows(args.val_pairs)}
    check_split_disjoint(splits)
    if any(not rows for rows in splits.values()):
        raise ValueError('Face train/val manifests must be nonempty')
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    request['pair_sha256'] = {split: digest(path) for split, path in (('train', args.train_pairs), ('val', args.val_pairs))}
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / 'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Existing Face preparation configuration differs; use a new output directory')
    request_path.write_text(json.dumps(request, indent=2))
    extractor = LocalFaceFeatureExtractor(insightface_root=ROOT / 'models/antelopev2',
                                         dino_model=ROOT / 'models/dinov2-giant', device='cpu')
    extractor._device = torch.device(args.device)
    extractor._dino.to(extractor._device)
    model_metadata = _extractor_metadata(extractor)
    by_sequence = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            by_sequence[row['sequence_id']].append((split, row))
    kept = {'train': [], 'val': []}
    exclusions, source_audit, cache_hashes = [], [], {}
    for sequence, pairs in sorted(by_sequence.items()):
        stems = sorted({row[side + '_stem'] for _, row in pairs for side in ('source', 'target')})
        sources = {row['source_stem']: row['source_rgb'] for _, row in pairs}
        cache_path = args.output / 'face_cache' / f'{sequence}.pt'
        params_path = args.assets_root / sequence / 'smplestx/params.pt'
        map_path = args.assets_root / sequence / 'smplestx/frame_map.json'
        source_hashes = {'params': digest(params_path), 'frame_map': digest(map_path),
                         **{stem: digest(path) for stem, path in sources.items()}}
        if cache_path.exists():
            cache = load_v65_face_cache(cache_path)
            if cache['frame_stems'] != stems or cache['metadata']['source_hashes'] != source_hashes:
                raise ValueError(f'Cached Face inputs changed: {sequence}')
        else:
            params = torch.load(params_path, map_location='cpu', weights_only=False, mmap=True)
            frames = json.loads(map_path.read_text())['frames']
            frame_stems = [str(frame.get('stem', f"{frame['index'] + 1:04d}")) for frame in frames]
            indices = {stem: i for i, stem in enumerate(frame_stems)}
            projected = project_tiktok_face72_to_rgb_normalized(params, raw_seq_dir=args.raw_root / sequence, stems=frame_stems)
            geometry = []
            for stem in stems:
                index = indices[stem]
                landmarks = projected[index].float()
                box, valid = compute_face_box(landmarks)
                jaw, expression = params['jaw_pose'][index].float(), params['expression'][index].float()
                valid = valid and bool(torch.isfinite(jaw).all() and torch.isfinite(expression).all())
                geometry.append(dict(face_landmarks=landmarks, face_boxes=box, jaw_pose=jaw,
                                     expression=expression, geometry_valid=valid))
            features, records = {}, []
            for stem, path in sorted(sources.items()):
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
                values, record = extract_source(extractor, rgb, geometry[stems.index(stem)],
                                                min_score=args.min_source_score, min_side=args.min_source_side)
                if values is not None:
                    features[stem] = values
                records.append(dict(sequence_id=sequence, source_stem=stem, **record))
            source_audit.extend(records)
            metadata = dict(schema_version=1, appearance=sequence, **model_metadata, source_hashes=source_hashes,
                            extraction_failures=[record for record in records if record['reason'] != 'accepted'],
                            identity_policy='Source only; target-only identity/texture are exact zero',
                            preprocess=dict(face_box_expansion=.15, dino_crop_scale=1.35, dino_grid=[16, 16],
                                            align_mode='resize', per_source=records))
            cache = assemble_cache(stems, geometry, features, metadata)
            atomic_save_cache(cache, cache_path)
            del params, projected
        cache_hashes[sequence] = cache['metadata']['cache_fingerprint']
        for split, row in pairs:
            si, ti = (cache['stem_to_idx'][row[side + '_stem']] for side in ('source', 'target'))
            box = cache['face_boxes'][ti]
            target_side = float(torch.minimum(box[2] - box[0], box[3] - box[1]) * args.resolution)
            if cache['valid'][si] and cache['geometry_valid'][ti] and target_side >= args.min_target_output_side:
                kept[split].append(row)
            else:
                exclusions.append(dict(split=split, pair_id=row['pair_id'], sequence_id=sequence,
                                       reason='invalid_source_or_target_face_geometry_or_size'))
        print(f'{sequence}: sources={len(sources)} valid={int(cache["valid"].sum())} frames={len(stems)}', flush=True)
    for split, rows in kept.items():
        write_rows(args.output / f'{split}_pairs.jsonl', rows)
    write_rows(args.output / 'exclusions.jsonl', exclusions)
    write_rows(args.output / 'source_audit.jsonl', source_audit)
    report = dict(architecture='qwen21_full_v1', counts={split: len(rows) for split, rows in kept.items()},
                  excluded_pairs=len(exclusions), source_identity_only=True, test_used=False,
                  cache_fingerprints=cache_hashes, request=request)
    (args.output / 'data_manifest.json').write_text(json.dumps(report, indent=2))
    if any(not rows for rows in kept.values()):
        raise RuntimeError('No eligible Face pairs in a split; inspect exclusions.jsonl')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
