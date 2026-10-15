#!/usr/bin/env python3
"""Prepare real SMPL-X geometry and source-only identity; never synthesize valid caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.data_runtime import pair_metadata
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json, sha256_file
from scripts.prepare_tiktok_qwen21_face import assemble_cache, extract_source
from scripts.precompute_iper_face_features_v65 import (
    LocalFaceFeatureExtractor, _extractor_metadata, atomic_save_cache,
    compute_face_box, load_v65_face_cache,
)
from scripts.precompute_tiktok_face_features_v65 import project_tiktok_face72_to_rgb_normalized

DEFAULT_SOURCE = Path('/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/data/face_cache')
DEFAULT_ASSETS = ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native'
DEFAULT_RAW = ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset'
SPLIT_FILES = {'train': 'train512_pairs.jsonl', 'val': 'val24_pairs.jsonl',
               'test': 'test24_pairs.jsonl', 'diagnostic': 'diagnostic6_pairs.jsonl'}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing split manifest: {path}')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def requested_frames(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = defaultdict(lambda: {'stems': set(), 'sources': {}})
    for row in rows:
        meta = pair_metadata(row)
        for side in ('source', 'target'):
            seq, stem = meta[f'{side}_appearance'], meta[f'{side}_stem']
            groups[seq]['stems'].add(stem)
            if side == 'source':
                old, path = groups[seq]['sources'].get(stem), meta['source_rgb']
                if old is not None and old != path:
                    raise ValueError('Source frame resolves to two different files')
                groups[seq]['sources'][stem] = path
    return groups


def validate_real_cache(cache):
    """A fingerprint checks bytes; also reject previously fabricated valid geometry."""
    if 'geometry_valid' not in cache or cache['geometry_valid'].tolist() != cache['metadata'].get('geometry_valid'):
        raise ValueError('Full Face requires fingerprinted geometry_valid')
    for i, valid in enumerate(cache['geometry_valid'].tolist()):
        if not valid:
            continue
        points, box = cache['face_landmarks'][i], cache['face_boxes'][i]
        if not torch.isfinite(points).all() or not torch.isfinite(box).all():
            raise ValueError('Nonfinite valid face geometry')
        if (points[:, :2].amax(0) - points[:, :2].amin(0)).min() <= 0:
            raise ValueError('Degenerate landmarks marked valid; regenerate from SMPL-X')
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Degenerate face box marked valid')
    for i, valid in enumerate(cache['valid'].tolist()):
        if valid and (cache['arcface'][i].float().norm() <= 0 or not torch.isfinite(cache['dino_patches'][i]).all()):
            raise ValueError('Invalid identity marked valid')


def prepare_study_face_cache(run_dir: Path, splits: list[str],
                             source_features_dir: Path | None = None,
                             assets_root: Path | None = None, raw_root: Path | None = None,
                             device: str = 'cpu') -> dict[str, Any]:
    data_dir = Path(run_dir) / 'data'
    assets_root, raw_root = Path(assets_root or DEFAULT_ASSETS), Path(raw_root or DEFAULT_RAW)
    source_features_dir = Path(source_features_dir or DEFAULT_SOURCE)
    rows = []
    for split in splits:
        if split not in SPLIT_FILES:
            raise ValueError(f'Unknown split: {split}')
        rows.extend(read_jsonl(data_dir / SPLIT_FILES[split]))
    if not rows:
        raise ValueError('No requested Face frames')
    groups = requested_frames(rows)
    face_dir, extractor = data_dir / 'face_cache', None
    records, audit = [], []
    for seq, info in sorted(groups.items()):
        params_path = assets_root / seq / 'smplestx/params.pt'
        frame_map_path = assets_root / seq / 'smplestx/frame_map.json'
        # Assert real inputs before writing anything for this appearance.
        hashes = {'params': sha256_file(params_path), 'frame_map': sha256_file(frame_map_path)}
        hashes.update({stem: sha256_file(path) for stem, path in sorted(info['sources'].items())})
        stems, dst = sorted(info['stems']), face_dir / f'{seq}.pt'
        if dst.is_file():
            cache = load_v65_face_cache(dst)
            validate_real_cache(cache)
            if cache['frame_stems'] != stems or cache['metadata'].get('source_hashes') != hashes:
                raise ValueError(f'Face inputs changed for {seq}; preserve old cache and use a new directory')
        else:
            params = torch.load(params_path, map_location='cpu', weights_only=False, mmap=True)
            frames = json.loads(frame_map_path.read_text())['frames']
            full_stems = [str(f.get('stem', f"{f['index'] + 1:04d}")) for f in frames]
            indices = {stem: i for i, stem in enumerate(full_stems)}
            if set(stems) - set(indices):
                raise ValueError(f'Missing SMPL-X frames for {seq}: {sorted(set(stems) - set(indices))}')
            projected = project_tiktok_face72_to_rgb_normalized(params, raw_seq_dir=raw_root / seq, stems=full_stems)
            geometry = []
            for stem in stems:
                index = indices[stem]
                points = projected[index].float()
                box, valid = compute_face_box(points)
                jaw, expression = params['jaw_pose'][index].float(), params['expression'][index].float()
                valid = valid and bool(torch.isfinite(jaw).all() and torch.isfinite(expression).all())
                geometry.append(dict(face_landmarks=points, face_boxes=box, jaw_pose=jaw,
                                     expression=expression, geometry_valid=valid))
            old = None
            old_path = source_features_dir / f'{seq}.pt'
            if old_path.is_file():
                old = load_v65_face_cache(old_path)
                validate_real_cache(old)
            features, source_records = {}, []
            model_metadata = {k: v for k, v in old['metadata'].items() if k.startswith(('arcface_', 'dino_'))} if old else {}
            old_hash = sha256_file(old_path) if old else None
            for stem, path in sorted(info['sources'].items()):
                old_index = old['stem_to_idx'].get(stem) if old else None
                if (old_index is not None and old['valid'][old_index]
                        and old['metadata'].get('source_hashes', {}).get(stem) == hashes[stem]):
                    features[stem] = old['arcface'][old_index].float(), old['dino_patches'][old_index].float()
                    source_records.append(dict(stem=stem, reason='reused_verified_source', original_cache_sha256=old_hash))
                    continue
                if extractor is None:
                    extractor = LocalFaceFeatureExtractor(insightface_root=ROOT / 'models/antelopev2',
                                                          dino_model=ROOT / 'models/dinov2-giant', device='cpu')
                    extractor._device = torch.device(device)
                    extractor._dino.to(extractor._device)
                actual_models = _extractor_metadata(extractor)
                if model_metadata and model_metadata != actual_models:
                    raise ValueError('Feature extractor weights differ from reusable source cache')
                model_metadata = actual_models
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
                values, result = extract_source(extractor, rgb, geometry[stems.index(stem)], min_score=.5, min_side=36.)
                if values is not None:
                    features[stem] = values
                source_records.append(dict(stem=stem, **result))
            metadata = dict(schema_version=1, appearance=seq, **model_metadata, source_hashes=hashes,
                            geometry_source='SMPLest-X projection into original normalized RGB coordinates',
                            coordinate_space='original_normalized', runtime_preprocessing='native_v1',
                            identity_policy='A sources only; target-only identity and texture are exact zero',
                            preprocess=dict(face_box_expansion=.15, dino_crop_scale=1.35, dino_grid=[16, 16],
                                            align_mode='native recognition crops', per_source=source_records))
            cache = assemble_cache(stems, geometry, features, metadata)
            validate_real_cache(cache)
            atomic_save_cache(cache, dst)
            audit.extend(dict(sequence_id=seq, **r) for r in source_records)
            del params, projected
        records.append(dict(sequence_id=seq, path=str(dst), sha256=sha256_file(dst),
                            cache_fingerprint=cache['metadata']['cache_fingerprint'],
                            requested_stems=stems, source_stems=sorted(info['sources']),
                            valid_sources=int(cache['valid'].sum()), valid_geometry=int(cache['geometry_valid'].sum())))
        print(f"Face {seq}: sources={len(info['sources'])}, valid={int(cache['valid'].sum())}, frames={len(stems)}", flush=True)
    manifest = dict(schema='qwen21_study_face_cache_v2', sequences=records,
                    total_sequences=len(records), source_identity_only=True, source_audit=audit)
    atomic_json(data_dir / 'face_assets.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=list(SPLIT_FILES))
    parser.add_argument('--source-features', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--assets-root', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-supervision', action='store_true', help='Prepare conditioning only')
    args = parser.parse_args()
    result = prepare_study_face_cache(args.run, args.splits, args.source_features,
                                     args.assets_root, args.raw_root, args.device)
    if not args.skip_supervision and 'train' in args.splits:
        from src.pose_control.qwen21_full.face_supervision import build_cache
        build_cache([args.run / 'data/train512_pairs.jsonl'], args.run / 'data/face_cache',
                    ROOT / 'models/antelopev2/scrfd_10g_bnkps.onnx',
                    args.run / 'data/face_supervision_native.json', align_mode='native')
    print(json.dumps(result, indent=2))


def validate_face_assets(run_dir: str | Path) -> None:
    data = Path(run_dir) / 'data'
    manifest = json.loads((data / 'face_assets.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'qwen21_study_face_cache_v2' or manifest.get('source_identity_only') is not True:
        raise ValueError('Missing real source-only Face cache provenance')
    rows = [r for name in SPLIT_FILES.values() for r in read_jsonl(data / name)]
    groups = requested_frames(rows)
    by_sequence = {r['sequence_id']: r for r in manifest['sequences']}
    if len(by_sequence) != manifest['total_sequences'] or set(by_sequence) != set(groups):
        raise ValueError('Face cache manifest does not cover all locked appearances')
    for seq, request in groups.items():
        entry = by_sequence[seq]
        path = data / 'face_cache' / f'{seq}.pt'
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'Changed Face cache bytes: {seq}')
        cache = load_v65_face_cache(path)
        validate_real_cache(cache)
        if (cache['frame_stems'] != sorted(request['stems'])
                or cache['metadata']['cache_fingerprint'] != entry['cache_fingerprint']):
            raise ValueError(f'Face cache frame/fingerprint differs: {seq}')
        for stem, index in cache['stem_to_idx'].items():
            if stem not in request['sources']:
                if cache['valid'][index] or cache['arcface'][index].any() or cache['dino_patches'][index].any():
                    raise ValueError(f'Target-only recognition features forbidden: {seq}/{stem}')
            elif sha256_file(request['sources'][stem]) != cache['metadata']['source_hashes'][stem]:
                raise ValueError(f'Source RGB changed: {seq}/{stem}')


if __name__ == '__main__':
    main()

