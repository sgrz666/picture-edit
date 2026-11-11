"""Read-only consumption of original Bilibili assets; all new files stay in the run."""
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from core import (audit_splits, digest, letterbox, pose_points, read_json, read_jsonl,
                  render_skeleton, select_pilot, sha256, transform_points, write_json, write_jsonl)


def prepare_data(config, name):
    root = Path(config['workspace_root'])
    out = root / 'data' / name
    out.mkdir(parents=True, exist_ok=False)
    prepared = root / 'input' / name
    prepared.mkdir(parents=True, exist_ok=False)
    size = tuple(config['canvas_wh'])
    raw = {s: read_jsonl(Path(config['dataset_root']) / s / 'pairs.jsonl') for s in ('train', 'val', 'test')}
    counts = audit_splits(raw)
    expected = {'train': 9520, 'val': 1490, 'test': 1146}
    if {s: len(r) for s, r in raw.items()} != expected:
        raise ValueError('Source split counts drifted from the approved 9520/1490/1146')
    images, poses, seen_pixels = {}, {}, {}
    source_manifests = {s: sha256(Path(config['dataset_root']) / s / 'pairs.jsonl') for s in raw}
    normalized = {}
    for split, rows in raw.items():
        converted = []
        for n, source in enumerate(rows):
            row = dict(source)
            row['group_id'] = row.get('group_id', row['clip_id'])
            for role in ('source', 'target'):
                fid = row[f'{role}_frame_id']
                path = Path(row[f'{role}_rgb_path'])
                if path.stem != fid:
                    raise ValueError(f'RGB/frame mismatch: {fid}: {path}')
                if fid not in images:
                    original_sha = sha256(path)
                    if original_sha in seen_pixels and seen_pixels[original_sha] != split:
                        raise ValueError(f'Identical RGB pixels across splits: {path}')
                    seen_pixels[original_sha] = split
                    with Image.open(path) as original:
                        rgb, affine = letterbox(original, size)
                    destination = prepared / f'{fid}.png'
                    rgb.save(destination)
                    images[fid] = {'path': str(destination), 'sha256': sha256(destination),
                                   'original_path': str(path), 'original_sha256': original_sha,
                                   'affine': affine, 'split': split}
                image = images[fid]
                if image['split'] != split or image['original_path'] != str(path):
                    raise ValueError(f'Frame identity drift: {fid}')
            fid = row['target_frame_id']
            if fid not in poses:
                pose_path = Path(row['target_pose_path'])
                if pose_path.stem != fid:
                    raise ValueError(f'Pose/frame mismatch: {pose_path}')
                record = read_json(pose_path)
                points, valid = pose_points(record, images[fid]['affine']['original'])
                mapped = transform_points(points, images[fid]['affine'])
                skeleton_path = prepared / f'{fid}_dwpose.png'
                render_skeleton(mapped, valid, size).save(skeleton_path)
                valid_points = np.where(valid[:, None], mapped, 0.0)
                geometry = {'keypoints_xy': valid_points.tolist(), 'valid': valid.tolist(),
                            'size': list(size), 'original_json': str(pose_path),
                            'original_json_sha256': sha256(pose_path),
                            'policy': record.get('keypoint_visibility_policy'),
                            'raw_response_semantics': record.get('raw_response_semantics')}
                metadata = prepared / f'{fid}_dwpose.json'
                write_json(metadata, geometry)
                poses[fid] = {'path': str(skeleton_path), 'sha256': sha256(skeleton_path),
                              'metadata': str(metadata)}
            source_image, target_image, pose = images[row['source_frame_id']], images[fid], poses[fid]
            row.update({'split': split, 'condition_images': [source_image['path'], pose['path']],
                        'condition_sha256': [source_image['sha256'], pose['sha256']],
                        'target_image': target_image['path'], 'target_sha256': target_image['sha256'],
                        'target_pose_metadata': pose['metadata'], 'prompt': config['prompt'],
                        'canvas_wh': list(size), 'condition_roles': ['appearance', 'target_dwpose']})
            converted.append(row)
            if n % 2000 == 0:
                print(f'prepare {split}: {n}/{len(rows)}, frames={len(images)}, poses={len(poses)}', flush=True)
        normalized[split] = converted
        write_jsonl(out / f'{split}.jsonl', converted)
    pilots = {
        'train': select_pilot(normalized['train'], config['train_videos'], config['train_targets_per_source'], config['seed']),
        'val': select_pilot(normalized['val'], config['val_videos'], config['val_targets_per_source'], config['seed'])}
    for split, rows in pilots.items():
        write_jsonl(out / f'pilot_{split}.jsonl', rows)
    # A genuine missing-pose diagnostic: both VLM pixels and VAE pixels become black.
    diagnostic = dict(pilots['val'][0])
    blank = prepared / 'blank_pose.png'
    Image.new('RGB', size, 'black').save(blank)
    diagnostic.update({'pair_id': diagnostic['pair_id'] + '_blank',
                       'condition_images': [diagnostic['condition_images'][0], str(blank)],
                       'condition_sha256': [diagnostic['condition_sha256'][0], sha256(blank)],
                       'diagnostic': 'blank_pose'})
    write_jsonl(out / 'diagnostics.jsonl', [diagnostic])
    write_json(out / 'assets.json', {'rgb': images, 'poses': poses})
    index = {'schema': 'bilibili_dual_edit_manifest_v1', 'config_sha256': digest(config),
             'source_manifests': source_manifests, 'splits': counts, 'unique_rgb': len(images),
             'unique_pose': len(poses), 'pilot_train': len(pilots['train']), 'pilot_val': len(pilots['val']),
             'render': 'original_dwpose134_validmask_letterbox_v1', 'canvas_wh': list(size),
             'pilot_manifests': {s: sha256(out / f'pilot_{s}.jsonl') for s in pilots},
             'prepared_at_unix': time.time()}
    write_json(out / 'index.json', index)
    preview = Image.new('RGB', (size[0] * 3, (size[1] + 30) * 4), 'white')
    draw = ImageDraw.Draw(preview)
    for i, row in enumerate([*pilots['train'][:2], *pilots['val'][:2]]):
        for j, path in enumerate([*row['condition_images'], row['target_image']]):
            with Image.open(path) as image:
                preview.paste(image.convert('RGB'), (j * size[0], i * (size[1] + 30) + 30))
        draw.text((4, i * (size[1] + 30) + 5), f'{row["split"]}: A | target DWPose | B RGB', fill='black')
    preview.save(out / 'preview.png')
    print(f'Prepared: {out}', flush=True)
    return out
