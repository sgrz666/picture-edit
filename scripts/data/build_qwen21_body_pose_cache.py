#!/usr/bin/env python3
"""在现有 SMPLest-X 环境从 params.pt 预计算 Body21；不重跑估计、不训练。

frames JSONL: {dataset: tiktok|iper, sequence_id, stem, role: target(默认)}。
只写新 output 目录。缓存相机原始像素，target CanvasSpec 只在读取时应用一次。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import inspect
import json

import numpy as np
from PIL import Image
import torch

from src.pose_control.qwen21_full.body_pose import POSE_SCHEMA, JOINT_NAMES, PARENTS_22, ROTATION_CONVENTION
from src.pose_control.qwen21_full.pose_cache import canonical_global, project_camera, record_id, sha256

ROOT = Path(__file__).resolve().parents[2]


def frame_sources(row, root=ROOT):
    dataset, seq, stem = row['dataset'], str(row['sequence_id']), str(row['stem'])
    role = row.get('role', 'target')
    if dataset == 'tiktok':
        assets = root / 'datasets/TikTokDataset/TikTok_3d_assets_native' / seq
        rgb = root / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset' / seq / 'images' / (stem + '.png')
        params = assets / 'smplestx/params.pt'
        paths = {'normal': assets / 'normal' / (stem + '.png'),
                 'human_mask': rgb.parent.parent / 'masks' / (stem + '.png'),
                 'smpl_mask': assets / 'smplx_mask' / (stem + '.png'),
                 'part': assets / 'part_512' / (stem + '.png'),
                 'depth_vis': assets / 'depth_vis' / (stem + '.png')}
    elif dataset == 'iper':
        sampled = root / 'datasets/iPER/iper_sampled_6src64tgt' / seq
        assets = root / 'datasets/iPER/iper_assets_512_nvdiffrast' / seq
        params = sampled / 'smplestx/params.pt'
        rgb = sampled / 'rgb_1024' / role / (stem + '.png')
        paths = {key: assets / folder / role / (stem + '.png') for key, folder in
                 (('normal', 'normal_512'), ('smpl_mask', 'mask_512'), ('part', 'part_512'), ('depth_vis', 'depth_vis_512'))}
    else:
        raise ValueError('Require TikTok/iPER SMPL-X; native SMPL69 is unsupported')
    frame_map = params.with_name('frame_map.json')
    mapping = json.loads(frame_map.read_text())
    matches = [f for f in mapping['frames'] if f['stem'] == stem and (dataset == 'tiktok' or f['role'] == role)]
    if len(matches) != 1:
        raise ValueError(f'Exact role/stem mapping required: {row}')
    index = int(matches[0]['index' if dataset == 'tiktok' else 'smplestx_index'])
    return index, {'params': params, 'frame_map': frame_map, 'rgb': rgb,
                   'legacy_conditions': assets / 'v6_conditions.pt', **paths}


def build(args):
    import smplx
    if args.output.exists():
        raise FileExistsError('Use a new output directory; old caches remain unchanged')
    rows = [json.loads(x) for x in args.frames.read_text().splitlines() if x.strip()]
    keys = [record_id(r['dataset'], r['sequence_id'], r.get('role', 'target'), r['stem']) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError('Frames must be nonempty and unique')
    torch.set_num_threads(4)
    model = smplx.create(str(args.model_root), model_type='smplx', gender='neutral', ext='npz',
                         use_pca=False, use_face_contour=True).eval().requires_grad_(False)
    model_file = args.model_root / 'smplx/SMPLX_NEUTRAL.npz'
    if tuple(model.parents[:22].tolist()) != PARENTS_22:
        raise ValueError('Model joint tree differs')
    sources_cache, params_cache = {}, {}
    def fingerprint(path):
        path = Path(path).resolve()
        if path not in sources_cache:
            sources_cache[path] = {'path': str(path), 'sha256': sha256(path)}
        return sources_cache[path]
    args.output.mkdir(parents=True)
    facts = []
    with (args.output / 'records.jsonl').open('x') as manifest, torch.no_grad():
        for row, key in zip(rows, keys):
            index, paths = frame_sources(row, args.repo)
            if paths['params'] not in params_cache:
                params_cache[paths['params']] = torch.load(paths['params'], map_location='cpu', weights_only=False)
            params = params_cache[paths['params']]
            with Image.open(paths['rgb']) as image:
                original_hw = (image.height, image.width)
            if (int(params['height'][index].item()), int(params['width'][index].item())) != original_hw:
                raise ValueError('Parameter camera/RGB dimensions differ')
            global26, camera = canonical_global(params, index, original_hw, dataset=row['dataset'])
            kwargs = {name: params[name][index:index+1].float() for name in
                      ('betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression')}
            for name in ('leye_pose', 'reye_pose'):
                if name in params:
                    kwargs[name] = params[name][index:index+1].float()
            output = model(**kwargs, transl=torch.zeros(1, 3))
            translation = torch.tensor(camera['translation'])[None, None]
            reconstructed = output.vertices + translation
            error = float((reconstructed[0] - params['mesh_cam'][index].float()).abs().max())
            if error > 1e-4 or not torch.isfinite(reconstructed).all():
                raise ValueError(f'FK/mesh mismatch {key}: {error}; threshold 1e-4')
            joints = output.joints[0, :22] + translation[0]
            # Canonical rest-pose height uses this frame's beta, zero root/expression/hands.
            rest = model(betas=kwargs['betas'], global_orient=torch.zeros(1, 3), body_pose=torch.zeros(1, 63),
                         left_hand_pose=torch.zeros(1, 45), right_hand_pose=torch.zeros(1, 45),
                         jaw_pose=torch.zeros(1, 3), expression=torch.zeros_like(kwargs['expression']), transl=torch.zeros(1, 3))
            height = rest.vertices[0, :, 1].max() - rest.vertices[0, :, 1].min()
            uv, projected = project_camera(joints[1:], torch.tensor(camera['focal_xy']),
                                           torch.tensor(camera['principal_xy']), original_hw)
            transforms = {}
            for kind in ('normal', 'smpl_mask', 'part', 'depth_vis'):
                with Image.open(paths[kind]) as image:
                    sh, sw = image.height, image.width
                transforms[kind] = {'source_hw': [sh, sw], 'original_hw': list(original_hw),
                                     'scale_xy': [original_hw[1] / sw, original_hw[0] / sh],
                                     'rule': 'normalised_full_frame_then_target_canvas_once'}
            with Image.open(paths['smpl_mask']) as image:
                mask = np.asarray(image.convert('L')) > 0
            yy, xx = np.nonzero(mask)
            if len(yy) == 0:
                raise ValueError('Empty same-source SMPL mask')
            sx, sy = transforms['smpl_mask']['scale_xy']
            bbox = [float(xx.min() * sx), float(yy.min() * sy), float((xx.max() + 1) * sx), float((yy.max() + 1) * sy)]
            provenance = {name: fingerprint(path) for name, path in paths.items()}
            provenance.update(model=fingerprint(model_file), fk_implementation=fingerprint(inspect.getfile(type(model))))
            data = {'body_pose': kwargs['body_pose'][0], 'joints_cam': joints, 'joint_uv': uv,
                    'fk_valid': torch.ones(22, dtype=torch.bool), 'projection_valid': projected,
                    'body_height': height, 'smplx_global': global26,
                    'metadata': {'schema': POSE_SCHEMA, 'record_id': key, 'dataset': row['dataset'],
                                 'sequence_id': str(row['sequence_id']), 'role': row.get('role', 'target'), 'stem': str(row['stem']),
                                 'param_index': index, 'joint_names': list(JOINT_NAMES), 'parents_22': list(PARENTS_22),
                                 'rotation_convention': ROTATION_CONVENTION, 'projection_space': 'original_camera_pixels',
                                 'original_hw': list(original_hw), 'camera': camera, 'sources': provenance,
                                 'condition_to_original': transforms, 'smpl_mask_bbox_original_xyxy': bbox,
                                 'mesh_max_abs_error': error, 'flat_hand_mean': False,
                                 'fk_model': 'SMPL-X neutral npz, use_pca=False, face_contour=True',
                                 'height_convention': 'same_beta_rest_mesh_y_extent',
                                 'pairing': 'target_frame_shape_pose_camera'}}
            filename = key.replace(':', '__') + '.pt'
            with (args.output / filename).open('xb') as stream:
                torch.save(data, stream)
            manifest.write(json.dumps({'record_id': key, 'path': filename, 'sha256': sha256(args.output / filename)}) + '\n')
            old = torch.load(paths['legacy_conditions'], map_location='cpu', weights_only=False)
            old_index = int(old['stem_to_idx'][str(row['stem'])])
            delta = (old['smplx_global'][old_index].float() - global26).abs()
            fact = {'record_id': key, 'mesh_max_abs_error': error, 'canonical_height_m': float(height),
                    'projectable_joints': int(projected.sum()), 'camera': camera['selection'],
                    'global26_legacy_max_abs': float(delta.max()), 'changed_indices': delta.gt(1e-6).nonzero().flatten().tolist()}
            facts.append(fact)
            print(json.dumps(fact), flush=True)
    report = {'schema': POSE_SCHEMA, 'count': len(facts), 'frames': facts, 'no_estimation': True,
              'no_optimizer': True, 'source_manifest_sha256': sha256(args.frames)}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--model-root', type=Path, default=ROOT / 'add/tools/SMPLest-X/human_models/human_model_files')
    build(p.parse_args())


if __name__ == '__main__':
    main()
