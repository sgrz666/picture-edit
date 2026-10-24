#!/usr/bin/env python3
"""从现有目标 mesh_cam 导出版本化 Camera-Z/valid 和绑定 Body21 的 Depth V2 manifest。

不估计人体、不训练、不覆盖旧资产。本轮只实现工具，不执行。
--kind relative_png 只为现有 PNG 建立明确来源记录，不伪造 Camera-Z。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import importlib.util
import json

import numpy as np
from PIL import Image
import torch

from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
from src.pose_control.qwen21_full.depth_v2_records import RECORD_SCHEMA, NORMAL_BASES, camera_fingerprint
from src.pose_control.qwen21_full.pose_cache import sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--pose-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=('camera_z', 'relative_png'), default='camera_z')
    p.add_argument('--normal-basis', choices=NORMAL_BASES, default='view_xright_yup_ztoward')
    p.add_argument('--polarity', choices=('near_one', 'near_zero'), default='near_one')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    target = args.output.resolve()
    data_root = Path('/data/shangguanrz').resolve()
    if not target.is_relative_to(data_root) or target == data_root:
        raise ValueError('New geometry products must stay under /data/shangguanrz in a dedicated directory')
    if target.exists():
        raise FileExistsError(target)
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    dataset = NativePoseDataset(args.pairs, args.pose_manifest)
    renderer_module = None
    renderer_path = ROOT / 'add/tools/SMPLest-X/scripts/build_iper_assets_512_nvdiffrast.py'
    renderers, params_cache, depths = {}, {}, {}
    if args.kind == 'camera_z':
        if not torch.cuda.is_available():
            raise RuntimeError('Existing SMPLest-X/nvdiffrast CUDA environment required; no package installation')
        spec = importlib.util.spec_from_file_location('_qwen_depth_v2_renderer', renderer_path)
        renderer_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = renderer_module
        spec.loader.exec_module(renderer_module)
    target.mkdir(parents=True, exist_ok=False)
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with (target / 'records.jsonl').open('x', encoding='utf8') as manifest:
        for i in range(count):
            sample = dataset[i]
            ref = sample['pose_provenance']
            pose = dataset.records.raw(ref['record_id'])
            meta = pose['metadata']
            original_h, original_w = meta['original_hw']
            if ref['record_id'] in depths:
                depth_path, valid_path, asset_hw = depths[ref['record_id']]
            elif args.kind == 'relative_png':
                depth_path = Path(meta['sources']['depth_vis']['path'])
                valid_path = Path(meta['sources']['smpl_mask']['path'])
                with Image.open(depth_path) as image:
                    asset_hw = (image.height, image.width)
                depths[ref['record_id']] = depth_path, valid_path, asset_hw
            else:
                params_path = meta['sources']['params']['path']
                if params_path not in params_cache:
                    params_cache[params_path] = torch.load(params_path, map_location='cpu', weights_only=False)
                mesh = params_cache[params_path]['mesh_cam'][meta['param_index']].detach().cpu().numpy().astype(np.float32)
                model_path = meta['sources']['model']['path']
                size = (max(original_h, original_w) + 31) // 32 * 32
                renderer_key = (model_path, size)
                if renderer_key not in renderers:
                    with np.load(model_path, allow_pickle=False) as model:
                        faces = model['f'].astype(np.int32)
                    renderers[renderer_key] = renderer_module.NvDiffRasterRenderer(faces, size, torch.device('cuda'))
                fx, fy = meta['camera']['focal_xy']
                cx, cy = meta['camera']['principal_xy']
                z, valid, _ = renderers[renderer_key].render(mesh, fx=fx, fy=fy, cx=cx, cy=cy)
                z, valid = z[:original_h, :original_w], valid[:original_h, :original_w] > 0
                if not np.isfinite(z[valid]).all() or (z[valid] <= 0).any():
                    raise ValueError('Renderer produced invalid camera-space Z')
                z = np.where(valid, z, 0).astype(np.float32)
                name = ref['record_id'].replace(':', '__')
                depth_path, valid_path = target / (name + '_camera_z.npy'), target / (name + '_valid.npy')
                for path, value in ((depth_path, z), (valid_path, valid)):
                    with path.open('xb') as stream:
                        np.save(stream, value, allow_pickle=False)
                asset_hw = (original_h, original_w)
                depths[ref['record_id']] = depth_path, valid_path, asset_hw
            sx, sy = original_w / asset_hw[1], original_h / asset_hw[0]
            row = {'schema': RECORD_SCHEMA, 'pair_id': sample['pair_metadata']['pair_id'],
                   'target_stem': sample['target_stem'], 'target_role': meta['role'], 'kind': args.kind,
                   'pose_record_id': ref['record_id'], 'pose_record_sha256': ref['record_sha256'],
                   'source_params_sha256': meta['sources']['params']['sha256'],
                   'source_model_sha256': meta['sources']['model']['sha256'],
                   'camera_sha256': camera_fingerprint(meta['camera']), 'camera_id': ref['record_id'],
                   'scale_group_id': ref['record_id'], 'normal_basis': args.normal_basis,
                   'units': 'model_unit' if args.kind == 'camera_z' else 'unknown',
                   'scale_status': 'model_scale' if args.kind == 'camera_z' else 'unknown',
                   'depth_path': str(depth_path), 'valid_path': str(valid_path),
                   'depth_sha256': sha256(depth_path), 'valid_sha256': sha256(valid_path),
                   'asset_to_original': [[sx, 0., 0.], [0., sy, 0.], [0., 0., 1.]],
                   'polarity': args.polarity if args.kind == 'relative_png' else 'near_one',
                   'renderer_path': str(renderer_path) if args.kind == 'camera_z' else None,
                   'renderer_sha256': sha256(renderer_path) if args.kind == 'camera_z' else None,
                   'tool_sha256': sha256(Path(__file__)), 'body_source': 'paired_target_frame'}
            manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
    (target / 'README.md').write_text(
        '# Depth V2 来源记录\n\n'
        f'本目录含 {count} 个 pair，类型为 {args.kind}。Camera-Z 是模型尺度，不自动宣称真实米制。\n'
        'records.jsonl 绑定原始资产、相机和 Body21 pose 的 SHA256。未执行人体估计或训练。\n', encoding='utf8')
    print(json.dumps({'records': count, 'manifest': str(target / 'records.jsonl'), 'kind': args.kind}))


if __name__ == '__main__':
    main()
