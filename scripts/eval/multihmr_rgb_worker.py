#!/usr/bin/env python3
"""Batch generated-RGB Multi-HMR inference in the existing isolated environment.

Python 3.9 compatible. No dataset, OpenCV, Qwen or evaluator imports are used.
All assets must exist locally. The model is loaded exactly once per request.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_request(request, request_path):
    if request.get('schema_version') != 'dual_v8_multihmr_request_v1':
        raise ValueError('unsupported Multi-HMR request schema')
    K = np.asarray(request.get('K512'), dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('worker K512 requires finite calibrated 512 intrinsics')
    partition = np.asarray(request.get('joint_to_part'))
    if partition.shape != (55,) or not np.issubdtype(partition.dtype, np.integer) or partition.min() < 0 or partition.max() >= 14:
        raise ValueError('worker requires the canonical 55-joint to 14-part table')
    if not _sha(request.get('geometry_sidecar_sha256')):
        raise ValueError('worker requires the actual camera sidecar SHA')
    rows = request.get('outputs')
    if not isinstance(rows, list) or not rows or len(rows) > 12:
        raise ValueError('worker request requires one to twelve generated PNGs')
    scope = Path(request_path).resolve().parent
    keys, destinations = set(), set()
    for row in rows:
        key = row.get('key')
        if not isinstance(key, str) or re.fullmatch('[a-zA-Z0-9_]+', key) is None or key in keys:
            raise ValueError('worker output keys must be distinct safe names')
        keys.add(key)
        image = Path(row['image']).resolve()
        if not Path(row['image']).is_absolute() or not Path(row['output_npz']).is_absolute():
            raise ValueError('worker image and output paths must be absolute')
        if not _sha(row.get('image_sha256')) or file_sha256(image) != row['image_sha256']:
            raise ValueError('worker generated image SHA mismatch')
        with Image.open(image) as value:
            if value.size != (512, 512) or value.format != 'PNG':
                raise ValueError('worker inputs must be actual 512 generated PNGs')
        if not _sha(row.get('original_configuration_sha256')) or not _sha(row.get('original_record_sha256')):
            raise ValueError('worker requires original generation record bindings')
        target = Path(row['output_npz']).resolve()
        if scope not in target.parents or target.suffix != '.npz' or target in destinations or target == image:
            raise ValueError('worker output scope must stay inside the request directory')
        destinations.add(target)
    return request


def _source_hashes(root):
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(root.rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    request_path = args.request.resolve()
    request = validate_request(json.loads(request_path.read_text()), request_path)
    manifest = args.output.resolve() if args.output is not None else request_path.with_name('results.json')
    if request_path.parent not in manifest.parents or manifest == request_path:
        raise ValueError('worker result manifest must stay inside the request directory')
    root = Path(request['root']).resolve()
    repo = root / 'add/tools/multi-hmr'
    local_dino = repo.parent / 'dinov2_e1277af'
    if not (repo / 'models/multiHMR/multiHMR_896_L.pt').is_file() or not (local_dino / 'hubconf.py').is_file():
        raise FileNotFoundError('existing Multi-HMR weights and local DINOv2 are required; downloads are disabled')
    os.environ['MULTIHMR_DINOV2_REPO'] = str(local_dino)
    os.environ['XFORMERS_DISABLED'] = '1'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from utils.constants import CACHE_DIR_MULTIHMR, SMPLX_DIR
    actual_weight = Path(CACHE_DIR_MULTIHMR) / 'multiHMR_896_L.pt'
    if not actual_weight.is_file():
        raise FileNotFoundError('actual Multi-HMR cache weight is absent; downloads are disabled')
    from demo_smplx_only import load_model, open_image, forward_model
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = load_model('multiHMR_896_L', device=torch.device(args.device)).eval()
    body = model.smpl_layer['neutral_10'].bm_x
    lbs = body.lbs_weights.detach().float().cpu().numpy()
    if lbs.shape != (10475, 55) or not np.isfinite(lbs).all():
        raise ValueError('Multi-HMR body must use 10475-vertex, 55-joint SMPL-X')
    labels = np.asarray(request['joint_to_part'], dtype=np.int16)[lbs.argmax(axis=1)]
    body_path = request_path.with_name('smplx_lbs_weights.npz')
    np.savez_compressed(body_path, lbs_weights=lbs, part_labels=labels, faces=np.asarray(body.faces, dtype=np.int32))
    versions = {'python': sys.version.split()[0]}
    for name in ('torch', 'torchvision', 'numpy', 'pillow', 'scipy', 'roma', 'smplx', 'einops', 'pyrender'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    provenance = dict(request_sha256=file_sha256(request_path), worker_source_sha256=file_sha256(__file__),
        executable=sys.executable, model='multiHMR_896_L', model_load_count=1, image_size=int(model.img_size),
        weight_path=str(actual_weight.resolve()), weight_sha256=file_sha256(actual_weight),
        runtime_versions=versions, multihmr_source_sha256=_source_hashes(repo),
        dinov2_source_sha256=_source_hashes(local_dino),
        smplx_asset_sha256={path.relative_to(Path(SMPLX_DIR)).as_posix(): file_sha256(path)
            for path in sorted((Path(SMPLX_DIR) / 'smplx').glob('*')) if path.is_file()},
        body_partition_path=str(body_path), body_partition_sha256=file_sha256(body_path),
        geometry_sidecar_sha256=request['geometry_sidecar_sha256'], generated_rgb_only=True,
        K512=request['K512'], inference=dict(det_thresh=.2, nms_kernel_size=3, seed=0,
            parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}),
            input_tensor_dtype='torch.float32_from_existing_normalize_rgb', output_array_dtype='float32',
            forward_autocast='cuda_float16_via_existing_demo_forward_model' if torch.device(args.device).type == 'cuda' else 'cuda_autocast_disabled_on_cpu'))
    records = []
    for row in request['outputs']:
        target = Path(row['output_npz']); target.parent.mkdir(parents=True, exist_ok=True)
        record = dict(key=row['key'], original_output_sha256=row['image_sha256'],
            original_configuration_sha256=row['original_configuration_sha256'], original_record_sha256=row['original_record_sha256'],
            request_sha256=provenance['request_sha256'], worker_provenance=provenance,
            output_npz=None, output_npz_sha256=None, reconstructed_people=None, failure=None)
        try:
            if file_sha256(row['image']) != row['image_sha256']:
                raise ValueError('generated image changed after worker request validation')
            image, _ = open_image(row['image'], model.img_size, device=torch.device(args.device))
            record['actual_input_tensor_dtype'] = str(image.dtype) if hasattr(image, 'dtype') else None
            K = np.asarray(request['K512'], np.float32); K[:2] *= model.img_size / 512
            with torch.inference_mode():
                humans = forward_model(model, image, torch.tensor(K, device=args.device)[None], det_thresh=.2, nms_kernel_size=3)
            arrays = dict(person_count=np.asarray(len(humans), np.int32), image_size=np.asarray(model.img_size, np.int32),
                K512=np.asarray(request['K512'], np.float32), part_labels=labels)
            for index, human in enumerate(humans):
                for name in ('v3d', 'j3d', 'j2d', 'shape', 'rotvec', 'transl_pelvis'):
                    if name in human:
                        arrays[f'person_{index}_{name}'] = human[name].detach().float().cpu().numpy()
            np.savez_compressed(target, **arrays)
            record.update(output_npz=str(target), output_npz_sha256=file_sha256(target), reconstructed_people=len(humans))
        except Exception as exc:
            record['failure'] = f'{type(exc).__name__}: {exc}'
        write_json(target.with_suffix('.json'), record)
        records.append(record)
        write_json(manifest, dict(schema_version='dual_v8_multihmr_result_v1', provenance=provenance, records=records))
        print(json.dumps(dict(key=row['key'], reconstructed_people=record['reconstructed_people'], failure=record['failure'])), flush=True)


if __name__ == '__main__':
    main()
