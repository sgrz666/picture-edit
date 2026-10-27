#!/usr/bin/env python3
"""Run Multi-HMR on Hi4D target photographs and dump per-person joints in native pixels.

Runs inside the isolated multi-HMR environment (Python 3.9, torch, smplx). It imports nothing
from this repository: everything it needs comes from the request JSON, and it writes one npz per
image plus a manifest. The model is loaded once for the whole request.

Why the geometry bookkeeping is explicit: ``open_image`` keeps the aspect ratio and centre-pads
the frame to the square model input, so a joint in model coordinates is only a pixel position in
the original photograph once the scale and the pad offset are undone. Getting that wrong would
shift every PCK number by a constant and still look plausible, so the worker re-projects its own
3D joints and records the median disagreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = 'multiHMR_896_L'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def pad_geometry(size, img_size):
    """Reproduce ImageOps.contain + ImageOps.pad exactly: scale, then integer centred pad.

    ImageOps.contain returns a new image rather than resizing in place, so the returned object's
    size is the one that matters; reading the copy's size silently reported scale 1.0 and shifted
    every joint by the pad, which no self-check inside the model could catch.
    """
    width, height = size
    contained = ImageOps.contain(Image.new('RGB', (width, height)), (img_size, img_size))
    scaled_width, scaled_height = contained.size
    scale_x = scaled_width / width
    scale_y = scaled_height / height
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError(f'contain did not keep one scale: {size} -> {(scaled_width, scaled_height)}')
    pad_x = (img_size - scaled_width) // 2
    pad_y = (img_size - scaled_height) // 2
    return scale_x, (float(pad_x), float(pad_y))


def model_intrinsics(k_native, scale, pad):
    """Native 3x3 intrinsics to the padded square input: u_model = s*u_native + pad."""
    k_native = np.asarray(k_native, dtype=np.float64)
    if k_native.shape != (3, 3) or not np.isfinite(k_native).all():
        raise ValueError('K_native must be a finite 3x3 matrix')
    k_model = k_native.copy()
    k_model[0, :2] *= scale
    k_model[1, :2] *= scale
    k_model[0, 2] = k_native[0, 2] * scale + pad[0]
    k_model[1, 2] = k_native[1, 2] * scale + pad[1]
    return k_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding='utf-8'))
    out_dir = Path(request['out_dir']).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(request['project_root']).resolve()
    repo = (root / 'add/tools/multi-hmr').resolve()
    dinov2 = (root / 'add/tools/dinov2_e1277af').resolve()
    weight = repo / 'models/multiHMR' / f'{MODEL_NAME}.pt'
    if not weight.is_file():
        raise FileNotFoundError(f'Multi-HMR checkpoint missing at {weight} (downloads are off)')
    if not (dinov2 / 'hubconf.py').is_file():
        raise FileNotFoundError(f'local DINOv2 repository missing at {dinov2}')
    os.environ['MULTIHMR_DINOV2_REPO'] = str(dinov2)
    os.environ['XFORMERS_DISABLED'] = '1'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from demo_smplx_only import forward_model, load_model, open_image
    from smplx.joint_names import JOINT_NAMES

    torch.set_num_threads(8)
    torch.manual_seed(0)
    started = time.perf_counter()
    model = load_model(MODEL_NAME, device=torch.device(args.device)).eval()
    img_size = int(model.img_size)
    det_thresh = float(request.get('det_thresh', 0.2))
    provenance = {
        'schema': 'dual_fit_worker_v1', 'model': MODEL_NAME, 'img_size': img_size,
        'det_thresh': det_thresh, 'checkpoint': str(weight.resolve()),
        'checkpoint_sha256': sha256_file(weight), 'torch': torch.__version__,
        'joint_order': 'smplx.joint_names.JOINT_NAMES', 'joint_count': len(JOINT_NAMES),
        'load_seconds': time.perf_counter() - started, 'python': sys.version.split()[0],
        'request_sha256': sha256_file(request_path)}
    records = []
    for item in request['images']:
        image_path = Path(item['image']).resolve()
        name = item['name']
        record = {'name': name, 'image': str(image_path), 'failure': None}
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f'no such image: {image_path}')
            record['image_sha256'] = sha256_file(image_path)
            with Image.open(image_path) as handle:
                native_size = handle.size
            scale, pad = pad_geometry(native_size, img_size)
            k_model = model_intrinsics(item['K_native'], scale, pad)
            tensor, _ = open_image(str(image_path), img_size, device=torch.device(args.device))
            with torch.inference_mode():
                humans = forward_model(model, tensor,
                    torch.tensor(k_model, dtype=torch.float32,
                        device=torch.device(args.device))[None],
                    det_thresh=det_thresh, nms_kernel_size=int(request.get('nms_kernel_size', 3)))
            j2d_model, j3d = [], []
            for human in humans:
                two_d = human.get('j2d_smplx', human.get('j2d'))
                three_d = human.get('j3d_smplx', human.get('j3d'))
                if two_d is None or three_d is None:
                    raise KeyError(f'human dict lacks 2D/3D joints, has {sorted(human)}')
                j2d_model.append(two_d.detach().float().cpu().numpy())
                j3d.append(three_d.detach().float().cpu().numpy())
            j2d_model = np.stack(j2d_model) if j2d_model else np.zeros((0, len(JOINT_NAMES), 2))
            j3d = np.stack(j3d) if j3d else np.zeros((0, len(JOINT_NAMES), 3))
            j2d_native = (j2d_model - np.asarray(pad)) / scale
            # Self-check: the model's own 3D joints projected through the same intrinsics must
            # land on its 2D output. A large median offset means the coordinate convention here
            # is not what the rest of the report assumes.
            check = None
            if len(j3d):
                projected = (k_model @ j3d[..., None]).squeeze(-1)
                depth = np.clip(projected[..., 2:], 1e-3, None)
                projected = projected[..., :2] / depth
                check = float(np.median(np.linalg.norm(projected - j2d_model, axis=-1)))
            target = out_dir / f'{name}.npz'
            np.savez_compressed(target, j2d_native=j2d_native, j2d_model=j2d_model, j3d=j3d,
                joint_names=np.asarray(JOINT_NAMES), scale=np.asarray(scale, np.float64),
                pad_xy=np.asarray(pad), native_size=np.asarray(native_size),
                K_native=np.asarray(item['K_native'], np.float64),
                K_model=k_model, person_count=np.asarray(len(humans)))
            record.update(person_count=int(len(humans)), native_size=list(native_size),
                scale=scale, pad_xy=list(pad), reproject_median_px=check,
                npz=target.name, npz_sha256=sha256_file(target))
        except Exception as error:  # one bad frame must not lose the rest of the batch
            record['failure'] = f'{type(error).__name__}: {error}'
        records.append(record)
        print(json.dumps({'name': name, 'person_count': record.get('person_count'),
            'reproject_median_px': record.get('reproject_median_px'),
            'failure': record['failure']}), flush=True)
    manifest = {'provenance': provenance, 'records': records}
    write_json(out_dir / 'fit_manifest.json', manifest)


if __name__ == '__main__':
    main()
