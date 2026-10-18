"""Run existing Multi-HMR weights using actual Hi4D intrinsics for all pilot views."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from geometry import resize_pad_affine, project, associate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/shangguanrz/project/pic-edit')
    parser.add_argument('--workdir', required=True)
    args = parser.parse_args()
    root, work = Path(args.root), Path(args.workdir)
    repo = root / 'add/tools/multi-hmr'
    os.environ['MULTIHMR_DINOV2_REPO'] = str(repo.parent / 'dinov2_e1277af')
    os.environ['XFORMERS_DISABLED'] = '1'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    from demo_smplx_only import load_model, open_image, forward_model
    torch.set_num_threads(8)
    torch.manual_seed(0)
    start = time.time()
    model = load_model('multiHMR_896_L').eval()
    output = work / 'initialization'
    output.mkdir(exist_ok=True)
    records = []
    joints = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    for action, frame in [('talk14', 32), ('hug14', 62)]:
        raw = root / 'datasets/Hi4D_pilot_v1/raw/pair14' / action
        cams = np.load(raw / 'cameras/rgb_cameras.npz')
        smpl = np.load(raw / 'smpl' / f'{frame:06d}.npz')
        for c, camera in enumerate(cams['ids']):
            camera = int(camera)
            image = work / 'inputs/pair14' / action / 'images' / str(camera) / f'{frame:06d}.jpg'
            x, original = open_image(str(image), model.img_size)
            small = ImageOps.contain(original, (model.img_size, model.img_size))
            pad = (round((model.img_size - small.width) / 2), round((model.img_size - small.height) / 2))
            A = resize_pad_affine(original.size, small.size, pad)
            K = A @ cams['intrinsics'][c]
            if np.max(np.abs(cams['dist_coeffs'][c])) > 1e-8:
                raise ValueError('This initializer needs undistorted Hi4D images')
            humans = forward_model(model, x, torch.tensor(K, device='cuda', dtype=torch.float32)[None], det_thresh=.2, nms_kernel_size=3)
            tensors = {'K_canvas': K, 'canvas_affine': A, 'K_native': cams['intrinsics'][c], 'extrinsics': cams['extrinsics'][c]}
            for p, human in enumerate(humans):
                for key, value in human.items():
                    if torch.is_tensor(value):
                        tensors[f'person_{p}_{key}'] = value.detach().float().cpu().numpy()
            truth = np.stack([project(person[joints], K, cams['extrinsics'][c]) for person in smpl['joints_3d']])
            candidates = np.stack([tensors[f'person_{p}_j2d'][joints] for p in range(len(humans))]) if humans else np.zeros((0, len(joints), 2))
            assignment, errors = associate(candidates, np.ones(candidates.shape[:2]), truth)
            tensors['actor_assignment'] = assignment
            tensors['assignment_error_px'] = errors
            target = output / f'{action}_{frame:06d}_cam{camera}.npz'
            np.savez_compressed(target, **tensors)
            record = {'action': action, 'frame': frame, 'camera': camera, 'person_count': len(humans), 'actor_assignment': assignment.tolist(), 'assignment_error_px': [float(e) if np.isfinite(e) else None for e in errors], 'path': str(target)}
            records.append(record)
            print(json.dumps(record), flush=True)
    weight = repo / 'models/multiHMR/multiHMR_896_L.pt'
    h = hashlib.sha256()
    with weight.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    metadata = {'model': 'multiHMR_896_L', 'weight_sha256': h.hexdigest(), 'torch': torch.__version__, 'elapsed_seconds': time.time() - start, 'camera_source': 'Hi4D calibrated intrinsics transformed by actual resize and padding', 'records': records}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
