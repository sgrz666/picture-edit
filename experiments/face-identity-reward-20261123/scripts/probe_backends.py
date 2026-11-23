#!/usr/bin/env python3
"""Probe face-identity backends on the evaluation host.

Checks: SCRFD detection, glintr100 (ArcFace R100), LVFace-B (ONNX),
AdaFace-KPRPE (torch + cvlface code, needs rpe_index_cpp), w600k_r50.
Prints shapes, timings and a same/different cosine sanity pair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MODELS = Path('/home/shangguanrz/project/pic-edit/models')
ANTELOPE = MODELS / 'antelopev2'
FACE_MODELS = MODELS / 'face'
CVLFACE = FACE_MODELS / 'cvlface_adaface_vit_base_kprpe_webface4m'
LVFACE_ONNX = FACE_MODELS / 'LVFace' / 'LVFace-B_Glint360K' / 'LVFace-B_Glint360K.onnx'


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'cannot read {path}')
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-a', required=True)
    parser.add_argument('--image-b', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    report = {'python': sys.version.split()[0], 'device': args.device}
    import torch
    report['torch'] = torch.__version__
    report['cuda_available'] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)

    from insightface import model_zoo
    from insightface.utils import face_align

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])
    ctx_id = 0 if args.device == 'cuda' else -1
    import onnxruntime as ort
    report['ort'] = ort.__version__
    report['ort_providers'] = ort.get_available_providers()

    detector = model_zoo.get_model(str(ANTELOPE / 'scrfd_10g_bnkps.onnx'), providers=providers)
    detector.prepare(ctx_id=ctx_id, input_size=(640, 640))

    def detect_and_align(path):
        bgr = read_bgr(path)
        boxes, landmarks = detector.detect(bgr)
        if not len(boxes):
            return bgr, None, None, None
        index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        aligned = face_align.norm_crop(bgr, landmark=landmarks[index], image_size=112)
        return bgr, boxes[index], landmarks[index], aligned

    bgr_a, box_a, kps_a, aligned_a = detect_and_align(args.image_a)
    bgr_b, box_b, kps_b, aligned_b = detect_and_align(args.image_b)
    report['det_a'] = None if box_a is None else [float(v) for v in box_a]
    report['det_b'] = None if box_b is None else [float(v) for v in box_b]
    if aligned_a is None or aligned_b is None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit('detection failed on a probe image')

    results = {}

    # --- glintr100 (current baseline) ---
    recognizer = model_zoo.get_model(str(ANTELOPE / 'glintr100.onnx'), providers=providers)
    recognizer.prepare(ctx_id=ctx_id)
    t0 = time.time()
    fa = recognizer.get_feat(aligned_a).reshape(-1).astype(np.float32)
    fb = recognizer.get_feat(aligned_b).reshape(-1).astype(np.float32)
    fa /= np.linalg.norm(fa)
    fb /= np.linalg.norm(fb)
    results['glintr100'] = {'dim': int(fa.size), 'same_pair_cosine': float(fa @ fb),
                            'ms_per_image': (time.time() - t0) * 500}

    # --- w600k_r50 (buffalo_l ArcFace R50) ---
    w600k_path = Path('/data/shangguanrz/models/insightface/models/buffalo_l/w600k_r50.onnx')
    if w600k_path.is_file():
        r50 = model_zoo.get_model(str(w600k_path), providers=providers)
        r50.prepare(ctx_id=ctx_id)
        t0 = time.time()
        fa = r50.get_feat(aligned_a).reshape(-1).astype(np.float32)
        fb = r50.get_feat(aligned_b).reshape(-1).astype(np.float32)
        fa /= np.linalg.norm(fa)
        fb /= np.linalg.norm(fb)
        results['w600k_r50'] = {'dim': int(fa.size), 'same_pair_cosine': float(fa @ fb),
                                'ms_per_image': (time.time() - t0) * 500}
    else:
        results['w600k_r50'] = {'status': 'missing', 'path': str(w600k_path)}

    # --- LVFace-B ONNX ---
    if LVFACE_ONNX.is_file():
        session = ort.InferenceSession(str(LVFACE_ONNX), providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        report['lvface_io'] = {'input': input_name, 'output': output_name,
                               'input_shape': session.get_inputs()[0].shape,
                               'providers_used': session.get_providers()}

        def lvface_embed(aligned_bgr):
            rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
            out = session.run([output_name], {input_name: tensor})[0]
            feat = np.asarray(out).reshape(-1).astype(np.float32)
            return feat / max(float(np.linalg.norm(feat)), 1e-12)

        t0 = time.time()
        fa, fb = lvface_embed(aligned_a), lvface_embed(aligned_b)
        results['lvface_b'] = {'dim': int(fa.size), 'same_pair_cosine': float(fa @ fb),
                               'ms_per_image': (time.time() - t0) * 500}
    else:
        results['lvface_b'] = {'status': 'missing', 'path': str(LVFACE_ONNX)}

    # --- AdaFace-KPRPE ---
    try:
        import rpe_index_cpp  # noqa: F401
        report['rpe_index_cpp'] = rpe_index_cpp.version()
        rpe_ok = True
    except Exception as error:  # noqa: BLE001
        report['rpe_index_cpp'] = f'missing: {type(error).__name__}: {error}'
        rpe_ok = False
    if rpe_ok:
        sys.path.insert(0, str(CVLFACE))
        os.chdir(CVLFACE)
        from models import get_model
        from omegaconf import OmegaConf
        import yaml
        config = OmegaConf.create(yaml.safe_load(open('pretrained_model/model.yaml')))
        model = get_model(config)
        model.load_state_dict_from_path('pretrained_model/model.pt')
        model.eval()
        if args.device == 'cuda':
            model = model.cuda()

        def kprpe_embed(aligned_bgr):
            # cvlface expects RGB in [-1, 1]; the 5 keypoints are re-detected on
            # the already-aligned 112x112 crop so they live in crop coordinates.
            rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
            crop_boxes, crop_kps = detector.detect(aligned_bgr)
            if not len(crop_boxes):
                return None
            index = int(np.argmax((crop_boxes[:, 2] - crop_boxes[:, 0]) *
                                  (crop_boxes[:, 3] - crop_boxes[:, 1])))
            points = np.asarray(crop_kps[index], dtype=np.float32)
            keypoints = torch.from_numpy(points)[None]
            if args.device == 'cuda':
                tensor, keypoints = tensor.cuda(), keypoints.cuda()
            with torch.no_grad():
                out = model(tensor, keypoints)
            feat = out[0] if isinstance(out, (tuple, list)) else out
            feat = feat.reshape(-1).float().cpu().numpy()
            return feat / max(float(np.linalg.norm(feat)), 1e-12)

        t0 = time.time()
        fa = kprpe_embed(aligned_a)
        fb = kprpe_embed(aligned_b)
        if fa is None or fb is None:
            results['adaface_kprpe'] = {'status': 'blocked', 'reason': 'no keypoints on aligned crop'}
        else:
            results['adaface_kprpe'] = {'dim': int(fa.size), 'same_pair_cosine': float(fa @ fb),
                                        'ms_per_image': (time.time() - t0) * 500}
    else:
        results['adaface_kprpe'] = {'status': 'blocked', 'reason': report['rpe_index_cpp']}

    report['results'] = results
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
