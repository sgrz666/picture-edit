#!/usr/bin/env python3
"""Sanity-check preprocessing: batch vs reference path, and LVFace input scaling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facekit  # noqa: E402


def lvface_auc(session, crops, same_idx, diff_idx, mode):
    name = session.get_inputs()[0].name

    def encode(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        if mode == 'minus1_1':
            rgb = rgb / 127.5 - 1.0
        elif mode == 'zero_1':
            rgb = rgb / 255.0
        elif mode == 'raw255':
            pass
        elif mode == 'bgr_minus1_1':
            rgb = crop.astype(np.float32) / 127.5 - 1.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
        feature = session.run(None, {name: tensor})[0].reshape(-1).astype(np.float32)
        return feature / max(float(np.linalg.norm(feature)), 1e-12)

    embeddings = np.stack([encode(crop) for crop in crops])
    same = np.array([float(embeddings[a] @ embeddings[b]) for a, b in same_idx])
    diff = np.array([float(embeddings[a] @ embeddings[b]) for a, b in diff_idx])
    return {'auc': float(roc_auc_score(np.r_[np.ones_like(same), np.zeros_like(diff)],
                                       np.r_[same, diff])),
            'same_mean': float(same.mean()), 'diff_mean': float(diff.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()

    bundle = np.load(args.crops, allow_pickle=False)
    ids = {key: index for index, key in enumerate(bundle['ids'].tolist())}
    crops, keypoints = bundle['crops'], bundle['keypoints']
    pairs = json.loads(args.pairs.read_text(encoding='utf-8'))['pairs']

    used = {}
    same_idx, diff_idx = [], []
    for pair in pairs:
        a, b = ids.get(pair['a']['key']), ids.get(pair['b']['key'])
        if a is None or b is None:
            continue
        if a not in used:
            used[a] = len(used)
        if b not in used:
            used[b] = len(used)
        (same_idx if pair['kind'] == 'same' else diff_idx).append((used[a], used[b]))
        if len(same_idx) >= args.limit and len(diff_idx) >= args.limit:
            break
    order = list(used)
    subset = [crops[index] for index in order]
    print(f'{len(subset)} faces, {len(same_idx)} same / {len(diff_idx)} diff pairs', flush=True)

    # --- glintr100: batched path must equal insightface get_feat ---
    from insightface import model_zoo
    reference = model_zoo.get_model(str(facekit.ANTELOPE / 'glintr100.onnx'),
                                    providers=['CPUExecutionProvider'])
    reference.prepare(ctx_id=-1)
    backend = facekit.InsightFaceBackend('glintr100', facekit.ANTELOPE / 'glintr100.onnx',
                                         providers=['CPUExecutionProvider'])
    batched = np.stack(backend.embed(subset[:32]))
    single = np.stack([reference.get_feat(crop).reshape(-1) for crop in subset[:32]])
    single /= np.linalg.norm(single, axis=1, keepdims=True)
    print('glintr100 batch-vs-get_feat max|diff| =', float(np.abs(batched - single).max()), flush=True)

    # --- LVFace input scaling ---
    import onnxruntime as ort
    session = ort.InferenceSession(str(facekit.LVFACE_ONNX),
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    for mode in ('minus1_1', 'zero_1', 'raw255', 'bgr_minus1_1'):
        print('lvface', mode, json.dumps(lvface_auc(session, subset, same_idx, diff_idx, mode)),
              flush=True)


if __name__ == '__main__':
    main()
