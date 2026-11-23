#!/usr/bin/env python3
"""Face-identity evaluation kit: detection, alignment, degradation, backends.

All backends consume the same 112x112 aligned crop produced by SCRFD 5-point
alignment with the ArcFace template, so differences are attributable to the
recognition network rather than to per-model pre-processing.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Import torch before onnxruntime so its bundled CUDA 12 libraries are already
# loaded; otherwise the ORT CUDA provider fails to resolve libcublasLt.so.12.
try:  # pragma: no cover - availability depends on the host
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass

MODELS = Path('/home/shangguanrz/project/pic-edit/models')
ANTELOPE = MODELS / 'antelopev2'
FACE_MODELS = MODELS / 'face'
CVLFACE = FACE_MODELS / 'cvlface_adaface_vit_base_kprpe_webface4m'
LVFACE_ONNX = FACE_MODELS / 'LVFace' / 'LVFace-B_Glint360K' / 'LVFace-B_Glint360K.onnx'
W600K_ONNX = Path('/data/shangguanrz/models/insightface/models/buffalo_l/w600k_r50.onnx')

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


# --------------------------------------------------------------------------- io
def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'cannot read {path}')
    return image


def stable_seed(text: str, base: int = 0) -> int:
    digest = hashlib.sha256(f'{base}:{text}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little')


# ------------------------------------------------------------------- alignment
def estimate_norm(keypoints, size: int = 112) -> np.ndarray:
    from skimage.transform import SimilarityTransform

    dst = ARCFACE_DST * (size / 112.0)
    transform = SimilarityTransform()
    if not transform.estimate(np.asarray(keypoints, dtype=np.float64), dst):
        raise RuntimeError('similarity transform failed')
    return transform.params[:2].copy()


def jitter_matrix(seed: int, rotation_deg: float, scale_pct: float,
                  shift_px: float, size: int = 112) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-rotation_deg, rotation_deg))
    scale = 1.0 + float(rng.uniform(-scale_pct, scale_pct))
    centre = (size / 2.0, size / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    matrix[0, 2] += float(rng.uniform(-shift_px, shift_px))
    matrix[1, 2] += float(rng.uniform(-shift_px, shift_px))
    return matrix


def align_face(bgr, keypoints, *, size=112, jitter=None, seed=0):
    """Return (crop, keypoints_in_crop, matrix). `jitter` is a dict of strengths."""
    matrix = estimate_norm(keypoints, size=size)
    if jitter:
        extra = jitter_matrix(seed, jitter.get('rotation', 0.0), jitter.get('scale', 0.0),
                              jitter.get('shift', 0.0), size=size)
        homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
        matrix = (np.vstack([extra, [0.0, 0.0, 1.0]]) @ homogeneous)[:2]
    crop = cv2.warpAffine(bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    points = np.asarray(keypoints, dtype=np.float64)
    moved = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) @ matrix.T
    return crop, moved.astype(np.float32), matrix


# ------------------------------------------------------------------ degradation
def degrade(crop, kind: str, seed: int = 0):
    """Degrade an aligned 112x112 BGR crop. Returns a new uint8 crop."""
    if kind in ('clean', 'none', ''):
        return crop
    rng = np.random.default_rng(seed)
    height, width = crop.shape[:2]
    if kind.startswith('blur'):
        sigma = float(kind[4:])
        return cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if kind.startswith('lowres'):
        side = int(kind[6:])
        small = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if kind.startswith('jpeg'):
        quality = int(kind[4:])
        ok, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else crop
    if kind == 'occl':
        output = crop.copy()
        band = int(height * 0.45)
        top = int(height * 0.55)
        output[top:top + band, :] = 128
        return output
    if kind == 'genlike':
        small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.1)
        ok, buffer = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else small
        upscaled = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)
        cast = upscaled.astype(np.float32) * np.array([1.02, 1.0, 0.97], np.float32)
        return np.clip(cast, 0, 255).astype(np.uint8)
    raise ValueError(f'unknown degradation {kind}')


# -------------------------------------------------------------------- detector
class Detector:
    def __init__(self, providers=None, det_size=640):
        from insightface import model_zoo

        providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = model_zoo.get_model(str(ANTELOPE / 'scrfd_10g_bnkps.onnx'), providers=providers)
        ctx = 0 if 'CUDAExecutionProvider' in providers else -1
        self.model.prepare(ctx_id=ctx, input_size=(det_size, det_size))
        self.det_size = det_size
        self.providers = providers

    def detect(self, bgr):
        boxes, landmarks = self.model.detect(bgr)
        output = []
        for box, points in zip(boxes, landmarks):
            x0, y0, x1, y1, score = [float(v) for v in box[:5]]
            output.append({'bbox': [x0, y0, x1, y1], 'score': score,
                           'side': float(min(x1 - x0, y1 - y0)),
                           'keypoints': np.asarray(points, dtype=np.float32)})
        output.sort(key=lambda item: item['side'], reverse=True)
        return output


def face_quality(crop, keypoints):
    """Cheap quality descriptors used for stratification."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_mid = (left_eye + right_eye) / 2.0
    inter_eye = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
    # Nose displacement from the eye midpoint, in inter-ocular units: 0 frontal,
    # grows with yaw. Sign gives the turning direction.
    yaw = float((nose[0] - eye_mid[0]) / inter_eye)
    roll = float(np.degrees(np.arctan2(*(right_eye - left_eye)[::-1])))
    return {'sharpness': sharpness, 'yaw_proxy': yaw, 'roll_deg': roll,
            'inter_eye': inter_eye}


# -------------------------------------------------------------------- backends
def _l2(array):
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    return array / max(float(np.linalg.norm(array)), 1e-12)


class InsightFaceBackend:
    """glintr100 (ArcFace R100, Glint360K) and w600k_r50 (ArcFace R50)."""

    def __init__(self, name, onnx_path, providers=None, batch_size=64):
        from insightface import model_zoo

        providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.name = name
        self.batch_size = batch_size
        self.model = model_zoo.get_model(str(onnx_path), providers=providers)
        self.model.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1)
        self.input_mean = getattr(self.model, 'input_mean', 127.5)
        self.input_std = getattr(self.model, 'input_std', 127.5)
        self.dim = 512

    def embed(self, crops):
        blobs = [cv2.dnn.blobFromImage(crop, 1.0 / self.input_std, (112, 112),
                                       (self.input_mean, self.input_mean, self.input_mean),
                                       swapRB=True) for crop in crops]
        output = []
        for start in range(0, len(blobs), self.batch_size):
            chunk = np.concatenate(blobs[start:start + self.batch_size], axis=0)
            features = self.model.session.run(None, {self.model.input_name: chunk})[0]
            output.extend(_l2(row) for row in features)
        return output


class LVFaceBackend:
    """LVFace-B (ViT-B, Glint360K) from the official ONNX export."""

    def __init__(self, providers=None, batch_size=32):
        import onnxruntime as ort

        self.name = 'lvface_b'
        self.batch_size = batch_size
        providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(LVFACE_ONNX), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.dim = 512

    def embed(self, crops):
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            tensors.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        output = []
        for start in range(0, len(tensors), self.batch_size):
            chunk = np.stack(tensors[start:start + self.batch_size])
            features = self.session.run([self.output_name], {self.input_name: chunk})[0]
            output.extend(_l2(row) for row in features)
        return output


class KPRPEBackend:
    """AdaFace ViT-B with keypoint relative position encoding (CVLface)."""

    def __init__(self, device='cuda', batch_size=32):
        import sys

        import torch
        import yaml
        from omegaconf import OmegaConf

        self.name = 'adaface_kprpe'
        self.batch_size = batch_size
        self.torch = torch
        self.device = device if torch.cuda.is_available() else 'cpu'
        sys.path.insert(0, str(CVLFACE))
        cwd = os.getcwd()
        os.chdir(CVLFACE)
        try:
            from models import get_model

            config = OmegaConf.create(yaml.safe_load(open('pretrained_model/model.yaml')))
            model = get_model(config)
            model.load_state_dict_from_path('pretrained_model/model.pt')
        finally:
            os.chdir(cwd)
        self.model = model.eval().to(self.device)
        self.dim = 512

    def embed(self, crops, keypoints):
        torch = self.torch
        output = []
        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start:start + self.batch_size]
            batch_points = keypoints[start:start + self.batch_size]
            tensors = []
            for crop in batch_crops:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                tensors.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            images = torch.from_numpy(np.stack(tensors)).to(self.device)
            # KP-RPE expects keypoints normalised to [0, 1] patch coordinates.
            points = torch.from_numpy(np.stack(batch_points).astype(np.float32) / 112.0).to(self.device)
            with torch.no_grad():
                features = self.model(images, points)
            features = features.float().cpu().numpy()
            output.extend(_l2(row) for row in features)
        return output


def build_backends(names, device='cuda'):
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda'
                 else ['CPUExecutionProvider'])
    available = {}
    for name in names:
        if name == 'glintr100':
            available[name] = InsightFaceBackend('glintr100', ANTELOPE / 'glintr100.onnx', providers)
        elif name == 'w600k_r50':
            available[name] = InsightFaceBackend('w600k_r50', W600K_ONNX, providers)
        elif name == 'lvface_b':
            available[name] = LVFaceBackend(providers)
        elif name == 'adaface_kprpe':
            available[name] = KPRPEBackend(device)
        else:
            raise ValueError(f'unknown backend {name}')
    return available


@dataclass(frozen=True)
class Condition:
    """One (alignment, degradation) cell of the ablation grid."""

    name: str
    jitter: dict | None = None
    degrade: str = 'clean'
