"""Tracked copies of the established four-arm CPU identity/pose evaluators.

Copied from sg:/home/shangguanrz/project/pic-edit/eval_iper_four_arm.py
on 2026-10-08 (source SHA256:
51a7b2d6ae88b94a64e818179fe781a5d117de65d9e7999384b8710fc473feb8).
Metric algorithms, visibility threshold, 512 coordinate grid and COCO-to-Body25
mapping are preserved. Model imports are lazy and model paths use repository
ROOT, allowing report/image helpers to run without ONNX or InsightFace installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COCO_TO_BODY25 = (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


class PoseMetric:
    def __init__(self, models_root: str | Path | None = None) -> None:
        import onnxruntime as ort
        paths = Path(models_root) / 'DWPose' if models_root is not None else ROOT / 'models/DWPose'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.det = ort.InferenceSession(str(paths / 'yolox_l.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.pose = ort.InferenceSession(str(paths / 'dw-ll_ucoco_384.onnx'), sess_options=options, providers=['CPUExecutionProvider'])

    def detect(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray] | None:
        import cv2
        from src.evaluation.dwpose_onnx.onnxdet import inference_detector
        from src.evaluation.dwpose_onnx.onnxpose import inference_pose
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        boxes = inference_detector(self.det, bgr)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        points, scores = inference_pose(self.pose, boxes[np.argmax(areas):np.argmax(areas) + 1], bgr)
        return points[0, :17], scores[0, :17]

    @staticmethod
    def compare(detected: tuple[np.ndarray, np.ndarray] | None, target: np.ndarray,
                image_size: tuple[int, int] | None = None) -> dict[str, float | bool | None]:
        if detected is None:
            return {'pose_detected': False, 'pose_nme_diagonal': None, 'pose_pck_002': 0.0, 'pose_pck_005': 0.0, 'pose_joint_recall': 0.0, 'pose_matched_joints': 0}
        points, scores = detected
        target_coco = target[np.array(COCO_TO_BODY25)]
        target_visible = target_coco[:, 2] > 0.3
        matched = target_visible & (scores > 0.3)
        count = int(matched.sum())
        denominator = int(target_visible.sum())
        if denominator == 0:
            return {'pose_detected': True, 'pose_nme_diagonal': None, 'pose_pck_002': None, 'pose_pck_005': None, 'pose_joint_recall': None, 'pose_matched_joints': 0}
        w, h = (512.0, 512.0) if image_size is None else (float(image_size[0]), float(image_size[1]))
        scale = np.array([w, h], dtype=np.float32)
        diag = float(np.hypot(w, h))
        distance = np.linalg.norm(points[matched] - target_coco[matched, :2] * scale, axis=1) / max(diag, 1e-12)
        return {'pose_detected': True, 'pose_nme_diagonal': float(distance.mean()) if count else None,
                'pose_pck_002': float((distance < 0.02).sum() / denominator),
                'pose_pck_005': float((distance < 0.05).sum() / denominator),
                'pose_joint_recall': float(count / denominator), 'pose_matched_joints': count}


class IdentityMetric:
    def __init__(self, models_root: str | Path | None = None) -> None:
        from insightface import model_zoo
        from insightface.utils import face_align
        root = Path(models_root) / 'antelopev2' if models_root is not None else ROOT / 'models/antelopev2'
        self.face_align = face_align
        self.det = model_zoo.get_model(str(root / 'scrfd_10g_bnkps.onnx'), providers=['CPUExecutionProvider'])
        self.recognition = model_zoo.get_model(str(root / 'glintr100.onnx'), providers=['CPUExecutionProvider'])
        self.det.prepare(ctx_id=-1, input_size=(640, 640))
        self.recognition.prepare(ctx_id=-1)

    def embedding(self, image: Image.Image) -> np.ndarray | None:
        import cv2
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        boxes, landmarks = self.det.detect(bgr)
        if len(boxes) == 0:
            return None
        index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        crop = self.face_align.norm_crop(bgr, landmark=landmarks[index])
        embedding = self.recognition.get_feat(crop).reshape(-1).astype(np.float32)
        return embedding / max(float(np.linalg.norm(embedding)), 1e-12)

    @staticmethod
    def cosine(source: np.ndarray | None, generated: np.ndarray | None) -> float | None:
        return float(source @ generated) if source is not None and generated is not None else None
