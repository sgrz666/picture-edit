"""SDPose/COCO17 pose scoring and CPU glintr100 identity scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COCO_TO_BODY25 = (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


class PoseMetric:
    def __init__(self, models_root: str | Path | None = None, *, device: str = 'cpu') -> None:
        from .sdpose import load_scoring_runtime

        # Keep the common evaluator constructor used by existing callers. SDPose
        # assets are pinned separately from the AntelopeV2/models_root directory.
        self.runtime = load_scoring_runtime(device=device)

    def detect_all(self, image: Image.Image):
        import cv2
        from .sdpose import score_people

        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        return score_people(bgr, self.runtime)

    def detect(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray] | None:
        from .sdpose import coco17

        people = self.detect_all(image)
        return coco17(people[0]) if people else None

    @staticmethod
    def compare(detected: tuple[np.ndarray, np.ndarray] | None, target: np.ndarray,
                image_size: tuple[int, int] | None = None) -> dict[str, float | bool | None]:
        if detected is None:
            return {'pose_detected': False, 'pose_nme_diagonal': None, 'pose_pck_002': 0.0, 'pose_pck_005': 0.0, 'pose_joint_recall': 0.0, 'pose_matched_joints': 0}
        from .sdpose import valid_joints

        points, scores = detected
        target_coco = target[np.array(COCO_TO_BODY25)]
        target_visible = target_coco[:, 2] > 0.3
        w, h = (512.0, 512.0) if image_size is None else (float(image_size[0]), float(image_size[1]))
        matched = target_visible & valid_joints(points, scores, w, h)
        count = int(matched.sum())
        denominator = int(target_visible.sum())
        if denominator == 0:
            return {'pose_detected': True, 'pose_nme_diagonal': None, 'pose_pck_002': None, 'pose_pck_005': None, 'pose_joint_recall': None, 'pose_matched_joints': 0}
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
