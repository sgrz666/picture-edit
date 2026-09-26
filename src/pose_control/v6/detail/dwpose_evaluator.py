import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Ensure onnxruntime and dwpose can be imported
ORT_SITE = "/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/lib/python3.12/site-packages"
if ORT_SITE not in sys.path and Path(ORT_SITE).exists():
    sys.path.insert(0, ORT_SITE)

DWROOT = Path("/home/shangguanrz/project/pic-edit/add/tools/dwpose_mimicmotion_official")
if str(DWROOT) not in sys.path and DWROOT.exists():
    sys.path.insert(0, str(DWROOT))


class DWPoseEvaluator:
    """Wrapper around DWPose wholebody detector for evaluation of face NME and hand PCK."""

    def __init__(self, device: str = "cpu"):
        det_model = str(DWROOT / "models/yolox_l.onnx")
        pose_model = str(DWROOT / "models/dw-ll_ucoco_384.onnx")
        from dwpose.wholebody import Wholebody
        self.estimator = Wholebody(det_model, pose_model, device=device)

    def detect(self, img_bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run DWPose wholebody on an image.

        Returns:
            keypoints: [134, 2] in pixel coordinates or None if no person detected
            scores: [134] confidence scores or None
        """
        if is_rgb:
            img_bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_bgr_or_rgb

        kps, scs = self.estimator(img_bgr)
        if kps is None or len(kps) == 0:
            return None, None

        # Select person with highest body confidence
        if kps.shape[0] > 1:
            body_conf = scs[:, :18].mean(axis=1)
            best_idx = int(np.argmax(body_conf))
            return kps[best_idx].copy(), scs[best_idx].copy()

        return kps[0].copy(), scs[0].copy()
