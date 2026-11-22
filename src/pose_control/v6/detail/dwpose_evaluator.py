"""Legacy import path for the shared SDPose face and hand keypoints."""

import cv2
import numpy as np

from src.evaluation.sdpose import load_scoring_runtime, score_people


class SDPoseEvaluator:
    def __init__(self, device: str = 'cpu'):
        self.runtime = load_scoring_runtime(device=device)

    def detect(self, img_bgr_or_rgb: np.ndarray, is_rgb: bool = True):
        """Return the highest-confidence body's geometry-134 points and scores."""
        bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR) if is_rgb else img_bgr_or_rgb
        people = score_people(bgr, self.runtime)
        if not people:
            return None, None
        person = max(people, key=lambda item: item['scores'][:18].mean())
        return person['points'].copy(), person['scores'].copy()


DWPoseEvaluator = SDPoseEvaluator
