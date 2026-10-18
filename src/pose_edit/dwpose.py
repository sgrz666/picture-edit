"""One home for the DWPose ONNX call convention: person boxes in, 134-slot landmarks out.

Both pipelines need this: the two-person evaluator re-detects the generated image, and the
single-person line now measures A's own torso instead of assuming a constant fraction of A's box
height. Keeping two copies of the convention is how the 133-vs-134 slot bug happened once already,
so both call sites go through here.

Conventions recorded once, here:
  * ONNX sessions run on CPU (the geometry stage already holds CUDA memory for the estimator).
  * ``inference_detector`` in this build returns xyxy boxes only, having dropped candidates under
    its own 0.3 threshold and kept class 0.
  * The model emits COCO-WholeBody **133** points; this repo renders and scores the **134**-slot
    OpenPose-compatible layout with a neck inserted at slot 1, via ``cocowholebody133_to_dwpose134``.
  * Slot 1 (neck) is a derived point, so its score is defined as the weaker of the two shoulder
    scores rather than something the model reported.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.evaluation.dwpose_onnx.onnxpose import inference_pose
from src.pose_edit.geometry import cocowholebody133_to_dwpose134, torso_length

# Slot 1 of the 134 layout is inserted, not measured; its confidence is the weaker shoulder.
NECK_FROM_SLOTS = (5, 6)
NECK_SLOT = 1


def load_sessions(config):
    """Detection and pose sessions for the paths named in ``config``."""
    return (ort.InferenceSession(config['dwpose_det_model'], providers=['CPUExecutionProvider']),
        ort.InferenceSession(config['dwpose_pose_model'], providers=['CPUExecutionProvider']))


def convert_person(raw_points, raw_scores):
    """One model output (133 slots) to this repo's 134-slot points and scores."""
    raw_points = np.asarray(raw_points, dtype=float)
    raw_scores = np.asarray(raw_scores, dtype=float)
    scores = cocowholebody133_to_dwpose134(raw_scores[:, None])[:, 0].copy()
    scores[NECK_SLOT] = float(min(raw_scores[NECK_FROM_SLOTS[0]], raw_scores[NECK_FROM_SLOTS[1]]))
    return cocowholebody133_to_dwpose134(raw_points), scores


def detect_people(config, image_bgr, sessions=None):
    """One entry per detected person: ``box_xyxy``, 134 ``points`` and their ``scores``."""
    det_session, pose_session = sessions if sessions is not None else load_sessions(config)
    detected = inference_detector(det_session, image_bgr)
    people = []
    if not len(detected):
        return people
    points, scores = inference_pose(pose_session, detected.copy(), image_bgr)
    for index in range(len(detected)):
        converted, slot_scores = convert_person(points[index], scores[index])
        people.append({'box_xyxy': [float(value) for value in detected[index][:4]],
            'points': converted, 'scores': slot_scores})
    return people


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return overlap / (area_a + area_b - overlap)


def pick_person(people, box_xyxy):
    """The detection that best matches a known box, or None when nothing overlaps it.

    Used when the subject box comes from another detector (the single-person line knows A's YOLOv8
    box and only needs A's landmarks): guessing by index would track detection order, which is not
    stable across runs.
    """
    ranked = sorted(((box_iou(person['box_xyxy'], box_xyxy), person) for person in people),
        key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] <= 0.0:
        return None
    return ranked[0][1]


def measure_torso(config, image_bgr, box_xyxy, sessions=None):
    """Neck-to-hip-centre length of the person inside ``box_xyxy``, or None when unmeasurable."""
    person = pick_person(detect_people(config, image_bgr, sessions), box_xyxy)
    if person is None:
        return None
    value = torso_length(person['points'])
    return float(value) if np.isfinite(value) and value > 0 else None
