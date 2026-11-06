"""Offline GT supervision regions. No model loading, inference or file writes."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.data.canvas_contract import CanvasSpec, native_pad

BODY_JOINTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
HAND_EDGES = tuple((start + k, start + k + 1) for start in (1, 5, 9, 13, 17) for k in range(3)) + tuple((0, i) for i in (1, 5, 9, 13, 17))


def canvas_for(original_hw, long_side=512):
    """Same integer resize/pad policy as the existing HIA/Bilibili assembler."""
    hw = tuple(int(v) for v in original_hw)
    if long_side == 0:
        return native_pad(hw)
    if long_side < 32 or long_side % 32:
        raise ValueError('long_side must be 0 or a positive multiple of 32')
    scale = min(1., long_side / max(hw))
    resized = tuple(max(1, round(v * scale)) for v in hw)
    canvas = tuple(math.ceil(v / 32) * 32 for v in resized)
    offset = ((canvas[1] - resized[1]) // 2, (canvas[0] - resized[0]) // 2)
    return CanvasSpec(hw, resized, canvas, offset, policy='aspect_preserving')


def valid_keypoints(frame):
    points = np.asarray(frame['candidate'], dtype=np.float32)
    score = np.asarray(frame.get('effective_confidence', frame.get('confidence')), dtype=np.float32)
    declared = np.asarray(frame.get('valid_keypoint_mask', [True] * len(points)), dtype=bool)
    if points.shape != (134, 2) or score.shape != (134,) or declared.shape != (134,):
        raise ValueError('DWPose 134 original-pixel points and explicit scores required')
    h, w = frame['original_hw']
    valid = declared & np.isfinite(points).all(1) & np.isfinite(score)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    return points, np.clip(np.nan_to_num(score), 0., 1.), valid


def build_prompts(frame, confidence=.3):
    """Existing person detection is a SAM prompt, never the resulting mask."""
    points, score, valid = valid_keypoints(frame)
    good = valid[:18] & (score[:18] >= confidence)
    positive = points[:18][good]
    if len(positive) < 3:
        raise ValueError('Fewer than three valid body points for SAM prompting')
    h, w = frame['original_hw']
    box = frame.get('detection', {}).get('primary_box')
    if box is None:
        box = [*positive.min(0), *positive.max(0)]
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError('Invalid original-pixel person detection box')
    size = box[2:] - box[:2]
    box = np.concatenate((box[:2] - .08 * size, box[2:] + .08 * size))
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
    return positive.copy(), np.ones(len(positive), dtype=np.int32), box


def canvas_mask(mask, spec):
    value = np.asarray(mask, dtype=np.float32)
    if value.shape != spec.original_hw or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError('Body mask shape/range differs from original image')
    image = Image.fromarray(value)
    if spec.resized_hw != spec.original_hw:
        image = image.resize(spec.resized_hw[::-1], Image.Resampling.BOX)
    out = np.zeros(spec.canvas_hw, dtype=np.float32)
    x, y = spec.offset_xy; h, w = spec.resized_hw
    out[y:y+h, x:x+w] = np.asarray(image)
    return out


def _gaussian(shape, point, sigma):
    y, x = np.ogrid[:shape[0], :shape[1]]
    distance = ((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * sigma ** 2)
    return np.where(distance <= 4.5, np.exp(-distance), 0).astype(np.float32)


def _hull(points):
    pts = sorted(set(tuple(float(v) for v in p) for p in points))
    if len(pts) < 3:
        return pts
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hand_region(frame, key, spec):
    zero = np.zeros(spec.canvas_hw, np.float32)
    mask = frame.get('loss_mask', {})
    raw = frame.get('hand_raw_pose_usable', {})
    if not mask.get(key, False) or not raw.get(key, False):
        return zero, dict(enabled=False, reason='existing_hand_supervision_gate')
    hand = np.asarray(frame.get('hand_landmarks21', {}).get(key, []), dtype=np.float32)
    if hand.shape != (21, 2):
        return zero, dict(enabled=False, reason='missing_original_hand21')
    h, w = spec.original_hw
    good = np.isfinite(hand).all(1) & (hand[:, 0] >= 0) & (hand[:, 0] < w) & (hand[:, 1] >= 0) & (hand[:, 1] < h)
    if good.sum() < 15 or not good[0]:
        return zero, dict(enabled=False, reason='hand_cropped_or_nonfinite')
    points = spec.pixel_points(hand)
    palm_indices = np.array([0, 1, 5, 9, 13, 17])
    palm = points[palm_indices[good[palm_indices]]]
    hull = _hull(palm)
    if len(hull) < 3:
        return zero, dict(enabled=False, reason='degenerate_palm')
    extent = np.ptp(points[good], axis=0)
    if max(extent) < 1 or max(extent) > .6 * max(spec.resized_hw):
        return zero, dict(enabled=False, reason='implausible_hand_extent')
    radius = float(max(1.5, min(5., .07 * max(extent))))
    image = Image.new('L', spec.canvas_hw[::-1]); draw = ImageDraw.Draw(image)
    draw.polygon(hull, fill=255)
    for a, b in HAND_EDGES:
        if good[a] and good[b]:
            draw.line([tuple(points[a]), tuple(points[b])], fill=255, width=max(2, round(radius*2)))
    for x, y in points[good]:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    value = np.asarray(image.filter(ImageFilter.GaussianBlur(max(.7, radius*.4))), dtype=np.float32) / 255.
    return value, dict(enabled=True, reason='existing_loss_and_raw_hand_gates', source=frame.get('hand_landmarks_source'), valid_points=int(good.sum()))


def regional_maps(frame, body_original, spec, *, confidence=.3, joint_sigma_fraction=.025):
    if tuple(frame['original_hw']) != spec.original_hw:
        raise ValueError('Frame original shape differs from CanvasSpec')
    body = canvas_mask(body_original, spec)
    # Feather only the boundary of SAM's observed foreground, not an inferred mesh.
    body = np.asarray(Image.fromarray(np.uint8(body*255)).filter(ImageFilter.GaussianBlur(.8)), dtype=np.float32) / 255.
    valid = spec.valid_mask()[0].numpy()
    body *= valid
    points, scores, good = valid_keypoints(frame)
    points = spec.pixel_points(points)
    box = frame.get('detection', {}).get('primary_box')
    if box is None:
        original,_,visible=valid_keypoints(frame)
        box=([*original[:18][visible[:18]].min(0),*original[:18][visible[:18]].max(0)]
             if visible[:18].any() else [0,0,spec.original_hw[1],spec.original_hw[0]])
    height = max(1., (box[3]-box[1]) * spec.scale_xy[1])
    sigma = float(np.clip(joint_sigma_fraction * height, 8., 24.))
    joint = np.zeros(spec.canvas_hw, np.float32)
    used = []
    if frame.get('loss_mask', {}).get('body', True):
        for i in BODY_JOINTS:
            if good[i] and scores[i] >= confidence:
                joint = np.maximum(joint, scores[i] * _gaussian(spec.canvas_hw, points[i], sigma))
                used.append(i)
    hands, statuses = [], {}
    for key in ('hand_a', 'hand_b'):
        value, statuses[key] = _hand_region(frame, key, spec)
        hands.append(value)
    hand = np.maximum(*hands) * valid
    return dict(body=body, joint=joint*valid, hand=hand, valid=valid), dict(joint_ids=used, joint_sigma_canvas_px=sigma, hands=statuses)
