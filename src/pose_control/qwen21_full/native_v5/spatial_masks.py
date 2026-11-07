"""Offline GT supervision regions. No model loading, inference or file writes."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.data.canvas_contract import CanvasSpec, native_pad

BODY_JOINTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
HAND_EDGES = tuple((start + k, start + k + 1) for start in (1, 5, 9, 13, 17) for k in range(3)) + tuple((0, i) for i in (1, 5, 9, 13, 17))
HAND_POLICIES = ('legacy_strict', 'visible_region_v2')
COARSE_HAND_GAIN = .65


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


def _hand_presence(frame,key):
    value=frame.get(key+'_present')
    return frame.get('hand_presence',{}).get(key,{}).get('present') if value is None else value


def _coarse_hand_region(frame, key, spec, body):
    """Visible GT region, not a source of finger-coordinate supervision."""
    zero=np.zeros(spec.canvas_hw,np.float32)
    def reject(reason,**evidence):
        return zero,dict(enabled=False,reason=reason,**evidence)
    if frame.get(key+'_status') in ('occluded','out_of_frame'):
        return reject('explicit_occluded_or_out_of_frame')
    present=_hand_presence(frame,key)
    if not present:
        return reject('hand_not_present')
    block=frame.get('hand_block_slices',{}).get(key)
    wrist=frame.get('hand_body_wrist_indices',{}).get(key)
    if (not isinstance(block,(list,tuple)) or len(block)!=2 or
            any(type(i)!=int for i in block) or block[1]-block[0]!=21 or not 18<=block[0]<block[1]<=134 or
            type(wrist)!=int or not 0<=wrist<18):
        return reject('missing_explicit_dwpose_hand_mapping')
    raw,score,visible=valid_keypoints(frame)
    ids=np.arange(*block);good=visible[ids]&(score[ids]>=.35)
    evidence=dict(valid_points=int(good.sum()),mean_score=float(score[ids][good].mean()) if good.any() else 0.)
    if good.sum()<8 or not good[0] or evidence['mean_score']<.45:
        return reject('insufficient_confident_dwpose_points',**evidence)
    if not visible[wrist] or score[wrist]<.3:
        return reject('body_wrist_unreliable',**evidence)
    points=spec.pixel_points(raw[ids]);anchor=spec.pixel_points(raw[wrist:wrist+1])[0]
    extent=np.ptp(points[good],axis=0);span=float(max(extent))
    box=frame.get('detection',{}).get('primary_box')
    person_height=float((box[3]-box[1])*spec.scale_xy[1]) if box is not None else float(spec.resized_hw[0])
    if span<1 or span>.3*person_height:
        return reject('implausible_dwpose_hand_extent',span_canvas_px=span,**evidence)
    root_distance=float(np.linalg.norm(points[0]-anchor))
    if root_distance>max(6.,.4*span):
        return reject('dwpose_hand_body_wrist_mismatch',root_distance_canvas_px=root_distance,**evidence)
    # A small halo tolerates segmentation edges, but does not turn arbitrary
    # background detections into high-weight regions.
    foreground=np.asarray(Image.fromarray(np.uint8(body*255)).filter(ImageFilter.MaxFilter(5)),dtype=np.float32)/255.
    xy=np.rint(points[good]).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,spec.canvas_hw[1]-1);xy[:,1]=np.clip(xy[:,1],0,spec.canvas_hw[0]-1)
    fraction=float((foreground[xy[:,1],xy[:,0]]>.1).mean())
    if fraction<.6:
        return reject('dwpose_hand_outside_gt_foreground',foreground_point_fraction=fraction,**evidence)
    hull=_hull(points[good])
    if len(hull)<3:
        return reject('degenerate_dwpose_hand_hull',**evidence)
    radius=float(np.clip(.12*span,2.5,6.))
    image=Image.new('L',spec.canvas_hw[::-1]);draw=ImageDraw.Draw(image)
    draw.polygon(hull,fill=255)
    for x,y in points[good]:
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=255)
    value=np.asarray(image.filter(ImageFilter.GaussianBlur(float(max(.8,radius*.35)))),dtype=np.float32)/255.
    value*=foreground*COARSE_HAND_GAIN
    return value,dict(enabled=bool(value.any()),reason='visible_confident_dwpose_region',
                      source='dwpose134',gain=COARSE_HAND_GAIN,foreground_point_fraction=fraction,
                      root_distance_canvas_px=root_distance,**evidence)


def visible_hand_region(frame,key,spec,body):
    coarse,coarse_status=_coarse_hand_region(frame,key,spec,body)
    fine,fine_status=_hand_region(frame,key,spec)
    # Explicit visibility negatives also protect the fine path from contradictory
    # upstream trainability flags. No original annotation is modified.
    if frame.get(key+'_status') in ('occluded','out_of_frame') or _hand_presence(frame,key) is False:
        fine=np.zeros_like(fine);fine_status=dict(enabled=False,reason='explicit_visibility_negative')
    combined=np.maximum(coarse,fine)
    return combined,dict(enabled=bool(combined.any()),reason='visible_region_v2',coarse=coarse_status,fine=fine_status)


def regional_maps(frame, body_original, spec, *, confidence=.3, joint_sigma_fraction=.025, hand_policy='legacy_strict'):
    if hand_policy not in HAND_POLICIES:
        raise ValueError('Unknown explicit hand region policy')
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
        value, statuses[key] = (visible_hand_region(frame,key,spec,body) if hand_policy=='visible_region_v2'
                                else _hand_region(frame,key,spec))
        hands.append(value)
    hand = np.maximum(*hands) * valid
    return dict(body=body, joint=joint*valid, hand=hand, valid=valid), dict(joint_ids=used, joint_sigma_canvas_px=sigma, hands=statuses)
