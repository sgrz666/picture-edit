"""Ground-truth-free metrics for the single-person pose-transfer workflow.

Every metric here answers one of two questions without an edited-image reference:
did the requested pose land (checked by a second, architecturally different estimator),
and did everything that should have stayed the same stay the same (identity, body shape,
background, other people).

Design rules this module follows:
- Heavy third-party models are imported lazily inside the method that needs them, so the
  math helpers stay importable and unit-testable on a machine without ONNX or InsightFace.
- The DWPose-134 layout from ``src.pose_edit.geometry`` is the only keypoint convention
  used here; ``cocowholebody133_to_dwpose134`` converts raw detector output.
- Nothing is silently zeroed. A joint the detector could not see is reported through
  ``joints_compared`` / ``joints_excluded_ratio``; an unavailable metric is ``None`` with a
  reason in the caller, because ``common.atomic_json`` refuses NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GROUPS = {'body': slice(0, 18), 'feet': slice(18, 24), 'face': slice(24, 92),
    'left_hand': slice(92, 113), 'right_hand': slice(113, 134)}
PCK_THRESHOLDS = (0.2, 0.5)
COCO_WHOLEBODY_133 = 133
DWSLOT_COUNT = 134


def num(value):
    """Finite float or None, so json.dumps(allow_nan=False) never sees NaN."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


_num = num


def points134(array):
    """Validate a (134, 2) keypoint array without copying when it already is one."""
    points = np.asarray(array, dtype=float)
    if points.ndim != 2 or points.shape != (DWSLOT_COUNT, 2):
        raise ValueError(f'expected (134, 2) DWPose points, got {points.shape}')
    return points


def validity(points, width, height):
    """Finite and inside the canvas, matching src.pose_edit.geometry.valid_points."""
    finite = np.isfinite(points).all(axis=1)
    inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (
        points[:, 1] < height)
    return finite & inside


def torso_px(points134_array):
    """Neck-to-hip-centre length in pixels; the normalisation the workflow already uses."""
    points = points134(points134_array)
    hips = np.nanmean(points[[8, 11]], axis=0)
    if not np.isfinite(points[1]).all() or not np.isfinite(hips).all():
        return None
    length = float(np.linalg.norm(points[1] - hips))
    return length if length > 1.0 else None


def joint_pairs(points134_array):
    """Bones available in the DWPose-134 layout, as (name, joint_a, joint_b)."""
    return (
        ('neck_head', 1, 0),
        ('shoulder_width', 2, 5), ('hip_width', 8, 11),
        ('right_upper_arm', 2, 3), ('right_forearm', 3, 4),
        ('left_upper_arm', 5, 6), ('left_forearm', 6, 7),
        ('right_thigh', 8, 9), ('right_shank', 9, 10),
        ('left_thigh', 11, 12), ('left_shank', 12, 13),
        ('torso', 1, 8), ('torso_left', 1, 11),
    )


def limb_ratios(points134_array):
    """Body proportions that a pose change must not alter.

    Ratios are dimensionless, so a generated frame can be compared against the source even
    though the body moved, resized and was re-cropped. They catch what PCK hides: a leg
    stretched 30% longer still scores well if every joint landed on the requested skeleton.

    Two families are kept apart because they behave differently under 2D projection:

    - ``invariant``: one bone over its own neighbour (forearm/upper arm, shank/thigh) and
      arm-chain over leg-chain. Both members of a ratio live on the same limb, so
      foreshortening shrinks numerator and denominator together and the ratio survives.
    - ``view_sensitive``: widths across the body and torso over a leg. A quarter turn of the
      torso collapses shoulder width while limb lengths barely move, so these change with
      the camera, not with the person. They are reported as diagnostics and never enter the
      drift headline.
    """
    points = points134(points134_array)
    lengths = {}
    for name, a, b in joint_pairs(points):
        if not np.isfinite(points[a]).all() or not np.isfinite(points[b]).all():
            continue
        lengths[name] = float(np.linalg.norm(points[a] - points[b]))
    invariant = {
        'right_forearm_over_upper_arm': ('right_forearm', 'right_upper_arm'),
        'left_forearm_over_upper_arm': ('left_forearm', 'left_upper_arm'),
        'right_shank_over_thigh': ('right_shank', 'right_thigh'),
        'left_shank_over_thigh': ('left_shank', 'left_thigh'),
    }
    view_sensitive = {
        'shoulder_over_hip_width': ('shoulder_width', 'hip_width'),
        'torso_over_leg_right': ('torso', 'right_thigh'),
        'torso_over_leg_left': ('torso_left', 'left_thigh'),
        'neck_over_torso': ('neck_head', 'torso'),
        # Cross-limb: bending or turning the arms alone foreshortens the arm chain while the
        # leg chain is untouched, so this reads as a body-shape change when it is not.
        'arm_over_leg': ('arm_chain', 'leg_chain'),
    }

    def chain(side, segment_a, segment_b):
        first, second = lengths.get(segment_a), lengths.get(segment_b)
        return None if first is None or second is None else first + second

    arms = [value for value in (chain('r', 'right_upper_arm', 'right_forearm'),
        chain('l', 'left_upper_arm', 'left_forearm')) if value]
    legs = [value for value in (chain('r', 'right_thigh', 'right_shank'),
        chain('l', 'left_thigh', 'left_shank')) if value]
    if arms and legs:
        lengths['arm_chain'] = float(np.mean(arms))
        lengths['leg_chain'] = float(np.mean(legs))
        view_sensitive['arm_over_leg'] = ('arm_chain', 'leg_chain')

    def build(spec):
        out = {}
        for key, (numerator, denominator) in spec.items():
            if numerator in lengths and denominator in lengths and lengths[denominator] > 1.0:
                out[key] = _num(lengths[numerator] / lengths[denominator])
        return out

    return {'lengths_px': {k: _num(v) for k, v in sorted(lengths.items())},
        'invariant': build(invariant), 'view_sensitive': build(view_sensitive),
        'families': {'invariant': sorted(invariant), 'view_sensitive': sorted(view_sensitive)}}


def _drift_over(source_ratios, generated_ratios):
    rows = {}
    for key in sorted(set(source_ratios) & set(generated_ratios)):
        before, after = source_ratios[key], generated_ratios[key]
        if before is None or after is None or before <= 0:
            continue
        rows[key] = _num(abs(after - before) / before)
    values = {key: value for key, value in rows.items() if value is not None}
    if not values:
        return {'ratios_compared': 0, 'mean_rel_change': None, 'max_rel_change': None,
            'worst_ratio': None, 'per_ratio': {}}
    worst = max(values, key=values.get)
    return {'ratios_compared': len(values), 'mean_rel_change': _num(np.mean(list(values.values()))),
        'max_rel_change': _num(np.max(list(values.values()))), 'worst_ratio': worst,
        'per_ratio': rows}


def ratio_drift(source, generated):
    """Relative change of each proportion between source and generated frame.

    The headline numbers come from the pose-invariant family only; the view-sensitive family
    is reported beside it so a reviewer can see a torso turn rather than mistake it for a
    stretched limb.
    """
    source = source or {}
    generated = generated or {}
    out = {}
    for family in ('invariant', 'view_sensitive'):
        block = _drift_over(source.get(family) or {}, generated.get(family) or {})
        prefix = '' if family == 'invariant' else 'view_'
        out[f'{prefix}ratios_compared'] = block['ratios_compared']
        out[f'{prefix}mean_rel_change'] = block['mean_rel_change']
        out[f'{prefix}max_rel_change'] = block['max_rel_change']
        out[f'{prefix}worst_ratio'] = block['worst_ratio']
        out[f'{prefix}per_ratio'] = block['per_ratio']
    return out


def scale_points(points, factor):
    """Map keypoints (N,2) or boxes (N,4) detected at one resolution into another.

    The source photo and the generated frame are different sizes: DWPose runs on
    ``A_rgb.png`` natively while the generated image is the letterbox content region. Any
    mask, torso length or displacement computed across the two frames must pass through this
    step first, or it compares two scales and reports preservation error that does not exist.
    """
    array = np.asarray(points, dtype=float)
    scale = np.asarray(factor, dtype=float).reshape(2)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f'factor must be two positive finite numbers, got {scale}')
    if array.size == 0:
        return array
    if array.ndim == 2 and array.shape[1] == 4:
        scaled = array.copy()
        scaled[:, :2] = scaled[:, :2] * scale
        scaled[:, 2:] = scaled[:, 2:] * scale
        return scaled
    if array.ndim == 2 and array.shape[1] == 2:
        return array * scale
    raise ValueError(f'unsupported shape {array.shape}; expected (N, 2) or (N, 4)')


def pose_error_stats(guide, guide_valid, estimate, estimate_valid, size, normaliser=None):
    """Guide-versus-estimate joint errors, with the out-of-frame accounting made explicit.

    ``evaluate.py`` drops guide joints whose estimate left the frame instead of scoring them
    as errors, which inflates PCK when the person is cropped. Both readings are returned:
    ``pck_*`` on compared joints, ``pck_*_strict`` charging every dropped joint as a miss,
    plus ``joints_excluded_ratio`` so the two can be told apart downstream.
    """
    guide, estimate = points134(guide), points134(estimate)
    width, height = int(size[0]), int(size[1])
    guide_valid = np.asarray(guide_valid, dtype=bool)
    compared = guide_valid & np.asarray(estimate_valid, dtype=bool)
    requested = int(guide_valid.sum())
    if requested == 0:
        return {'joints_requested': 0, 'joints_compared': 0, 'joints_excluded_ratio': None,
            'torso_normalisation_px': None, 'groups': {}, 'summary': {}}
    distances = np.linalg.norm(guide - estimate, axis=1)
    distances = np.where(compared, distances, np.nan)
    scale = normaliser if normaliser else torso_px(guide)
    if not scale:
        scale = float(np.hypot(width, height)) / 4.0
    groups = {}
    for name, block in GROUPS.items():
        errors = distances[block]
        finite = errors[np.isfinite(errors)]
        group_requested = int(guide_valid[block].sum())
        if finite.size == 0:
            groups[name] = {'joints': 0, 'joints_requested': group_requested,
                'joints_dropped': group_requested}
            continue
        normalised = finite / scale
        groups[name] = {
            'joints': int(finite.size),
            'joints_requested': group_requested,
            'joints_dropped': int(group_requested - finite.size),
            'mpjpe_px': _num(finite.mean()),
            'mpjpe_norm': _num(normalised.mean()),
            'p95_norm': _num(np.percentile(normalised, 95)),
            **{f'pck@{threshold}': _num(float((normalised <= threshold).mean()))
                for threshold in PCK_THRESHOLDS},
            # Strict variant: a requested joint the estimator could not place inside the frame
            # counts as a miss, so a person cropped out of the picture cannot score well. The
            # denominator is the joints this group requested, never the whole 134-point body.
            **{f'pck@{threshold}_strict': _num(float((normalised <= threshold).sum())
                / group_requested) for threshold in PCK_THRESHOLDS if group_requested},
            'worst_joint_offset_px': _num(finite.max())}
    body = groups['body']
    summary = {'body_pck@0.2': body.get('pck@0.2'), 'body_pck@0.2_strict': body.get(
        'pck@0.2_strict'), 'body_mpjpe_norm': body.get('mpjpe_norm'),
        'face_mpjpe_norm': groups['face'].get('mpjpe_norm'),
        'hand_mpjpe_norm': _num(np.mean([groups[key].get('mpjpe_norm')
            for key in ('left_hand', 'right_hand')
            if groups[key].get('mpjpe_norm') is not None] or [np.nan]))}
    return {'final_size': [width, height], 'torso_normalisation_px': _num(scale),
        'joints_requested': requested, 'joints_compared': int(compared.sum()),
        'joints_dropped_by_estimator': int(requested - compared.sum()),
        'joints_excluded_ratio': _num(1.0 - int(compared.sum()) / requested),
        'groups': groups, 'summary': summary}


def estimator_agreement(smplx_points, dwpose_points, size, normaliser=None):
    """Do two different estimators agree on where this generated body is?

    The workflow scores pose with the same SMPLest-X that produced the guide, so estimator
    bias cancels into both sides of the comparison. DWPose is a top-down 2D detector with a
    different failure mode; disagreement between the two on the same image is the only cheap
    signal that the pose number itself is not trustworthy.
    """
    left, right = points134(smplx_points), points134(dwpose_points)
    width, height = int(size[0]), int(size[1])
    valid = validity(left, width, height) & validity(right, width, height)
    scale = normaliser if normaliser else torso_px(left)
    if not scale:
        scale = float(np.hypot(width, height)) / 4.0
    out = {'joints_compared': int(valid.sum()), 'normaliser_px': _num(scale), 'groups': {}}
    if int(valid.sum()) == 0:
        out['mean_norm_distance'] = None
        return out
    distance = np.linalg.norm(left - right, axis=1)
    distance = np.where(valid, distance, np.nan)
    for name, block in GROUPS.items():
        finite = distance[block][np.isfinite(distance[block])]
        if finite.size == 0:
            out['groups'][name] = {'joints': 0}
            continue
        normalised = finite / scale
        out['groups'][name] = {'joints': int(finite.size),
            'mean_norm_distance': _num(normalised.mean()),
            'agree@0.1': _num(float((normalised <= 0.1).mean())),
            'agree@0.2': _num(float((normalised <= 0.2).mean()))}
    body = out['groups']['body']
    out['body_mean_norm_distance'] = body.get('mean_norm_distance')
    out['body_agree@0.2'] = body.get('agree@0.2')
    out['verdict'] = ('consistent' if (body.get('agree@0.2') or 0) >= 0.8
        else 'review' if (body.get('agree@0.2') or 0) >= 0.5 else 'conflict')
    return out


def mask_from_mesh(mesh_cam, focal_xy, principal_xy, faces, size):
    """Rasterise the fitted SMPL-X mesh into a person mask for the estimated image.

    ``mesh_cam`` is the camera-space vertex array SMPLest-X exports, ``faces`` comes from
    SMPLX_NEUTRAL. Projection reuses the estimator's ``Z + 1e-4`` convention so the mask
    lands exactly where the skeleton does.
    """
    import cv2

    from src.pose_edit.geometry import project_camera

    vertices = np.asarray(mesh_cam, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32)
    projected = project_camera(vertices, np.asarray(focal_xy, dtype=float),
        np.asarray(principal_xy, dtype=float))
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    finite = np.isfinite(projected).all(axis=1)
    for triangle in faces:
        corners = projected[triangle]
        if not np.isfinite(corners).all():
            continue
        cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    return mask.astype(bool)


def mask_from_limbs(points134_array, size, torso=None):
    """Approximate person mask from the skeleton when no mesh is available.

    Limbs are drawn with a width derived from the torso, which is coarse but keeps the
    background region conservative: it over-covers the body rather than leaking body pixels
    into the background metric. Callers must record ``mask_kind`` because the two masks are
    not interchangeable.
    """
    from PIL import Image, ImageDraw

    points = points134(points134_array)
    width, height = int(size[0]), int(size[1])
    scale = torso if torso else torso_px(points)
    if not scale:
        scale = float(np.hypot(width, height)) / 40.0
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    edges = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13), (2, 5), (8, 11), (0, 1), (10, 18), (13, 20), (10, 21),
        (13, 22)]
    stroke = max(3, int(round(0.55 * scale)))
    valid = validity(points, width, height)
    for a, b in edges:
        if a < len(points) and b < len(points) and valid[a] and valid[b]:
            draw.line((tuple(points[a]), tuple(points[b])), fill=1, width=stroke)
    radius = max(2, int(round(0.3 * scale)))
    for index in (0, 8, 11, 10, 13):
        if valid[index]:
            x, y = points[index]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    return np.asarray(canvas, dtype=bool)


def dilate(mask, pixels):
    """Grow a mask so thin limbs and hair do not leak into the 'background' region."""
    import cv2

    radius = max(1, int(round(pixels)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def region_metrics(source, generated, mask_source, mask_generated, ssim_function=None):
    """Preservation numbers measured against the source image, not against a GT edit.

    ``source`` and ``generated`` are float RGB arrays of identical shape. Background is
    everything outside the union of both person masks, exactly the convention
    ``scripts/eval/eval_iper_four_arm.py`` records as ``background_mae_source``; the person
    region is that union, deliberately loose because the body moved between the two frames.
    """
    source = np.asarray(source, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if source.shape != generated.shape:
        raise ValueError(f'source {source.shape} and generated {generated.shape} must match')
    both_outside = ~(mask_source | mask_generated)
    either_inside = mask_source | mask_generated
    difference = np.abs(source - generated)
    out = {}
    for name, region in (('background', both_outside), ('person', either_inside)):
        pixels = int(region.sum())
        if pixels == 0:
            out[f'{name}_mae'] = None
            out[f'{name}_pixels'] = 0
            continue
        out[f'{name}_mae'] = _num(difference[region].mean())
        out[f'{name}_pixels'] = pixels
        out[f'{name}_mse'] = _num((difference[region] ** 2).mean())
        squared = out[f'{name}_mse']
        out[f'{name}_psnr'] = _num(10.0 * np.log10(255.0 ** 2 / squared)) if squared and squared > 0 else None
    if ssim_function is not None:
        for name, region in (('background', both_outside), ('person', either_inside)):
            if int(region.sum()) == 0:
                out[f'{name}_ssim'] = None
                continue
            window = _bounding_box(region)
            if window is None:
                continue
            crop_source = _crop(source, window)
            crop_generated = _crop(generated, window)
            out[f'{name}_ssim'] = _num(ssim_function(crop_source, crop_generated,
                data_range=255, channel_axis=2))
    return out


def _bounding_box(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _crop(array, window):
    top, bottom, left, right = window
    return array[top:bottom, left:right]


def beta_distance(betas_source, betas_generated, standard_deviation=None):
    """SMPL-X shape parameters are pose-invariant, so they proxy body and garment volume."""
    left = np.asarray(betas_source, dtype=float).reshape(-1)
    right = np.asarray(betas_generated, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError('betas must have the same length')
    delta = right - left
    out = {'dimensions': int(left.size), 'beta_l2': _num(np.linalg.norm(delta)),
        'beta_mean_abs': _num(np.abs(delta).mean()),
        'beta_max_abs': _num(np.abs(delta).max()),
        'per_beta': [_num(value) for value in delta]}
    if standard_deviation is not None:
        deviation = np.asarray(standard_deviation, dtype=float).reshape(-1)
        if deviation.shape == delta.shape and np.all(deviation > 0):
            out['beta_mahalanobis'] = _num(np.linalg.norm(delta / deviation))
    return out


def nonsubject_regions(boxes_a, boxes_generated, subject_index_a, iou_threshold=0.3):
    """What happened to the *other* people in the frame.

    ``--subject`` picks who the transfer applies to, but everyone else stays in the picture,
    and the workflow README records that they are not evaluated. Boxes are matched by IoU to
    the source detections; matched regions give a per-person displacement and area change.
    """
    boxes_a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    boxes_b = np.asarray(boxes_generated, dtype=float).reshape(-1, 4)
    if boxes_a.shape[0] <= subject_index_a:
        raise ValueError('subject index outside the source detections')
    subject = boxes_a[subject_index_a]
    others = [index for index in range(boxes_a.shape[0]) if index != subject_index_a]
    rows = []
    for index in others:
        box = boxes_a[index]
        best, best_iou = None, 0.0
        for candidate in boxes_b:
            iou = _iou(box, candidate)
            if iou > best_iou:
                best, best_iou = candidate, iou
        matched = best is not None and best_iou >= iou_threshold
        centre_a = (box[:2] + box[2:]) / 2
        area_a = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        if matched:
            centre_b = (best[:2] + best[2:]) / 2
            area_b = max(1.0, float((best[2] - best[0]) * (best[3] - best[1])))
            rows.append({'source_index': index, 'matched': True, 'iou': _num(best_iou),
                'displacement_px': _num(np.linalg.norm(centre_b - centre_a)),
                'area_ratio': _num(area_b / area_a)})
        else:
            rows.append({'source_index': index, 'matched': False, 'iou': _num(best_iou),
                'displacement_px': None, 'area_ratio': None})
    return {'nonsubject_count': len(others), 'matched_count': int(sum(row['matched']
        for row in rows)), 'regions': rows}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(
        0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


DWPOSE_BODY_FROM_COCO133 = [0, 1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 4, 1, 3]


def dwpose134_from_coco133(values):
    """Remap any per-keypoint COCO-WholeBody-133 array into the repo's 134-slot order.

    Mirrors ``src.pose_edit.geometry.cocowholebody133_to_dwpose134`` for arrays that are not
    coordinates (confidence scores, visibility flags), where the neck slot has to be derived
    from the shoulders by hand. The test suite pins the two implementations against each
    other so a change in the geometry mapping cannot silently desynchronise this one.
    """
    array = np.asarray(values, dtype=float)
    flat = array.ndim == 1
    rows = array.reshape(-1, 1) if flat else array
    if rows.shape[0] != COCO_WHOLEBODY_133:
        raise ValueError(f'expected {COCO_WHOLEBODY_133} slots, got {rows.shape[0]}')
    converted = [rows[index] for index in DWPOSE_BODY_FROM_COCO133]
    converted[1] = (rows[5] + rows[6]) / 2.0
    converted += [rows[index] for index in range(17, COCO_WHOLEBODY_133)]
    out = np.asarray(converted, dtype=float)
    return out.reshape(-1) if flat else out


def dwpose134_on_image(image, det_session, pose_session):
    """Run the DWPose ONNX pair and return the repo's 134-slot layout with scores.

    Returns ``(points134, scores134, boxes)`` for the largest detected person, or
    ``(None, None, boxes)`` when nothing was detected. Raw detector output is
    COCO-WholeBody-133 and is converted by ``cocowholebody133_to_dwpose134``, which inserts a
    synthesised neck at slot 1 (shoulder midpoint) exactly as the geometry stage does, so
    guides and estimates share one frame. Scores are remapped with the same body index list;
    ``tests/test_no_gt_metrics.py`` pins the two orderings together.
    """
    import cv2

    from src.evaluation.dwpose_onnx.onnxdet import inference_detector
    from src.evaluation.dwpose_onnx.onnxpose import inference_pose
    from src.pose_edit.geometry import cocowholebody133_to_dwpose134

    bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    boxes = np.asarray(inference_detector(det_session, bgr), dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return None, None, boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(areas))
    points, scores = inference_pose(pose_session, boxes[index:index + 1], bgr)
    raw = np.asarray(points[0], dtype=float).reshape(-1, 2)
    if raw.shape[0] != COCO_WHOLEBODY_133:
        raise ValueError(f'DWPose returned {raw.shape[0]} points, expected '
            f'{COCO_WHOLEBODY_133}')
    confidence = np.asarray(scores[0], dtype=float).reshape(-1)
    if confidence.shape[0] != COCO_WHOLEBODY_133:
        raise ValueError(f'DWPose returned {confidence.shape[0]} scores, expected '
            f'{COCO_WHOLEBODY_133}')
    extended = dwpose134_from_coco133(confidence)
    return cocowholebody133_to_dwpose134(raw), extended, boxes


def smplx_faces(smpl_root):
    """Load SMPL-X triangle indices from the estimator checkout (neutral model)."""
    smpl_root = Path(smpl_root)
    candidates = sorted(smpl_root.glob('human_models/human_model_files/smplx/SMPLX_NEUTRAL*'))
    for path in candidates:
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=True) as bundle:
                if 'faces' in bundle:
                    return np.asarray(bundle['faces'], dtype=np.int32)
        if path.suffix == '.pkl':
            import pickle

            with path.open('rb') as stream:
                model = pickle.load(stream, encoding='latin1')
            faces = getattr(model, 'faces', None)
            if faces is not None:
                return np.asarray(faces, dtype=np.int32)
    raise FileNotFoundError(f'no SMPLX_NEUTRAL faces found under {smpl_root}')
