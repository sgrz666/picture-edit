"""Generated-RGB metrics for the known-geometry dual V8 pilot.

Identity is never used to assign detections to actors. Optional model imports
are lazy so metric bookkeeping and its tests work without CUDA or model files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value)
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationTarget:
    rgb: np.ndarray
    person_masks: np.ndarray
    metadata: dict

    @classmethod
    def read(cls, manifest, index=0, *, generation_manifest=None):
        """Read GT through the same strict binding validator as supervision.

        Reading is deferred by the runner until its main comparisons finish.
        A sibling generation.json is the default for standalone metric callers.
        """
        from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset
        manifest = Path(manifest)
        generation_manifest = manifest.with_name('generation.json') if generation_manifest is None else generation_manifest
        target = Hi4DSMPLXDataset(generation_manifest, manifest).supervision(index)
        rgb = (target.rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype(np.uint8)
        masks = target.person_masks[0, :, 0].detach().cpu().numpy()
        return cls(rgb, masks, dict(target.metadata))

    def changed_in_memory(self):
        """Actually change both evaluator truth arrays without touching disk."""
        return EvaluationTarget(255 - self.rgb, 1. - self.person_masks,
                                dict(self.metadata, target_replacement_probe=True))

    @property
    def sha256(self):
        return array_sha256(self.rgb, self.person_masks)


@dataclass(frozen=True)
class PersonDetection:
    box: np.ndarray
    xy: np.ndarray
    scores: np.ndarray

    def as_dict(self):
        return dict(box_at_512=self.box.tolist(), wholebody134_at_512=self.xy.tolist(),
                    wholebody134_scores=self.scores.tolist())


def reorder_dwpose(keypoints, scores):
    """Official Wholebody convention: 18 body + 6 feet + face/hands = 134."""
    info = np.concatenate([keypoints, scores[..., None]], axis=-1)
    neck = info[:, [5, 6]].mean(axis=1)
    neck[:, 2] = np.logical_and(info[:, 5, 2] > .3, info[:, 6, 2] > .3)
    info = np.insert(info, 17, neck, axis=1)
    dest = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    origin = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    info[:, dest] = info[:, origin]
    return info[..., :2], info[..., 2]


class DWPosePeople:
    def __init__(self, models_root, threads=4):
        import onnxruntime as ort
        paths = Path(models_root) / 'DWPose'
        names = ['yolox_l.onnx', 'dw-ll_ucoco_384.onnx']
        for name in names:
            if not (paths / name).is_file():
                raise FileNotFoundError(paths / name)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.det, self.pose = [ort.InferenceSession(str(paths / n), sess_options=options,
                             providers=['CPUExecutionProvider']) for n in names]
        self.metadata = dict(provider='CPUExecutionProvider', convention='dwpose134_official_reorder',
                             weights={n: file_sha256(paths / n) for n in names})

    def detect(self, image):
        import cv2
        from src.evaluation.dwpose_onnx.onnxdet import inference_detector
        from src.evaluation.dwpose_onnx.onnxpose import inference_pose
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        boxes = inference_detector(self.det, bgr)
        if boxes is None or len(boxes) == 0:
            return []
        xy, scores = reorder_dwpose(*inference_pose(self.pose, boxes, bgr))
        return [PersonDetection(np.asarray(box[:4]), points, confidence)
                for box, points, confidence in zip(boxes, xy, scores)]


def boxes_from_masks(masks):
    boxes = []
    for mask in masks:
        y, x = np.nonzero(np.asarray(mask).squeeze() > .5)
        if not len(x):
            raise ValueError('target geometry actor has an empty amodal mask')
        boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
    return np.asarray(boxes, dtype=np.float32)


def heatmap_observations(heatmaps):
    values = np.asarray(heatmaps)
    count, joints, height, width = values.shape
    flat = values.reshape(count, joints, -1)
    indices = flat.argmax(-1)
    return np.stack((indices % width, indices // width), -1).astype(np.float32), flat.max(-1)


def _iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
    return float(intersection / max(float(union), 1e-8))


def assign_by_geometry(detections, target_boxes, target_xy, target_scores, image_size=(512, 512)):
    """Two actor assignment using external boxes/pose; no appearance argument.

    A gated missing slot is preferable to associating a remote extra detection.
    The returned indices freeze UID correspondence before any identity scoring.
    """
    diagonal = float(np.hypot(*image_size))
    cost = np.full((2, len(detections)), np.inf)
    for actor in range(2):
        box = np.asarray(target_boxes[actor])
        for candidate, detection in enumerate(detections):
            count = min(len(target_xy[actor]), len(detection.xy), 24)
            valid = (np.asarray(target_scores[actor])[:count] > .3) & (detection.scores[:count] > .3)
            valid &= np.isfinite(detection.xy[:count]).all(-1) & np.isfinite(target_xy[actor, :count]).all(-1)
            overlap = _iou(box, detection.box)
            centre_error = np.linalg.norm((box[:2] + box[2:] - detection.box[:2] - detection.box[2:]) / 2) / diagonal
            pose_error = float(np.linalg.norm(detection.xy[:count][valid] - target_xy[actor, :count][valid], axis=-1).mean() / diagonal) if valid.any() else None
            if overlap <= 0 or (pose_error is not None and pose_error > .25) or (pose_error is None and centre_error > .2):
                continue
            cost[actor, candidate] = .25 * (1 - overlap) + .5 * centre_error + (2 * pose_error if pose_error is not None else .2)
    best, best_cost = (-1, -1), float('inf')
    for indices in product(range(-1, len(detections)), repeat=2):
        if indices[0] >= 0 and indices[0] == indices[1]:
            continue
        candidate_cost = sum(.75 if index < 0 else cost[actor, index] for actor, index in enumerate(indices))
        if candidate_cost < best_cost:
            best, best_cost = indices, candidate_cost
    return dict(indices=list(best), costs=[None if index < 0 else float(cost[a, index]) for a, index in enumerate(best)],
                unassigned_detection_indices=[i for i in range(len(detections)) if i not in best],
                assignment_basis='external_target_geometry_bbox_and_pose_without_identity')


def pck(detection, target_xy, target_scores, image_size=(512, 512)):
    target_xy, target_scores = np.asarray(target_xy), np.asarray(target_scores)
    valid = (target_scores > .3) & np.isfinite(target_xy).all(-1)
    denominator = int(valid.sum())
    result = dict(eligible_joints=denominator, detected_joints=0, confidence_threshold=.3,
                  normalization='512_canvas_diagonal', failure=None, nme_diagonal=None,
                  pck_002=None, pck_005=None, pck_002_hits=0, pck_005_hits=0)
    if detection is None:
        result['failure'] = 'person_not_detected'
        return result
    if denominator == 0:
        result['failure'] = 'no_reliable_target_joints'
        return result
    count = min(len(target_xy), len(detection.xy))
    matched = valid[:count] & (detection.scores[:count] > .3) & np.isfinite(detection.xy[:count]).all(-1)
    distance = np.linalg.norm(detection.xy[:count][matched] - target_xy[:count][matched], axis=-1) / np.hypot(*image_size)
    result['detected_joints'] = int(matched.sum())
    result['nme_diagonal'] = float(distance.mean()) if len(distance) else None
    for threshold in (.02, .05):
        key = 'pck_' + ('002' if threshold == .02 else '005')
        result[key + '_hits'] = int((distance < threshold).sum())
        result[key] = result[key + '_hits'] / denominator
    result['joint_recall'] = int(matched.sum()) / denominator
    return result


def appearance_binding(source, generated, *, metric_kind='clothing_proxy_dinov2'):
    similarities = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            if source[i] is not None and generated[j] is not None:
                a, b = np.asarray(source[i]), np.asarray(generated[j])
                similarities[i][j] = float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))
    usable = all(value is not None for row in similarities for value in row)
    own = (similarities[0][0] + similarities[1][1]) / 2 if usable else None
    swapped = (similarities[0][1] + similarities[1][0]) / 2 if usable else None
    return dict(metric_kind=metric_kind, similarity_matrix_source_by_geometry_assigned_actor=similarities,
                actor_similarity=[similarities[0][0], similarities[1][1]],
                original_correspondence_mean=own, swapped_correspondence_mean=swapped,
                margin_original_minus_swapped=None if not usable else own - swapped,
                swap_detected=None if not usable else bool(swapped > own),
                failure=None if usable else 'missing_source_or_generated_embedding',
                assignment_changed_by_identity=False)


def background_error(output, source, source_masks, target_gt_masks, *, dilation=8, valid_canvas=None):
    import cv2
    output, source = np.asarray(output), np.asarray(source)
    if output.shape != source.shape:
        raise ValueError('background RGB inputs must share one letterbox canvas')
    union = np.maximum(np.asarray(source_masks).max(0), np.asarray(target_gt_masks).max(0)) > .5
    if dilation:
        union = cv2.dilate(union.astype(np.uint8), np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)) > 0
    valid = ~union
    if valid_canvas is not None:
        valid &= np.asarray(valid_canvas, dtype=bool)
    difference = np.abs(output.astype(np.float32) - source.astype(np.float32)) / 255
    count = int(valid.sum())
    return dict(mae_rgb_0_1=float(difference[valid].mean()) if count else None,
                pixel_denominator=count, dilation_rgb_pixels=dilation,
                target_gt_masks_used_only_by_evaluator=True,
                failure=None if count else 'empty_background_roi')


def generated_person_masks(detections, assignment, size=512):
    """DWPose-only spatial crop proxy; never uses control or GT segmentation.

    Competing detector boxes are partitioned by distance to detected body joints.
    This is not a clothing segmentation, and its scores are labelled as proxies.
    """
    yy, xx = np.mgrid[:size, :size]
    distances = np.full((2, size, size), np.inf, dtype=np.float32)
    for actor, index in enumerate(assignment):
        if index < 0:
            continue
        detected = detections[index]
        valid = (detected.scores[:18] > .3) & np.isfinite(detected.xy[:18]).all(-1)
        points = detected.xy[:18][valid]
        if not len(points):
            points = ((detected.box[:2] + detected.box[2:]) / 2)[None]
        distance = np.full((size, size), np.inf, np.float32)
        for x, y in points:
            distance = np.minimum(distance, (xx - x) ** 2 + (yy - y) ** 2)
        x0, y0, x1, y1 = detected.box
        inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        distances[actor] = np.where(inside, distance, np.inf)
    winner = distances.argmin(0)
    return np.stack([(winner == actor) & np.isfinite(distances[actor]) for actor in range(2)]).astype(np.float32)


def whole_square_dino_inputs(processor, crop):
    image = crop.resize((224, 224), Image.Resampling.BICUBIC)
    return processor(images=image, return_tensors='pt', do_resize=False, do_center_crop=False)


class Dinov2Appearance:
    def __init__(self, path, device='cpu'):
        from transformers import AutoModel, AutoImageProcessor
        path = Path(path)
        if not (path / 'config.json').is_file():
            raise FileNotFoundError(path / 'config.json')
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
        self.device = device
        self.metadata = dict(path=str(path), config_sha256=file_sha256(path / 'config.json'),
            weight_sha256=file_sha256(path / 'model.safetensors') if (path / 'model.safetensors').exists() else None,
            crop='source_person_crops: square_1.2_exclusive_grey128_then_224_bicubic',
            processor_do_resize=False, processor_do_center_crop=False,
            pooling='mean_valid_patch_coverage_gt_0.5_then_l2', identity_claim='clothing_appearance_proxy_only')

    def encode(self, rgb, masks):
        import torch
        import torch.nn.functional as F
        from src.data.hi4d_smplx_dataset import source_person_crops
        embeddings, failures, crops = [None, None], [None, None], [None, None]
        # Run each actor separately so one failed crop does not erase the other.
        for actor in range(2):
            try:
                if (np.asarray(masks[actor]) > .5).sum() < 16:
                    raise ValueError('insufficient_generated_or_source_person_pixels')
                other = masks[1 - actor]
                # source_person_crops validates both masks, so construct its unused
                # second slot outside this actor if the other person is missing.
                # This cannot remove any actual pixels from the scored actor.
                if ((other > .5) & (masks[actor] <= .5)).sum() < 16:
                    outside = np.flatnonzero(np.asarray(masks[actor]).reshape(-1) <= .5)
                    if len(outside) < 16:
                        raise ValueError('no_exclusive_space_for_missing_other_person')
                    other = np.zeros_like(other)
                    other.reshape(-1)[outside[:16]] = 1
                ordered = np.stack([masks[actor], other])
                crop_array, coverage, _ = source_person_crops(np.asarray(rgb, dtype=np.uint8), ordered)
                crop = Image.fromarray(crop_array[0]); crops[actor] = crop
                inputs = whole_square_dino_inputs(self.processor, crop)
                with torch.inference_mode():
                    tokens = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).last_hidden_state[0, 1:].float()
                if tuple(tokens.shape) != (256, 1536):
                    raise ValueError(f'expected source-bank DINOv2-G patch shape (256,1536), got {tuple(tokens.shape)}')
                valid = F.adaptive_avg_pool2d(torch.from_numpy(coverage[0])[None, None], (16, 16)).flatten() > .5
                if not valid.any():
                    raise ValueError('no_valid_person_dino_patches')
                value = tokens[valid.to(tokens.device)].mean(0).cpu().numpy()
                embeddings[actor] = value / max(float(np.linalg.norm(value)), 1e-12)
            except Exception as exc:
                failures[actor] = f'{type(exc).__name__}: {exc}'
        return embeddings, failures, crops


def mesh_contact_proxy(vertices, labels, assignment, contact_endpoints):
    result = dict(mesh_source='generated_rgb_multihmr_camera_space', reconstructed_people=len(vertices),
                  part_pair_distances=None, failure=None, metric_units='camera_space_metric_scale_proxy',
                  scale_normalization='mean_generated_mesh_y_extent', signed_penetration=None)
    if len(vertices) < 2 or len(assignment) != 2 or min(assignment) < 0:
        result['failure'] = 'fewer_than_two_generated_meshes'
        return result
    from scipy.spatial import cKDTree
    a, b = (np.asarray(vertices[index], dtype=np.float64) for index in assignment)
    labels = np.asarray(labels)
    if len(labels) != len(a) or len(a) != len(b) or not np.isfinite([a, b]).all():
        result['failure'] = 'invalid_generated_mesh_or_smplx_part_labels'
        return result
    height = max(float((np.ptp(a[:, 1]) + np.ptp(b[:, 1])) / 2), 1e-8)
    records = []
    for first, second in np.asarray(contact_endpoints, dtype=int).reshape(-1, 2):
        first_actor, second_actor = first // 14, second // 14
        va, vb = (a if first_actor == 0 else b)[labels == first % 14], (a if second_actor == 0 else b)[labels == second % 14]
        distance = float(cKDTree(vb).query(va)[0].min()) if len(va) and len(vb) else None
        records.append(dict(part_pair=[int(first), int(second)], distance_m_proxy=distance,
                            distance_over_mean_height=None if distance is None else distance / height))
    result['part_pair_distances'] = records
    nearest_a, nearest_b = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    result['surface_proximity_fraction_under_2cm'] = float((np.mean(nearest_a < .02) + np.mean(nearest_b < .02)) / 2)
    result['surface_proximity_interpretation'] = 'unsigned_near_surface_proxy_cannot_establish_penetration'
    result['mean_generated_height_m_proxy'] = height
    result['mean_camera_z_by_actor'] = [float(a[:, 2].mean()), float(b[:, 2].mean())]
    return result


def generated_mesh_quality(humans, labels, K512, image_size, target_boxes, target_xy, target_scores, endpoints):
    """Score generated reconstructions identically for local and isolated inference."""
    K512 = np.asarray(K512)
    vertices, detections = [], []
    smpl_to_dw = [(17, 2), (19, 3), (21, 4), (16, 5), (18, 6), (20, 7),
                  (2, 8), (5, 9), (8, 10), (1, 11), (4, 12), (7, 13)]
    for human in humans:
        mesh = np.asarray(human['v3d']); vertices.append(mesh)
        joints = np.asarray(human['j2d']) * 512 / image_size
        xy, scores = np.zeros((24, 2)), np.zeros(24)
        for smpl, dw in smpl_to_dw:
            xy[dw], scores[dw] = joints[smpl], 1.
        projected = mesh @ K512.T
        projected = projected[:, :2] / projected[:, 2:].clip(1e-8)
        box = np.r_[projected.min(0), projected.max(0)]
        detections.append(PersonDetection(box, xy, scores))
    assignment = assign_by_geometry(detections, target_boxes, target_xy, target_scores)['indices']
    result = mesh_contact_proxy(vertices, labels, assignment, endpoints)
    result.update(actor_assignment=assignment, anatomy_left_right_limb_ratio_proxy=[])
    for index in assignment:
        if index < 0:
            result['anatomy_left_right_limb_ratio_proxy'].append(None)
            continue
        joints = np.asarray(humans[index]['j3d'])
        pairs = [((16, 18), (17, 19)), ((18, 20), (19, 21)), ((1, 4), (2, 5)), ((4, 7), (5, 8))]
        ratios = [float(np.linalg.norm(joints[a] - joints[b]) / max(float(np.linalg.norm(joints[c] - joints[d])), 1e-8))
                  for (a, b), (c, d) in pairs]
        result['anatomy_left_right_limb_ratio_proxy'].append(ratios)
    return result


def _weighted(values):
    usable = [(float(value), int(weight)) for value, weight in values if value is not None and weight > 0]
    denominator = sum(weight for _, weight in usable)
    return dict(mean=sum(value * weight for value, weight in usable) / denominator if denominator else None,
                denominator=denominator, successes=len(usable), failures=len(values) - len(usable))


def aggregate_quality(rows):
    counts = [row['detected_people'] for row in rows if row.get('detected_people') is not None]
    result = dict(outputs=len(rows), third_person_rate=None if not counts else sum(count > 2 for count in counts) / len(counts),
                  third_person_denominator=len(counts), detection_failed_outputs=len(rows) - len(counts),
                  two_person_detection_rate=None if not counts else sum(count == 2 for count in counts) / len(counts))
    for name in ('body', 'wholebody'):
        scores = [person[name] for row in rows for person in row.get('people', []) if name in person]
        scored = [score for score in scores if score.get('pck_005') is not None]
        denominator = sum(score['eligible_joints'] for score in scored)
        result[name] = dict(scored_people=len(scored), failed_people=len(scores) - len(scored),
            eligible_joint_denominator_all=sum(score['eligible_joints'] for score in scores), scored_joint_denominator=denominator,
            pck_002=sum(score['pck_002_hits'] for score in scored) / denominator if denominator else None,
            pck_005=sum(score['pck_005_hits'] for score in scored) / denominator if denominator else None,
            failures=[score.get('failure') for score in scores if score.get('failure')])
    for name in ('appearance', 'face_identity'):
        values = [row.get(name, {}).get('original_correspondence_mean') for row in rows]
        result[name] = _weighted([(value, 1) for value in values])
        swaps = [row.get(name, {}).get('swap_detected') for row in rows]
        valid = [value for value in swaps if value is not None]
        result[name]['swap_rate'] = sum(valid) / len(valid) if valid else None
        result[name]['swap_denominator'] = len(valid)
        result[name]['actors'] = [_weighted([(row.get(name, {}).get('actor_similarity', [None, None])[actor], 1)
                                             for row in rows]) for actor in range(2)]
    result['background'] = _weighted([(row.get('background', {}).get('mae_rgb_0_1'), row.get('background', {}).get('pixel_denominator', 0)) for row in rows])
    contacts = [(entry['distance_over_mean_height'], 1) for row in rows for entry in (row.get('mesh', {}).get('part_pair_distances') or [])]
    result['generated_mesh_contact_distance_over_height'] = _weighted(contacts)
    result['mesh_failed_outputs'] = sum(bool(row.get('mesh', {}).get('failure')) for row in rows)
    return result
