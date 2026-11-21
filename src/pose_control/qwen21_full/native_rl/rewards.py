"""Fixed target eligibility and identity-blind matching for rendered images."""

from dataclasses import dataclass
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


SINGLE_WEIGHTS = {'pose': .5, 'identity': .5}
DUAL_WEIGHTS = {'pose': .30, 'identity': .25, 'binding': .20, 'layout': .15, 'contact': .10}
OKS_SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072,
                       .062, .062, .107, .107, .087, .087, .089, .089])
PART_POINTS = ((0,), (5, 6, 11, 12), (5, 7), (7, 9), (9,), (6, 8), (8, 10),
               (10,), (11, 13), (13, 15), (15,), (12, 14), (14, 16), (16,))
EVALUATOR_ASSETS = ('DWPose/yolox_l.onnx', 'DWPose/dw-ll_ucoco_384.onnx',
                    'antelopev2/scrfd_10g_bnkps.onnx', 'antelopev2/glintr100.onnx')


@dataclass(frozen=True)
class PersonPose:
    points: np.ndarray
    valid: np.ndarray
    box: np.ndarray


@dataclass(frozen=True)
class RewardTarget:
    poses: tuple
    identities: tuple
    content_hw: tuple
    contacts: tuple = ()


class UnscorableTarget(ValueError):
    """Fixed target pose is missing or ambiguous."""


def _assign(scores, minimum):
    """One-to-one slot assignment, allowing a slot to remain unmatched."""
    slots, count = scores.shape
    allowed = np.where(scores >= minimum, scores, -1.)
    _, columns = linear_sum_assignment(np.concatenate((allowed, np.zeros((slots, slots))), axis=1), maximize=True)
    return tuple(int(i) if i < count else None for i in columns)


def _iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return intersection / max(1., np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection)


def _pose_score(target, predicted, diagonal):
    valid = target.valid
    if not valid.any():
        raise ValueError('Pose reward requires visible target joints')
    if predicted is None:
        return 0., 0., 0.
    matched = valid & predicted.valid
    distance = np.linalg.norm(predicted.points[matched] - target.points[matched], axis=1)
    area = max(1., (target.box[2:] - target.box[:2]).prod())
    oks = np.exp(-distance**2 / (8 * area * OKS_SIGMAS[matched]**2)).sum() / valid.sum()
    return float(oks), float((distance < .02 * diagonal).sum() / valid.sum()), float(matched.sum() / valid.sum())


def _part(pose, part):
    ids = list(PART_POINTS[part])
    if pose is None or not pose.valid[ids].all():
        return None
    return pose.points[ids].mean(axis=0)


def _cosine(a, b):
    cosine = float(a @ b)
    if not np.isfinite(cosine):
        raise ValueError('Non-finite identity similarity')
    return float(np.clip(cosine, -1, 1))


def score_candidate(target, poses, identities):
    if len(poses) != len(identities):
        raise ValueError('Generated face features must follow the detected pose order')
    diagonal = float(np.hypot(*target.content_hw))
    scores = np.empty((len(target.poses), len(poses)))
    for i, gt in enumerate(target.poses):
        for j, pose in enumerate(poses):
            oks, _, _ = _pose_score(gt, pose, diagonal)
            scores[i, j] = .6 * _iou(gt.box, pose.box) + .4 * oks
    match = _assign(scores, .10)
    assigned = [poses[j] if j is not None else None for j in match]
    faces = [identities[j] if j is not None else None for j in match]
    pose_rows = [_pose_score(gt, pred, diagonal) for gt, pred in zip(target.poses, assigned)]
    oks, pck, recall = np.asarray(pose_rows).T
    reward = {'pose': float(.7 * oks.mean() + .3 * oks.min())}
    eligible = [i for i, source in enumerate(target.identities) if source is not None]
    seen = [i for i in eligible if faces[i] is not None]
    cosines = {i: _cosine(faces[i], target.identities[i]) for i in seen}
    details = {'person_pose': oks.tolist(), 'pck_002': float(pck.mean()), 'joint_recall': float(recall.mean()),
               'matched_people': sum(j is not None for j in match), 'detections': len(poses),
               'id_eligible': len(eligible), 'id_missing': len(eligible) - len(seen),
               'id_cosine': [cosines.get(i) for i in range(len(target.poses))]}
    if eligible:
        reward['identity'] = sum((v + 1.) / 2. for v in cosines.values()) / len(eligible)
    if len(target.poses) == 2:
        reward['layout'] = 0.
        if all(p is not None for p in assigned):
            def layout(pair):
                a, b = (pose.box for pose in pair)
                shift = (b[:2] + b[2:] - a[:2] - a[2:]) / (2 * diagonal)
                scales = np.log(np.maximum(1, np.stack((a[2:] - a[:2], b[2:] - b[:2]))) / diagonal)
                return shift, scales

            gd, gs = layout(target.poses)
            pd, ps = layout(assigned)
            reward['layout'] = float(np.exp(-.5 * (np.sum(((gd - pd) / .10)**2) + np.mean(((gs - ps) / .35)**2))))
        if len(eligible) == 2:
            reward['binding'] = 0.
            details['id_swap'] = None
            if len(seen) == 2:
                margin = cosines[0] + cosines[1] - _cosine(faces[0], target.identities[1]) - _cosine(faces[1], target.identities[0])
                reward['binding'] = float((1 + np.tanh(margin / .4)) / 2)
                details['id_swap'] = bool(margin < 0)
        if target.contacts:
            contacts = []
            for a, b, expected in target.contacts:
                p0, p1 = _part(assigned[0], a), _part(assigned[1], b)
                contacts.append(0. if p0 is None or p1 is None else
                                np.exp(-.5 * ((np.linalg.norm(p0 - p1) - expected) / (.06 * diagonal))**2))
            reward['contact'] = float(np.mean(contacts))
    return reward, details


class ImageRewardBank:
    def __init__(self, models_root):
        from src.evaluation.joint_pilot_metrics import IdentityMetric, PoseMetric

        self.pose = PoseMetric(models_root)
        self.identity = IdentityMetric(models_root)

    def _poses(self, image, canvas):
        import cv2
        from src.evaluation.dwpose_onnx.onnxdet import inference_detector
        from src.evaluation.dwpose_onnx.onnxpose import inference_pose

        if image.size != canvas.canvas_hw[::-1]:
            raise ValueError('Reward image and target Canvas differ')
        x0, y0, x1, y1 = canvas.valid_rect
        bgr = cv2.cvtColor(np.asarray(image.crop(canvas.valid_rect).convert('RGB')), cv2.COLOR_RGB2BGR)
        boxes = np.asarray(inference_detector(self.pose.det, bgr))[:16]
        if not len(boxes):
            return ()
        points, confidence = inference_pose(self.pose.pose, boxes, bgr)
        output = []
        for xy, score, box in zip(points[:, :17], confidence[:, :17], boxes):
            valid = ((score > .3) & np.isfinite(xy).all(1) & (xy >= 0).all(1)
                     & (xy < np.array([x1 - x0, y1 - y0])).all(1))
            if valid.any():
                box = np.clip(box[:4], 0, [x1 - x0, y1 - y0, x1 - x0, y1 - y0])
                output.append(PersonPose(xy + [x0, y0], valid, box + [x0, y0, x0, y0]))
        return tuple(output)

    def _faces(self, image, boxes):
        import cv2

        if not boxes:
            return ()
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        detected, landmarks = self.identity.det.detect(bgr)
        if not len(detected):
            return (None,) * len(boxes)
        scores = np.array([[_iou(box, face[:4]) if box is not None else 0. for face in detected] for box in boxes])
        output = []
        for i in _assign(scores, .10):
            embedding = None
            if i is not None:
                crop = self.identity.face_align.norm_crop(bgr, landmark=landmarks[i])
                feature = self.identity.recognition.get_feat(crop).reshape(-1).astype(np.float32)
                norm = np.linalg.norm(feature)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError('Invalid glintr100 embedding')
                embedding = feature / norm
            output.append(embedding)
        return tuple(output)

    def target(self, sample, condition):
        from src.data.canvas_contract import CanvasSpec
        from ..native_v5.conditioning import reference_images
        from ..native_v62.conditions import part_contact_permissions
        from ..qwen_runtime import dataset_image_to_pil

        canvas = CanvasSpec.from_record(sample['target_canvas'])
        local = sample['local_condition_v4']
        slots = local.person_valid[0].nonzero().flatten().tolist()
        if slots not in ([0], [0, 1]):
            raise ValueError('Native RL requires canonical active person slots')
        target_image = dataset_image_to_pil(sample['tgt_image'])
        detections = self._poses(target_image, canvas)
        masks = [sample['human_mask' if i == 0 else 'human_mask_b'].squeeze().bool().cpu().numpy() for i in slots]
        scores = np.empty((len(slots), len(detections)))
        for i, mask in enumerate(masks):
            body_box = _mask_box(mask)
            for j, pose in enumerate(detections):
                x, y = pose.points[pose.valid].astype(int).T
                scores[i, j] = .6 * mask[y, x].mean() + .4 * (_iou(body_box, pose.box) if body_box is not None else 0.)
        match = _assign(scores, .15)
        if any(j is None or detections[j].valid.sum() < 4 for j in match):
            raise UnscorableTarget('No reliable DWPose match for each Native person slot')
        poses = tuple(detections[j] for j in match)
        gt_faces = self._faces(target_image, [_head_box(p) for p in poses])
        bank = sample.get('source_bank_v5')
        if bank is None:
            source = dataset_image_to_pil(sample['src_image'])
            masks = sample['source_head_mask']
            source_faces = self._faces(source, [_mask_box(masks[int(local.source_indices[0, i])]) for i in slots])
        else:
            images = reference_images(condition)
            source_faces = []
            for i in slots:
                source = bank.sources[int(bank.target_qwen_image_indices[0, i])]
                physical = int(bank.target_person_indices[0, i])
                source_faces.append(self._faces(images[source.qwen_image_index], [_mask_box(source.head_mask[0, physical])])[0])
        faces = tuple(src if gt is not None else None for src, gt in zip(source_faces, gt_faces))
        contacts = []
        if len(slots) == 2 and 'dual_depth' in sample:
            positive, _ = part_contact_permissions(sample['dual_depth'].part_contact[0])
            for a, b in positive.nonzero().tolist():
                p0, p1 = _part(poses[0], a), _part(poses[1], b)
                if p0 is not None and p1 is not None:
                    contacts.append((a, b, float(np.linalg.norm(p0 - p1))))
        return RewardTarget(poses, faces, canvas.resized_hw, tuple(contacts)), canvas

    def score(self, target, canvas, image):
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (128, 128, 128, 255))
            image = Image.alpha_composite(background, image).convert('RGB')
        poses = self._poses(image, canvas)
        return score_candidate(target, poses, self._faces(image, [_head_box(p) for p in poses]))


def _mask_box(mask):
    y, x = np.where(np.asarray(mask).squeeze())
    if not len(x):
        return None
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1])


def _head_box(pose):
    points = pose.points[:5][pose.valid[:5]]
    if not len(points):
        return None
    lo, hi = points.min(0), points.max(0)
    margin = max(4., (pose.box[3] - pose.box[1]) * .09)
    return np.concatenate([lo - margin, hi + margin])
