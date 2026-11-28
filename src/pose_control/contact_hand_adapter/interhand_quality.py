"""InterHand L1 pairing policy using camera-space articulation and RGB quality."""

from collections import Counter, defaultdict
from functools import lru_cache
import io
import math
import random

import numpy as np
from PIL import Image


PAIR_QUOTAS = {"left": .20, "right": .20, "interacting": .35, "interacting_partial": .25}
THRESHOLDS = {"finger_degrees": 10., "palm_degrees": 12., "wrist_relative": .16,
              "inter_hand_relative": .30, "minimum_hand_pixels": 500., "minimum_sharpness": 12.}


def angles(a, b):
    numerator = (a * b).sum(axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(numerator / denom, -1., 1.)))


def movement(a, b, valid):
    """Coordinates are wrist-first, in camera meters. Normalize lengths by palm span."""
    bends = [(start + j - 1 if j > 0 else 0, start + j, start + j + 1)
             for start in range(1, 21, 4) for j in range(3)]
    valid_sides, scale = [], []
    wrist_relative, finger, palm = [], [], []
    for side in range(a.shape[0]):
        keep = valid[side]
        if not keep[0] or keep.sum() < 6:
            continue
        x, y = a[side] - a[side, 0], b[side] - b[side, 0]
        length = np.median(np.r_[np.linalg.norm(x[keep], axis=-1), np.linalg.norm(y[keep], axis=-1)])
        if not np.isfinite(length) or length <= 1e-4:
            continue
        valid_sides.append(side)
        scale.append(length)
        wrist_relative.append(float(np.sqrt(np.square(x[keep] - y[keep]).sum(axis=-1).mean()) / length))
        for root, joint, tip in bends:
            if keep[[root, joint, tip]].all():
                diff = abs(angles(x[root] - x[joint], x[tip] - x[joint])
                           - angles(y[root] - y[joint], y[tip] - y[joint]))
                if np.isfinite(diff):
                    finger.append(float(diff))
        if keep[[5, 17]].all():
            turn = angles(np.cross(x[5], x[17]), np.cross(y[5], y[17]))
            if np.isfinite(turn):
                palm.append(float(turn))
    inter = None
    if len(valid_sides) == 2:
        inter = float(np.linalg.norm((a[1, 0] - a[0, 0]) - (b[1, 0] - b[0, 0])) / np.mean(scale))
    return {"wrist_relative": max(wrist_relative, default=None),
            "finger_degrees": float(np.percentile(finger, 80)) if finger else None,
            "palm_degrees": max(palm, default=None), "inter_hand_relative": inter}


def rgb_quality(data, points, valid, hands):
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        areas, sharpness = [], []
        for side in np.where(hands)[0]:
            xy = points[side, valid[side]]
            if len(xy) < 4:
                return 0., 0.
            low, high = xy.min(axis=0), xy.max(axis=0)
            areas.append(float(np.prod(high - low)))
            if (high <= low).any():
                return 0., 0.
            x0, y0 = np.floor(low).astype(int)
            x1, y1 = np.ceil(high).astype(int)
            region = np.asarray(gray.crop((x0, y0, x1 + 1, y1 + 1)).resize((64, 64)), np.float32)
            center = region[1:-1, 1:-1]
            laplace = region[:-2, 1:-1] + region[2:, 1:-1] + region[1:-1, :-2] + region[1:-1, 2:] - 4 * center
            sharpness.append(float(laplace.var()))
    return min(areas), min(sharpness)


class HandPriorScreen:
    def __init__(self, limit, *, seed, max_candidates=None):
        self.limit = limit
        self.seed = seed
        self.max_candidates = max_candidates
        self.stats = Counter()
        self.rgb = lru_cache(maxsize=512)(self.rgb)

    @property
    def exhausted(self):
        return self.max_candidates is not None and self.stats["considered"] >= self.max_candidates

    @staticmethod
    def rgb(path, corners, read):
        xy, mask, hands = corners
        return rgb_quality(read(path), np.asarray(xy), np.asarray(mask), np.asarray(hands))

    def measure(self, source, target, a, b, joint_order, read):
        self.stats["considered"] += 1
        hands = b.present_hands()[:2].numpy()
        common = (a.joint_3d_valid[:2] & b.joint_3d_valid[:2]).numpy()
        metrics = movement(a.joints_3d[:2].numpy(), b.joints_3d[:2].numpy(), common)
        ratios = [metrics[name] / THRESHOLDS[name] for name in metrics if metrics[name] is not None]
        strength = max(ratios, default=0.)
        if strength < 1.:
            self.stats["low_articulation"] += 1
            return None
        per_view = []
        for frame, condition in ((source, a), (target, b)):
            view = condition.projected_joints()[:2].numpy() & frame.view_joints[joint_order]
            if (view[hands].sum(axis=1) < 4).any() or not view[hands, 0].all():
                self.stats["weak_view_support"] += 1
                return None
            flat_xy = tuple(tuple(map(tuple, hand)) for hand in condition.joints[:2, :, :2].numpy())
            area, sharp = self.rgb(frame.path, (flat_xy, tuple(map(tuple, view)), tuple(hands)), read)
            if area < THRESHOLDS["minimum_hand_pixels"] or sharp < THRESHOLDS["minimum_sharpness"]:
                self.stats["weak_rgb"] += 1
                return None
            per_view.append({"min_hand_pixels": area, "min_sharpness": sharp,
                             "visible_joints": [int(n) for n in view.sum(axis=1)]})
        category = target.hand_type
        if category == "interacting" and min(per_view[1]["visible_joints"]) <= 17:
            category = "interacting_partial"
        motion = "small" if strength < 2 else "medium" if strength < 4 else "large"
        return {**metrics, "motion_strength": strength, "motion_bin": motion,
                "category": category, "source_rgb": per_view[0], "target_rgb": per_view[1]}

    def sample(self, proposals):
        groups, seen = defaultdict(list), Counter()
        rng = random.Random(self.seed)
        for item in proposals:
            fields = item[2]
            key = fields["category"], fields["motion_bin"]
            seen[key] += 1
            group = groups[key]
            if len(group) < self.limit:
                group.append(item)
            else:
                other = rng.randrange(seen[key])
                if other < self.limit:
                    group[other] = item
        quotas = {category: math.floor(self.limit * weight) for category, weight in PAIR_QUOTAS.items()}
        for category in sorted(quotas, key=lambda c: (-(self.limit * PAIR_QUOTAS[c] - quotas[c]), c))[
                :self.limit - sum(quotas.values())]:
            quotas[category] += 1
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        used, bins = Counter(), Counter()
        while len(chosen) < self.limit:
            available = [key for key, group in groups.items() if group]
            if not available:
                break
            key = min(available, key=lambda k: (used[k[0]] >= quotas[k[0]], used[k[0]] / PAIR_QUOTAS[k[0]],
                                                  bins[k], k))
            chosen.append(groups[key].pop())
            used[key[0]] += 1
            bins[key] += 1
        self.audit = {"thresholds": THRESHOLDS, "quota_target": quotas, "seed": self.seed,
                      "partial_definition": "at most 17 trusted projected joints on one target hand",
                      "proposed": {"|".join(k): v for k, v in sorted(seen.items())},
                      "selected": {"|".join(k): v for k, v in sorted(bins.items())}, "counters": dict(self.stats)}
        return chosen
