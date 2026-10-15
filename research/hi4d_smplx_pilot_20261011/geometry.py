"""Pure geometry utilities; coordinates are OpenCV camera or Hi4D world, in metres."""
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment


def resize_pad_affine(native_wh, resized_wh, padding_xy):
    w, h = native_wh
    rw, rh = resized_wh
    px, py = padding_xy
    return np.array([[rw / w, 0., px], [0., rh / h, py], [0., 0., 1.]])


def camera_to_world(points, extrinsics):
    E = np.asarray(extrinsics)
    return (np.asarray(points) - E[:, 3]) @ E[:, :3]


def project(points, K, E):
    camera = np.asarray(points) @ E[:, :3].T + E[:, 3]
    p = camera @ K.T
    return p[..., :2] / p[..., 2:3]


def associate(candidate_xy, candidate_scores, reference_xy, max_error=120.):
    """Return detection index per fixed reference slot, preserving absent slots as -1."""
    slots = len(reference_xy)
    assignment = np.full(slots, -1, dtype=np.int64)
    errors = np.full(slots, np.inf)
    if len(candidate_xy) == 0:
        return assignment, errors
    cost = np.full((slots, len(candidate_xy)), 1e9)
    for s, truth in enumerate(reference_xy):
        for c, (xy, scores) in enumerate(zip(candidate_xy, candidate_scores)):
            valid = (scores > .3) & np.isfinite(xy).all(-1)
            if not valid.any():
                continue
            distances = np.linalg.norm(xy[valid] - truth[valid], axis=-1)
            cost[s, c] = np.average(np.minimum(distances, 400), weights=scores[valid])
    row, col = linear_sum_assignment(cost)
    for s, c in zip(row, col):
        if cost[s, c] <= max_error:
            assignment[s], errors[s] = c, cost[s, c]
    return assignment, errors


def wholebody134_indices():
    """DWPose reordered layout: body18, feet6, face68, left21, right21."""
    body = [55, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59]
    feet = [60, 61, 62, 63, 64, 65]
    face = list(range(127, 144)) + list(range(76, 127))
    left = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
    right = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]
    return np.array(body + feet + face + left + right, dtype=np.int64)


def _dlt(xy, Ps):
    A = np.stack([row for uv, P in zip(xy, Ps) for row in (uv[0] * P[2] - P[0], uv[1] * P[2] - P[1])])
    _, _, V = np.linalg.svd(A)
    if abs(V[-1, 3]) < 1e-10:
        return None
    return V[-1, :3] / V[-1, 3]


def triangulate(xy, scores, Ks, Es, threshold=12., min_score=.5):
    """Robust calibrated triangulation, with per-landmark visibility and inlier counts."""
    V, J, _ = xy.shape
    Ps = Ks @ Es
    points = np.full((J, 3), np.nan)
    valid = np.zeros(J, dtype=bool)
    error = np.full(J, np.nan)
    counts = np.zeros(J, dtype=np.int32)
    for j in range(J):
        ids = np.flatnonzero((scores[:, j] >= min_score) & np.isfinite(xy[:, j]).all(-1))
        if len(ids) < 2:
            continue
        best = None
        for pair in combinations(ids, 2):
            X = _dlt(xy[list(pair), j], Ps[list(pair)])
            if X is None:
                continue
            cams = X @ Es[ids, :, :3].transpose(0, 2, 1) + Es[ids, :, 3]
            if np.any(cams[:, 2] <= 0):
                continue
            hom = np.einsum('vij,j->vi', Ps[ids], np.r_[X, 1.])
            residual = np.linalg.norm(hom[:, :2] / hom[:, 2:3] - xy[ids, j], axis=-1)
            inliers = ids[residual < threshold]
            if len(inliers) < 2:
                continue
            rank = (len(inliers), -float(np.median(residual[residual < threshold])))
            if best is None or rank > best[0]:
                best = (rank, inliers)
        if best is None:
            continue
        ids = best[1]
        X = _dlt(xy[ids, j], Ps[ids])
        hom = np.einsum('vij,j->vi', Ps[ids], np.r_[X, 1.])
        points[j] = X
        valid[j] = True
        error[j] = np.median(np.linalg.norm(hom[:, :2] / hom[:, 2:3] - xy[ids, j], axis=-1))
        counts[j] = len(ids)
    return points, valid, error, counts
