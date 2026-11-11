"""Shape metrics for the cross-shape benchmark.

No torch and no CUDA: numpy + scipy + cv2.  The measurement implementation is
the CPU convex-hull version used by the October 2026 HBW run; it is kept for
reconciliation with the official SHAPY CPU branch (see --measure-impl later).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MM = 1000.0


# ---------------------------------------------------------------------------
# SMPL-X shape decoding
# ---------------------------------------------------------------------------

def load_smplx_npz(path: str | Path):
    """Return (v_template, shapedirs, faces) from a SMPL-X .npz."""
    z = np.load(path, allow_pickle=True)
    template = np.asarray(z["v_template"], dtype=np.float64)
    shapedirs = np.asarray(z["shapedirs"], dtype=np.float64)
    faces = np.asarray(z["f"], dtype=np.int32)
    if template.shape != (10475, 3):
        raise ValueError(f"unexpected template shape {template.shape}")
    if shapedirs.shape[:2] != (10475, 3) or shapedirs.shape[2] < 10:
        raise ValueError(f"unexpected shapedirs shape {shapedirs.shape}")
    if faces.shape[1] != 3:
        raise ValueError(f"unexpected faces shape {faces.shape}")
    return template, shapedirs, faces


def beta_to_verts(template, shapedirs, betas, num_betas=None):
    """v = template + sum_i beta_i * shapedirs[:, :, i], no pose blendshapes."""
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim == 1:
        b = b[None]
    k = int(num_betas or b.shape[1])
    if k > shapedirs.shape[2]:
        raise ValueError(f"requested {k} betas but only {shapedirs.shape[2]} available")
    return template[None] + np.einsum("bi,vci->bvc", b[:, :k], shapedirs[:, :, :k])


# ---------------------------------------------------------------------------
# Surface metrics
# ---------------------------------------------------------------------------

def v2v_mm(pred, gt, align: str = "translation") -> float:
    """Mean per-vertex distance in mm.

    ``translation`` subtracts the centroid from both meshes, matching the
    official SHAPY point_error protocol.  No rotation and no scale alignment:
    height errors must stay visible.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {g.shape}")
    if align == "translation":
        p = p - p.mean(0, keepdims=True)
        g = g - g.mean(0, keepdims=True)
    elif align != "none":
        raise ValueError(f"unknown align {align!r}")
    return float(np.linalg.norm(p - g, axis=-1).mean() * MM)


def p2p20k_mm(pred, gt, regressor) -> float:
    """P2P-20K via SHAPY's fixed HD_SMPLX_from_SMPL sparse regressor."""
    p = np.asarray(regressor.dot(np.asarray(pred, dtype=np.float64)))
    g = np.asarray(regressor.dot(np.asarray(gt, dtype=np.float64)))
    return v2v_mm(p, g, align="translation")


def beta_l2(b1, b2) -> float:
    """Diagnostic only; never the primary endpoint."""
    return float(np.linalg.norm(np.asarray(b1) - np.asarray(b2)))


# ---------------------------------------------------------------------------
# Anthropometric measurements (CPU convex hull; old implementation)
# ---------------------------------------------------------------------------

def barycenter_vertex(tri, item):
    return (tri[int(item["face_idx"])] * np.asarray(item["bc"], np.float32).reshape(3, 1)).sum(axis=0)


def slice_hull_perimeter(tri, y):
    """CPU implementation of the SHAPY cross-section convex-hull perimeter."""
    from scipy.spatial import ConvexHull

    ydist = tri[:, :, 1] - y
    chosen = (ydist.min(axis=1) < -1e-7) & (ydist.max(axis=1) > 1e-7)
    t = tri[chosen]
    d = ydist[chosen]
    edges = ((0, 1), (1, 2), (2, 0))
    sections = []
    for a, b in edges:
        active = ((d[:, a] < 0) & (d[:, b] > 0)) | ((d[:, a] > 0) & (d[:, b] < 0))
        if not active.any():
            continue
        alpha = d[active, a] / (d[active, a] - d[active, b])
        points = t[active, a] + alpha[:, None] * (t[active, b] - t[active, a])
        sections.append(points[:, [0, 2]])
    if not sections:
        raise RuntimeError("no plane-triangle intersections")
    points = np.concatenate(sections, axis=0)
    if len(points) < 3:
        raise RuntimeError("not enough points for slice hull")
    hull = ConvexHull(points)
    cycle = points[hull.vertices]
    return float(np.linalg.norm(np.roll(cycle, -1, axis=0) - cycle, axis=1).sum())


def mesh_measurements(verts, faces, defns, landmarks):
    """Height, chest, waist, hips, mass from a template-pose mesh."""
    t = verts[faces]
    htop = barycenter_vertex(t, landmarks["HeadTop"])
    heel = barycenter_vertex(t, landmarks["HeelLeft"])
    height = float(abs(htop[1] - heel[1]))
    xyz = t.astype(np.float64)
    volume = float(abs(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()) / 6.0)
    output = {"height": height, "mass": volume * 985.0}
    for name, code in (("chest", "CW_p"), ("waist", "BW_p"), ("hips", "IW_p")):
        entry = landmarks[defns[code][0]]
        plane_height = barycenter_vertex(t, entry)[1]
        output[name] = slice_hull_perimeter(t, plane_height)
    return output


def load_measurement_resources(utility_dir: str | Path):
    """Load the official 20K regressor and the SHAPY measurement definitions."""
    import yaml

    utility_dir = Path(utility_dir)
    with open(utility_dir / "HD_SMPLX_from_SMPL.pkl", "rb") as f:
        regressor = pickle.load(f)
    defns = yaml.safe_load((utility_dir / "measurement_defitions.yaml").read_text())
    landmarks = yaml.safe_load((utility_dir / "smplx_measurements.yaml").read_text())
    return regressor, defns, landmarks


# ---------------------------------------------------------------------------
# Silhouette metrics
# ---------------------------------------------------------------------------

def mask_iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def _boundary_map(mask):
    import cv2

    m = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m - eroded).astype(bool)


def contour_chamfer_px(a, b) -> float:
    """Symmetric mean distance between the two mask boundaries, in pixels."""
    import cv2

    ba, bb = _boundary_map(a), _boundary_map(b)
    if not ba.any() or not bb.any():
        return float("nan")
    dt_b = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    dt_a = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (dt_b[ba].mean() + dt_a[bb].mean()))


def mask_height_px(mask) -> float:
    ys = np.where(np.asarray(mask, dtype=bool))[0]
    if ys.size == 0:
        return float("nan")
    return float(ys.max() - ys.min() + 1)


def contour_metrics(a, b):
    iou = mask_iou(a, b)
    chamfer = contour_chamfer_px(a, b)
    h = mask_height_px(b)
    return {
        "iou": iou,
        "chamfer_px": chamfer,
        "chamfer_norm": (chamfer / h) if (np.isfinite(chamfer) and h > 0) else float("nan"),
        "mask_a_px": int(np.asarray(a, dtype=bool).sum()),
        "mask_b_px": int(np.asarray(b, dtype=bool).sum()),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_summary(a, b, n_boot: int = 10000, seed: int = 0):
    """Subject-level paired summary of b - a (negative means b is better)."""
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired_summary expects two 1-D arrays of equal length")
    diff = b - a
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    wins = int((diff < 0).sum())
    # Exact two-sided sign test against 0.5; ties are ignored.
    nonzero = diff[diff != 0]
    p = float(binomtest(int((nonzero < 0).sum()), len(nonzero), 0.5).pvalue) if len(nonzero) else 1.0
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "sd_diff": sd,
        "dz": float(diff.mean() / sd) if sd and np.isfinite(sd) else float("nan"),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wins_b": wins,
        "ties": int((diff == 0).sum()),
        "sign_test_p": p,
        "per_subject_diff": diff.tolist(),
    }


def icc11(values_by_subject):
    """One-way random-effects ICC(1,1) for subjects with >=2 observations."""
    groups = [np.asarray(v, dtype=np.float64) for v in values_by_subject if len(v) >= 2]
    k = len(groups)
    if k < 2:
        return float("nan")
    n_total = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else float("nan")


def stability_summary(betas_by_subject):
    """Per-subject beta std / mean norm, plus ICC(1,1) per beta dimension."""
    out = {}
    iccs = []
    for sid, betas in sorted(betas_by_subject.items()):
        b = np.asarray(betas, dtype=np.float64)
        out[sid] = {
            "n_images": int(len(b)),
            "mean_beta": b.mean(0).tolist(),
            "std_beta": b.std(0, ddof=1).tolist() if len(b) > 1 else [0.0] * b.shape[1],
            "mean_std": float(b.std(0, ddof=1).mean()) if len(b) > 1 else 0.0,
            "norm_mean": float(np.linalg.norm(b.mean(0))),
        }
    dims = max((np.asarray(b).shape[1] for b in betas_by_subject.values()), default=0)
    for d in range(dims):
        iccs.append(icc11([np.asarray(b)[:, d] for b in betas_by_subject.values()]))
    out["_icc_by_dim"] = iccs
    out["_icc_mean"] = float(np.nanmean(iccs)) if iccs else float("nan")
    return out
