"""Synthetic cameras, silhouettes and a small CPU shaded renderer.

The camera follows the same pinhole / Z+epsilon convention as
src/pose_edit/geometry.py:project_camera, so masks and 3D projections agree.
"""
from __future__ import annotations

import math

import numpy as np

Z_EPS = 1e-4


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("zero vector")
    return v / n


def _rotation_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def make_camera(points, azimuth_deg=0.0, size=512, margin=0.06, f_factor=1.2):
    """Frame a 3-D point cloud from a given azimuth; returns camera dict."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2.0
    extent = float(np.max(hi - lo))
    extent = max(extent, 1e-3)
    f = float(size) * f_factor
    # Start with a distance that fits the largest extent, then verify.
    d = f * extent / (size * (1.0 - 2.0 * margin))
    d = max(d, 1.0)
    up = np.array([0.0, 1.0, 0.0])
    for _ in range(12):
        cam = _camera_at(center, d, azimuth_deg, f, size)
        xy, valid = project(pts, cam)
        good = valid & np.isfinite(xy).all(1)
        if good.all():
            lo_xy = xy[good].min(0)
            hi_xy = xy[good].max(0)
            if (lo_xy >= margin * size).all() and (hi_xy <= (1.0 - margin) * size).all():
                break
        d *= 1.12
    return cam


def _camera_at(center, distance, azimuth_deg, f, size):
    center = np.asarray(center, dtype=np.float64)
    offset = _rotation_y(azimuth_deg) @ np.array([0.0, 0.0, distance], dtype=np.float64)
    eye = center + offset
    forward = _unit(center - eye)
    right = _unit(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    # Image y points down: camera +y must be world-down, not world-up.
    down = _unit(np.cross(forward, right))
    r_cw = np.stack([right, down, forward], axis=1)  # camera-to-world columns
    world_to_camera = np.concatenate([r_cw.T, (-r_cw.T @ eye)[:, None]], axis=1)
    k = np.array([[f, 0.0, size / 2.0], [0.0, f, size / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return {
        "K": k,
        "world_to_camera": world_to_camera,
        "image_hw": (int(size), int(size)),
        "focal_xy": (float(f), float(f)),
        "principal_xy": (size / 2.0, size / 2.0),
        "eye": eye.tolist(),
        "center": center.tolist(),
        "azimuth_deg": float(azimuth_deg),
        "rotation_y_deg": float(azimuth_deg),
    }


def to_camera(points, cam):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(cam["world_to_camera"], dtype=np.float64)
    return pts @ r[:, :3].T + r[:, 3]


def project(points, cam, z_epsilon=Z_EPS):
    """Return (xy, valid) with the Z+epsilon convention."""
    xyz = to_camera(points, cam)
    f = np.asarray(cam["focal_xy"], dtype=np.float64)
    c = np.asarray(cam["principal_xy"], dtype=np.float64)
    valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    xy = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    depth = xyz[valid, 2] + z_epsilon
    xy[valid, 0] = xyz[valid, 0] / depth * f[0] + c[0]
    xy[valid, 1] = xyz[valid, 1] / depth * f[1] + c[1]
    xy[~np.isfinite(xy).all(1)] = np.nan
    return xy, valid


def jaw_from_angle(angle_deg):
    return math.radians(angle_deg)


def mask_from_verts(points, faces, cam, size=None, dilate=0):
    import cv2

    size = size or cam["image_hw"][0]
    xy, valid = project(points, cam)
    faces = np.asarray(faces, dtype=np.int32)
    mask = np.zeros((int(size), int(size)), dtype=np.uint8)
    tri_valid = valid[faces].all(1) & np.isfinite(xy[faces]).all(axis=(1, 2))
    for tri in faces[tri_valid]:
        corners = xy[tri]
        cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate) + 1, 2 * int(dilate) + 1))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def render_shaded(points, faces, cam, size=None, base_color=(0.72, 0.74, 0.78),
                  light=(0.35, 0.55, 0.75), vertex_colors=None,
                  background=(0.10, 0.11, 0.13)):
    """Painter's algorithm; adequate for QA/panels, not photorealistic."""
    import cv2

    size = size or cam["image_hw"][0]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32)
    xy, valid = project(pts, cam)
    xyz = to_camera(pts, cam)
    canvas = np.zeros((int(size), int(size), 3), dtype=np.float64)
    canvas[:] = np.asarray(background, dtype=np.float64)
    if not valid.any():
        return (canvas * 255).astype(np.uint8)

    tri = pts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-12, None)
    light = _unit(light)
    shade = 0.25 + 0.75 * np.abs(normals @ light)
    depth = xyz[faces, 2].mean(1)
    order = np.argsort(-depth)
    base = np.asarray(base_color, dtype=np.float64)
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float64)
        face_colors = vc[faces].mean(1)
    else:
        face_colors = None
    for i in order:
        tri_idx = faces[i]
        if not valid[tri_idx].all():
            continue
        corners = xy[tri_idx]
        if not np.isfinite(corners).all():
            continue
        color = face_colors[i] if face_colors is not None else base
        color = np.clip(np.asarray(color) * shade[i], 0.0, 1.0)
        cv2.fillPoly(canvas, [np.round(corners).astype(np.int32)], color.tolist())
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def error_colors(err_mm, vmin=0.0, vmax=40.0):
    """Simple blue->yellow->red colormap for per-vertex errors (mm)."""
    e = np.clip((np.asarray(err_mm, dtype=np.float64) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 * e, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * e - 1.0) * 1.5, 0, 1)
    b = np.clip(1.0 - 1.5 * e, 0, 1)
    return np.stack([r, g, b], axis=1)
