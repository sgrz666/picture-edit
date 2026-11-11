"""P0': choose 6 shared real pose templates from the 705 HBW estimates.

Clusters body_pose (63-D) after PCA, picks medoids from diverse clusters and
renders a QA image per template.  Global orientation is zeroed so the three
benchmark views are controlled by the camera, not by the source photo.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

import config
import render
import smplx_pose


def load_params():
    # Materialise every array once: NpzFile decompresses the whole member on
    # each __getitem__, and mesh_cam is 80 MB.
    with np.load(config.SMPLESTX_PARAMS, allow_pickle=True) as f:
        z = {k: f[k] for k in f.files}
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    if len(z["betas"]) != len(manifest):
        raise RuntimeError("params/manifest length mismatch")
    return z, manifest


def validity(z, idx, j_regressor):
    """Project exported camera-space joints; joint_proj_model is heatmap-grid
    coordinates (divided by input_body_shape), not image pixels."""
    mesh = z["mesh_cam"][idx].astype(np.float64)
    if not np.isfinite(mesh).all() or (mesh[:, 2] <= 0).any():
        return False, "camera_depth"
    joints = np.asarray(j_regressor @ mesh)
    f = z["focal_length_xy"][idx].astype(np.float64)
    c = z["princpt"][idx].astype(np.float64)
    xy = joints[:, :2] / (joints[:, 2:3] + 1e-4) * f + c
    xy = xy[:25]
    w = float(z["width"][idx][0])
    h = float(z["height"][idx][0])
    if not np.isfinite(xy).all() or w <= 0 or h <= 0:
        return False, "nonfinite"
    margin_x, margin_y = 0.01 * w, 0.01 * h
    if (xy[:, 0] < margin_x).any() or (xy[:, 0] > w - margin_x).any():
        return False, "x_out"
    if (xy[:, 1] < margin_y).any() or (xy[:, 1] > h - margin_y).any():
        return False, "y_out"
    bbox_w = xy[:, 0].max() - xy[:, 0].min()
    bbox_h = xy[:, 1].max() - xy[:, 1].min()
    if bbox_h < 0.40 * h or bbox_w < 0.10 * w:
        return False, "too_small"
    # Upright: head joint (15) above pelvis (0); feet (10,11) below knees (4,5).
    if not (xy[15, 1] < xy[0, 1] < xy[4, 1]):
        return False, "not_upright"
    if not (xy[4, 1] < xy[10, 1] and xy[5, 1] < xy[11, 1]):
        return False, "legs_not_down"
    return True, "ok"


def medoid(features, members):
    sub = features[members]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    return members[int(np.argmin(d.mean(1)))]


def main():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    z, manifest = load_params()
    smplx_npz = np.load(config.SMPLX_NEUTRAL, allow_pickle=True)
    j_regressor = smplx_npz["J_regressor"]
    valid_idx, reasons = [], Counter()
    for i in range(len(manifest)):
        ok, why = validity(z, i, j_regressor)
        reasons[why] += 1
        if ok:
            valid_idx.append(i)
    if len(valid_idx) < 60:
        raise SystemExit(f"only {len(valid_idx)} valid poses; reasons={dict(reasons)}")

    body = z["body_pose"][valid_idx].astype(np.float64)
    body = body - body.mean(0, keepdims=True)
    pca = PCA(n_components=min(20, body.shape[1]), random_state=config.SEED)
    feat = pca.fit_transform(body)
    k = min(12, max(6, len(valid_idx) // 20))
    km = KMeans(n_clusters=k, random_state=config.SEED, n_init=20).fit(feat)
    labels = km.labels_
    clusters = {c: np.where(labels == c)[0] for c in range(k)}
    clusters = {c: m for c, m in clusters.items() if len(m) >= 5}

    cands = []
    for c, members in clusters.items():
        m = medoid(feat, members)
        idx = valid_idx[int(m)]
        sid = manifest[idx]["subject_id"]
        cands.append({"cluster": int(c), "size": int(len(members)), "member": int(m),
                      "index": int(idx), "subject_id": sid,
                      "image_name": manifest[idx]["image_name"]})
    if not cands:
        raise SystemExit("no usable clusters")

    # Greedy diversity: start from the largest, then maximise distance to the
    # selected set while preferring a new subject.
    selected = [max(cands, key=lambda c: (c["size"], c["subject_id"]))]
    pool = [c for c in cands if c is not selected[0]]
    while len(selected) < config.POSE_COUNT and pool:
        def score(c):
            dmin = min(float(np.linalg.norm(feat[c["member"]] - feat[s["member"]])) for s in selected)
            new_subject = c["subject_id"] not in {s["subject_id"] for s in selected}
            return (new_subject, dmin * (1.0 + 0.05 * np.log1p(c["size"])))
        nxt = max(pool, key=score)
        selected.append(nxt)
        pool.remove(nxt)
    if len(selected) < config.POSE_COUNT:
        raise SystemExit(f"only {len(selected)} diverse pose clusters")

    selected = sorted(selected, key=lambda c: c["cluster"])
    wrapper = smplx_pose.SMPLXWrapper(config.SMPLX_DIR, num_betas=10, device="cpu")
    qa_dir = config.OUT_P0 / "poses"
    qa_dir.mkdir(parents=True, exist_ok=True)
    templates = []
    for j, c in enumerate(selected):
        idx = c["index"]
        body_pose = z["body_pose"][idx].astype(np.float64)
        body_pose = body_pose - 0.0
        verts = wrapper.vertices(np.zeros((1, 10), np.float32), body_pose[None], np.zeros((1, 3), np.float32))[0]
        cam = render.make_camera(verts, azimuth_deg=0.0, size=config.PANEL_SIZE, margin=0.06)
        rgba = render.render_shaded(verts, np.load(config.SMPLX_NEUTRAL, allow_pickle=True)["f"],
                                    cam, size=config.PANEL_SIZE)
        from PIL import Image
        Image.fromarray(rgba).save(qa_dir / f"qa_pose_{j:02d}.png")
        templates.append({
            "pose_id": f"pose_{j:02d}",
            "cluster": c["cluster"],
            "cluster_size": c["size"],
            "source_index": idx,
            "source_image": c["image_name"],
            "subject_id": c["subject_id"],
            "body_pose": body_pose.tolist(),
            "global_orient": [0.0, 0.0, 0.0],
            "transl": [0.0, 0.0, 0.0],
            "qa_camera": cam,
        })
        print(f"[POSE {j}] cluster={c['cluster']} size={c['size']} subject={c['subject_id']} image={c['image_name']}")

    payload = {
        "seed": config.SEED,
        "n_valid": len(valid_idx),
        "validity_reasons": dict(reasons),
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "clusters_kept": len(clusters),
        "templates": templates,
    }
    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    config.POSES.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf8",
    )
    print("[P0'] wrote", config.POSES, "valid", len(valid_idx), "reasons", dict(reasons))


if __name__ == "__main__":
    main()
