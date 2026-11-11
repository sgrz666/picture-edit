"""P0: controlled A->B shape swap in six shared real poses and three views.

Arms:
  G0  source A's SMPLest-X mean beta (no swap)
  G1  B's SMPLest-X mean beta
  G2  B's SHAPY mean beta
  G3  B's SHAPY median beta
  G4  B's GT beta (identity / reference)

The GT reference for posed comparisons is SMPL-X(beta_GT, theta*).  The raw
scan is T-pose and is used only for the canonical measurement floor.
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

import numpy as np
import yaml

import config
import metrics
import render
import smplx_pose


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_betas(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["subject_id"]].append(np.asarray(r["betas"], dtype=np.float64))
    return {k: np.stack(v) for k, v in out.items()}


def jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def compose_panel(images, labels, title, path):
    from PIL import Image, ImageDraw

    h, w = images[0].shape[:2]
    header = 34
    canvas = Image.new("RGB", (w * len(images), h + header), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(240, 240, 240))
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(img), (i * w, header))
        draw.text((i * w + 8, header + 6), label, fill=(240, 220, 120))
    canvas.save(path)


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    genders = yaml.safe_load(config.GENDERS.read_text())
    gt = np.load(config.GT_BETAS_10, allow_pickle=True)
    subjects = [str(s) for s in gt["subject_id"]]
    gt_betas = {str(s): b for s, b in zip(gt["subject_id"], gt["betas"])}
    g1 = group_betas(load_jsonl(config.SMPLESTX_PRED))
    g2 = group_betas(load_jsonl(config.PRED_SHAPY))
    for sid in subjects:
        if sid not in g1 or sid not in g2:
            raise RuntimeError(f"missing predictions for {sid}")

    variants = {
        "G1": {sid: g1[sid].mean(0) for sid in subjects},
        "G2": {sid: g2[sid].mean(0) for sid in subjects},
        "G3": {sid: np.median(g2[sid], axis=0) for sid in subjects},
        "G4": {sid: np.asarray(gt_betas[sid], dtype=np.float64) for sid in subjects},
    }
    np.savez_compressed(
        config.OUT_P0 / "p0_betas.npz",
        subject_id=np.asarray(subjects),
        **{f"{k}_{sid}": v for k, d in variants.items() for sid, v in d.items()},
    )

    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    reg, defns, landmarks = metrics.load_measurement_resources(config.UTILITY_DIR)
    poses = json.loads(config.POSES.read_text(encoding="utf8"))["templates"]
    if len(poses) != config.POSE_COUNT:
        raise RuntimeError(f"expected {config.POSE_COUNT} poses, got {len(poses)}")

    # Canonical meshes and measurements (pose-independent).
    canon = {v: {sid: metrics.beta_to_verts(template, shapedirs, variants[v][sid])[0]
                 for sid in subjects} for v in variants}
    gt_meas = {}
    arm_meas = {v: {} for v in variants}
    raw_gt = {}
    for sid in subjects:
        rec = next(r for r in manifest if r["subject_id"] == sid)
        raw = np.load(rec["gt_path"]).astype(np.float64)
        raw_gt[sid] = raw
        gt_meas[sid] = metrics.mesh_measurements(raw, faces, defns, landmarks)
        for v in variants:
            arm_meas[v][sid] = metrics.mesh_measurements(canon[v][sid], faces, defns, landmarks)

    # Posed meshes: 4 variants x 10 subjects x 6 poses.
    wrapper = smplx_pose.SMPLXWrapper(config.SMPLX_DIR, num_betas=10, device="cpu")
    meshes = {v: {sid: [] for sid in subjects} for v in variants}
    for v in variants:
        for sid in subjects:
            for pose in poses:
                bp = np.asarray(pose["body_pose"], dtype=np.float32)[None]
                verts = wrapper.vertices(
                    variants[v][sid][None], bp,
                    np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32),
                )[0]
                meshes[v][sid].append(verts)
    print("[P0] meshes ready")

    # Cameras per pose and view, framed on the union of all variants/subjects.
    cams_mask = []
    cams_panel = []
    for pi in range(len(poses)):
        union = np.concatenate([meshes[v][sid][pi] for v in variants for sid in subjects], axis=0)
        cm, cp = [], []
        for az in config.VIEW_AZIMUTHS:
            cm.append(render.make_camera(union, azimuth_deg=az, size=config.MASK_SIZE, margin=0.07))
            cp.append(render.make_camera(union, azimuth_deg=az, size=512, margin=0.07))
        cams_mask.append(cm)
        cams_panel.append(cp)
    print("[P0] cameras ready")

    # Masks: 4 variants x 10 subjects x 6 poses x 3 views.
    masks = {v: {sid: [] for sid in subjects} for v in variants}
    for v in variants:
        for sid in subjects:
            for pi in range(len(poses)):
                view_masks = []
                for vi in range(len(config.VIEW_AZIMUTHS)):
                    m = render.mask_from_verts(meshes[v][sid][pi], faces, cams_mask[pi][vi], size=config.MASK_SIZE)
                    if not m.any():
                        raise RuntimeError(f"empty mask {v} {sid} pose{pi} view{vi}")
                    view_masks.append(m)
                masks[v][sid].append(view_masks)
    print("[P0] masks ready")

    # Directed pairs and canonical shape difference.
    pairs = [(a, b) for a in subjects for b in subjects if a != b]
    d_shape = {(a, b): metrics.v2v_mm(canon["G4"][a], canon["G4"][b]) for a, b in pairs}
    pair_rows = []
    arm_spec = {
        "G0": ("G1", "A"),
        "G1": ("G1", "B"),
        "G2": ("G2", "B"),
        "G3": ("G3", "B"),
        "G4": ("G4", "B"),
    }
    for a, b in pairs:
        group = "same" if genders[a] == genders[b] else "cross"
        for pi, pose in enumerate(poses):
            gt_mesh = meshes["G4"][b][pi]
            for vi, az in enumerate(config.VIEW_AZIMUTHS):
                gt_mask = masks["G4"][b][pi][vi]
                for arm, (v, who) in arm_spec.items():
                    sid = a if who == "A" else b
                    mesh = meshes[v][sid][pi]
                    mask = masks[v][sid][pi][vi]
                    cm = metrics.contour_metrics(mask, gt_mask)
                    pair_rows.append({
                        "pair_id": f"{a}->{b}",
                        "A": a, "B": b, "group": group,
                        "d_shape_canonical_mm": round(float(d_shape[(a, b)]), 4),
                        "pose_id": pose["pose_id"], "view": az, "arm": arm,
                        "v2v_mm": round(metrics.v2v_mm(mesh, gt_mesh), 4),
                        "p2p20k_mm": round(metrics.p2p20k_mm(mesh, gt_mesh, reg), 4),
                        "iou": round(cm["iou"], 5),
                        "chamfer_px": round(cm["chamfer_px"], 4),
                        "chamfer_norm": round(cm["chamfer_norm"], 6),
                        "mask_area": cm["mask_a_px"],
                    })
    print("[P0] pair rows", len(pair_rows))

    with open(config.OUT_P0 / "p0_per_pair.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    # Canonical measurement errors: 90 pairs x 5 arms.
    meas_rows = []
    for a, b in pairs:
        group = "same" if genders[a] == genders[b] else "cross"
        for arm, (v, who) in arm_spec.items():
            sid = a if who == "A" else b
            m = arm_meas[v][sid]
            g = gt_meas[b]
            meas_rows.append({
                "pair_id": f"{a}->{b}", "A": a, "B": b, "group": group, "arm": arm,
                "height_mm": round(abs(m["height"] - g["height"]) * 1000, 4),
                "chest_mm": round(abs(m["chest"] - g["chest"]) * 1000, 4),
                "waist_mm": round(abs(m["waist"] - g["waist"]) * 1000, 4),
                "hips_mm": round(abs(m["hips"] - g["hips"]) * 1000, 4),
                "mass_kg": round(abs(m["mass"] - g["mass"]), 5),
            })
    with open(config.OUT_P0 / "p0_measurements.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meas_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meas_rows)

    # Per-subject summary (B is the unit; average over A/pose/view).
    per_subject = []
    for b in subjects:
        for arm in arm_spec:
            sel = [r for r in pair_rows if r["B"] == b and r["arm"] == arm]
            ms = [r for r in meas_rows if r["B"] == b and r["arm"] == arm]
            per_subject.append({
                "B": b, "arm": arm, "n": len(sel),
                "v2v_mm": round(float(np.mean([r["v2v_mm"] for r in sel])), 4),
                "p2p20k_mm": round(float(np.mean([r["p2p20k_mm"] for r in sel])), 4),
                "iou": round(float(np.mean([r["iou"] for r in sel])), 5),
                "chamfer_norm": round(float(np.mean([r["chamfer_norm"] for r in sel])), 6),
                "height_mm": round(float(np.mean([r["height_mm"] for r in ms])), 4),
                "chest_mm": round(float(np.mean([r["chest_mm"] for r in ms])), 4),
                "waist_mm": round(float(np.mean([r["waist_mm"] for r in ms])), 4),
                "hips_mm": round(float(np.mean([r["hips_mm"] for r in ms])), 4),
                "mass_kg": round(float(np.mean([r["mass_kg"] for r in ms])), 5),
            })
    with open(config.OUT_P0 / "p0_per_subject.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_subject[0].keys()))
        writer.writeheader()
        writer.writerows(per_subject)

    # Summary and gates.
    from scipy.stats import spearmanr

    same_pairs = [p for p in pairs if genders[p[0]] == genders[p[1]]]
    cross_pairs = [p for p in pairs if genders[p[0]] != genders[p[1]]]
    summary = {
        "n_subjects": len(subjects),
        "n_pairs_same": len(same_pairs),
        "n_pairs_cross": len(cross_pairs),
        "n_pose": len(poses),
        "n_views": len(config.VIEW_AZIMUTHS),
        "n_pair_rows": len(pair_rows),
        "n_measurement_rows": len(meas_rows),
        "arms": {},
        "identity_v2v_max_mm": max(r["v2v_mm"] for r in pair_rows if r["arm"] == "G4"),
        "identity_iou_min": min(r["iou"] for r in pair_rows if r["arm"] == "G4"),
        "mask_area_min": min(r["mask_area"] for r in pair_rows),
        "mask_area_max": max(r["mask_area"] for r in pair_rows),
        "gt_fit_floor_canonical_mm": {
            sid: round(float(np.linalg.norm(metrics.beta_to_verts(template, shapedirs, variants["G4"][sid])[0] - raw_gt[sid], axis=1).mean() * 1000), 4)
            for sid in subjects
        },
    }
    for arm in arm_spec:
        sel = [r for r in pair_rows if r["arm"] == arm]
        summary["arms"][arm] = {
            "v2v_mm": round(float(np.mean([r["v2v_mm"] for r in sel])), 4),
            "p2p20k_mm": round(float(np.mean([r["p2p20k_mm"] for r in sel])), 4),
            "iou": round(float(np.mean([r["iou"] for r in sel])), 5),
            "chamfer_norm": round(float(np.mean([r["chamfer_norm"] for r in sel])), 6),
        }
    g0 = [r for r in pair_rows if r["arm"] == "G0"]
    rho, pval = spearmanr([r["v2v_mm"] for r in g0], [r["d_shape_canonical_mm"] for r in g0])
    summary["g0_vs_shape_difference_spearman"] = {"rho": float(rho), "p": float(pval), "n": len(g0)}
    # Per-pair mean version (avoids pose/view pseudo-replication).
    g0_by_pair = defaultdict(list)
    d_by_pair = {}
    for r in g0:
        g0_by_pair[r["pair_id"]].append(r["v2v_mm"])
        d_by_pair[r["pair_id"]] = r["d_shape_canonical_mm"]
    rho2, p2 = spearmanr([np.mean(v) for v in g0_by_pair.values()],
                         [d_by_pair[k] for k in g0_by_pair])
    summary["g0_vs_shape_difference_spearman_by_pair"] = {"rho": float(rho2), "p": float(p2), "n": len(g0_by_pair)}
    summary["gates"] = {
        "pairs_same_42": len(same_pairs) == 42,
        "pairs_cross_48": len(cross_pairs) == 48,
        "rows_complete": len(pair_rows) == len(pairs) * len(poses) * len(config.VIEW_AZIMUTHS) * len(arm_spec),
        "identity_zero": summary["identity_v2v_max_mm"] < 1e-6,
        "identity_iou_one": summary["identity_iou_min"] > 1 - 1e-6,
        "g0_positive_shape_corr": rho2 > 0.5,
        "mask_area_sane": summary["mask_area_min"] > 0 and summary["mask_area_max"] < config.MASK_SIZE * config.MASK_SIZE * 0.6,
    }
    (config.OUT_P0 / "p0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable))

    # Panels: three same-gender pairs (max/median/min shape difference), front view.
    same_sorted = sorted(same_pairs, key=lambda p: d_shape[p], reverse=True)
    chosen = [same_sorted[0], same_sorted[len(same_sorted) // 2], same_sorted[-1]]
    panel_dir = config.OUT_P0 / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for a, b in chosen:
        for pi, pose in enumerate(poses):
            gt_mesh = meshes["G4"][b][pi]
            imgs, labels = [], []
            for arm in ("G0", "G1", "G2", "G3"):
                v, who = arm_spec[arm]
                sid = a if who == "A" else b
                imgs.append(render.render_shaded(meshes[v][sid][pi], faces, cams_panel[pi][0], size=512))
                labels.append(f"{arm} ({sid})")
            imgs.append(render.render_shaded(gt_mesh, faces, cams_panel[pi][0], size=512))
            labels.append("GT beta")
            err = np.linalg.norm(meshes["G2"][b][pi] - gt_mesh, axis=1) * 1000
            imgs.append(render.render_shaded(meshes["G2"][b][pi], faces, cams_panel[pi][0],
                                             size=512, vertex_colors=render.error_colors(err)))
            labels.append("G2 error mm")
            compose_panel(imgs, labels,
                          f"{a} -> {b} | {pose['pose_id']} | front | d_shape={d_shape[(a,b)]:.1f}mm",
                          panel_dir / f"{a}_to_{b}_{pose['pose_id']}.png")
    # Three-view grid of the largest-difference pair (GT and G2 side by side).
    a, b = chosen[0]
    for pi, pose in enumerate(poses):
        for vi, az in enumerate(config.VIEW_AZIMUTHS):
            imgs = [
                render.render_shaded(meshes["G4"][b][pi], faces, cams_panel[pi][vi], size=512),
                render.render_shaded(meshes["G2"][b][pi], faces, cams_panel[pi][vi], size=512),
            ]
            compose_panel(imgs, ["GT beta", "SHAPY mean beta"],
                          f"{a} -> {b} | {pose['pose_id']} | view {az}",
                          panel_dir / f"views_{a}_to_{b}_{pose['pose_id']}_{az}.png")
    print("[P0] panels", panel_dir)


if __name__ == "__main__":
    main()
