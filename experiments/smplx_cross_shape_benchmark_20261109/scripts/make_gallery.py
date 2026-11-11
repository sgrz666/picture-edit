"""Human-readable comparison sheets: lineup, real-photo swap, pose grid, views."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

import config
import metrics
import render
import smplx_pose

GALLERY = config.RUN_ROOT / "output" / "gallery"
CELL = 360
SMALL = 320


def font(size=14):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = font(14)
FONT_SMALL = font(12)
FONT_TITLE = font(20)


def compose(title, headers, rows, path, cell, label_w=250, cap_h=20):
    ncol = len(headers)
    row_h = cell + cap_h + 8
    top = 58
    W = label_w + ncol * cell + 16
    H = top + len(rows) * row_h + 10
    canvas = Image.new("RGB", (W, H), (16, 17, 20))
    d = ImageDraw.Draw(canvas)
    d.text((10, 12), title, font=FONT_TITLE, fill=(245, 245, 245))
    for j, h in enumerate(headers):
        d.text((label_w + j * cell + 8, 38), h, font=FONT, fill=(240, 210, 120))
    for i, (row_label, cells) in enumerate(rows):
        y = top + i * row_h
        d.text((10, y + cell // 2 - 8), row_label, font=FONT, fill=(220, 220, 220))
        for j, (img, cap) in enumerate(cells):
            x = label_w + j * cell
            if img is not None:
                canvas.paste(Image.fromarray(img), (x, y))
            if cap:
                d.text((x + 6, y + cell + 2), cap, font=FONT_SMALL, fill=(180, 220, 180))
    canvas.save(path)
    print("[gallery]", path)


def crop_person(rec, size):
    with open(rec["keypoint_path"], "r", encoding="utf8") as f:
        data = json.load(f)
    kp = np.asarray(data["people"][0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    ok = kp[:, 2] > 0.05
    if ok.sum() < 4:
        raise RuntimeError(f"no keypoints for {rec['image_path']}")
    x0, y0 = kp[ok, :2].min(0)
    x1, y1 = kp[ok, :2].max(0)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w, h) * 1.25
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = int(cx + side / 2), int(cy + side / 2)
    img = Image.open(rec["image_path"]).convert("RGB")
    W, H = img.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    img = img.crop((x0, y0, x1, y1))
    # letterbox to square
    side = max(img.size)
    square = Image.new("RGB", (side, side), (16, 17, 20))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return np.asarray(square.resize((size, size), Image.LANCZOS))


def load_p0_betas():
    z = np.load(config.OUT_P0 / "p0_betas.npz", allow_pickle=True)
    variants = defaultdict(dict)
    subjects = [str(s) for s in z["subject_id"]]
    for k in z.files:
        if k == "subject_id":
            continue
        v, sid = k.split("_", 1)
        variants[v][sid] = np.asarray(z[k], dtype=np.float64)
    return {v: dict(d) for v, d in variants.items()}, subjects


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    variants, subjects = load_p0_betas()
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    genders = yaml.safe_load(config.GENDERS.read_text())
    poses = json.loads(config.POSES.read_text(encoding="utf8"))["templates"]
    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    gt = np.load(config.GT_BETAS_10, allow_pickle=True)
    gt_betas = {str(s): np.asarray(b, dtype=np.float64) for s, b in zip(gt["subject_id"], gt["betas"])}
    variants["G4"] = gt_betas

    # P1 per-subject metrics for labels.
    p1 = {}
    for r in csv.DictReader(open(config.OUT_P1 / "p1_per_subject.csv", encoding="utf8")):
        if r["arm"] in ("G1", "G2"):
            p1[(r["arm"], r["subject_id"])] = (float(r["p2p20k_mm"]), float(r["height_mm"]))
    # P0 per-pair metrics for labels.
    p0 = {}
    for r in csv.DictReader(open(config.OUT_P0 / "p0_per_pair.csv", encoding="utf8")):
        p0[(r["pair_id"], r["pose_id"], int(r["view"]), r["arm"])] = (
            float(r["v2v_mm"]), float(r["iou"]), float(r["p2p20k_mm"]))

    wrapper = smplx_pose.SMPLXWrapper(config.SMPLX_DIR, num_betas=10, device="cpu")
    canon = {v: {sid: metrics.beta_to_verts(template, shapedirs, variants[v][sid])[0]
                 for sid in subjects} for v in ("G1", "G2", "G4")}
    posed = {v: {sid: [] for sid in subjects} for v in ("G1", "G2", "G4")}
    for v in ("G1", "G2", "G4"):
        for sid in subjects:
            for pose in poses:
                bp = np.asarray(pose["body_pose"], dtype=np.float32)[None]
                posed[v][sid].append(wrapper.vertices(
                    variants[v][sid][None], bp, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))[0])
    print("[gallery] meshes")

    # ---------- 1) canonical body-shape lineup ----------
    union = np.concatenate([canon[v][sid] for v in ("G1", "G2", "G4") for sid in subjects], axis=0)
    cam_line = render.make_camera(union, azimuth_deg=0, size=CELL, margin=0.06)
    headers = ["GT beta (10-D fit)", "SMPLest-X mean", "SHAPY mean"]
    for part, chunk in (("A", subjects[:5]), ("B", subjects[5:])):
        rows = []
        for sid in chunk:
            h_gt = float(np.linalg.norm(canon["G4"][sid][:, 1].max() - canon["G4"][sid][:, 1].min()) * 1000)
            cells = []
            for v, name in (("G4", "GTbeta"), ("G1", "SMPLest-X"), ("G2", "SHAPY")):
                img = render.render_shaded(canon[v][sid], faces, cam_line, size=CELL)
                if v == "G4":
                    cap = f"GTbeta  h={h_gt/1000:.3f} m"
                else:
                    p2p, he = p1[(v, sid)]
                    cap = f"{name}  P2P {p2p:.1f}  h {he:.1f} mm"
                cells.append((img, cap))
            rows.append((f"{sid}  GTbeta h={h_gt/1000:.3f} m", cells))
        compose(f"HBW body-shape lineup {part}: same scale, canonical T-pose",
                headers, rows, GALLERY / f"gallery_lineup_{part}.png", CELL)

    # ---------- 2) real photo + swap + GT + error ----------
    by_subject = defaultdict(list)
    for r in manifest:
        by_subject[r["subject_id"]].append(r)
    def photo(sid):
        recs = [r for r in by_subject[sid] if r["scene"] == "Photos_Lab"] or by_subject[sid]
        return crop_person(recs[0], CELL)

    pairs = [("017", "020"), ("018", "020"), ("023", "022")]
    # Shared camera for pose 0 / front.
    union0 = np.concatenate([posed[v][sid][0] for v in ("G1", "G2", "G4") for sid in subjects], axis=0)
    cam0 = render.make_camera(union0, azimuth_deg=0, size=CELL, margin=0.06)
    rows = []
    headers2 = ["A photo", "B photo", "G0: A beta", "G1: SMPLest-X", "G2: SHAPY", "GT beta"]
    for a, b in pairs:
        gt_mesh = posed["G4"][b][0]
        cells = [(photo(a), f"A {a} real photo"), (photo(b), f"B {b} real photo")]
        for arm, v, sid in (("G0", "G1", a), ("G1", "G1", b), ("G2", "G2", b), ("G4", "G4", b)):
            mesh = posed[v][sid][0]
            img = render.render_shaded(mesh, faces, cam0, size=CELL)
            if arm == "G4":
                cap = "GT reference"
            else:
                v2v, iou, p2p = p0[(f"{a}->{b}", poses[0]["pose_id"], 0, arm)]
                cap = f"{arm} v2v {v2v:.1f}  IoU {iou:.3f}"
            cells.append((img, cap))
        rows.append((f"{a} -> {b}", cells))
    compose("A/B real photos + shape swap + GT (shared pose 00, front; G1/G2 depend only on B)",
            headers2, rows, GALLERY / "gallery_swap_pairs.png", CELL)

    # Error heatmaps under the same grid.
    err_rows = []
    headers3 = ["", "", "G0 error", "G1 error", "G2 error", "GT"]
    for a, b in pairs:
        gt_mesh = posed["G4"][b][0]
        cells = [(None, ""), (None, "")]
        for v, sid in (("G1", a), ("G1", b), ("G2", b)):
            err = np.linalg.norm(posed[v][sid][0] - gt_mesh, axis=1) * 1000
            cells.append((render.render_shaded(posed[v][sid][0], faces, cam0, size=CELL,
                                               vertex_colors=render.error_colors(err, vmin=0, vmax=60)), "0-60 mm"))
        cells.append((render.render_shaded(gt_mesh, faces, cam0, size=CELL), "reference"))
        err_rows.append((f"{a} -> {b} error", cells))
    compose("Per-vertex error vs GT beta (blue 0 -> red 60 mm)",
            headers3, err_rows, GALLERY / "gallery_swap_errors.png", CELL)

    # ---------- 3) pose grid for the hardest subject 020 ----------
    sid = "020"
    headers4 = ["G1 SMPLest-X", "G2 SHAPY", "GT beta"]
    rows = []
    for pi, pose in enumerate(poses):
        union_p = np.concatenate([posed[v][s][pi] for v in ("G1", "G2", "G4") for s in subjects], axis=0)
        cam = render.make_camera(union_p, azimuth_deg=0, size=SMALL, margin=0.06)
        gt_mesh = posed["G4"][sid][pi]
        cells = []
        for v, name in (("G1", "G1"), ("G2", "G2")):
            err = np.linalg.norm(posed[v][sid][pi] - gt_mesh, axis=1) * 1000
            img = render.render_shaded(posed[v][sid][pi], faces, cam, size=SMALL,
                                       vertex_colors=render.error_colors(err, vmin=0, vmax=60))
            cells.append((img, f"{name} err 0-60 mm"))
        cells.append((render.render_shaded(gt_mesh, faces, cam, size=SMALL), "GT"))
        rows.append((f"{pose['pose_id']}  {pose['source_image'].split('/')[-2]}", cells))
    compose("Subject 020 across 6 shared poses (front): predicted vs GT beta",
            headers4, rows, GALLERY / "gallery_pose_grid_020.png", SMALL)

    # ---------- 4) three views for 018 -> 020 ----------
    a, b = "018", "020"
    headers5 = ["G1 SMPLest-X", "G2 SHAPY", "GT beta"]
    rows = []
    pose = poses[0]
    for vi, az in enumerate(config.VIEW_AZIMUTHS):
        union_v = np.concatenate([posed[v][s][0] for v in ("G1", "G2", "G4") for s in subjects], axis=0)
        cam = render.make_camera(union_v, azimuth_deg=az, size=CELL, margin=0.06)
        gt_mesh = posed["G4"][b][0]
        cells = []
        for v, name in (("G1", "G1"), ("G2", "G2")):
            v2v, iou, p2p = p0[(f"{a}->{b}", pose["pose_id"], int(az), name)]
            cells.append((render.render_shaded(posed[v][b][0], faces, cam, size=CELL),
                          f"{name} v2v {v2v:.1f}  IoU {iou:.3f}"))
        cells.append((render.render_shaded(gt_mesh, faces, cam, size=CELL), "GT"))
        rows.append((f"view {az} deg", cells))
    compose("018 -> 020: three views, pose 00",
            headers5, rows, GALLERY / "gallery_three_views_018_020.png", CELL)

    print("[gallery] done", GALLERY)


if __name__ == "__main__":
    main()
