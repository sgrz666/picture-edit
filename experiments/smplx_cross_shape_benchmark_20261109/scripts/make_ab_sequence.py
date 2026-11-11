"""Five-panel A->B sequence: A photo, B photo, A mesh, B mesh, A+B beta.

Only A's beta is estimated (SMPLest-X and SHAPY, multi-image mean).  B's beta
is the GT fit and is treated as known.  All meshes use the same shared pose, so
the last panel is identically B's mesh; that is the point of the identity
check, not a shape-accuracy result.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

import config
import metrics
import render
import smplx_pose
from make_gallery import compose, crop_person

OUT = config.RUN_ROOT / "output" / "ab_sequence"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    by_subject = defaultdict(list)
    for r in manifest:
        by_subject[r["subject_id"]].append(r)
    pose = next(p for p in json.loads(config.POSES.read_text(encoding="utf8"))["templates"]
                if p["pose_id"] == "pose_04")
    body_pose = np.asarray(pose["body_pose"], dtype=np.float32)[None]

    z = np.load(config.OUT_P0 / "p0_betas.npz", allow_pickle=True)
    betas = defaultdict(dict)
    for k in z.files:
        if k == "subject_id":
            continue
        v, sid = k.split("_", 1)
        betas[v][sid] = np.asarray(z[k], dtype=np.float64)

    p1 = {}
    for r in csv.DictReader(open(config.OUT_P1 / "p1_per_subject.csv", encoding="utf8")):
        if r["arm"] in ("G1", "G2"):
            p1[(r["arm"], r["subject_id"])] = (float(r["p2p20k_mm"]), int(r["n_images"]))
    fit = {}
    for r in csv.DictReader(open(config.DATA / "gt_fit_report.csv", encoding="utf8")):
        if int(r["num_betas"]) == 10:
            fit[r["subject_id"]] = float(r["residual_mean_mm"])

    wrapper = smplx_pose.SMPLXWrapper(config.SMPLX_DIR, num_betas=10, device="cpu")
    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    reg, _, _ = metrics.load_measurement_resources(config.UTILITY_DIR)

    def mesh(beta):
        return wrapper.vertices(beta[None], body_pose, np.zeros((1, 3), np.float32),
                                np.zeros((1, 3), np.float32))[0]

    pairs = [("017", "020"), ("018", "020"), ("023", "022")]
    headers = ["A real photo", "B real photo", "A mesh (estimated beta)",
               "B mesh (GT beta)", "A + B beta (swap)"]
    for a, b in pairs:
        b_gt = betas["G4"][b]
        b_mesh = mesh(b_gt)
        # Shared camera for this pair.
        union = np.concatenate([mesh(betas["G1"][a]), mesh(betas["G2"][a]), b_mesh, mesh(b_gt)], axis=0)
        cam = render.make_camera(union, azimuth_deg=0, size=360, margin=0.06)
        b_mask = render.mask_from_verts(b_mesh, faces, cam, size=360)
        photo_a = crop_person(([r for r in by_subject[a] if r["scene"] == "Photos_Lab"] or by_subject[a])[0], 360)
        photo_b = crop_person(([r for r in by_subject[b] if r["scene"] == "Photos_Lab"] or by_subject[b])[0], 360)
        rows = []
        for arm, arm_name in (("G1", "SMPLest-X"), ("G2", "SHAPY")):
            a_est = betas[arm][a]
            a_mesh = mesh(a_est)
            a_mask = render.mask_from_verts(a_mesh, faces, cam, size=360)
            p2p_a_gt, n_img = p1[(arm, a)]
            p2p_a_b = metrics.p2p20k_mm(a_mesh, b_mesh, reg)
            swap = mesh(b_gt)  # A's pose is the shared pose, B's beta
            swap_mask = render.mask_from_verts(swap, faces, cam, size=360)
            p2p_swap_b = metrics.p2p20k_mm(swap, b_mesh, reg)
            iou_a_b = metrics.mask_iou(a_mask, b_mask)
            cells = [
                (photo_a, f"A {a} (context)"),
                (photo_b, f"B {b} (context)"),
                (render.render_shaded(a_mesh, faces, cam, size=360, base_color=(0.55, 0.68, 0.90)),
                 f"A est {arm_name}\nvs A GT {p2p_a_gt:.1f} | vs B {p2p_a_b:.1f}"),
                (render.render_shaded(b_mesh, faces, cam, size=360),
                 f"B GT beta\nfit {fit[b]:.1f} mm"),
                (render.render_shaded(swap, faces, cam, size=360, base_color=(0.55, 0.82, 0.62)),
                 f"swap vs B {p2p_swap_b:.2f}\nIoU {metrics.mask_iou(swap_mask, b_mask):.3f}"),
            ]
            rows.append((f"{arm_name} (A: {n_img} imgs)", cells))
        title = (f"A->{b}: A's beta estimated, B's beta assumed GT; "
                 f"shared {pose['pose_id']} front. Panel 5 = panel 4 by construction.")
        path = OUT / f"ab_sequence_{a}_{b}.png"
        compose(title, headers, rows, path, 360, cap_h=38)
        print("[ab]", path)

    print("[ab] done", OUT)


if __name__ == "__main__":
    main()
