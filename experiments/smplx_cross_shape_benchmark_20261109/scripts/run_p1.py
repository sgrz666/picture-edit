"""P1: per-image and per-subject shape errors for SMPLest-X / SHAPY / fusion.

Primary unit is the subject: images within a subject are not independent.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
import metrics


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_vertices(v, gt, gt_meas, gt_beta, beta, reg, defns, landmarks):
    meas = metrics.mesh_measurements(v, GT_FACES, defns, landmarks)
    return {
        "v2v_mm": metrics.v2v_mm(v, gt),
        "p2p20k_mm": metrics.p2p20k_mm(v, gt, reg),
        "height_mm": abs(meas["height"] - gt_meas["height"]) * metrics.MM,
        "chest_mm": abs(meas["chest"] - gt_meas["chest"]) * metrics.MM,
        "waist_mm": abs(meas["waist"] - gt_meas["waist"]) * metrics.MM,
        "hips_mm": abs(meas["hips"] - gt_meas["hips"]) * metrics.MM,
        "mass_kg": abs(meas["mass"] - gt_meas["mass"]),
        "beta_l2": metrics.beta_l2(beta, gt_beta),
    }


GT_FACES = None


def main():
    global GT_FACES
    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    GT_FACES = faces
    reg, defns, landmarks = metrics.load_measurement_resources(config.UTILITY_DIR)

    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    gt_b = np.load(config.GT_BETAS_10, allow_pickle=True)
    gt_betas = {str(s): b for s, b in zip(gt_b["subject_id"], gt_b["betas"])}
    gt_cache = {}
    gt_meas = {}
    for sid, path in sorted({r["subject_id"]: r["gt_path"] for r in manifest}.items()):
        g = np.load(path).astype(np.float64)
        gt_cache[sid] = g
        gt_meas[sid] = metrics.mesh_measurements(g, faces, defns, landmarks)

    g1 = {r["image_name"]: np.asarray(r["betas"], dtype=np.float64) for r in load_jsonl(config.SMPLESTX_PRED)}
    arms = {"G1": g1}
    if config.PRED_SHAPY.is_file():
        g2rows = load_jsonl(config.PRED_SHAPY)
        g2 = {}
        for r in g2rows:
            b = r.get("betas")
            if b is not None and len(b) == 10 and np.isfinite(b).all():
                g2[r["image_name"]] = np.asarray(b, dtype=np.float64)
        if g2:
            arms["G2"] = g2
    print("[P1] arms", {k: len(v) for k, v in arms.items()})

    per_image = []
    betas_by_arm_subject = defaultdict(lambda: defaultdict(list))
    for arm, preds in arms.items():
        for rec in manifest:
            name = rec["image_name"]
            if name not in preds:
                continue
            beta = preds[name]
            sid = rec["subject_id"]
            v = metrics.beta_to_verts(template, shapedirs, beta)[0]
            row = eval_vertices(v, gt_cache[sid], gt_meas[sid], gt_betas[sid], beta, reg, defns, landmarks)
            row.update({"arm": arm, "image_name": name, "subject_id": sid, "scene": rec["scene"]})
            per_image.append(row)
            betas_by_arm_subject[arm][sid].append(beta)

    # Fusion arms (mean/median of per-image betas), one row per subject.
    fusion_rows = []
    for arm, by_subject in betas_by_arm_subject.items():
        for sid, betas in sorted(by_subject.items()):
            b = np.stack(betas)
            for kind, beta in (("mean", b.mean(0)), ("median", np.median(b, 0))):
                v = metrics.beta_to_verts(template, shapedirs, beta)[0]
                row = eval_vertices(v, gt_cache[sid], gt_meas[sid], gt_betas[sid], beta, reg, defns, landmarks)
                row.update({"arm": f"{arm}_{kind}", "image_name": f"__subject_{kind}__",
                            "subject_id": sid, "scene": "all"})
                fusion_rows.append(row)

    # GT-beta floor.
    floor_rows = []
    for sid in sorted(gt_betas):
        beta = gt_betas[sid]
        v = metrics.beta_to_verts(template, shapedirs, beta)[0]
        row = eval_vertices(v, gt_cache[sid], gt_meas[sid], gt_betas[sid], beta, reg, defns, landmarks)
        row.update({"arm": "G4", "image_name": "__gt_beta__", "subject_id": sid, "scene": "all"})
        floor_rows.append(row)

    rows = per_image + fusion_rows + floor_rows
    fields = ["arm", "subject_id", "scene", "image_name", "v2v_mm", "p2p20k_mm",
              "height_mm", "chest_mm", "waist_mm", "hips_mm", "mass_kg", "beta_l2"]
    with open(config.OUT_P1 / "p1_per_image.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in fields})

    # Per-subject means (single-image arms) and single rows for fusion/floor.
    per_subject = []
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_subject = defaultdict(list)
        for r in arm_rows:
            by_subject[r["subject_id"]].append(r)
        for sid, items in sorted(by_subject.items()):
            d = {"arm": arm, "subject_id": sid, "n_images": len(items)}
            for k in ("v2v_mm", "p2p20k_mm", "height_mm", "chest_mm", "waist_mm", "hips_mm", "mass_kg", "beta_l2"):
                d[k] = round(float(np.mean([x[k] for x in items])), 6)
            per_subject.append(d)
    pfields = ["arm", "subject_id", "n_images", "v2v_mm", "p2p20k_mm", "height_mm",
               "chest_mm", "waist_mm", "hips_mm", "mass_kg", "beta_l2"]
    with open(config.OUT_P1 / "p1_per_subject.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=pfields)
        writer.writeheader()
        writer.writerows(per_subject)

    # Image-weighted vs subject-weighted summaries.
    summary = {"arms": sorted({r["arm"] for r in rows}), "n_images": len(manifest), "n_subjects": len(gt_betas)}
    for arm in summary["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        subj_rows = [r for r in per_subject if r["arm"] == arm]
        summary[arm] = {}
        for k in ("v2v_mm", "p2p20k_mm", "height_mm", "chest_mm", "waist_mm", "hips_mm", "mass_kg", "beta_l2"):
            summary[arm]["image_weighted_" + k] = round(float(np.mean([r[k] for r in arm_rows])), 4)
            summary[arm]["subject_weighted_" + k] = round(float(np.mean([r[k] for r in subj_rows])), 4)
        summary[arm]["n"] = len(arm_rows)

    # Stability and G3 gain.
    stab = {}
    for arm in ("G1", "G2"):
        if arm in betas_by_arm_subject:
            stab[arm] = metrics.stability_summary(betas_by_arm_subject[arm])
    (config.OUT_P1 / "p1_stability.json").write_text(json.dumps(stab, ensure_ascii=False, indent=2), encoding="utf8")

    def subject_values(arm, key):
        return np.asarray([r[key] for r in per_subject if r["arm"] == arm], dtype=np.float64)

    paired = {}
    if "G2" in arms:
        for key in ("p2p20k_mm", "v2v_mm", "height_mm", "chest_mm", "waist_mm", "hips_mm", "mass_kg"):
            a = subject_values("G1", key)
            b = subject_values("G2", key)
            if len(a) == len(b) and len(a) >= 2:
                paired[f"G2_vs_G1_{key}"] = metrics.paired_summary(a, b, seed=config.SEED)
    for arm in ("G1", "G2"):
        for kind in ("mean", "median"):
            name = f"{arm}_{kind}"
            if name in summary and arm in summary:
                paired[f"{name}_vs_{arm}_p2p20k_mm"] = metrics.paired_summary(
                    subject_values(arm, "p2p20k_mm"), subject_values(name, "p2p20k_mm"), seed=config.SEED)
    summary["paired"] = paired
    summary["stability_icc_mean"] = {k: v.get("_icc_mean") for k, v in stab.items()}

    # A1 audit against the October 2026 run (G1 only).
    audit = {}
    if config.SMPLESTX_PER_IMAGE.is_file():
        old = {}
        with open(config.SMPLESTX_PER_IMAGE, "r", encoding="utf8") as f:
            for r in csv.DictReader(f):
                old[r["image_name"]] = r
        diffs_v2v, diffs_p2p = [], []
        for r in per_image:
            if r["arm"] != "G1" or r["image_name"] not in old:
                continue
            diffs_v2v.append(abs(r["v2v_mm"] - float(old[r["image_name"]]["v2v_mm"])))
            diffs_p2p.append(abs(r["p2p20k_mm"] - float(old[r["image_name"]]["p2p20k_mm"])))
        audit["n_compared"] = len(diffs_v2v)
        audit["v2v_max_abs_diff_mm"] = float(max(diffs_v2v)) if diffs_v2v else None
        audit["p2p20k_max_abs_diff_mm"] = float(max(diffs_p2p)) if diffs_p2p else None
        audit["v2v_mean_abs_diff_mm"] = float(np.mean(diffs_v2v)) if diffs_v2v else None
        audit["p2p20k_mean_abs_diff_mm"] = float(np.mean(diffs_p2p)) if diffs_p2p else None
        old_summary = json.loads(config.SMPLESTX_SUMMARY.read_text(encoding="utf8"))
        audit["old_subject_weighted"] = {k: old_summary.get("subject_weighted_" + k)
                                         for k in ("v2v_mm", "p2p20k_mm", "height_mm")}
        audit["new_subject_weighted"] = {k: summary["G1"].get("subject_weighted_" + k)
                                         for k in ("v2v_mm", "p2p20k_mm", "height_mm")}
    summary["audit_g1"] = audit
    (config.OUT_P1 / "p1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[P1] wrote", config.OUT_P1)


if __name__ == "__main__":
    config.OUT_P1.mkdir(parents=True, exist_ok=True)
    main()
