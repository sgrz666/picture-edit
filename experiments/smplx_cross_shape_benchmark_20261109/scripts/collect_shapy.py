"""Collect SHAPY demo npz into predictions.jsonl and check coverage/template."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import config
import metrics

DEMO = config.SHAPY_RUN / "demo_out"


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    by_name = {r["image_name"]: r for r in manifest}
    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    rows = []
    missing = []
    bad = []
    max_vdiff = 0.0
    checked = 0
    for p in sorted(DEMO.rglob("*.npz")):
        rel = p.relative_to(DEMO)
        if len(rel.parts) != 3:
            bad.append(str(rel))
            continue
        subject, img_type, stem = rel.parts[0], rel.parts[1], p.stem
        image_name = f"val/{subject}/{img_type}/{stem}.png"
        rec = by_name.get(image_name)
        if rec is None:
            missing.append(image_name)
            continue
        z = np.load(p, allow_pickle=True)
        if "betas" not in z.files:
            bad.append(f"{rel}: no betas")
            continue
        beta = np.asarray(z["betas"], dtype=np.float64).reshape(-1)
        if beta.shape != (10,) or not np.isfinite(beta).all():
            bad.append(f"{rel}: bad betas {beta.shape}")
            continue
        if checked < 10 and "v_shaped" in z.files:
            ours = metrics.beta_to_verts(template, shapedirs, beta)[0]
            v = np.asarray(z["v_shaped"], dtype=np.float64)
            if v.shape == ours.shape:
                max_vdiff = max(max_vdiff, float(np.abs(v - ours).max()))
                checked += 1
        rows.append({"image_name": image_name, "subject_id": rec["subject_id"],
                     "betas": beta.tolist()})
    rows.sort(key=lambda r: r["image_name"])
    config.PRED_SHAPY.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PRED_SHAPY, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    coverage = {
        "total_manifest": len(manifest),
        "predictions": len(rows),
        "coverage_pct": round(len(rows) * 100.0 / len(manifest), 2),
        "missing_images": missing,
        "bad_outputs": bad,
        "per_subject": dict(sorted(Counter(r["subject_id"] for r in rows).items())),
        "template_max_abs_diff_vs_shapy_v_shaped_m": max_vdiff,
        "template_checks": checked,
    }
    config.SHAPY_COVERAGE.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
