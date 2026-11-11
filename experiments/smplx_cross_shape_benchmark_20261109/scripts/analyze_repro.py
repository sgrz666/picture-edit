"""Quantify the 10-image SHAPY rerun difference in beta and surface space."""
from __future__ import annotations

import json

import numpy as np

import config
import metrics

REPRO = config.RUN_ROOT / "runs" / "shapy_repro" / "demo_out"


def main():
    template, shapedirs, faces = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    full = {r["image_name"]: np.asarray(r["betas"], dtype=np.float64)
            for r in (json.loads(line) for line in open(config.PRED_SHAPY, encoding="utf8") if line.strip())}
    rows = []
    for p in sorted(REPRO.rglob("*.npz")):
        rel = p.relative_to(REPRO)
        if len(rel.parts) != 3:
            continue
        subject, img_type, stem = rel.parts[0], rel.parts[1], p.stem
        name = f"val/{subject}/{img_type}/{stem}.png"
        b2 = np.load(p)["betas"].astype(np.float64).reshape(-1)
        b1 = full[name]
        v1 = metrics.beta_to_verts(template, shapedirs, b1)[0]
        v2 = metrics.beta_to_verts(template, shapedirs, b2)[0]
        rows.append({
            "image_name": name,
            "beta_max_abs_diff": float(np.abs(b1 - b2).max()),
            "beta_l2": float(np.linalg.norm(b1 - b2)),
            "v2v_mm": metrics.v2v_mm(v2, v1),
            "vertex_max_mm": float(np.linalg.norm(v2 - v1, axis=1).max() * 1000),
        })
    rows.sort(key=lambda r: r["v2v_mm"], reverse=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("max beta", max(r["beta_max_abs_diff"] for r in rows),
          "max v2v", max(r["v2v_mm"] for r in rows),
          "max vertex", max(r["vertex_max_mm"] for r in rows))


if __name__ == "__main__":
    main()
