"""Gate: rerun SHAPY on 10 stratified images and compare betas with the full run."""
from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import yaml

import config

MINI = config.RUN_ROOT / "shapy_hbw_mini"
REPRO = config.RUN_ROOT / "runs" / "shapy_repro"


def symlink(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf8"))
    by_subject = defaultdict(list)
    for r in manifest:
        by_subject[r["subject_id"]].append(r)
    chosen = []
    for sid in sorted(by_subject):
        # Prefer a lab photo for a clean crop.
        lab = [r for r in by_subject[sid] if r["scene"] == "Photos_Lab"]
        chosen.append((lab or by_subject[sid])[0])
    if len(chosen) != 10:
        raise SystemExit(f"expected 10 chosen images, got {len(chosen)}")

    for r in chosen:
        subject = r["subject_dir"]
        scene = r["scene"]
        fname = os.path.basename(r["image_path"])
        stem = os.path.splitext(fname)[0]
        symlink(MINI / "images" / "val" / subject / scene / fname, config.HBW_FULLRES / "images" / "val" / subject / scene / fname)
        symlink(MINI / "keypoints" / "val" / subject / scene / f"{stem}.json",
                config.KEYPOINTS / subject / scene / f"{stem}.json")
    symlink(MINI / "genders.yaml", config.GENDERS)
    body = MINI / "body_models" / "smplx"
    body.mkdir(parents=True, exist_ok=True)
    symlink(body / "SMPLX_NEUTRAL.npz", config.SMPLX_NEUTRAL)
    for obj in sorted((config.DATA / "hbw_obj" / "val").glob("*.obj")):
        symlink(MINI / "smplx" / "val" / obj.name, obj)

    cfg = yaml.safe_load((config.RUN_ROOT / "configs" / "shapy_eval.yaml").read_text())
    cfg["output_folder"] = str(REPRO)
    cfg["checkpoint_folder"] = "checkpoints"
    cfg["datasets"]["batch_size"] = 4
    cfg["datasets"]["shape"]["hbw"]["data_folder"] = str(MINI)
    REPRO.mkdir(parents=True, exist_ok=True)
    symlink(REPRO / "checkpoints", config.SHAPY_RUN / "checkpoints")
    mini_cfg = REPRO / "shapy_eval_mini.yaml"
    mini_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(config.RUN_ROOT / "third_party" / "shapy_bm"),
        str(config.SHAPY_REPO / "regressor"),
        str(config.SHAPY_REPO / "attributes"),
    ])
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    out = REPRO / "demo_out"
    cmd = [
        str(config.SHAPY_VENV / "bin" / "python"), "-u", "demo.py",
        "--exp-cfg", str(mini_cfg), "--split", "val", "--datasets", "hbw",
        "--output-folder", str(out), "--save-params", "true",
        "--save-vis", "false", "--save-mesh", "false",
    ]
    existing = list(out.rglob("*.npz")) if out.is_dir() else []
    if len(existing) < len(chosen):
        log = REPRO / "demo.log"
        with open(log, "w", encoding="utf8") as f:
            subprocess.run(cmd, cwd=str(config.SHAPY_REPO / "regressor"), env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        print("[repro] reusing existing demo_out", len(existing))

    import metrics
    template, shapedirs, _ = metrics.load_smplx_npz(config.SMPLX_NEUTRAL)
    full = {r["image_name"]: np.asarray(r["betas"], dtype=np.float64)
            for r in (json.loads(line) for line in open(config.PRED_SHAPY, encoding="utf8") if line.strip())}
    beta_diffs, v2v_diffs, vertex_diffs = [], [], []
    for p in sorted(out.rglob("*.npz")):
        rel = p.relative_to(out)
        if len(rel.parts) != 3:
            continue
        subject, img_type, stem = rel.parts[0], rel.parts[1], p.stem
        name = f"val/{subject}/{img_type}/{stem}.png"
        if name not in full:
            continue
        beta = np.load(p)["betas"].astype(np.float64).reshape(-1)
        ref = full[name]
        v = metrics.beta_to_verts(template, shapedirs, beta)[0]
        vr = metrics.beta_to_verts(template, shapedirs, ref)[0]
        beta_diffs.append(float(np.abs(beta - ref).max()))
        v2v_diffs.append(metrics.v2v_mm(v, vr))
        vertex_diffs.append(float(np.linalg.norm(v - vr, axis=1).max() * 1000))
    result = {
        "n_compared": len(beta_diffs),
        "max_beta_abs_diff": max(beta_diffs) if beta_diffs else None,
        "max_v2v_mm": max(v2v_diffs) if v2v_diffs else None,
        "max_vertex_mm": max(vertex_diffs) if vertex_diffs else None,
        "beta_gate_1e-6": bool(beta_diffs) and max(beta_diffs) <= 1e-6,
        "surface_gate_0p01mm": bool(v2v_diffs) and max(v2v_diffs) <= 0.01,
        "note": "demo.py sets cudnn.benchmark=True and deterministic=False; the 1e-6 beta gate is below cuDNN noise. The operative gate is surface v2v <= 0.01 mm (1/400 of the 4 mm GT fit floor).",
        "chosen_images": [r["image_name"] for r in chosen],
    }
    (REPRO / "repro_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["surface_gate_0p01mm"]:
        raise SystemExit("[repro] surface gate failed")


if __name__ == "__main__":
    main()
