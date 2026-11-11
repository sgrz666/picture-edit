"""D0: prepare HBW validation data, manifest, coverage and SMPLest-X params.

Read-only with respect to the original HBW tree.  Extracts only the full-res
keypoints, genders and GT OBJ files into RUN_ROOT; images and smplx meshes are
symlinked.  The 18 SMPLest-X parameter shards are concatenated and checked
against predictions.jsonl (must be bit-identical in betas).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config

OFFICIAL_VAL_IMAGES = 781


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_tree(records, key: str) -> str:
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r["image_name"]):
        h.update(rec["image_name"].encode())
        h.update(bytes.fromhex(rec[key]))
    return h.hexdigest()


def extract_zip_member(z: zipfile.ZipFile, member: str, dest: Path):
    info = z.getinfo(member)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == info.file_size:
        return
    tmp = dest.with_name(dest.name + ".extracting")
    with z.open(info) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if tmp.stat().st_size != info.file_size:
        raise RuntimeError(f"size mismatch after extract: {dest}")
    os.replace(tmp, dest)


def ensure_symlink(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink {link}")
    link.symlink_to(target, target_is_directory=True)


def load_predictions():
    rows = []
    with open(config.SMPLESTX_PRED, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len({r["image_name"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate image_name in predictions.jsonl")
    return rows


def load_params():
    import torch

    paths = sorted(glob.glob(str(config.SMPLESTX_PARAMS_GLOB)))
    if not paths:
        raise RuntimeError(f"no SMPLest-X params shards match {config.SMPLESTX_PARAMS_GLOB}")
    keys = ("global_orient", "body_pose", "betas", "joint_proj_model", "mesh_cam", "transl",
            "focal_length_xy", "princpt", "width", "height", "render_bbox")
    acc = defaultdict(list)
    for p in paths:
        params = torch.load(p, map_location="cpu", weights_only=False)
        for k in keys:
            if k not in params:
                raise RuntimeError(f"{p} missing {k}")
            acc[k].append(params[k].detach().cpu().numpy())
    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    return out, paths


def main():
    config.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config.DATA.mkdir(parents=True, exist_ok=True)
    config.LOGS.mkdir(parents=True, exist_ok=True)
    config.CACHE.mkdir(parents=True, exist_ok=True)

    ensure_symlink(config.DATA / "hbw_images", config.HBW_FULLRES / "images" / "val")
    ensure_symlink(config.DATA / "hbw_smplx", config.HBW_FULLRES / "smplx" / "val")
    ensure_symlink(config.DATA / "smplx_model", config.SMPLX_DIR)

    preds = load_predictions()
    if len(preds) != 705:
        raise RuntimeError(f"expected 705 predictions, got {len(preds)}")

    params, shard_paths = load_params()
    if len(params["betas"]) != len(preds):
        raise RuntimeError(f"params rows {len(params['betas'])} != predictions {len(preds)}")
    pred_betas = np.asarray([r["betas"] for r in preds], dtype=np.float32)
    diff = float(np.abs(params["betas"].astype(np.float32) - pred_betas).max())
    if diff != 0.0:
        raise RuntimeError(f"params betas do not match predictions.jsonl: maxdiff={diff}")

    # Extract keypoints / genders / GT OBJ from the official archive.
    extracted_kp = extracted_obj = 0
    with zipfile.ZipFile(config.HBW_ZIP) as z:
        for member in z.namelist():
            if member.startswith("HBW/keypoints/val/") and member.endswith(".json"):
                rel = member[len("HBW/keypoints/val/"):]
                extract_zip_member(z, member, config.KEYPOINTS / rel)
                extracted_kp += 1
            elif member == "HBW/genders.yaml":
                extract_zip_member(z, member, config.GENDERS)
            elif member.startswith("HBW/smplx/val/") and member.endswith(".obj"):
                rel = member[len("HBW/smplx/val/"):]
                extract_zip_member(z, member, config.DATA / "hbw_obj" / "val" / rel)
                extracted_obj += 1

    records = []
    for idx, pred in enumerate(preds):
        image_name = pred["image_name"]
        parts = Path(image_name).parts
        if len(parts) != 4 or parts[0] != "val":
            raise RuntimeError(f"unexpected image_name {image_name!r}")
        subject_dir, scene, filename = parts[1], parts[2], parts[3]
        subject_id = subject_dir.split("_")[0]
        stem = Path(filename).stem
        image_path = config.HBW_FULLRES / "images" / "val" / subject_dir / scene / filename
        gt_path = config.HBW_FULLRES / "smplx" / "val" / f"{subject_id}.npy"
        kp_path = config.KEYPOINTS / subject_dir / scene / f"{stem}.json"
        for p in (image_path, gt_path, kp_path):
            if not p.is_file():
                raise RuntimeError(f"missing {p} for {image_name}")
        records.append({
            "index": idx,
            "image_name": image_name,
            "image_path": str(image_path),
            "subject_id": subject_id,
            "subject_dir": subject_dir,
            "scene": scene,
            "gt_path": str(gt_path),
            "keypoint_path": str(kp_path),
            "sha256_image": sha256_file(image_path),
            "sha256_gt": sha256_file(gt_path),
            "sha256_keypoint": sha256_file(kp_path),
        })

    gt_ids = sorted({r["subject_id"] for r in records})
    if len(gt_ids) != 10:
        raise RuntimeError(f"expected 10 subjects, got {gt_ids}")
    per_subject = Counter(r["subject_id"] for r in records)
    per_scene = Counter(r["scene"] for r in records)
    if min(per_subject.values()) < 20:
        raise RuntimeError(f"subject with <20 images: {per_subject}")

    config.MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")

    coverage = {
        "n_images": len(records),
        "n_subjects": len(gt_ids),
        "subjects": gt_ids,
        "per_subject": dict(sorted(per_subject.items())),
        "per_scene": dict(sorted(per_scene.items())),
        "official_val_images": OFFICIAL_VAL_IMAGES,
        "coverage_pct": round(len(records) * 100.0 / OFFICIAL_VAL_IMAGES, 2),
        "keypoints_extracted": extracted_kp,
        "keypoints_used": len(records),
        "keypoints_extra_unmatched": extracted_kp - len(records),
        "gt_obj_extracted": extracted_obj,
        "note": "HBW.zip contains 705 val images; the published val split is 781, so 90.27% coverage is a hard ceiling for this run.",
    }
    config.COVERAGE.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf8")

    np.savez_compressed(
        config.SMPLESTX_PARAMS,
        image_name=np.asarray([r["image_name"] for r in records]),
        **{k: v.astype(np.float32) for k, v in params.items()},
    )

    sha = {
        "hbw_zip": sha256_file(config.HBW_ZIP),
        "smplx_neutral": sha256_file(config.SMPLX_NEUTRAL),
        "predictions_jsonl": sha256_file(config.SMPLESTX_PRED),
        "params_shards": {str(Path(p).name): sha256_file(Path(p)) for p in shard_paths},
        "manifest_images_digest": sha256_tree(records, "sha256_image"),
        "manifest_gt_digest": sha256_tree(records, "sha256_gt"),
        "manifest_keypoints_digest": sha256_tree(records, "sha256_keypoint"),
        "gt_betas_10": None,
        "gt_betas_300": None,
    }
    config.SHA256.write_text(json.dumps(sha, ensure_ascii=False, indent=2), encoding="utf8")

    print("[D0] subjects", len(gt_ids), per_subject)
    print("[D0] scenes", dict(per_scene))
    print("[D0] coverage", coverage["coverage_pct"], "%")
    print("[D0] keypoints extracted/used/extra", extracted_kp, len(records), extracted_kp - len(records))
    print("[D0] params rows", len(params["betas"]), "betas maxdiff", diff)
    print("[D0] manifest", config.MANIFEST)


if __name__ == "__main__":
    main()
