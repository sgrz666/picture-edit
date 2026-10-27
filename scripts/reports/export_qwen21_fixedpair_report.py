"""Export existing fixed-pair evidence and original images; never run a model."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import tarfile

ROOTS = {
    "tiktok": Path("/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1"),
    "hi4d": Path("/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def latest(paths):
    return max(paths, key=lambda p: p.stat().st_mtime_ns, default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".tar.gz").exists():
        raise FileExistsError("Choose a new version; evidence packages are immutable")
    if not a.output.resolve().is_relative_to("/data/shangguanrz"):
        raise ValueError("New remote artifacts belong on the persistent volume")
    a.output.mkdir(parents=True)
    assets = a.output / "assets"
    evidence = a.output / "evidence"
    assets.mkdir()
    evidence.mkdir()
    images = {}

    def image(path, expected=None):
        from PIL import Image
        path = Path(path)
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError("Image SHA conflict: " + str(path))
        name = digest + path.suffix.lower()
        if digest not in images:
            dest = assets / name
            shutil.copyfile(path, dest)
            with Image.open(dest) as im:
                size = list(im.size)
            images[digest] = {"asset": "assets/" + name, "sha256": digest,
                              "size": size, "original_paths": []}
        if str(path) not in images[digest]["original_paths"]:
            images[digest]["original_paths"].append(str(path))
        return digest

    result = {"schema": "native_v5_fixedpair_visual_evidence_v1",
              "exported_at": datetime.now(timezone.utc).isoformat(),
              "model_or_optimizer_executed": False, "experiments": {}, "images": images}
    for kind, root in ROOTS.items():
        selection = read(root / "data/selection.json")
        baseline_path = latest((root / "report").glob("*/baseline_generation.json"))
        if baseline_path is None:
            raise ValueError("Baseline missing: " + kind)
        baseline = read(baseline_path)
        records = read(root / "data/native_v5_v1/qwen_records.json")
        audits = list((root / "report").glob("completion_audit_*.json"))
        audit_path = latest(audits)
        final_control = latest((root / "report").glob("*/final_controls.json"))
        final_seeds = latest((root / "report").glob("*/final_extra_seeds.json"))
        completed = latest((root / "report").glob("*/completed.json"))
        if a.require_complete and any(x is None for x in (audit_path, final_control, final_seeds, completed)):
            raise ValueError("Final evidence incomplete: " + kind)
        exp = {"root": str(root), "selection": selection, "pairs": [],
               "audit": None if audit_path is None else read(audit_path),
               "completed": None if completed is None else read(completed), "evidence": {}}
        if a.require_complete:
            if final_control.parent != completed.parent or final_seeds.parent != completed.parent:
                raise ValueError("Final observations belong to a different orchestration run")
            if exp["audit"]["checkpoint"] != exp["completed"]["checkpoint"]["checkpoint"]:
                raise ValueError("Completion audit/checkpoint differs")
        paths = [root / "data/selection.json", root / "data/qwen_frozen_v2/index.json",
                 baseline_path, audit_path, final_control, final_seeds, completed]
        paths += list((root / "configs").glob("*.json"))
        paths += list((root / "report").glob("*/*_data.json"))
        paths += list((root / "report").glob("*/*_grad.json"))
        paths += list((root / "report").glob("cpu_regression*.txt"))
        paths += list((root / "source").glob("delivery_*/manifest.json"))
        paths += list((root / "source").glob("drift_*/manifest.json"))
        for path in paths:
            if path is None:
                continue
            name = kind + "__" + str(path.relative_to(root)).replace("/", "__")
            shutil.copyfile(path, evidence / name)
            exp["evidence"][str(path)] = {"file": "evidence/" + name, "sha256": sha(path)}
        for case in selection["pairs"]:
            record = next(r for r in records if r["pair_id"] == case["pair_id"])
            manifest = Path(record["v4_manifest"]["path"])
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            old = next(r for r in rows if r["pair_id"] == case["pair_id"])
            condition = read(record["qwen_condition"]["path"])
            pair = {"pair_id": case["pair_id"], "role": case["role"],
                    "prompt": case["prompt"], "condition": condition,
                    "source": image(old["source_rgb"]["path"], old["source_rgb"]["sha256"]),
                    "target": image(old["target_rgb"]["path"], old["target_rgb"]["sha256"]),
                    "inputs": [], "baseline": {}, "progress": {}, "seeds": {}, "controls": {},
                    "v4_record": old}
            # Decode only small raw-condition fields, without importing torch,
            # constructing networks, or evaluating a learned feature.
            import numpy as np
            if sha(old["local"]["path"]) != old["local"]["sha256"]:
                raise ValueError("Local raw-condition SHA conflict")
            with np.load(old["local"]["path"], allow_pickle=False) as raw:
                tree = json.loads(str(raw["record_json"]))["tree"]["fields"]
                keys = ("person_valid", "source_indices", "face_boxes", "face_roi_valid",
                        "hand_boxes", "hand_roi_valid", "jaw_valid", "expression_valid",
                        "hand_pose_valid", "mesh_valid")
                pair["local_summary"] = {k: raw[tree[k]["array"]].tolist() for k in keys}
            for item in condition["images"]:
                pair["inputs"].append({**item, "image": image(item["file"]["path"], item["file"]["sha256"])})
            for item in baseline["records"]:
                if item["pair_id"] == case["pair_id"]:
                    pair["baseline"][item["variant"]] = image(item["path"], item["sha256"])
            for update in range(30, 301, 30):
                path = root / f"training/output/images/update_{update:06d}" / case["role"] / case["pair_id"] / "native_seed42.png"
                if path.is_file():
                    bound = [r for g in (exp["audit"] or {}).get("committed_generation_records", [])
                             if g["update"] == update for r in g["records"] if r["pair_id"] == case["pair_id"]]
                    if a.require_complete and len(bound) != 1:
                        raise ValueError("Scheduled image lacks unique committed journal provenance")
                    pair["progress"][str(update)] = image(path, bound[0]["sha256"] if bound else None)
                elif a.require_complete:
                    raise ValueError("Missing scheduled image: " + str(path))
            for path, field in ((final_seeds, "seeds"), (final_control, "controls")):
                if path:
                    for item in read(path)["records"]:
                        if item["pair_id"] == case["pair_id"]:
                            pair[field][item["variant"] + ":" + str(item["seed"])] = image(item["path"], item["sha256"])
            exp["pairs"].append(pair)
        result["experiments"][kind] = exp
    if a.require_complete:
        starts = {e["audit"]["initialization"]["sha256"] for e in result["experiments"].values()}
        if len(starts) != 1:
            raise ValueError("The independent experiments did not use the same initialization")
    for path in (ROOTS["tiktok"] / "initialization").glob("*.json"):
        name = "initialization__" + path.name
        shutil.copyfile(path, evidence / name)
    (a.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    (a.output / "README.md").write_text(
        "# V5固定样本实验图文证据包\n\nassets为原始文件的只读副本，未做修图。manifest.json记录原路径、SHA和图片尺寸。"
        "evidence保存配置、数据及梯度报告。此导出工具不调用模型或optimizer。\n", encoding="utf8")
    archive = a.output.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(a.output, arcname=a.output.name)
    print(json.dumps({"directory": str(a.output), "archive": str(archive),
                      "sha256": sha(archive), "unique_images": len(images),
                      "complete_required": a.require_complete}, ensure_ascii=False))


if __name__ == "__main__":
    main()
