"""Stage 28 -- extend the compliant face cache to every appearance in the training pool.

Runs the stage26 shim build + stage27 promote per sequence, then rebuilds the face manifest across
ALL cached appearances (stage27 only serves its own sequence, so the last write would otherwise
silently narrow the pool). Everything it claims goes through the cache files themselves, not stdout.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
REPO = Path("/home/shangguanrz/project/pic-edit")
PY = "/home/shangguanrz/nihaochen/envs/qwen21/bin/python"
FEAT = PILOT / "CHAMP_v1/features/face_v65"
SHADOW = PILOT / "qc/face_shadow"
TRAINABLE = PILOT / "CHAMP_v1/manifests/pairs_trainable/train_pairs.jsonl"

rows = [json.loads(l) for l in TRAINABLE.read_text(encoding="utf-8").splitlines() if l.strip()]
appearances = sorted({str(r["sequence_id"]) for r in rows})
print("APPEARANCES", appearances, flush=True)

# ---- shadow trees first, so a missing RGB is caught before any GPU work ----
link_report = {}
for seq in appearances:
    fmap = PILOT / f"assets/{seq}/smplestx/frame_map.json"
    if not fmap.is_file():
        link_report[seq] = {"error": f"missing {fmap}"}
        continue
    frames = json.loads(fmap.read_text(encoding="utf-8")).get("frames") or []
    stems = [str(f.get("stem", f"{f['index']+1:04d}")) for f in frames]
    d = SHADOW / seq / "images"
    d.mkdir(parents=True, exist_ok=True)
    made, missing = 0, []
    for st in stems:
        src = PILOT / f"frames/{seq}/rgb/{st}.png"
        if not src.is_file():
            missing.append(st)
            continue
        link = d / f"{st}.png"
        if not link.exists():
            link.symlink_to(src)
        made += 1
    link_report[seq] = {"stems": len(stems), "links": made, "missing_rgb": len(missing),
                        "first_missing": missing[:3]}
    print(seq, json.dumps(link_report[seq], ensure_ascii=False), flush=True)

runnable = [s for s in appearances if link_report[s].get("missing_rgb", 1) == 0]
print("RUNNABLE", runnable, "SKIPPED (rgb absent)",
      [s for s in appearances if s not in runnable], flush=True)

# ---- build + promote per sequence ----
per_seq = {}
for seq in runnable:
    if (FEAT / f"{seq}.pt").is_symlink() or (FEAT / f"{seq}.pt").exists():
        cache = torch_load = None
        import torch
        torch_load = torch.load(FEAT / f"{seq}.pt", map_location="cpu", weights_only=False)
        md = torch_load.get("metadata", {})
        if int(md.get("champ_dino_nonzero") or 0) > 0 and "geometry_valid" in torch_load:
            per_seq[seq] = {"skipped": "already promoted", "arcface_nonzero": md.get("champ_arcface_nonzero"),
                            "dino_nonzero": md.get("champ_dino_nonzero")}
            print(seq, "already promoted, skipping", flush=True)
            continue
    log = PILOT / f"logs/stage28_{seq}.log"
    out = subprocess.run([PY, str(PILOT / "scripts/champ_stage26_face_build.py"), seq],
                         cwd=str(PILOT), capture_output=True, text=True)
    ok26 = out.returncode == 0
    out2 = subprocess.run([PY, str(PILOT / "scripts/champ_stage27_face_promote.py"), seq],
                          cwd=str(PILOT), capture_output=True, text=True)
    log.write_text(out.stdout + "\n--- stderr ---\n" + out.stderr + "\n=== stage27 ===\n"
                   + out2.stdout + "\n--- stderr ---\n" + out2.stderr, encoding="utf-8")
    if out2.returncode != 0:
        per_seq[seq] = {"build_exit": out.returncode, "promote_exit": out2.returncode,
                        "tail": (out2.stderr or out.stderr)[-400:]}
        print(seq, "FAILED", per_seq[seq]["tail"][:200], flush=True)
        continue
    rep = json.loads((PILOT / "qc/face_cache/stage27_report.json").read_text(encoding="utf-8"))
    per_seq[seq] = {"build_exit": out.returncode, "promote_exit": 0,
                    "arcface_nonzero": rep["arcface_nonzero"], "dino_nonzero": rep["dino_nonzero"],
                    "valid_true": rep["valid_true"], "geometry_valid_true": rep["geometry_valid_true"],
                    "store_check": rep["store_check"]}
    print(seq, json.dumps(per_seq[seq], ensure_ascii=False), flush=True)

# ---- merged manifest: every cached appearance, filtered by what the store can serve ----
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))  # data_runtime imports `from src.data...`, so the repo root must be on path
# the store imports scripts.precompute_iper_face_features_v65, which only exists under scripts/data/
_alias = "scripts.precompute_iper_face_features_v65"
if _alias not in sys.modules:
    try:
        importlib.import_module(_alias)
    except ModuleNotFoundError:
        _spec = importlib.util.spec_from_file_location(
            _alias, REPO / "scripts/data/precompute_iper_face_features_v65.py")
        _m = importlib.util.module_from_spec(_spec)
        sys.modules[_alias] = _m
        _spec.loader.exec_module(_m)
import torch  # noqa: E402
from pose_control.qwen21_full.data_runtime import FaceConditionStore  # noqa: E402

store = FaceConditionStore(FEAT)
caches = {}
for seq in appearances:
    p = FEAT / f"{seq}.pt"
    if p.is_file() or p.is_symlink():
        caches[seq] = store._load(seq)   # raises if a cache is not contract-compliant

kept, drops = [], {"appearance_not_cached": 0, "stem_not_in_cache": 0, "face_invalid": 0}
for r in rows:
    seq = str(r["sequence_id"])
    if seq not in caches:
        drops["appearance_not_cached"] += 1
        continue
    c = caches[seq]
    s, t = str(r["source_stem"]), str(r["target_stem"])
    if s not in c["stem_to_idx"] or t not in c["stem_to_idx"]:
        drops["stem_not_in_cache"] += 1
        continue
    si, ti = c["stem_to_idx"][s], c["stem_to_idx"][t]
    if not (bool(c["valid"][si]) and bool(c["geometry_valid"][ti])):
        drops["face_invalid"] += 1
        continue
    row = dict(r)
    row["source_appearance"] = seq
    row["target_appearance"] = seq
    row["face_condition_lineage"] = ("v65 cache built under the pilot compat shim; geometry_valid from "
                                     "measured qc/geometry_qc.jsonl; fingerprint re-stamped after adding it")
    kept.append(row)
out_dir = PILOT / "CHAMP_v1/manifests/pairs_face"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "train_pairs.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept),
                                            encoding="utf-8")
report = {"appearances_in_pool": appearances, "shadow_links": link_report, "per_sequence": per_seq,
          "caches_passing_store": sorted(caches), "merged_manifest": {
              "rows_in": len(rows), "rows_kept": len(kept), "drops": drops,
              "distinct_source_appearances": sorted({str(r['source_appearance']) for r in kept}),
              "path": str(out_dir / "train_pairs.jsonl")}}
(PILOT / "qc/face_cache/stage28_batch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
print(json.dumps(report["merged_manifest"], ensure_ascii=False, indent=1), flush=True)
print("WROTE", PILOT / "qc/face_cache/stage28_batch.json", flush=True)
