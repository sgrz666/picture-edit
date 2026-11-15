"""Control groups for the LoRA pose benchmark (read-only, CPU only).

Question: does PCK-against-B have any power to detect pose-transfer quality,
or does simply reproducing A's pose already score at the same level?

Arms, all scored with the experiment's own pose_stats() and the same reference:
  copyA      detect pose on the letterboxed A condition image, score vs its own B
  shuffleA   detect pose on A_i, score vs B_j (fixed random derangement) -> chance floor
  baseline   read from the existing report (original Qwen, no LoRA)
  lora       read from the existing report (checkpoint-009520)

Also reports the raw A-vs-B keypoint distance so pair difficulty is visible.
Writes nothing into the experiment tree.
"""
import json
import os
import random
import statistics as st
import sys

import numpy as np
from PIL import Image

ROOT = "/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora"
PROJECT = "/home/shangguanrz/project/pic-edit"
REPORT = os.path.join(ROOT, "report/full_report_20261111_v1")
COMP = os.path.join(ROOT, "output/full_comparison_20261111_v1")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, PROJECT)
from core import pose_stats, DWPOSE_TO_COCO  # noqa: E402
from src.evaluation.joint_pilot_metrics import PoseMetric  # noqa: E402


def jl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


rows = jl(os.path.join(COMP, "outputs.jsonl"))
per_image = jl(os.path.join(REPORT, "per_image.jsonl"))
reported = {(r["method"], r["case"]): r for r in per_image}

# one row per case (baseline and lora share the same original_row / reference)
cases = {}
for r in rows:
    if r["intervention"] != "normal":
        continue
    cases.setdefault(r["case"], r)
names = sorted(cases)
print("normal cases: %d" % len(names))

pose = PoseMetric(os.path.join(PROJECT, "models"))
det = {}
for name in names:
    r = cases[name]
    a_path = r["original_row"]["condition_images"][0]
    with Image.open(a_path) as im:
        a_size = im.size
        det[name] = pose.detect(im)
    ref = json.load(open(r["cache_row"]["target_pose_metadata"], "r", encoding="utf-8"))
    assert list(ref["size"]) == list(a_size), (name, ref["size"], a_size)
print("A detections: %d/%d succeeded ; canvas assertion passed for all"
      % (sum(v is not None for v in det.values()), len(names)))

# original resolutions: if they differ, letterbox scale differs and the
# cross-case shuffle control is partly confounded -- report it, do not hide it.
orig_sizes = {}
for name in names:
    with Image.open(cases[name]["original_row"]["source_rgb_path"]) as im:
        orig_sizes[name] = im.size
print("original A resolutions:", sorted(set(orig_sizes.values())))
print("clip ids:", sorted(set(cases[n]["original_row"]["clip_id"] for n in names)))

refs = {}
for name in names:
    r = cases[name]
    m = json.load(open(r["cache_row"]["target_pose_metadata"], "r", encoding="utf-8"))
    refs[name] = (np.asarray(m["keypoints_xy"]), np.asarray(m["valid"], dtype=bool), m["size"])

# fixed derangement: no case keeps its own reference
rng = random.Random(20261112)
for _ in range(10000):
    perm = names[:]
    rng.shuffle(perm)
    if all(perm[i] != names[i] for i in range(len(names))):
        break
else:
    raise SystemExit("no derangement found")
partner = dict(zip(names, perm))
print("shuffle derangement (seed 20261112):",
      ", ".join("%s->%s" % (k, partner[k]) for k in names))

FIELDS = ["pck_002_diagonal", "pck_005_diagonal", "pck_010_diagonal", "nme_diagonal"]
arms = {"copyA": {}, "shuffleA": {}, "baseline": {}, "lora": {}}
for name in names:
    ref_xy, ref_valid, size = refs[name]
    arms["copyA"][name] = pose_stats(det[name], ref_xy, ref_valid, size)
    # mismatched arm: A from another case, scored against THIS case's B reference
    p = partner[name]
    arms["shuffleA"][name] = pose_stats(det[p], ref_xy, ref_valid, size)
    for meth in ("baseline", "lora"):
        arms[meth][name] = reported[(meth, name)]

# raw A-vs-B keypoint gap, same normalization as pose_stats
print("\nA-vs-B reference keypoint gap (diagonal-normalized, valid COCO17 joints):")
gaps = []
for name in names:
    ref_xy, ref_valid, size = refs[name]
    rc = ref_xy[DWPOSE_TO_COCO]
    vc = ref_valid[DWPOSE_TO_COCO]
    if det[name] is None:
        continue
    xy, sc = det[name]
    good = vc & (np.asarray(sc) > 0.3) & np.isfinite(xy).all(1)
    d = np.linalg.norm(np.asarray(xy)[good] - rc[good], axis=1) / float(np.hypot(*size))
    gaps.append((name, cases[name]["original_row"]["split"], float(d.mean()),
                 float((d < 0.02).mean()), int(good.sum())))
for name, split, mean, frac, n in gaps:
    print("  %-11s %-5s meanA-B=%.4f  frac<0.02=%.3f  n=%d" % (name, split, mean, frac, n))

for split in ("val", "train"):
    sel = [n for n in names if cases[n]["original_row"]["split"] == split]
    if not sel:
        continue
    print("\n" + "=" * 76)
    print("split=%s  n=%d cases" % (split, len(sel)))
    print("=" * 76)
    print("%-10s %10s %10s %10s %10s" % ("arm", "PCK@.02", "PCK@.05", "PCK@.10", "NME"))
    for arm in ("shuffleA", "copyA", "baseline", "lora"):
        vals = {f: [arms[arm][n][f] for n in sel if arms[arm][n].get(f) is not None]
                for f in FIELDS}
        print("%-10s %10.5f %10.5f %10.5f %10.5f"
              % (arm, *[st.mean(vals[f]) if vals[f] else float("nan") for f in FIELDS]))
    print("\nper-case PCK@.02 and PCK@.05:")
    print("%-11s %-18s %-18s %-18s %-18s" % ("case", "shuffleA", "copyA", "baseline", "lora"))
    for n in sel:
        cells = []
        for arm in ("shuffleA", "copyA", "baseline", "lora"):
            a = arms[arm][n]
            cells.append("%.3f/%.3f" % (a["pck_002_diagonal"], a["pck_005_diagonal"]))
        print("%-11s %-18s %-18s %-18s %-18s" % (n, *cells))
    # paired: does the model beat the trivial copy-A control?
    for arm in ("baseline", "lora"):
        for f, sign in (("pck_002_diagonal", 1), ("pck_005_diagonal", 1), ("nme_diagonal", -1)):
            d = [arms[arm][n][f] - arms["copyA"][n][f] for n in sel
                 if arms[arm][n].get(f) is not None and arms["copyA"][n].get(f) is not None]
            if not d:
                continue
            win = sum(1 for x in d if x * sign > 0)
            lose = sum(1 for x in d if x * sign < 0)
            print("  %-8s minus copyA  %-18s mean %+.5f  better=%d worse=%d tie=%d"
                  % (arm, f, st.mean(d), win, lose, len(d) - win - lose))

print("\njoint_recall / matched joints per arm (val):")
sel = [n for n in names if cases[n]["original_row"]["split"] == "val"]
for arm in ("shuffleA", "copyA", "baseline", "lora"):
    jr = [arms[arm][n].get("joint_recall") for n in sel]
    mj = [arms[arm][n].get("matched_joints") for n in sel]
    tj = [arms[arm][n].get("target_joints") for n in sel]
    print("  %-10s recall=%.4f matched=%.1f target=%.1f"
          % (arm, st.mean([x for x in jr if x is not None]),
             st.mean([x for x in mj if x is not None]),
             st.mean([x for x in tj if x is not None])))
print("\nDONE")