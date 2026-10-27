"""Stage 14b: score the fields Stage 2 already stores against the 134 labels.

Stage 14 falsified the hand-designed limb-contour detector (AUC ~0.49 on real
labels even though painted blur moves it 2/3 of the time).  Before concluding
"nothing works", check the signals that are already computed for every frame --
above all the plan's fifth row, 视频连续性 / 邻帧对比, which is a different
mechanism from any single-frame sharpness statistic.
"""
import json
from pathlib import Path

import numpy as np

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
OUT = PILOT / "qc" / "sharpness_calibration"
GOOD, BAD = {"clear"}, {"motionblur", "ghost", "soft"}
KEYS = ["roi_lap_var", "lap_var_full", "lap_var_center", "tenengrad", "hf_ratio",
        "spectral_slope", "rel_sharp_vs_neighbors", "rel_sharp_vs_neighbors_pctl",
        "roi_grad_ratio", "roi_grad_mean", "motion_prev", "face_width_px",
        "person_height_px", "body_mean_conf", "hand_mean_conf", "face_mean_conf",
        "dwpose_kpt_count", "person_area_share"]


def rank(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a))
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 8 or len(set(np.round(x, 9))) < 3 or len(set(y.tolist())) < 2:
        return None
    return round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 3)


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main():
    fq = {}
    for line in (PILOT / "manifests" / "frame_quality.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            fq[(r["sequence_id"], int(r["frame_idx"]))] = r
    items = []
    sm = {str(s["idx"]): s for s in json.loads((OUT / "samples.json").read_text())}
    for idx, lab in json.loads((OUT / "labels.json").read_text()).items():
        s = sm.get(idx)
        if s:
            items.append((s["sequence_id"], int(s["frame_idx"]), (lab.get("label") or "").lower()))
    d = json.loads((OUT / "disagree_samples.json").read_text())
    sm2 = {str(s["idx"]): s for s in d["samples"]}
    for idx, lab in json.loads((OUT / "disagree_labels.json").read_text()).items():
        s = sm2.get(idx)
        if s:
            items.append((s["sequence_id"], int(s["frame_idx"]), (lab.get("label") or "").lower()))

    rows = []
    for seq, fi, label in items:
        if label not in GOOD | BAD:
            continue
        q = fq.get((seq, fi))
        if q:
            rows.append((label, q))
    clear = [q for l, q in rows if l in GOOD]
    badr = [q for l, q in rows if l in BAD]

    def col(subset, key):
        return [float(r[key]) for r in subset if isinstance(r.get(key), (int, float))]

    out = {}
    for k in KEYS:
        allv = [(float(q[k]), 1 if l in BAD else 0) for l, q in rows
                if isinstance(q.get(k), (int, float))]
        x = [a for a, _ in allv]
        y = [b for _, b in allv]
        a = auc(col(badr, k), col(clear, k))
        out[k] = {"n": len(allv), "auc_bad_gt_clear": a,
                  "auc_clear_gt_bad": None if a is None else round(1 - a, 3),
                  "spearman_vs_bad": spearman(x, y) if len(allv) >= 8 else None,
                  "separable": None if a is None else bool(max(a, 1 - a) >= 0.70)}
    best = sorted(((k, max(v["auc_bad_gt_clear"] or 0, 1 - (v["auc_bad_gt_clear"] or 0.5)))
                   for k, v in out.items() if v["auc_bad_gt_clear"] is not None),
                  key=lambda kv: -kv[1])
    report = {
        "labelled": len(rows), "clear_n": len(clear), "bad_n": len(badr),
        "reading": "auc 0.5 = no separation; >=0.70 is the bar for calling a field usable "
                   "as a gate. auc_bad_gt_clear < 0.5 means the field runs the other way, "
                   "so the useful direction is 1-auc.",
        "fields": out,
        "ranking": [{"field": k, "best_direction_auc": round(v, 3)} for k, v in best[:8]],
        "nothing_crosses_0_70": all(not v["separable"] for v in out.values()
                                    if v["separable"] is not None),
        "caveat": "labels are a model's vision pass, not human; n is 36 clear vs 98 bad and "
                  "the disagreement half was sampled on purpose, so these AUCs are fine for "
                  "ranking fields against each other and not for quoting corpus accuracy.",
    }
    (OUT / "stored_fields_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("labelled", "clear_n", "bad_n", "ranking", "nothing_crosses_0_70")},
                     indent=2))
    for k, v in out.items():
        print(f"  {k:32s} auc_bad={v['auc_bad_gt_clear']} rho={v['spearman_vs_bad']} n={v['n']}")


if __name__ == "__main__":
    main()
