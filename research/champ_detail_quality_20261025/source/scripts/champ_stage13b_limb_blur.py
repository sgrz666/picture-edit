"""Stage 13b: does a LOCAL limb-blur measure separate what roi_lap_var cannot?

Measured in stage 13: the shipped gate (roi_lap_var >= corpus p25) misses 63.2% of
frames that are visibly blurry at native resolution while dropping 11.8% of the
clear ones, and blurry frames appear in every roi_lap_var bin including the
sharpest.  The reason is structural -- in these clips the torso stays frozen while
the arms/hands sweep, so a whole-body average is dominated by sharp clothing and
background texture.

So compute sharpness per body region and test the ratio hand/torso.  Held to the
same false-positive budget as the current gate (must keep >= 88% of clear frames),
otherwise "catches more blur" is trivially winnable by rejecting everything.
"""
import json
from pathlib import Path

import cv2
import numpy as np

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
OUT = PILOT / "qc" / "sharpness_calibration"
GOOD, BAD = {"clear"}, {"motionblur", "ghost", "soft"}
MIN_SIDE = 24          # a 10x10 "hand" has no meaningful Laplacian


def lap(img):
    if img is None or min(img.shape[:2]) < MIN_SIDE:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def box_from(pts, pad=0.0, shape=None):
    p = np.array([q for q in pts if q is not None and len(q) == 2], float)
    if len(p) < 2:
        return None
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    if pad:
        w, h = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - w, y0 - h, x1 + w, y1 + h
    if shape:
        H, W = shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    return None


def crop(img, box):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a))
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 6 or len(set(np.round(x, 9))) < 3 or len(set(y.tolist())) < 2:
        return None
    return round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 3)


def main():
    samples = {str(s["idx"]): s for s in json.loads((OUT / "samples.json").read_text())}
    labels = json.loads((OUT / "labels.json").read_text())
    quality = {}
    for line in (PILOT / "manifests" / "frame_quality.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        quality[(r["sequence_id"], int(r["frame_idx"]))] = r

    rows = []
    for idx, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
        name = (lab.get("label") or "").strip().lower()
        if name not in GOOD | BAD | {"multi", "truncated", "filter"}:
            continue
        s = samples.get(idx)
        if not s:
            continue
        q = quality.get((s["sequence_id"], int(s["frame_idx"])))
        if not q:
            continue
        path = PILOT / "frames" / s["sequence_id"] / "rgb" / f"{int(s['frame_idx']):05d}.png"
        img = cv2.imread(str(path))
        if img is None:
            continue
        kpts = q.get("dwpose_kpts") or []
        conf = q.get("dwpose_conf") or []

        def kp(i):
            if i < len(kpts) and kpts[i] and (not conf or conf[i] > 0.3):
                return kpts[i]
            if i < len(kpts) and kpts[i]:
                return kpts[i]
            return None

        torso = box_from([kp(2), kp(5), kp(8), kp(11)], 0.15, img.shape)
        face = box_from([kp(0), kp(14), kp(15), kp(16), kp(17)], 0.25, img.shape)
        hands, arms = [], []
        for wrist, sh in ((4, 3), (7, 6)):
            w, e = kp(wrist), kp(sh)
            if not w:
                continue
            reach = 0.5 * (np.array(w) - np.array(e)) if e is not None else None
            pts = [w]
            if reach is not None:
                pts.append(w + reach)
            hands.append(box_from(pts, 0.35, img.shape))
        if len(kpts) >= 134:
            hands = [box_from(kpts[92:113], 0.1, img.shape), box_from(kpts[113:134], 0.1, img.shape)]
        for a, b in ((3, 4), (6, 7)):
            arms.append(box_from([kp(a), kp(b)], 0.3, img.shape))

        lap_torso, lap_face = lap(crop(img, torso)), lap(crop(img, face))
        lap_hand = [lap(crop(img, h)) for h in hands]
        lap_arm = [lap(crop(img, a)) for a in arms]
        hv = [v for v in lap_hand if v is not None]
        av = [v for v in lap_arm if v is not None]
        row = {"idx": int(idx), "label": name, "seq": s["sequence_id"],
               "frame": int(s["frame_idx"]),
               "roi_lap_var": float(q.get("roi_lap_var") or 0.0),
               "lap_torso": lap_torso, "lap_face": lap_face,
               "lap_hand": hv, "lap_arm": av,
               "min_hand": min(hv) if hv else None,
               "ratio_hand_torso": (min(hv) / lap_torso) if hv and lap_torso else None,
               "ratio_arm_torso": (min(av) / lap_torso) if av and lap_torso else None}
        rows.append(row)

    def col(key):
        return [r for r in rows if isinstance(r.get(key), (int, float))]

    binary_ok = [r for r in rows if r["label"] in GOOD | BAD]
    y = np.array([0 if r["label"] in GOOD else 1 for r in binary_ok])
    corr = {}
    for m in ("roi_lap_var", "lap_torso", "lap_face", "min_hand",
              "ratio_hand_torso", "ratio_arm_torso"):
        pair = [(float(r[m]), int(b)) for r in binary_ok if isinstance(r.get(m), (int, float))
                for b in [0 if r["label"] in GOOD else 1]]
        if len(pair) >= 6:
            x = np.array([p[0] for p in pair])
            yy = np.array([p[1] for p in pair])
            corr[m] = {"rho": spearman(x, yy), "n": len(pair),
                       "clear_mean": round(float(x[yy == 0].mean()), 4),
                       "bad_mean": round(float(x[yy == 1].mean()), 4)}
        else:
            corr[m] = {"rho": None, "n": len(pair), "why": "region too small / not measured"}

    # candidate gate, held to the current gate's clear-keep budget
    pool = [float(r["roi_lap_var"]) for r in
            (json.loads(l) for l in (PILOT / "manifests" / "frame_quality.jsonl")
             .read_text().splitlines() if l.strip()) if r.get("roi_lap_var") is not None]
    p25 = float(np.percentile(pool, 25))
    clear = [r for r in rows if r["label"] in GOOD]
    badr = [r for r in rows if r["label"] in BAD]
    budget = 1.0 - 11.8 / 100.0     # current gate drops 11.8% of clear frames

    def eval_gate(fn):
        kept_clear = sum(1 for r in clear if fn(r))
        caught = sum(1 for r in badr if not fn(r))
        return {"clear_kept_pct": round(100.0 * kept_clear / max(len(clear), 1), 1),
                "bad_caught_pct": round(100.0 * caught / max(len(badr), 1), 1),
                "clear_kept": kept_clear, "clear_n": len(clear),
                "bad_caught": caught, "bad_n": len(badr)}

    gates = {"current roi_lap_var>=p25": eval_gate(lambda r: r["roi_lap_var"] >= p25)}
    # sweep a relative-hand-blur threshold, keep the best that respects the budget
    sweep = []
    for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7):
        g = eval_gate(lambda r, t=t: (r["ratio_hand_torso"] is None)
                      or (r["ratio_hand_torso"] >= t))
        g["threshold"] = t
        sweep.append(g)
    ok = [g for g in sweep if g["clear_kept_pct"] >= 100 * budget]
    best = max(ok, key=lambda g: g["bad_caught_pct"]) if ok else None
    gates["relative hand blur (best within budget)"] = best or {"why": "no threshold met the budget"}
    gates["absolute hand blur min_hand>=p25"] = eval_gate(
        lambda r: (r["min_hand"] is None) or
        (r["min_hand"] >= float(np.percentile([x for x in (q["min_hand"] for q in rows)
                                               if isinstance(x, (int, float))], 25))))

    report = {"annotator": "qoder agent vision pass, not human",
              "rows": len(rows), "clear_n": len(clear), "bad_n": len(badr),
              "labels": {r["idx"]: r["label"] for r in rows},
              "spearman": corr,
              "budget_clear_keep_pct": round(100 * budget, 1),
              "gates": gates,
              "threshold_sweep": sweep,
              "rows_detail": rows,
              "caveat": "hand ROIs are often tiny or out of frame; ratio_hand_torso is None "
                        "for those and they are treated as passing, which understates the "
                        "gate's reach rather than flattering it."}
    (OUT / "limb_blur_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "clear_n", "bad_n", "spearman",
                                             "budget_clear_keep_pct", "gates",
                                             "threshold_sweep")}, indent=2))


if __name__ == "__main__":
    main()
