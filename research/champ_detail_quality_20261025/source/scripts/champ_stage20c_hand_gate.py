"""Stage 20c -- per-side hand/face ROI gate, with the index table pinned by eye.

Stage 20/20b settled the layout on a manually inspected frame (qc/layout/sheet_*.png, zoom_*.png):
  0 nose, 1 neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow, 7 LWrist,
  8 RHip, 9 RKnee, 10 RAnkle, 11 LHip, 12 LKnee, 13 LAnkle, 14 REye, 15 LEye, 16 REar, 17 LEar,
  18:24 foot6, 24:92 face68, 92:113 hand A, 113:134 hand B.
The legs are side-grouped, which is NOT the COCO/DWPose interleaved order, so every anchor below
comes from this table rather than from memory.

Layout being right does not make every hand window usable: in zoom_handB_000333_00056 the 21 points
of window B are scattered over the jacket while window A is a textbook hand. So each (frame, side)
gets its own verdict -- the hand root must sit on a body wrist, the 21 points must stay inside a
hand-sized box beyond that wrist, and the two windows must not both claim the same wrist. Stage 19
shipped masks for every side unconditionally; this produces the list of sides that may be used.
"""
from pathlib import Path
import json

import numpy as np

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
OUT = PILOT / "qc" / "hand_gate"
CONF = 0.3
NOSE, NECK, RSH, RELB, RWR, LSH, LELB, LWR = 0, 1, 2, 3, 4, 5, 6, 7
RHIP, RKNE, RANK, LHIP, LKNE, LANK, REYE, LEYE, REAR, LEAR = 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
FACE, FOOT, HAND = slice(24, 92), slice(18, 24), {"hand_a": slice(92, 113), "hand_b": slice(113, 134)}
ARMS = (("R", RELB, RWR), ("L", LELB, LWR))
LEGS = (("R", RKNE, RANK), ("L", LKNE, LANK))


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def load(seq, stem):
    rec = json.loads((PILOT / "assets" / seq / "dwpose" / f"{stem}.json").read_text())
    return np.asarray(rec["candidate"], np.float64)[:, :2], np.asarray(rec["subset"], np.float64)


def hand_gate(pts, conf, sl):
    kp = pts[sl][conf[sl] >= CONF]
    if kp.shape[0] < 12:
        return {"usable": False, "why": f"only {kp.shape[0]} points over conf {CONF}"}
    root = pts[sl.start]
    best = None
    for side, elbow, wrist in ARMS:
        if conf[elbow] < CONF or conf[wrist] < CONF:
            continue
        fore = dist(pts[wrist], pts[elbow])
        if fore < 8:
            continue
        attach = dist(root, pts[wrist]) / fore
        u = (pts[wrist] - pts[elbow]) / fore
        v = kp.mean(0) - pts[wrist]
        nv = dist(kp.mean(0), pts[wrist])
        align = float(np.dot(v, u) / nv) if nv > 1e-6 else 0.0
        cand = {"side": side, "attach_forearm": attach, "beyond_wrist": align,
                "span_forearm": float(np.max(np.hypot(*(kp - kp.mean(0)).T)) / fore),
                "reach_forearm": nv / fore}
        if best is None or cand["attach_forearm"] < best["attach_forearm"]:
            best = cand
    if best is None:
        return {"usable": False, "why": "no usable arm anchor (elbow or wrist below conf)"}
    best["n_visible"] = int(kp.shape[0])
    best["conf_mean"] = round(float(conf[sl].mean()), 3)
    best["usable"] = bool(best["attach_forearm"] <= 0.35 and best["span_forearm"] <= 0.9
                          and best["beyond_wrist"] >= 0.0)
    if not best["usable"]:
        best["why"] = "hand root not on a wrist, or the 21 points spread past hand size"
    return best


def face_gate(pts, conf):
    kp = pts[FACE][conf[FACE] >= CONF]
    if kp.shape[0] < 40 or min(conf[NOSE], conf[REAR], conf[LEAR]) < CONF:
        return {"usable": False, "why": "not enough visible face points or head anchors"}
    ear_mid = (pts[REAR] + pts[LEAR]) / 2
    head = dist(pts[REAR], pts[LEAR])
    if head < 20:
        return {"usable": False, "why": "ears too close together to size the head"}
    c = kp.mean(0)
    out = {"off_ear_line": dist(c, ear_mid) / head, "span": float(np.max(np.hypot(*(kp - c).T)) / head),
           "n_visible": int(kp.shape[0]), "conf_mean": round(float(conf[FACE].mean()), 3)}
    out["usable"] = bool(out["off_ear_line"] <= 0.6 and out["span"] <= 0.9)
    if not out["usable"]:
        out["why"] = "face centroid not inside the head box defined by the two ears"
    return out


def foot_gate(pts, conf):
    kp = pts[FOOT][conf[FOOT] >= CONF]
    if kp.shape[0] < 3:
        return {"usable": False, "why": "fewer than 3 visible foot points"}
    best = None
    for side, knee, ankle in LEGS:
        if conf[knee] < CONF or conf[ankle] < CONF:
            continue
        shin = dist(pts[ankle], pts[knee])
        if shin < 10:
            continue
        d = dist(kp.mean(0), pts[ankle]) / shin
        if best is None or d < best["d_shin"]:
            best = {"side": side, "d_shin": d, "below_ankle": bool(kp[:, 1].mean() > pts[ankle][1])}
    if best is None:
        return {"usable": False, "why": "no usable leg anchor"}
    best["usable"] = bool(best["d_shin"] <= 1.0 and best["below_ankle"])
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, unparsed = [], []
    for detail in sorted((PILOT / "assets").glob("*/detail")):
        seq = detail.parent.name
        stems = {}
        for png in detail.glob("*.png"):
            for side in ("face", "hand_a", "hand_b"):
                if png.stem.endswith("_" + side):
                    stems.setdefault(png.stem[:-(len(side) + 1)], set()).add(side)
                    break
            else:
                unparsed.append(png.name)
        for stem, sides in sorted(stems.items()):
            frames.append((seq, stem, sides))
    if not frames:
        raise SystemExit("no stage-19 detail masks found -- nothing to gate")
    rows, skipped = [], 0
    for seq, stem, sides in frames:
        try:
            pts, conf = load(seq, stem)
        except (ValueError, FileNotFoundError):
            skipped += 1
            continue
        gates = {"hand_a": hand_gate(pts, conf, HAND["hand_a"]),
                 "hand_b": hand_gate(pts, conf, HAND["hand_b"]),
                 "face": face_gate(pts, conf), "foot": foot_gate(pts, conf)}
        a, b = gates["hand_a"], gates["hand_b"]
        same_wrist = bool("side" in a and "side" in b and a["side"] == b["side"])
        if same_wrist:
            # Two hands cannot hang off one wrist: at most one window is real, and there is no
            # evidence for which, so both sides are dropped rather than guessed.
            for side in ("hand_a", "hand_b"):
                if gates[side].get("usable"):
                    gates[side]["usable"] = False
                    gates[side]["why"] = "both 21-point windows attach to the same wrist"
        rows.append({"seq": seq, "stem": stem, "gates": gates, "both_hands_same_wrist": same_wrist})
    usable_sides = [r for r in rows for s in ("hand_a", "hand_b") if r["gates"][s].get("usable")]
    attach = [r["gates"][s]["attach_forearm"] for r in rows for s in ("hand_a", "hand_b")
              if "attach_forearm" in r["gates"][s]]
    span = [r["gates"][s]["span_forearm"] for r in rows for s in ("hand_a", "hand_b")
            if "span_forearm" in r["gates"][s]]
    wrong_window = []
    for seq, stem, _ in frames[:200]:
        try:
            pts, conf = load(seq, stem)
        except (ValueError, FileNotFoundError):
            continue
        wrong_window.append(hand_gate(pts, conf, slice(60, 81)).get("usable", False))

    def stat(vals):
        if not vals:
            return {"n": 0}
        arr = np.asarray(vals, np.float64)
        return {"n": int(arr.size), "p50": round(float(np.median(arr)), 4),
                "p90": round(float(np.quantile(arr, 0.9)), 4), "max": round(float(arr.max()), 4)}

    report = {
        "index_table": "pinned by eye on qc/layout/sheet_000000_00112.png and zoom_* for 000444_00024",
        "frames_gated": len(rows), "frames_skipped_missing_json": skipped,
        "unparsed_detail_names": {"count": len(unparsed), "examples": unparsed[:5]},
        "hand_attach_forearm": stat(attach), "hand_span_forearm": stat(span),
        "hand_a_usable": sum(1 for r in rows if r["gates"]["hand_a"].get("usable")),
        "hand_b_usable": sum(1 for r in rows if r["gates"]["hand_b"].get("usable")),
        "face_usable": sum(1 for r in rows if r["gates"]["face"].get("usable")),
        "foot_usable": sum(1 for r in rows if r["gates"]["foot"].get("usable")),
        "both_hands_claim_same_wrist": sum(1 for r in rows if r["both_hands_same_wrist"]),
        "usable_side_rows": len(usable_sides),
        "counterexample_face_slice_scored_as_hand": {
            "n": len(wrong_window), "passed_hand_gate": int(sum(wrong_window)),
            "expected": 0, "note": "a face sub-slice fed to the hand gate must not pass"},
        "thresholds": {"attach_forearm_max": 0.35, "span_forearm_max": 0.9, "beyond_wrist_min": 0.0,
                       "conf_min": CONF, "chosen_from": "hand-sized geometry, not tuned on these rows"},
    }
    with (OUT / "roi_usable.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            for side in ("hand_a", "hand_b"):
                if r["gates"][side].get("usable"):
                    fh.write(json.dumps({"sequence_id": r["seq"], "stem": r["stem"], "roi": side,
                                         "wrist_side": r["gates"][side]["side"],
                                         "attach_forearm": round(r["gates"][side]["attach_forearm"], 4),
                                         "span_forearm": round(r["gates"][side]["span_forearm"], 4),
                                         "conf_mean": r["gates"][side]["conf_mean"]}) + "\n")
    (OUT / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    with (OUT / "per_frame.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
