"""Stage 19 路 Face / Hand ROI masks -- with the CHAMP DWPose file format read correctly.

Format fact established by probing four files: `candidate` already IS the 134 whole-body points in
DWPose order (body18, foot6, face68, hand21, hand21) and `subset` is a per-point CONFIDENCE row
(0.2..1.06, can exceed 1). Indexing candidate by subset values collapses every keypoint onto the
first two points, which is what stage 19's first version and stage 15b's skeleton renderer did.
Both are fixed here / re-rendered by champ_stage19c_reskeleton.py.

Masks are written in ORIGINAL image pixels (the space the V5 detail sidecar demands); the canvas
box is original + native_pad offset, since native padding never scales.
"""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
OUT = PILOT / "qc" / "detail_roi"
FACE_SLICE, HAND_A, HAND_B, BODY_SLICE = slice(24, 92), slice(92, 113), slice(113, 134), slice(0, 18)
CONF = 0.3


def first_image(dir_path, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = dir_path / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_points(seq, stem):
    rec = json.loads((PILOT / "assets" / seq / "dwpose" / f"{stem}.json").read_text())
    pts = np.asarray(rec["candidate"], np.float32)
    conf = np.asarray(rec["subset"], np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or conf.shape != pts.shape[:1]:
        raise ValueError(f"unexpected dwpose layout for {seq}/{stem}: {pts.shape} {conf.shape}")
    return pts[:, :2], conf


def window(pts, conf, sl):
    keep = conf[sl] >= CONF
    return pts[sl], keep


def hull(mask_shape, pts, keep, min_visible):
    mask = np.zeros(mask_shape, np.uint8)
    kept = pts[keep]
    if len(kept) < min_visible:
        return mask, 0, None
    h = cv2.convexHull(np.rint(kept).astype(np.int32))
    cv2.fillConvexPoly(mask, h, 1)
    return mask, int(mask.sum()), (int(h[:, 0, 0].min()), int(h[:, 0, 0].max()),
                                   int(h[:, 0, 1].min()), int(h[:, 0, 1].max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
    from src.data.canvas_contract import native_pad

    quality = {}
    for line in (PILOT / "manifests" / "frame_quality.jsonl").read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            quality[(str(r["sequence_id"]), int(r["frame_idx"]))] = r

    needed = {}
    for split in ("train", "val", "test"):
        for line in (PILOT / "manifests" / "pairs" / f"{split}_pairs.jsonl").read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seq = str(r.get("sequence_id") or r.get("appearance"))
            for side in ("source", "target"):
                needed.setdefault(seq, set()).add(str(r[f"{side}_stem"]))

    rows, ratios, hand_ratios, empty_face, empty_both = [], [], [], 0, 0
    for seq, stems in sorted(needed.items()):
        out_dir = PILOT / "assets" / seq / "detail"
        out_dir.mkdir(exist_ok=True)
        for stem in sorted(stems):
            rgb = first_image(PILOT / "frames" / seq / "rgb", stem)
            if rgb is None or not (PILOT / "assets" / seq / "dwpose" / f"{stem}.json").is_file():
                continue
            pts, conf = load_points(seq, stem)
            with Image.open(rgb) as im:
                w, h = im.size
            spec = native_pad((h, w))
            ox, oy = spec.offset_xy
            entry = {"seq": seq, "stem": stem, "frame_wh": [w, h], "canvas_hw": list(spec.canvas_hw)}
            face, face_keep = window(pts, conf, FACE_SLICE)
            body, body_keep = window(pts, conf, BODY_SLICE)
            for name, sl in (("face", FACE_SLICE), ("hand_a", HAND_A), ("hand_b", HAND_B)):
                p, k = window(pts, conf, sl)
                mask, area, box = hull((h, w), p, k, 20 if name == "face" else 10)
                Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{stem}_{name}.png")
                entry[f"{name}_px"], entry[f"{name}_conf_min"] = area, round(float(conf[sl][k].min()), 3) if k.any() else None
                entry[f"{name}_box_original"] = list(box) if box else None
                entry[f"{name}_box_canvas"] = [box[0] + ox, box[1] + ox, box[2] + oy, box[3] + oy] if box else None
                if name == "face" and area == 0:
                    empty_face += 1
            if entry["hand_a_px"] == 0 and entry["hand_b_px"] == 0:
                empty_both += 1
            measured = (quality.get((seq, int(stem))) or {}).get("face_width_px")
            if measured and entry["face_box_original"] and entry["face_px"]:
                x0, x1, _, _ = entry["face_box_original"]
                ratios.append(float(x1 - x0) / float(measured))
            # hands must be near the wrists, not near the nose -- an independent layout check
            anchors = [i for i in (10, 13, 9, 12) if body_keep[i]]
            wrists = body[anchors] if anchors else None
            for name in ("hand_a", "hand_b"):
                box = entry[f"{name}_box_original"]
                if box and wrists is not None and entry[f"{name}_px"]:
                    cx, cy = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0
                    d = float(np.linalg.norm(wrists - np.array([cx, cy]), axis=1).min())
                    scale = float(np.linalg.norm(body[body_keep].max(0) - body[body_keep].min(0))) or 1.0
                    hand_ratios.append((name, round(d / scale, 4)))
            rows.append(entry)

    ratio = np.asarray(ratios, np.float32)
    hr = np.asarray([v for _, v in hand_ratios], np.float32)
    per_side = {}
    for nm, v in hand_ratios:
        per_side.setdefault(nm, []).append(v)
    report = {"sequences": len(needed), "frames": len(rows),
              "format": {"candidate": "134x2 points, DWPose order", "subset": "per-point confidence"},
              "index_layout": {"face": [FACE_SLICE.start, FACE_SLICE.stop],
                               "hand_a": [HAND_A.start, HAND_A.stop], "hand_b": [HAND_B.start, HAND_B.stop]},
              "conf_gate": CONF, "frames_without_face_mask": empty_face,
              "frames_without_either_hand": empty_both,
              "share_without_face": round(empty_face / max(1, len(rows)), 4),
              "share_without_either_hand": round(empty_both / max(1, len(rows)), 4)}
    if ratio.size:
        report["face_hull_over_measured_width"] = {
            "n": int(ratio.size), "p05": round(float(np.percentile(ratio, 5)), 3),
            "p50": round(float(np.percentile(ratio, 50)), 3), "p95": round(float(np.percentile(ratio, 95)), 3)}
        p50 = float(np.percentile(ratio, 50))
        report["layout_guard"] = {"pass": bool(0.5 < p50 < 2.0), "p50": round(p50, 4),
                                  "meaning": "the 68-point hull width must agree with the detector's own "
                                             "face box; 0.14 (the previous run) meant points had collapsed"}
        report["face_guard_verdict"] = "PASS" if 0.5 < p50 < 2.0 else "FAIL"
    if hr.size:
        report["hand_center_to_nearest_wrist_over_body_diag"] = {
            "n": int(hr.size), "p50": round(float(np.percentile(hr, 50)), 4),
            "p95": round(float(np.percentile(hr, 95)), 4),
            "per_side_p95": {k: round(float(np.percentile(v, 95)), 4) for k, v in per_side.items()},
            "meaning": "a real hand ROI sits at the wrist, so this must be small; a wrong slice "
                       "would put the hull across the frame"}
        p95 = float(np.percentile(hr, 95))
        report["hand_guard_verdict"] = "PASS" if p95 < 0.25 else (
            f"FAIL p95={round(p95,3)} of the body diagonal -- the 92:113 / 113:134 windows do not "
            "sit at the wrists, so hand ROIs are NOT shipped as verified")
    (OUT / "detail_roi.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:2200])


main()
