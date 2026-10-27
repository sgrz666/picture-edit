"""CHAMP HQ-Pair Stage 2+3: DWPose (CPU onnx) + person-ROI quality metrics.

One decode pass yields both the numeric keypoints the plan wants and the
person-region sharpness numbers, because the ROI metrics are meaningless without
a body box (a sharp background hides a blurred person).

Writes manifests/frame_quality.jsonl and prints a validation table against the
four frames already read by eye in qc/000777.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DWROOT = Path("/home/shangguanrz/project/pic-edit/add/tools/dwpose_mimicmotion_official")
sys.path.insert(0, str(DWROOT))
from dwpose.wholebody import Wholebody  # noqa: E402

PILOT = Path("/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024")
DET = DWROOT / "models/yolox_l.onnx"
POSE = DWROOT / "models/dw-ll_ucoco_384.onnx"
BODY, FOOT, FACE, HAND = 18, 6, 68, 42
# dw-ll_ucoco_384 emits 134 = body18 + foot6 + face68 + hand42. The six foot points sit
# between body and face, so slicing face at BODY inflates face_width with toe positions.
FACE_AT, HAND_AT = BODY + FOOT, BODY + FOOT + FACE

# hand-checked from qc/000777.png: (sequence, frame_idx, label)
EYE_LABELS = [("000777", 224, "blur"), ("000777", 120, "blur"),
              ("000777", 176, "sharp"), ("000777", 264, "sharp")]


def roi_from_body(body_xy, body_sc, shape):
    """Convex hull of confident body joints, dilated, as a uint8 mask."""
    pts = body_xy[body_sc > 0.3]
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.float32))
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    return cv2.dilate(m, k).astype(np.uint8)


def roi_sharpness(gray, mask):
    """Sharpness restricted to the person, plus a two-scale gradient ratio.

    grad_ratio compares the |grad| energy a 1px operator sees against a 3px one:
    crisp edges keep a high ratio, motion blur evens the two out.
    """
    pix = mask > 0
    if pix.sum() < 500:
        return {}
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    g1 = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1))
    g3 = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    out = {
        "roi_lap_var": round(float(lap[pix].var()), 2),
        "roi_grad_mean": round(float(g1[pix].mean()), 2),
        "roi_px": int(pix.sum()),
    }
    out["roi_grad_ratio"] = round(float(g1[pix].mean() / (g3[pix].mean() + 1e-6)), 4)
    return out


def main():
    rows = [json.loads(l) for l in (PILOT / "manifests/frame_index.jsonl").read_text().splitlines() if l.strip()]
    print(f"frames to process: {len(rows)}", flush=True)
    est = Wholebody(str(DET), str(POSE), device="cpu")

    out = []
    for n, r in enumerate(rows):
        p = PILOT / "frames" / r["sequence_id"] / "rgb" / f"{r['frame_idx']:05d}.png"
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        try:
            kpts, scores = est(img)
        except Exception as e:
            print(f"  {r['sequence_id']} f{r['frame_idx']} DWPose failed: {type(e).__name__} {e}", flush=True)
            kpts, scores = np.zeros((0, 0, 2)), np.zeros((0, 0))

        rec = dict(r)
        nk = int(scores.shape[1]) if scores.ndim == 2 else 0
        rec["dwpose_kpt_count"] = nk
        rec["n_persons"] = int(scores.shape[0]) if scores.ndim == 2 else 0

        if rec["n_persons"] and nk >= BODY:
            body_sc = scores[:, :BODY]
            body_xy = kpts[:, :BODY, :]
            areas = []
            for i in range(rec["n_persons"]):
                pts = body_xy[i][body_sc[i] > 0.3]
                areas.append(float((pts.max(0)[0] - pts.min(0)[0]) * (pts.max(0)[1] - pts.min(0)[1])) if len(pts) > 2 else 0.0)
            mi = int(np.argmax(areas))
            rec["dwpose_kpts"] = [[round(float(x), 2), round(float(y), 2)]
                                  for x, y in kpts[mi][:BODY]]
            rec["dwpose_conf"] = [round(float(c), 4) for c in scores[mi][:BODY]]
            asset = PILOT / "assets" / r["sequence_id"] / "dwpose"
            asset.mkdir(parents=True, exist_ok=True)
            (asset / f"{r['frame_idx']:05d}.json").write_text(json.dumps({
                "candidate": kpts[mi].tolist(), "subset": scores[mi].tolist(),
                "n_persons": rec["n_persons"], "kpt_count": nk,
            }), encoding="utf-8")
            bx = body_xy[mi][body_sc[mi] > 0.3]
            rec["main_person"] = mi
            rec["body_mean_conf"] = round(float(body_sc[mi].mean()), 4)
            if len(bx) >= 2:
                x0, y0 = bx.min(0)
                x1, y1 = bx.max(0)
                rec["person_bbox"] = [round(float(x0), 1), round(float(y0), 1),
                                      round(float(x1 - x0), 1), round(float(y1 - y0), 1)]
                rec["person_height_px"] = int(round(float(y1 - y0)))
                rec["person_width_px"] = int(round(float(x1 - x0)))
                # truncation: main body box pinned to >=2 image borders
                touch = sum([x0 <= 2, y0 <= 2, x1 >= W - 3, y1 >= H - 3])
                rec["bbox_border_touch"] = int(touch)
                mask = roi_from_body(body_xy[mi], body_sc[mi], gray.shape)
                if mask is not None:
                    rec.update(roi_sharpness(gray, mask))
            if nk >= FACE_AT + FACE:
                fs = scores[mi][FACE_AT:FACE_AT + FACE]
                fx = kpts[mi][FACE_AT:FACE_AT + FACE, 0][fs > 0.3]
                fy = kpts[mi][FACE_AT:FACE_AT + FACE, 1][fs > 0.3]
                rec["face_mean_conf"] = round(float(fs.mean()), 4)
                rec["face_width_px"] = int(round(float(fx.max() - fx.min()))) if len(fx) >= 4 else 0
                # Invariant: the face block must sit on the head. Measured on 149 frames,
                # this fires on 85/149 with the face slice wrongly at index 18 (the six
                # foot points drag the chin to the floor) and 0/149 at the correct 24.
                # A median-distance version fired on neither and was a dead gate.
                nose_y = float(kpts[mi][0][1])
                hip_y = float(body_xy[mi][11][1]) if body_sc[mi][11] > 0.3 else None
                rec["face_layout_suspect"] = bool(
                    hip_y is not None and len(fy) >= 4 and float(fy.max()) > hip_y)
                if rec["face_layout_suspect"]:
                    rec["face_layout_nose_y"] = round(nose_y, 1)
                    rec["face_layout_hip_y"] = round(hip_y, 1)
            if nk >= HAND_AT + HAND:
                hs = scores[mi][HAND_AT:HAND_AT + HAND]
                rec["hand_mean_conf"] = round(float(hs.mean()), 4)
                # per-hand: 21 each, keep the better one as source-quality signal
                rec["hand_best_conf"] = round(float(max(hs[:21].mean(), hs[21:].mean())), 4)
            rec["person_area_share"] = round(float(sorted(areas)[-1] / (W * H)), 4) if areas else None
            rec["second_person_area_share"] = round(float(sorted(areas)[-2] / (W * H)), 4) if len(areas) > 1 else 0.0
        else:
            rec["person_height_px"] = 0
            rec["face_width_px"] = 0

        out.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(rows)} done", flush=True)

    (PILOT / "manifests" / "frame_quality.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")

    print(f"\nwrote {len(out)} rows")
    if out:
        print("kpt counts seen:", sorted({o.get("dwpose_kpt_count") for o in out}))
        print("n_persons histogram:", sorted(__import__("collections").Counter(o.get("n_persons", 0) for o in out).items()))
        for key in ("roi_lap_var", "roi_grad_ratio", "body_mean_conf", "face_mean_conf", "person_height_px"):
            vals = [o[key] for o in out if isinstance(o.get(key), (int, float))]
            if vals:
                print(f"{key}: p5={np.percentile(vals,5):.3f} p50={np.percentile(vals,50):.3f} "
                      f"p95={np.percentile(vals,95):.3f}")

    print("\n=== metric check against the 4 hand-read frames of 000777 ===")
    idx = {(o["sequence_id"], o["frame_idx"]): o for o in out}
    for seq, fi, label in EYE_LABELS:
        o = idx.get((seq, fi))
        if not o:
            print(f"{label:5s} f{fi}: MISSING")
            continue
        print(f"{label:5s} f{fi}: roi_lap={o.get('roi_lap_var')} grad_ratio={o.get('roi_grad_ratio')} "
              f"body_conf={o.get('body_mean_conf')} face_conf={o.get('face_mean_conf')} "
              f"face_w={o.get('face_width_px')} n_persons={o.get('n_persons')}")


if __name__ == "__main__":
    main()
